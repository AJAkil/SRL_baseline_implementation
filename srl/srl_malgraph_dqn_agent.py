#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified SRL-MalGraph DQN Agent

This module implements the DQN agent for adversarial malware generation.
Uses MalGraph's pre-computed CFG embeddings directly - no custom encoder needed.

Architecture:
    1. State: CFG block embeddings from MalGraph (num_blocks, 200)
    2. Block selection: Top-k via L2 norm (done in environment)
    3. Q-Network: Selects which NOP to insert into ALL top-k blocks
    4. Action: Single NOP index (0 to num_nops-1)
    5. Experience replay buffer

Based on SRL paper adapted for MalGraph's embedding format.

Author: Md Ajwad Akil
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import deque, namedtuple

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Note: CFGGraphEncoder and SortPoolingLayer removed
# These are handled by the environment using MalGraph's internal model


class QNetwork(nn.Module):
    """
    Simple Q-Network for NOP selection.
    
    Takes MalGraph's CFG block embeddings and outputs Q-values for each NOP type.
    The selected NOP is inserted into ALL top-k blocks by the environment.
    
    Action space: Single NOP index (0 to num_nops-1)
    """
    
    def __init__(
        self,
        embedding_dim: int = 200,  # MalGraph's block embedding dimension
        num_nops: int = 20,        # Number of semantic NOP types
        hidden_dim: int = 128,
        max_blocks: int = 1250     # Maximum blocks to process (subsample if needed)
    ):
        super(QNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nops = num_nops
        self.action_dim = num_nops  # Action is just NOP selection
        self.max_blocks = max_blocks
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_nops)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each NOP type from POOLED embeddings.
        
        Args:
            pooled_embeddings: [embedding_dim] or [batch, embedding_dim]
                              Already mean-pooled CFG embeddings
        
        Returns:
            q_values: [num_nops] or [batch, num_nops]
                     Q-value for each NOP action
        """
        # Input is already pooled to fixed size [embedding_dim] or [batch, embedding_dim]
        x = pooled_embeddings
        
        # MLP to predict Q-values for each NOP
        # [embedding_dim] -> [128]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # [128] -> [128]
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # [128] -> [num_nops]
        q_values = self.fc3(x)
        
        return q_values


class  SimplifiedDQNAgent:
    """
    Simplified DQN agent for SRL-MalGraph attack.
    
    Uses MalGraph's pre-computed embeddings from the environment.
    
    Components:
        1. Q-network (NOP selection)
        2. Experience replay
        3. Target network for stable training
    
    Note: Block importance ranking and CFG encoding handled by environment.
    """
    
    def __init__(
        self,
        num_nops: int,
        embedding_dim: int = 200,  # MalGraph's embedding dimension
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 500,
        target_update_freq: int = 10,
        memory_capacity: int = 1000,
        batch_size: int = 32,
        max_blocks: int = 1250,  # Max blocks to process
        optimizer_type: str = 'adam',  # 'adam' or 'rmsprop'
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize simplified DQN agent.
        
        Args:
            num_nops: Number of semantic NOP types
            embedding_dim: MalGraph's block embedding dimension (default: 200)
            hidden_dim: Q-network hidden dimension
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay steps
            target_update_freq: Target network update frequency
            memory_capacity: Replay buffer size
            batch_size: Training batch size
            optimizer_type: 'adam' or 'rmsprop' (SRL paper uses rmsprop)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.num_nops = num_nops
        self.action_dim = num_nops  # Action is just NOP selection
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        
        # Build Q-networks
        self.q_network = QNetwork(embedding_dim, num_nops, hidden_dim, max_blocks).to(device)
        self.target_q_network = QNetwork(embedding_dim, num_nops, hidden_dim, max_blocks).to(device)
        
        self._update_target_network()
        
        # Optimizer (SRL paper uses RMSProp)
        if optimizer_type.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.q_network.parameters(),
                lr=lr
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=lr
            )
        
        # Replay buffer
        self.memory = deque(maxlen=memory_capacity)
        
        # Training stats
        self.steps_done = 0
        self.learn_step_counter = 0
        
        print(f"Simplified DQN Agent initialized on {device}")
        print(f"Optimizer: {optimizer_type.upper()}, Batch size: {batch_size}, Memory: {memory_capacity}")
        print(f"Action space: {num_nops} NOPs")
        print(f"Embedding dimension: {embedding_dim}")
    
    def _update_target_network(self):
        """Copy weights from training network to target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def _get_epsilon(self) -> float:
        """Compute current epsilon for epsilon-greedy."""
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
              np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        return eps
    
    def get_embeddings_from_state(self, state_dict: Dict, detach: bool = False) -> torch.Tensor:
        """
        Extract block embeddings from state.
        Embeddings are already computed by the environment using MalGraph.
        
        Args:
            state_dict: State from environment with 'block_embeddings'
            detach: Whether to detach from computation graph (for training)
        
        Returns:
            embeddings: [num_blocks, embedding_dim]
        """
        # The environment should provide embeddings in the state
        # If not, we need to call the environment's embedding method
        if 'block_embeddings' in state_dict:
            embeddings = state_dict['block_embeddings']
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            else:
                embeddings = embeddings.to(self.device)
            
            # Detach from MalGraph's computation graph when training Q-network
            if detach:
                embeddings = embeddings.detach()
            
            return embeddings
        else:
            raise ValueError("State must contain 'block_embeddings' from environment")
    
    def select_action(self, state_dict: Dict, explore: bool = True) -> int:
        """
        Select action (NOP index) using epsilon-greedy policy.
        
        NOTE: This is for MDP action selection - uses CURRENT state from environment.
        Completely separate from replay buffer (which is only for Q-network training).
        
        Args:
            state_dict: Current state with block embeddings
            explore: Whether to use epsilon-greedy (False for evaluation)
        
        Returns:
            nop_idx: Index of NOP to insert (0 to num_nops-1)
        """
        eps = self._get_epsilon() if explore else 0.0
        self.steps_done += 1
        
        if random.random() < eps:
            # Random NOP
            nop_idx = random.randint(0, self.num_nops - 1)
        else:
            # Greedy NOP selection
            with torch.no_grad():
                # Get raw embeddings from environment: [num_blocks, embedding_dim]
                embeddings = self.get_embeddings_from_state(state_dict)
                # Pool to fixed size: [num_blocks, embedding_dim] -> [embedding_dim]
                pooled = embeddings.mean(dim=0)
                # Forward through Q-network: [embedding_dim] -> [num_nops]
                q_values = self.q_network(pooled)
                nop_idx = q_values.argmax().item()
        
        return nop_idx
    
    def store_experience(
        self,
        state: Dict,
        action: int,
        reward: float,
        next_state: Dict,
        done: bool
    ):
        """
        Store experience in replay buffer with POOLED embeddings.
        
        This fixes batching: instead of storing variable-length [num_blocks, embedding_dim],
        we pool to fixed-size [embedding_dim] so all experiences can be batched together.
        
        SRL paper: Drop negative rewards with 50% probability to focus on successful mutations.
        
        NOTE: This is ONLY for Q-network training, not for MDP progression.
        The environment still uses raw embeddings for block importance computation.
        """
        # SRL paper: Drop negative reward experiences with 50% probability
        if reward < 0 and random.random() < 0.5:
            return  # Don't store this experience
        
        # Extract raw embeddings: [num_blocks, embedding_dim]
        state_embeddings = self.get_embeddings_from_state(state, detach=True)
        next_state_embeddings = self.get_embeddings_from_state(next_state, detach=True)
        
        # Pool to fixed size: [num_blocks, embedding_dim] -> [embedding_dim]
        state_pooled = state_embeddings.mean(dim=0).cpu()  # Store on CPU to save GPU memory
        next_state_pooled = next_state_embeddings.mean(dim=0).cpu()
        
        # Store pooled embeddings (fixed size) instead of raw embeddings (variable size)
        self.memory.append(Experience(state_pooled, action, reward, next_state_pooled, done))
    
    def clear_memory(self):
        """Clear replay buffer. Use when switching to a different malware sample."""
        self.memory.clear()
        print(f"Replay buffer cleared")
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experiences from replay buffer.
        
        NOTE: This is ONLY for Q-network training, not for MDP state transitions.
        The buffer stores past experiences (s, a, r, s') which may be from many episodes ago.
        
        Returns:
            Loss value or None if not enough experiences
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random minibatch from replay buffer
        # (decorrelates experiences for stable training)
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract pooled embeddings from experiences
        # Each exp.state is [embedding_dim] (already pooled during storage)
        states = torch.stack([exp.state for exp in batch]).to(self.device)  # [batch, embedding_dim]
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)  # [batch, embedding_dim]
        
        # Extract other components
        # actions: [batch] - NOP index for each experience
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long, device=self.device)
        # rewards: [batch] - reward for each transition
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        # dones: [batch] - whether episode terminated
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)
        
        # ========== Compute Q(s, a) for actions taken ==========
        # Forward through Q-network: [batch, embedding_dim] -> [batch, num_nops]
        q_values_all = self.q_network(states)  # Q-values for ALL actions
        
        # Select Q-values for actions that were actually taken
        # q_values_all: [batch, num_nops] - Q-values for ALL actions
        # actions: [batch] - which action was taken in each experience
        # We need to index q_values_all to get Q(s, a) for the specific action taken
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch]
        # ========== Compute target Q-values: r + γ max_a' Q(s', a') ==========
        with torch.no_grad():
            # Forward through target network: [batch, embedding_dim] -> [batch, num_nops]
            next_q_values_all = self.target_q_network(next_states)
            # Get max Q-value for next state: [batch, num_nops] -> [batch]
            max_next_q = next_q_values_all.max(1)[0]  # max over action dimension
            # Bellman target: r + γ * max_a' Q(s', a') if not done, else just r
            target_q = rewards + (1 - dones) * self.gamma * max_next_q  # [batch]
        
        # ========== Compute loss and optimize ==========
        # MSE loss: (Q(s,a) - target)^2
        loss = F.mse_loss(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=10.0
        )
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss.item()
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'learn_step_counter': self.learn_step_counter,
            'num_nops': self.num_nops,
            'embedding_dim': self.embedding_dim
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    # Test agent initialization
    agent = SimplifiedDQNAgent(
        num_nops=20,  # Example: 20 semantic NOPs
        embedding_dim=200,  # MalGraph's embedding dimension
        batch_size=16
    )
    
    print(f"\nAgent architecture:")
    print(f"  Q-Network: {sum(p.numel() for p in agent.q_network.parameters())} parameters")
    print(f"  Total trainable: {sum(p.numel() for p in agent.q_network.parameters())} parameters")
    
    # Test forward pass with dummy embeddings
    dummy_embeddings = torch.randn(15, 200).to(agent.device)  # 15 blocks, 200-dim embeddings
    q_values = agent.q_network(dummy_embeddings)
    print(f"\n  Q-values shape: {q_values.shape}  (expected: [{agent.num_nops}])")
    print(f"  Selected NOP: {q_values.argmax().item()}")
