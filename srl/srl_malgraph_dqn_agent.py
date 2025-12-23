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
        hidden_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nops = num_nops
        self.action_dim = num_nops  # Action is just NOP selection
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_nops)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, block_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each NOP type.
        
        Args:
            block_embeddings: [num_blocks, embedding_dim] or [batch, num_blocks, embedding_dim]
                             Embeddings from MalGraph's CFG encoder
        
        Returns:
            q_values: [num_nops] or [batch, num_nops]
        """
        # Global mean pooling over all blocks to get graph-level representation
        if len(block_embeddings.shape) == 2:
            # Single graph: [num_blocks, embedding_dim] → [embedding_dim]
            x = block_embeddings.mean(dim=0)
        else:
            # Batch: [batch, num_blocks, embedding_dim] → [batch, embedding_dim]
            x = block_embeddings.mean(dim=1)
        
        # MLP to predict Q-values for each NOP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
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
        self.q_network = QNetwork(embedding_dim, num_nops, hidden_dim).to(device)
        self.target_q_network = QNetwork(embedding_dim, num_nops, hidden_dim).to(device)
        
        self._update_target_network()
        
        # Optimizer
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
    
    def get_embeddings_from_state(self, state_dict: Dict) -> torch.Tensor:
        """
        Extract block embeddings from state.
        Embeddings are already computed by the environment using MalGraph.
        
        Args:
            state_dict: State from environment with 'block_embeddings'
        
        Returns:
            embeddings: [num_blocks, embedding_dim]
        """
        # The environment should provide embeddings in the state
        # If not, we need to call the environment's embedding method
        if 'block_embeddings' in state_dict:
            embeddings = state_dict['block_embeddings']
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            return embeddings
        else:
            raise ValueError("State must contain 'block_embeddings' from environment")
    
    def select_action(self, state_dict: Dict, explore: bool = True) -> int:
        """
        Select action (NOP index) using epsilon-greedy policy.
        
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
                embeddings = self.get_embeddings_from_state(state_dict)
                q_values = self.q_network(embeddings)
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
        """Store experience in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if enough experiences available).
        
        Returns:
            Loss value or None if not enough experiences
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Get embeddings from states
        states_embeddings = torch.stack([
            self.get_embeddings_from_state(exp.state) for exp in batch
        ], dim=0)  # [batch_size, num_blocks, embedding_dim]
        
        next_states_embeddings = torch.stack([
            self.get_embeddings_from_state(exp.next_state) for exp in batch
        ], dim=0)
        
        # Prepare tensors
        actions = torch.tensor([
            exp.action for exp in batch
        ], dtype=torch.long, device=self.device)
        
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)
        
        # Compute Q(s, a)
        q_values = self.q_network(states_embeddings)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states_embeddings)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=10.0
        )
        self.optimizer.step()
        
        # Update target network
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
