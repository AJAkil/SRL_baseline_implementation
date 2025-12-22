#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRL-MalGraph DQN Agent with Trainable SortPooling

This module implements the complete DQN agent for adversarial malware generation,
including:
    1. Graph neural network encoder for CFG
    2. Trainable SortPooling layer for block importance ranking
    3. Q-network for action selection (block, NOP)
    4. Experience replay buffer
    5. Training procedures

Based on SRL paper's DGCNN architecture adapted for MalGraph's ACFG format.

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


class CFGGraphEncoder(nn.Module):
    """
    Graph encoder for CFG using message passing layers.
    Extracts node embeddings that will be used by SortPooling.
    """
    
    def __init__(
        self,
        input_dim: int = 11,  # MalGraph's 11-dim block features
        latent_dims: List[int] = [32, 32, 32],
        dropout: float = 0.1
    ):
        super(CFGGraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.num_layers = len(latent_dims)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        in_dim = input_dim
        for out_dim in latent_dims:
            self.conv_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        self.dropout = nn.Dropout(dropout)
        self.total_latent_dim = sum(latent_dims)
    
    def forward(
        self,
        node_features: torch.Tensor,  # [num_nodes, 11]
        edge_index: torch.Tensor,     # [2, num_edges]
        batch_indices: torch.Tensor   # [num_nodes] - which graph each node belongs to
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode CFG nodes with graph convolutions.
        
        Args:
            node_features: Node feature matrix [num_nodes, 11]
            edge_index: Edge connectivity [2, num_edges]
            batch_indices: Graph assignment for each node [num_nodes]
        
        Returns:
            node_embeddings: Concatenated embeddings [num_nodes, total_latent_dim]
            graph_sizes: Number of nodes in each graph
        """
        num_nodes = node_features.size(0)
        num_graphs = batch_indices.max().item() + 1
        
        # Compute adjacency matrix (sparse)
        adj = self._edge_index_to_adj(edge_index, num_nodes)
        
        # Graph convolution layers
        layer_outputs = []
        x = node_features
        
        for conv in self.conv_layers:
            # Message passing: (A + I) * X
            x_agg = torch.sparse.mm(adj, x) + x
            
            # Linear transformation
            x = conv(x_agg)
            
            # Activation
            x = torch.tanh(x)
            x = self.dropout(x)
            
            layer_outputs.append(x)
        
        # Concatenate all layer outputs
        node_embeddings = torch.cat(layer_outputs, dim=1)  # [num_nodes, total_latent_dim]
        
        # Compute graph sizes
        graph_sizes = []
        for i in range(num_graphs):
            mask = (batch_indices == i)
            graph_sizes.append(mask.sum().item())
        
        return node_embeddings, graph_sizes
    
    def _edge_index_to_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
        """
        Convert edge_index to sparse adjacency matrix with self-loops.
        
        Args:
            edge_index: [2, num_edges]
            num_nodes: Total number of nodes
        
        Returns:
            Sparse adjacency matrix (A + I)
        """
        # Add self-loops
        self_loop_index = torch.arange(num_nodes, device=edge_index.device)
        self_loops = torch.stack([self_loop_index, self_loop_index], dim=0)
        edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
        
        # Create sparse adjacency matrix
        num_edges = edge_index_with_loops.size(1)
        values = torch.ones(num_edges, device=edge_index.device)
        adj = torch.sparse.FloatTensor(
            edge_index_with_loops,
            values,
            torch.Size([num_nodes, num_nodes])
        )
        
        return adj


class SortPoolingLayer(nn.Module):
    """
    SortPooling layer from DGCNN (non-trainable operation).
    
    Ranks nodes by their importance (last feature dimension) and selects top-k.
    The ranking changes indirectly as the encoder (GNN) learns better embeddings.
    This layer itself has NO learnable parameters - it's just a sorting operation.
    """
    
    def __init__(self, k: int = 10):
        """
        Args:
            k: Number of top nodes to select
        """
        super(SortPoolingLayer, self).__init__()
        self.k = k
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # [num_nodes, embedding_dim]
        graph_sizes: List[int]          # Number of nodes per graph
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply SortPooling to select top-k important nodes per graph.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embedding_dim]
            graph_sizes: List of graph sizes
        
        Returns:
            pooled_graphs: [num_graphs, k, embedding_dim]
            topk_indices_list: List of top-k node indices per graph
        """
        num_graphs = len(graph_sizes)
        embedding_dim = node_embeddings.size(1)
        
        # Use last dimension as sorting criterion (continuous WL color)
        sort_channel = node_embeddings[:, -1]
        
        # Initialize output
        pooled_graphs = torch.zeros(
            num_graphs, self.k, embedding_dim,
            device=node_embeddings.device
        )
        
        topk_indices_list = []
        accum_count = 0
        
        for i in range(num_graphs):
            graph_size = graph_sizes[i]
            
            # Extract nodes for this graph
            to_sort = sort_channel[accum_count: accum_count + graph_size]
            
            # Determine actual k (may be less than self.k for small graphs)
            actual_k = min(self.k, graph_size)
            
            # Get top-k indices
            _, topk_indices = to_sort.topk(actual_k)
            topk_indices += accum_count  # Adjust to global indices
            
            topk_indices_list.append(topk_indices)
            
            # Select top-k nodes
            sorted_nodes = node_embeddings.index_select(0, topk_indices)
            
            # Pad if necessary
            if actual_k < self.k:
                padding = torch.zeros(
                    self.k - actual_k, embedding_dim,
                    device=node_embeddings.device
                )
                sorted_nodes = torch.cat([sorted_nodes, padding], dim=0)
            
            pooled_graphs[i] = sorted_nodes
            accum_count += graph_size
        
        return pooled_graphs, topk_indices_list


class QNetwork(nn.Module):
    """
    Q-Network that takes sorted graph embeddings and outputs Q-values for actions.
    
    Action space: (block_selection_idx, nop_idx)
    - block_selection_idx: Which of the top-k blocks to mutate (0 to k-1)
    - nop_idx: Which semantic NOP to insert (0 to num_nops-1)
    """
    
    def __init__(
        self,
        k: int,                    # Number of top blocks
        embedding_dim: int,        # Embedding dimension per node
        num_nops: int,            # Number of semantic NOP types
        hidden_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.num_nops = num_nops
        self.action_dim = k * num_nops  # Total action space
        
        # Flatten sorted graph: [k, embedding_dim] -> [k * embedding_dim]
        input_dim = k * embedding_dim
        
        # 1D convolution over sorted nodes (like DGCNN)
        self.conv1d_1 = nn.Conv1d(1, 16, kernel_size=embedding_dim, stride=embedding_dim)
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(k, embedding_dim)
        
        # MLP for Q-values
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def _get_conv_output_size(self, k, embedding_dim):
        """Calculate output size after conv layers."""
        # After conv1d_1: (k * embedding_dim) -> k
        size = k
        # After conv1d_2: k -> (k - 5 + 1) = k - 4
        size = size - 4
        # After multiplying by channels
        size = size * 32
        return size
    
    def forward(self, pooled_graphs: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Args:
            pooled_graphs: [batch_size, k, embedding_dim]
        
        Returns:
            q_values: [batch_size, action_dim]
        """
        batch_size = pooled_graphs.size(0)
        
        # Flatten: [batch_size, k, embedding_dim] -> [batch_size, 1, k * embedding_dim]
        x = pooled_graphs.view(batch_size, 1, -1)
        
        # 1D convolutions
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc3(x)
        
        return q_values


class DQNAgent:
    """
    Complete DQN agent for SRL-MalGraph attack.
    
    Combines:
        1. CFG encoder (GNN)
        2. SortPooling layer (trainable)
        3. Q-network (action selection)
        4. Experience replay
        5. Target network for stable training
    """
    
    def __init__(
        self,
        num_nops: int,
        k: int = 10,
        input_dim: int = 11,
        latent_dims: List[int] = [32, 32, 32],
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
        Initialize DQN agent.
        
        Args:
            num_nops: Number of semantic NOP types
            k: Number of top blocks to consider
            input_dim: Input feature dimension (11 for MalGraph)
            latent_dims: GNN layer dimensions
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
        self.k = k
        self.num_nops = num_nops
        self.action_dim = k * num_nops
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Build networks
        embedding_dim = sum(latent_dims)
        
        self.encoder = CFGGraphEncoder(input_dim, latent_dims).to(device)
        self.sortpool = SortPoolingLayer(k).to(device)
        self.q_network = QNetwork(k, embedding_dim, num_nops, hidden_dim).to(device)
        
        # Target network (for stable training)
        self.target_encoder = CFGGraphEncoder(input_dim, latent_dims).to(device)
        self.target_sortpool = SortPoolingLayer(k).to(device)
        self.target_q_network = QNetwork(k, embedding_dim, num_nops, hidden_dim).to(device)
        
        self._update_target_network()
        
        # Optimizer (SortPooling has no parameters, so only encoder + q_network)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.q_network.parameters()),
            lr=lr
        )
        
        # Replay buffer
        self.memory = deque(maxlen=memory_capacity)
        
        # Training stats
        self.steps_done = 0
        self.learn_step_counter = 0
        
        print(f"DQN Agent initialized on {device}")
        print(f"Action space: {k} blocks × {num_nops} NOPs = {self.action_dim} actions")
    
    def _update_target_network(self):
        """Copy weights from training networks to target networks."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_sortpool.load_state_dict(self.sortpool.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def _get_epsilon(self) -> float:
        """Compute current epsilon for epsilon-greedy."""
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
              np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        return eps
    
    def encode_state(self, state_dict: Dict) -> torch.Tensor:
        """
        Encode state (ACFG) into sorted node embeddings.
        
        Args:
            state_dict: State from environment with 'acfg', 'score', etc.
        
        Returns:
            pooled_graph: [1, k, embedding_dim]
        """
        acfg = state_dict['acfg']
        
        # Convert ACFG to PyTorch tensors
        node_features, edge_index, batch_indices = self._acfg_to_tensors(acfg)
        
        # Encode with GNN
        node_embeddings, graph_sizes = self.encoder(
            node_features, edge_index, batch_indices
        )
        
        # Apply SortPooling
        pooled_graph, _ = self.sortpool(node_embeddings, graph_sizes)
        
        return pooled_graph
    
    def _acfg_to_tensors(self, acfg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert ACFG JSON to PyTorch tensors.
        
        Returns:
            node_features: [num_nodes, 11]
            edge_index: [2, num_edges]
            batch_indices: [num_nodes]
        """
        all_features = []
        all_edges = []
        batch_indices = []
        node_offset = 0
        
        for func_idx, acfg_func in enumerate(acfg['acfg_list']):
            # Node features
            block_features = acfg_func['block_features']
            all_features.extend(block_features)
            
            # Batch indices (all nodes in this function belong to same graph)
            num_blocks = len(block_features)
            batch_indices.extend([func_idx] * num_blocks)
            
            # Edges
            block_edges = acfg_func['block_edges']
            src_indices = block_edges[0]
            dst_indices_list = block_edges[1]
            
            for src_idx, dst_indices in enumerate(zip(src_indices, dst_indices_list)):
                for dst_idx in dst_indices[1]:  # dst_indices[1] is the list
                    all_edges.append([node_offset + src_idx, node_offset + dst_idx])
            
            node_offset += num_blocks
        
        # Convert to tensors
        node_features = torch.tensor(all_features, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(all_edges, dtype=torch.long, device=self.device).t().contiguous()
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        
        return node_features, edge_index, batch_indices
    
    def select_action(self, state_dict: Dict, explore: bool = True) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_dict: Current state
            explore: Whether to use epsilon-greedy (False for evaluation)
        
        Returns:
            (block_selection_idx, nop_idx)
        """
        eps = self._get_epsilon() if explore else 0.0
        self.steps_done += 1
        
        if random.random() < eps:
            # Random action
            block_idx = random.randint(0, self.k - 1)
            nop_idx = random.randint(0, self.num_nops - 1)
        else:
            # Greedy action
            with torch.no_grad():
                pooled_graph = self.encode_state(state_dict)
                q_values = self.q_network(pooled_graph)
                action_idx = q_values.argmax().item()
                
                block_idx = action_idx // self.num_nops
                nop_idx = action_idx % self.num_nops
        
        return (block_idx, nop_idx)
    
    def store_experience(
        self,
        state: Dict,
        action: Tuple[int, int],
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
        
        # Encode states
        states_pooled = torch.cat([
            self.encode_state(exp.state) for exp in batch
        ], dim=0)  # [batch_size, k, embedding_dim]
        
        next_states_pooled = torch.cat([
            self.encode_state(exp.next_state) for exp in batch
        ], dim=0)
        
        # Prepare tensors
        actions = torch.tensor([
            exp.action[0] * self.num_nops + exp.action[1]
            for exp in batch
        ], dtype=torch.long, device=self.device)
        
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)
        
        # Compute Q(s, a)
        q_values = self.q_network(states_pooled)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states_pooled)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.q_network.parameters()),
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
            'encoder': self.encoder.state_dict(),
            'sortpool': self.sortpool.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_sortpool': self.target_sortpool.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'learn_step_counter': self.learn_step_counter
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.sortpool.load_state_dict(checkpoint['sortpool'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_sortpool.load_state_dict(checkpoint['target_sortpool'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    # Test agent initialization
    agent = DQNAgent(
        num_nops=200,  # MalGuise has ~200 NOPs
        k=10,
        input_dim=11,
        latent_dims=[32, 32, 32],
        batch_size=16
    )
    
    print(f"\nAgent architecture:")
    print(f"  Encoder (GNN): {sum(p.numel() for p in agent.encoder.parameters())} parameters")
    print(f"  SortPooling: 0 parameters (non-trainable sorting operation)")
    print(f"  Q-Network: {sum(p.numel() for p in agent.q_network.parameters())} parameters")
    print(f"  Total trainable: {sum(p.numel() for p in list(agent.encoder.parameters()) + list(agent.q_network.parameters()))} parameters")
