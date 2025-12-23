#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRL-MalGraph RL Environment

Reinforcement learning environment for adversarial malware generation
using semantic NOP insertion at the CFG basic block level.

Based on:
    - SRL paper: "Adversarial Deep Learning in Cyber Security"
    - MalGraph: Hierarchical GNN for Windows malware detection

Author: Md Ajwad Akil
Date: December 2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import copy
import json

from srl_malgraph_nop_mapping import SemanticNOPMapper


class SRLMalGraphEnvironment:
    """
    RL environment for SRL + MalGraph integration.
    
    State: ACFG with block_features (11-dim per block)
    Action: (block_selection_method, nop_type)
        - block_selection_method: index into top-k blocks from SortPooling
        - nop_type: index into semantic NOP list
    Reward: Score reduction (continuous) or binary bypass indicator
    """
    
    def __init__(
        self,
        malgraph_classifier,  # MalgraphServerFeature, DirectMalgraphClient, or adapter
        nop_mapper: SemanticNOPMapper,
        threshold: float = 0.14346,  # 100fpr: 0.14346, 1000fpr: 0.91276
        max_mutations: int = 50,
        top_k_blocks: int = 6,
        reward_type: str = 'continuous',  # 'continuous', 'binary', 'sparse'
        terminal_bonus: float = 10.0,
        sortpooling_method: str = 'l2_norm',  # 'l2_norm' or 'trainable'
        embedding_dim: int = 64  # For trainable attention
    ):
        """
        Initialize the SRL-MalGraph environment.
        
        Args:
            malgraph_classifier: Classifier with predict(acfg_json) → score
                                 Can be MalgraphServerFeature, DirectMalgraphClient,
                                 or SRLMalGraphClassifierAdapter
            nop_mapper: Semantic NOP mapper for feature increments
            threshold: Classification threshold for bypass (100fpr: 0.14346)
            max_mutations: Maximum number of mutations allowed
            top_k_blocks: Number of top important blocks to consider
            reward_type: Type of reward ('continuous', 'binary', 'sparse')
            terminal_bonus: Bonus reward for successful bypass
            sortpooling_method: Method for block importance ranking
                               'l2_norm': Use L2 norm of embeddings (non-trainable)
                               'trainable': Use learnable attention weights (trainable)
            embedding_dim: Dimension of node embeddings (for trainable method)
        """
        self.classifier = malgraph_classifier
        self.nop_mapper = nop_mapper
        self.threshold = threshold
        self.max_mutations = max_mutations
        self.top_k_blocks = top_k_blocks
        self.reward_type = reward_type
        self.terminal_bonus = terminal_bonus
        self.sortpooling_method = sortpooling_method
        self.embedding_dim = embedding_dim
        
        # Initialize trainable attention layer if needed
        if sortpooling_method == 'trainable':
            self.attention_layer = torch.nn.Linear(embedding_dim, 1)
            torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        else:
            self.attention_layer = None
        
        # Generate NOP action space
        self.nop_list = nop_mapper.generate_malguise_nop_list()
        self.num_nop_actions = len(self.nop_list)
        
        # State tracking
        self.original_acfg = None
        self.current_acfg = None
        self.current_score = None
        self.previous_score = None
        self.num_mutations = 0
        self.mutation_history = []
        self.important_blocks = []
        
        print(f"Environment initialized with {self.num_nop_actions} semantic NOPs")
    
    def reset(self, acfg_json: Dict) -> Dict:
        """
        Reset environment with new malware sample.
        
        Args:
            acfg_json: ACFG dictionary from IDA Pro extraction
        
        Returns:
            Initial state dictionary
        """
        # Deep copy to preserve original
        self.original_acfg = copy.deepcopy(acfg_json)
        self.current_acfg = copy.deepcopy(acfg_json)
        
        # Get initial classification score
        self.current_score = self.classifier.predict(self.current_acfg)
        self.previous_score = self.current_score
        print(f"Environment reset: Initial score = {self.current_score:.6f}")
        print(f"The classifier model is {self.classifier.classifier.model}")
        
        # Reset counters
        self.num_mutations = 0
        self.mutation_history = []
        
        # Compute block importance using SortPooling
        self.important_blocks = self._compute_block_importance()
        
        # Return initial state
        state = self._get_state()
        
        return state
    
    def _compute_block_importance(self) -> List[Tuple[int, int, float]]:
        """
        Compute importance scores for all basic blocks using MalGraph's embeddings.
        Dispatches to appropriate method based on sortpooling_method.
        
        Returns:
            List of (func_idx, block_idx, importance_score) sorted by importance
        """
        if self.sortpooling_method == 'trainable':
            return self._compute_block_importance_trainable()
        else:
            return self._compute_block_importance_l2_norm()
    
    def _compute_block_importance_l2_norm(self) -> List[Tuple[int, int, float]]:
        """
        Non-trainable SortPooling: Use L2 norm of MalGraph embeddings.
        
        Unlike DGCNN which concatenates layers and uses the last channel (designed as
        a sort score), MalGraph replaces layers at each step. Therefore, we use L2 norm
        across all embedding dimensions to measure node importance, similar to how
        attention mechanisms measure significance.
        
        Returns:
            List of (func_idx, block_idx, importance_score) sorted by importance
        """
        # Convert ACFG to PyTorch Geometric Data

        # Get CFG embeddings from MalGraph
        data = self.classifier.classifier.model.get_cfg_embedding(self.current_acfg)
        
        with torch.no_grad():
            # data is a PyTorch Geometric Batch object with:
            #   - data.x: [num_blocks, embedding_dim] - the actual embeddings
            #   - data.edge_index: [2, num_edges] - the graph edges
            #   - data.batch: [num_blocks] - batch assignment
            
            # Move to model device if needed
            data = data.to(self.classifier.classifier.model.device)
            
            # Compute importance as L2 norm of embeddings (magnitude across all dimensions)
            # This captures how "significant" each block is in the learned representation
            # data.x has shape [num_blocks, embedding_dim], we compute norm across dim=1 (embedding dimension)
            importance_scores = torch.norm(data.x, dim=1).cpu().numpy()
            
            # print(f"Block embeddings shape: {data.x.shape}")
            # print(f"Importance scores shape: {importance_scores.shape}")
            # print(f"Top 5 importance scores: {sorted(importance_scores, reverse=True)[:5]}")
        
        # Map scores back to (func_idx, block_idx)
        block_scores = []
        block_idx = 0
        for func_idx, acfg in enumerate(self.current_acfg['acfg_list']):
            num_blocks = acfg['block_number']
            for local_block_idx in range(num_blocks):
                score = importance_scores[block_idx]
                block_scores.append((func_idx, local_block_idx, score))
                block_idx += 1
        
        # Sort by importance (descending) and take top-k
        block_scores.sort(key=lambda x: x[2], reverse=True)

        # # print the sorted block scores
        # print("Top 10 important blocks (func_idx, block_idx, score):")
        # for i, (func_idx, block_idx, score) in enumerate(block_scores[:10]):
        #     print(f"  {i+1}. Function {func_idx}, Block {block_idx} (importance: {score:.4f})") 
        
        return block_scores[:self.top_k_blocks]
    
    def _compute_block_importance_trainable(self) -> List[Tuple[int, int, float]]:
        """
        Trainable SortPooling: Use learnable attention weights over embeddings.
        
        This allows the RL agent to learn which blocks are most important
        for successful mutations through backpropagation.
        
        Returns:
            List of (func_idx, block_idx, importance_score) sorted by importance
        """
        # Convert ACFG to PyTorch Geometric Data
        data = self._acfg_to_pytorch_data(self.current_acfg)
        
        # Get node embeddings from MalGraph's CFG-level GNN
        # Note: No torch.no_grad() here - we need gradients for training
        x = data.x  # Block features: [num_blocks, 11]
        edge_index = data.edge_index
        
        # Forward through MalGraph's CFG GNN layers
        x = self.classifier.model.forward_cfg_gnn_layers(data)
        
        # Apply attention layer: [num_blocks, embedding_dim] → [num_blocks, 1]
        attention_logits = self.attention_layer(x)  # [num_blocks, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits.squeeze(-1), dim=0)  # [num_blocks]
        
        # Importance scores are the attention weights
        importance_scores = attention_weights.detach().cpu().numpy()
        
        # Map scores back to (func_idx, block_idx)
        block_scores = []
        block_idx = 0
        for func_idx, acfg in enumerate(self.current_acfg['acfg_list']):
            num_blocks = acfg['block_number']
            for local_block_idx in range(num_blocks):
                score = importance_scores[block_idx]
                block_scores.append((func_idx, local_block_idx, score))
                block_idx += 1
        
        # Sort by importance (descending) and take top-k
        block_scores.sort(key=lambda x: x[2], reverse=True)
        
        return block_scores[:self.top_k_blocks]
    
    def _acfg_to_pytorch_data(self, acfg_json: Dict):
        """
        Convert ACFG JSON to PyTorch Geometric Data.
        
        This should match MalGraph's preprocessing in ModelPredForAttack.py
        """
        # This is a simplified version - you'll need to use the actual
        # conversion function from MalGraph
        from torch_geometric.data import Data
        
        all_block_features = []
        for acfg in acfg_json['acfg_list']:
            all_block_features.extend(acfg['block_features'])
        
        x = torch.tensor(all_block_features, dtype=torch.float32)
        
        # Build edge_index from block_edges
        # (Simplified - actual implementation more complex)
        edge_list = []
        offset = 0
        for acfg in acfg_json['acfg_list']:
            block_edges = acfg['block_edges']
            for src_idx, dst_indices in enumerate(block_edges[0]):
                for dst_idx in block_edges[1]:
                    edge_list.append([offset + src_idx, offset + dst_idx])
            offset += acfg['block_number']
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        data = Data(x=x, edge_index=edge_index)
        return data
    

    def get_current_state_embedding(self) -> torch.Tensor:
          # Get CFG embeddings from MalGraph
        data = self.classifier.classifier.model.get_cfg_embedding(self.current_acfg)
        data = data.to(self.classifier.classifier.model.device)
       
        return data.x  # Return block embeddings       
    
    def _get_state(self) -> Dict:
        """
        Get current state representation for DQN.
        
        Returns:
            State dictionary with:
                - acfg: Current ACFG
                - score: Current malware score
                - mutations: Number of mutations applied
                - important_blocks: Top-k important blocks
        """
        return {
            'acfg': self.current_acfg,
            'block_embeddings': self.get_current_state_embedding(),
            'score': self.current_score,
            'num_mutations': self.num_mutations,
            'important_blocks': self.important_blocks,
            'terminated': self._is_terminal()
        }
    
    def step_single_block(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one mutation step.
        
        Args:
            action: (block_selection_idx, nop_idx)
                - block_selection_idx: Index into important_blocks (0 to top_k-1)
                - nop_idx: Index into nop_list (0 to num_nop_actions-1)
        
        Returns:
            (next_state, reward, done, info)
        """


        block_selection_idx, nop_idx = action
        
        # Validate action
        if block_selection_idx >= len(self.important_blocks):
            # Invalid block selection
            return self._get_state(), -1.0, True, {'error': 'Invalid block index'}
        
        if nop_idx >= self.num_nop_actions:
            # Invalid NOP index
            return self._get_state(), -1.0, True, {'error': 'Invalid NOP index'}
        
        # Get target block
        func_idx, block_idx, importance = self.important_blocks[block_selection_idx]
        
        # Get NOP data
        nop_data = self.nop_list[nop_idx]
        nop_str = nop_data['nop_str']
        
        # Apply mutation using NOP mapper
        self._mutate_block(func_idx, block_idx, nop_str)
        
        # Update score
        self.previous_score = self.current_score
        self.current_score = self.classifier.predict(self.current_acfg)
        
        # Update counters
        self.num_mutations += 1
        self.mutation_history.append({
            'func_idx': func_idx,
            'block_idx': block_idx,
            'nop_idx': nop_idx,
            'nop_str': nop_data['nop_str'],
            'score_before': self.previous_score,
            'score_after': self.current_score
        })
        
        # Recompute important blocks (graph structure unchanged, but embeddings change)
        self.important_blocks = self._compute_block_importance()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = self._is_terminal()
        
        # Info dict
        info = {
            'score': self.current_score,
            'score_delta': self.previous_score - self.current_score,
            'num_mutations': self.num_mutations,
            'bypassed': self.current_score < self.threshold
        }
        
        return self._get_state(), reward, done, info
    


    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one mutation step.
        
        Args:
            action: nop_idx
                - nop_idx: Index into nop_list (0 to num_nop_actions-1)
        
        Returns:
            (next_state, reward, done, info)
        """

        nop_idx = action
        
        # Validate action
        # if top_k_important_blocks_indices > len(self.important_blocks):
        #     # Invalid block selection
        #     return self._get_state(), -1.0, True, {'error': 'Invalid block index'}
        
        if nop_idx >= self.num_nop_actions:
            # Invalid NOP index
            return self._get_state(), -1.0, True, {'error': 'Invalid NOP index'}
        
        # # Get target block
        # func_idx, block_idx, importance = self.important_blocks[block_selection_idx]
        
        # Get NOP data
        nop_data = self.nop_list[nop_idx]
        nop_str = nop_data['nop_str']

        #print("chosen nop_str: ", nop_str)

        # Store embeddings BEFORE mutation for verification
        embeddings_before = self.get_current_state_embedding().detach().cpu()

        for block_selection_idx in range(len(self.important_blocks)):
            func_idx, block_idx, importance = self.important_blocks[block_selection_idx]

            self._mutate_block(func_idx, block_idx, nop_str)
        
        # # Apply mutation using NOP mapper
        # self._mutate_block(func_idx, block_idx, nop_str)
        
        # Verify embeddings AFTER mutation have changed
        embeddings_after = self.get_current_state_embedding().detach().cpu()
        
        # Check if embeddings changed
        embeddings_diff = torch.norm(embeddings_after - embeddings_before)
        if embeddings_diff < 1e-6:
            print(f"  ⚠️  WARNING: Embeddings unchanged after mutation! Diff: {embeddings_diff:.8f}")
        # else:
        #     print(f"  ✓ Embeddings changed after mutation. L2 diff: {embeddings_diff:.6f}")
        
        # Update score
        self.previous_score = self.current_score
        self.current_score = self.classifier.predict(self.current_acfg)
        
        # Update counters
        self.num_mutations += 1
        self.mutation_history.append({
            'func_indices': [self.important_blocks[block_selection_idx][0] for block_selection_idx in range(len(self.important_blocks))],
            'block_indices': [self.important_blocks[block_selection_idx][1] for block_selection_idx in range(len(self.important_blocks))],
            'block_importances': [self.important_blocks[block_selection_idx][2] for block_selection_idx in range(len(self.important_blocks))],
            'nop_idx': nop_idx,
            'nop_str': nop_data['nop_str'],
            'score_before': self.previous_score,
            'score_after': self.current_score
        })
        
        # Recompute important blocks (graph structure unchanged, but embeddings change)
        self.important_blocks = self._compute_block_importance()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = self._is_terminal()
        
        # Info dict
        info = {
            'score': self.current_score,
            'score_delta': self.previous_score - self.current_score,
            'num_mutations': self.num_mutations,
            'bypassed': self.current_score < self.threshold
        }
        
        return self._get_state(), reward, done, info
    


    def _mutate_block(self, func_idx: int, block_idx: int, nop_str: str):
        """
        Apply semantic NOP to target basic block using the NOP mapper.
        
        Args:
            func_idx: Function index in acfg_list
            block_idx: Block index in block_features
            nop_str: Assembly string of the NOP to inject
        """
        # Get reference to block features (this is a list, modified in-place)
        # Python lists are mutable: modifying target_features modifies self.current_acfg
        target_features = self.current_acfg['acfg_list'][func_idx]['block_features'][block_idx]
        
        # Debug: Print before mutation
        features_before = target_features.copy()
        
        # Apply NOP using the mapper (modifies target_features in-place via element assignment)
        # The mapper does: target_features[i] += increment[i] for each i
        self.nop_mapper.apply_nop_to_block_features(target_features, nop_str)
        
        # Debug: Verify mutation actually happened
        features_after = target_features
        # if features_before != features_after:
        #     print(f"  ✓ Block F{func_idx}.B{block_idx} mutated: {features_before} → {features_after}")
        # else:
        #     print(f"  ✗ WARNING: Block F{func_idx}.B{block_idx} NOT mutated!")
        
        # Verify self.current_acfg was updated (should be same reference)
        assert self.current_acfg['acfg_list'][func_idx]['block_features'][block_idx] is target_features, \
            "ERROR: Block features reference mismatch!"
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on score change.
        
        Returns:
            Reward value
        """
        score_delta = self.previous_score - self.current_score
        
        if self.reward_type == 'continuous':
            # Reward proportional to score reduction
            reward = score_delta
            if self.current_score < self.threshold:
                reward += self.terminal_bonus
        
        elif self.reward_type == 'binary':
            # Binary reward for bypass
            reward = 1.0 if self.current_score < self.threshold else 0.0
        
        elif self.reward_type == 'sparse':
            # SRL paper's original: +1 if improved, 0 otherwise
            reward = 1.0 if score_delta > 0 else 0.0
            if self.current_score < self.threshold:
                reward += self.terminal_bonus
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        return reward
    
    def _is_terminal(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if terminal state reached
        """
        # Terminal if bypassed
        if self.current_score < self.threshold:
            return True
        
        # Terminal if max mutations reached
        if self.num_mutations >= self.max_mutations:
            return True
        
        return False
    
    def get_action_space_size(self) -> Tuple[int, int]:
        """
        Get dimensions of action space.
        
        Returns:
            (num_block_selections, num_nop_types)
        """
        return (self.top_k_blocks, self.num_nop_actions)
    
    def get_state_dim(self) -> int:
        """
        Get dimension of state representation for DQN.
        
        For simplicity, we use MalGraph's graph embeddings.
        """
        # This depends on how you represent state to DQN
        # Option 1: Use MalGraph's final embedding
        # Option 2: Use concatenated block features
        # Option 3: Use graph-level statistics
        
        # Placeholder: 128-dim embedding from MalGraph
        return 128
    
    def save_mutation_history(self, filepath: str):
        """Save mutation history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'original_score': self.previous_score,
                'final_score': self.current_score,
                'bypassed': self.current_score < self.threshold,
                'num_mutations': self.num_mutations,
                'mutations': self.mutation_history
            }, f, indent=2)
    
    def get_mutated_acfg(self) -> Dict:
        """Get current mutated ACFG."""
        return copy.deepcopy(self.current_acfg)


# Example usage
if __name__ == "__main__":
    # This is pseudocode - you need actual MalGraph classifier
    
    from srl_malgraph_nop_mapping import SemanticNOPMapper
    
    # Initialize components
    nop_mapper = SemanticNOPMapper()
    # malgraph_classifier = load_malgraph_classifier()  # Your classifier
    
    # Create environment
    # env = SRLMalGraphEnvironment(
    #     malgraph_classifier=malgraph_classifier,
    #     nop_mapper=nop_mapper,
    #     threshold=0.14346,  # 100fpr threshold
    #     max_mutations=50,
    #     top_k_blocks=10,
    #     reward_type='continuous'
    # )
    
    # # Load malware sample
    # with open('extracted_acfg.json', 'r') as f:
    #     acfg = json.load(f)
    
    # # Reset environment
    # state = env.reset(acfg)
    # print(f"Initial score: {state['score']}")
    
    # # Take action (block 0, NOP type 5)
    # next_state, reward, done, info = env.step((0, 5))
    # print(f"Reward: {reward}, New score: {info['score']}, Done: {done}")
    
    print("SRL-MalGraph Environment ready for DQN training!")
