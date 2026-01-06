#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Example: Using Hybrid NOP Distribution in SRL Training

This shows how to integrate the corrected NOP frequency analysis
into your existing SRL-MalGraph RL training pipeline.
"""

import numpy as np
from typing import Dict, List
from nop_frequency_analyzer import SemanticNOPFrequencyAnalyzer


class HybridNOPSampler:
    """
    Wrapper for sampling NOPs according to hybrid distribution during RL training.
    
    Integrates with your existing SimplifiedDQNAgent and SRLMalGraphEnvironment.
    """
    
    def __init__(self, use_hybrid: bool = True, method: str = 'geometric'):
        """
        Args:
            use_hybrid: If True, use hybrid distribution; if False, uniform sampling
            method: 'geometric' or 'arithmetic'
        """
        self.use_hybrid = use_hybrid
        self.method = method
        self.analyzer = SemanticNOPFrequencyAnalyzer()
        self.hybrid_distribution = None
        self.opcode_to_nop_idx = {}  # Map opcode types to NOP indices (0-27)
        
    
    def load_distributions_from_logs(self, log_files: Dict[str, str]):
        """
        Load and analyze log files from multiple classifiers.
        
        Args:
            log_files: {classifier_name: log_file_path}
        
        Example:
            sampler.load_distributions_from_logs({
                'MalConv': 'logs/malconv_bypass.json',
                'MalGraph_100fpr': 'logs/malgraph_100fpr_bypass.json',
                'EMBER': 'logs/ember_bypass.json'
            })
        """
        print(f"Loading distributions from {len(log_files)} classifiers...")
        
        for clf_name, log_path in log_files.items():
            print(f"  Processing: {clf_name}")
            self.analyzer.parse_single_log_file(log_path, clf_name)
        
        # Compute hybrid distribution
        if self.method == 'geometric':
            self.hybrid_distribution = self.analyzer.compute_hybrid_distribution_geometric()
        else:
            self.hybrid_distribution = self.analyzer.compute_hybrid_distribution_arithmetic()
        
        print(f"\n✅ Loaded hybrid distribution ({self.method} mean)")
        print(f"   Top 5 opcodes:")
        for opcode, prob in sorted(self.hybrid_distribution.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:5]:
            print(f"     {opcode:15s}: {prob:.4f} ({prob*100:5.2f}%)")
    
    
    def set_opcode_to_nop_mapping(self, mapping: Dict[str, int]):
        """
        Set the mapping from opcode types to your 28 NOP indices.
        
        Args:
            mapping: {opcode_type: nop_idx}
        
        Example:
            mapping = {
                'nop': 0,
                'add/sub': 1,
                'lea': 2,
                ...
                'mov': 27
            }
        """
        self.opcode_to_nop_idx = mapping
        print(f"✅ Loaded {len(mapping)} opcode-to-NOP mappings")
    
    
    def sample_nop_action(self) -> int:
        """
        Sample a NOP action (index 0-27) for the RL agent.
        
        Returns:
            NOP index (0-27) according to hybrid distribution or uniform
        
        Usage in RL training:
            # During agent.select_action() or epsilon-greedy exploration:
            if random.random() < epsilon:
                action = sampler.sample_nop_action()  # Use hybrid sampling
            else:
                action = agent.select_action(state)   # Use learned policy
        """
        if not self.use_hybrid or self.hybrid_distribution is None:
            # Fallback to uniform sampling
            return np.random.randint(0, 28)
        
        # Sample opcode type from hybrid distribution
        opcode_type = self.analyzer.sample_nops(self.hybrid_distribution, n_samples=1)[0]
        
        # Map to NOP index
        nop_idx = self.opcode_to_nop_idx.get(opcode_type, np.random.randint(0, 28))
        
        return nop_idx
    
    
    def get_nop_probabilities(self) -> np.ndarray:
        """
        Get probability array for all 28 NOPs (useful for weighted sampling).
        
        Returns:
            Array of shape (28,) with probabilities
        """
        if not self.use_hybrid or self.hybrid_distribution is None:
            # Uniform distribution
            return np.ones(28) / 28
        
        probs = np.zeros(28)
        for opcode_type, prob in self.hybrid_distribution.items():
            nop_idx = self.opcode_to_nop_idx.get(opcode_type, -1)
            if nop_idx >= 0:
                probs[nop_idx] = prob
        
        # Normalize (in case some NOPs weren't mapped)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(28) / 28
        
        return probs


def create_opcode_to_nop_mapping() -> Dict[str, int]:
    """
    Create the mapping from opcode types to your 28 NOP indices.
    
    Based on srl_malgraph_nop_mapping.py generate_malguise_nop_list()
    
    Returns:
        Dictionary mapping opcode types to NOP indices (0-27)
    """
    # This should match the 28 NOPs from SRL paper Table 5
    # Order matches generate_malguise_nop_list() output
    mapping = {
        'nop': 0,           # ID 1: NOP
        'add/sub': 1,       # ID 2-3: SUB, ADD (combined as add/sub)
        'lea': 3,           # ID 4: LEA
        'test': 4,          # ID 5: TEST
        'cmp': 5,           # ID 6: CMP
        'and': 6,           # ID 7: AND
        'or': 7,            # ID 8: OR
        'xor': 8,           # ID 9: XOR
        'mov': 9,           # ID 10: MOV
        'push/pop': 10,     # ID 11: PUSH,POP
        # ... continue for all 28 NOPs
        # See srl_malgraph_nop_mapping.py for complete list
    }
    
    # For now, return partial mapping
    # YOU SHOULD COMPLETE THIS based on your actual 28 NOPs
    return mapping


# ============================================================================
# INTEGRATION EXAMPLE: Modified Training Loop
# ============================================================================

def example_training_with_hybrid_nops():
    """
    Example showing how to modify your existing training loop
    to use hybrid NOP sampling.
    """
    
    # 1. Initialize hybrid NOP sampler
    print("="*80)
    print("STEP 1: Initialize Hybrid NOP Sampler")
    print("="*80)
    
    sampler = HybridNOPSampler(
        use_hybrid=True,      # Set to False for baseline comparison
        method='geometric'    # Or 'arithmetic'
    )
    
    # 2. Load distributions from your experiments
    # IMPORTANT: These should be from SUCCESSFUL bypasses!
    sampler.load_distributions_from_logs({
        'MalConv': 'logs/malconv_successful_bypasses.json',
        'MalGraph_100fpr': 'logs/malgraph_100fpr_successful.json',
        # Add more classifiers as needed
    })
    
    # 3. Set opcode-to-NOP index mapping
    opcode_mapping = create_opcode_to_nop_mapping()
    sampler.set_opcode_to_nop_mapping(opcode_mapping)
    
    # 4. Modified training loop
    print("\n" + "="*80)
    print("STEP 2: Modified Training Loop")
    print("="*80)
    
    # Your existing imports and setup
    # from srl_malgraph_training import SimplifiedDQNAgent, ...
    
    # PSEUDO-CODE (replace with your actual training loop)
    """
    agent = SimplifiedDQNAgent(state_dim=11, action_dim=28)
    env = SRLMalGraphEnvironment(...)
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        
        for step in range(MAX_STEPS):
            # MODIFIED: Epsilon-greedy with hybrid sampling
            if random.random() < epsilon:
                # OLD: action = random.randint(0, 27)
                # NEW: Sample from hybrid distribution
                action = sampler.sample_nop_action()
            else:
                # Use learned policy
                action = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > BATCH_SIZE:
                agent.train_step()
            
            state = next_state
            if done:
                break
    """
    
    print("\n✅ Training complete with hybrid NOP sampling!")
    
    # 5. Compare results
    print("\n" + "="*80)
    print("STEP 3: Compare Uniform vs Hybrid Sampling")
    print("="*80)
    
    # You should run two experiments:
    # Experiment A: use_hybrid=False (baseline, uniform sampling)
    # Experiment B: use_hybrid=True (hybrid geometric mean)
    # Then compare ASR, average mutations, etc.
    
    print("""
    Suggested Comparison:
    
    Baseline (Uniform):
        - ASR: ??.??%
        - Avg Mutations: ??
        - Training Time: ??
    
    Hybrid (Geometric):
        - ASR: ??.??%  (Should be HIGHER)
        - Avg Mutations: ??  (Should be LOWER or EQUAL)
        - Training Time: ??  (Should be SIMILAR)
    
    This validates the paper's hypothesis!
    """)


if __name__ == "__main__":
    # Run the example
    example_training_with_hybrid_nops()
    
    print("\n" + "="*80)
    print("INTEGRATION STEPS:")
    print("="*80)
    print("""
    1. Run baseline experiments to generate bypass logs
    2. Use SemanticNOPFrequencyAnalyzer to compute hybrid distributions
    3. Integrate HybridNOPSampler into your training loop
    4. Run comparative experiments (uniform vs hybrid)
    5. Report results in your paper!
    
    Files you'll need to modify:
        - srl_malgraph_training.py: Add HybridNOPSampler
        - srl_malgraph_environment.py: Maybe add sampling method parameter
        - evaluate_trained_agent.py: Compare results
    """)
