#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration example showing how to use SRL-MalGraph with your existing MalGraph code.

This demonstrates:
1. Loading your existing MalGraph classifier (MalgraphServerFeature or DirectMalgraphClient)
2. Creating SRL environment with the classifier
3. Creating DQN agent
4. Training loop

Author: Md Ajwad Akil
Date: December 2025
"""

import sys
from pathlib import Path
import os
import torch
import json

# Add paths
p = Path(os.path.abspath(__file__))
srl_base = str(p.parents[0])
sys.path.append(srl_base)

from malgraph_classifier_adapter import SRLMalGraphClassifierAdapter
from srl_malgraph_nop_mapping import SemanticNOPMapper
from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_dqn_agent import DQNAgent
from srl_malgraph_training import SRLMalGraphTrainer


def load_sample_acfg(sample_path: str) -> dict:
    """
    Load ACFG JSON for testing.
    
    Args:
        sample_path: Path to ACFG JSON file
    
    Returns:
        ACFG dictionary
    """
    with open(sample_path, 'r') as f:
        acfg = json.load(f)
    return acfg


def main():
    print("=" * 80)
    print("SRL-MALGRAPH INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # ============================================================================
    # STEP 1: Initialize MalGraph classifier (your existing code!)
    # ============================================================================
    print("\n[STEP 1] Initialize MalGraph classifier")
    print("-" * 80)
    
    # Option A: Use adapter (recommended for SRL)
    malgraph = SRLMalGraphClassifierAdapter(
        use_direct_client=True,      # True = DirectMalgraphClient, False = MalgraphServerFeature
        threshold_type='100fpr',      # '100fpr' or '1000fpr'
        device=None,                  # Auto-detect GPU/CPU
        server_port=5001              # IDA Pro API port
    )
    
    # Option B: Use your existing code directly
    # from MalgraphModel import MalgraphServerFeature, MalgraphModelParams
    # malgraph = MalgraphServerFeature(your_params, device)
    
    print(f"✓ MalGraph initialized (threshold: {malgraph.threshold:.4f})")
    
    # ============================================================================
    # STEP 2: Initialize semantic NOP mapper
    # ============================================================================
    print("\n[STEP 2] Initialize semantic NOP mapper")
    print("-" * 80)
    
    nop_mapper = SemanticNOPMapper()
    nop_list = nop_mapper.generate_malguise_nop_list()
    
    print(f"✓ Generated {len(nop_list)} semantic NOPs")
    print(f"  Example NOPs:")
    for i in range(min(3, len(nop_list))):
        nop_str = nop_list[i]
        increment = nop_mapper.compute_feature_increment(nop_str)
        print(f"    [{i}] {nop_str.strip()[:40]:40s} → {increment}")
    
    # ============================================================================
    # STEP 3: Initialize RL environment
    # ============================================================================
    print("\n[STEP 3] Initialize RL environment")
    print("-" * 80)
    
    env = SRLMalGraphEnvironment(
        malgraph_classifier=malgraph,  # Your existing classifier!
        nop_mapper=nop_mapper,
        threshold=malgraph.threshold,
        max_mutations=50,
        top_k_blocks=10,
        reward_type='continuous',      # or 'sparse' or 'binary'
        terminal_bonus=10.0
    )
    
    print(f"✓ Environment initialized")
    print(f"  Threshold: {env.threshold:.4f}")
    print(f"  Max mutations: {env.max_mutations}")
    print(f"  Top-k blocks: {env.top_k}")
    print(f"  Reward type: {env.reward_type}")
    
    # ============================================================================
    # STEP 4: Initialize DQN agent
    # ============================================================================
    print("\n[STEP 4] Initialize DQN agent")
    print("-" * 80)
    
    agent = DQNAgent(
        num_nops=len(nop_list),
        k=10,
        input_dim=11,               # MalGraph's 11-dim block features
        latent_dims=[32, 32, 32],   # GNN layers
        conv_sizes=[16, 32],        # 1D conv filters
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=1000,
        device=None                 # Auto-detect
    )
    
    print(f"✓ DQN agent initialized")
    print(f"  Action space: {agent.num_actions} (k={agent.k} × nops={agent.num_nops})")
    print(f"  Device: {agent.device}")
    
    # ============================================================================
    # STEP 5: Test single episode (optional)
    # ============================================================================
    print("\n[STEP 5] Test single episode")
    print("-" * 80)
    
    # Load test ACFG (replace with your actual path)
    test_acfg_path = "/path/to/test_malware_acfg.json"
    
    if os.path.exists(test_acfg_path):
        print(f"Loading test ACFG: {test_acfg_path}")
        test_acfg = load_sample_acfg(test_acfg_path)
        
        # Reset environment
        state = env.reset(test_acfg)
        print(f"  Initial score: {state['score']:.4f}")
        print(f"  Bypassed: {state['score'] < env.threshold}")
        print(f"  Important blocks: {len(state['important_blocks'])}")
        
        # Take one action
        action = agent.select_action(state)
        block_idx, nop_idx = action
        print(f"\n  Agent action: block={block_idx}, nop={nop_idx}")
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        print(f"  New score: {next_state['score']:.4f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
    else:
        print(f"⚠ Test ACFG not found: {test_acfg_path}")
        print("  Skipping episode test (provide actual ACFG path to test)")
    
    # ============================================================================
    # STEP 6: Create trainer
    # ============================================================================
    print("\n[STEP 6] Create trainer")
    print("-" * 80)
    
    # Load training samples (replace with your actual paths)
    train_acfg_dir = "/path/to/train_acfgs/"
    val_acfg_dir = "/path/to/val_acfgs/"
    checkpoint_dir = "/path/to/checkpoints/"
    
    if os.path.exists(train_acfg_dir):
        # Load ACFG paths
        train_samples = [
            os.path.join(train_acfg_dir, f)
            for f in os.listdir(train_acfg_dir)
            if f.endswith('.json')
        ]
        val_samples = [
            os.path.join(val_acfg_dir, f)
            for f in os.listdir(val_acfg_dir)
            if f.endswith('.json')
        ] if os.path.exists(val_acfg_dir) else []
        
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        
        trainer = SRLMalGraphTrainer(
            agent=agent,
            environment=env,
            train_samples=train_samples,
            val_samples=val_samples,
            checkpoint_dir=checkpoint_dir
        )
        
        print(f"✓ Trainer initialized")
        
        # ============================================================================
        # STEP 7: Train (commented out - uncomment to actually train)
        # ============================================================================
        print("\n[STEP 7] Training (commented out)")
        print("-" * 80)
        print("  To train, uncomment the trainer.train() call below")
        print("  This will run 2500 episodes and take several hours!")
        
        # trainer.train(
        #     num_episodes=2500,
        #     eval_interval=100,
        #     checkpoint_interval=500,
        #     batch_size=32
        # )
        
    else:
        print(f"⚠ Training directory not found: {train_acfg_dir}")
        print("  Trainer not created (provide actual ACFG directory)")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: What You Have Now")
    print("=" * 80)
    print("""
✓ MalGraph classifier loaded (your existing code works!)
✓ Semantic NOP mapper created (~200-300 NOPs)
✓ RL environment initialized
✓ DQN agent with trainable SortPooling ready
✓ Trainer ready (just provide ACFG paths)

WHAT THE AGENT DOES:
1. Takes original ACFG (from IDA Pro)
2. Selects important blocks (via SortPooling)
3. Mutates block_features by adding semantic NOPs
4. Passes mutated ACFG to your MalGraph classifier
5. Gets new score and computes reward
6. Learns which mutations reduce malware score

INTEGRATION FLOW:
    ACFG (original)
         ↓
    Environment.reset(acfg)
         ↓
    MalGraph.predict(acfg) → initial_score
         ↓
    Agent.select_action(state) → (block_idx, nop_idx)
         ↓
    Environment.step(action)
         ↓
    Mutate: block_features[block_idx] += nop_increment
         ↓
    MalGraph.predict(mutated_acfg) → new_score
         ↓
    Reward = (old_score - new_score) or bypass_bonus
         ↓
    Agent.train_step() → update Q-network

NO BINARY REWRITING (for proof of concept):
- Environment modifies ACFG JSON only
- block_features arrays are incremented
- Graph structure (edges) unchanged
- Pass mutated JSON directly to MalGraph

TO START TRAINING:
1. Collect 100-500 malware ACFGs (IDA Pro extraction)
2. Update paths in this script
3. Uncomment trainer.train()
4. Run for 2500 episodes (~4-8 hours)
5. Evaluate on test set

EXPECTED RESULTS (from SRL paper):
- Attack Success Rate (ASR): 60-80%
- Average mutations per success: 10-30
- Training time: 4-8 hours (2500 episodes)
""")
    
    print("=" * 80)
    print("✓ Integration complete! Ready to train.")
    print("=" * 80)


if __name__ == "__main__":
    main()
