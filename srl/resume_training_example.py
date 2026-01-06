#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Resume SRL-MalGraph Training from Checkpoint

This script shows how to resume training if it was interrupted.

Usage:
    python resume_training_example.py --checkpoint ./checkpoints/checkpoint_ep500.pt

Author: Md Ajwad Akil
Date: January 2026
"""

import argparse
import os

# Import your training modules
from srl_malgraph_training import (
    SRLMalGraphTrainer,
    load_acfg_dataset,
    SemanticNOPMapper,
    SRLMalGraphClassifierAdapter,
    SRLMalGraphEnvironment,
    SimplifiedDQNAgent
)


def resume_training(checkpoint_path: str):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., './checkpoints/checkpoint_ep500.pt')
    """
    
    print("="*80)
    print("RESUMING SRL-MALGRAPH TRAINING FROM CHECKPOINT")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Configuration (must match original training)
    ACFG_DIR = './data/acfgs'
    NUM_TRAIN_SAMPLES = 2000
    THRESHOLD = 0.14346  # 100fpr threshold
    MAX_STEPS = 30
    K_TOP_BLOCKS = 6
    BATCH_SIZE = 512
    REPLAY_MEMORY = 3000
    EPSILON_DECAY = 3000
    LR = 0.001
    GAMMA = 0.99
    NUM_EPISODES = 10000  # Total episodes (will resume from checkpoint episode)
    LOG_DIR = './logs/srl_malgraph'
    CHECKPOINT_DIR = './checkpoints'
    
    # 1. Load dataset (same as original training)
    print("1. Loading ACFG dataset...")
    acfgs, acfgs_file_names = load_acfg_dataset(ACFG_DIR, max_samples=NUM_TRAIN_SAMPLES)
    print(f"   Loaded {len(acfgs)} samples\n")
    
    # 2. Initialize NOP mapper
    print("2. Initializing semantic NOP mapper...")
    nop_mapper = SemanticNOPMapper()
    nop_list = nop_mapper.generate_malguise_nop_list()
    print(f"   Generated {len(nop_list)} semantic NOPs\n")
    
    # 3. Initialize classifier
    print("3. Initializing MalGraph classifier...")
    classifier = SRLMalGraphClassifierAdapter(
        use_direct_client=True,
        threshold_type='100fpr',
        device=None,
        server_port=5001
    )
    print("   ✓ Classifier ready\n")
    
    # 4. Initialize environment
    print("4. Initializing environment...")
    env = SRLMalGraphEnvironment(
        malgraph_classifier=classifier,
        nop_mapper=nop_mapper,
        threshold=THRESHOLD,
        max_mutations=MAX_STEPS,
        top_k_blocks=K_TOP_BLOCKS,
        reward_type='sparse',
        terminal_bonus=10.0,
        sortpooling_method='l2_norm',
        debug=False
    )
    print("   ✓ Environment ready\n")
    
    # 5. Initialize agent (will be overwritten by checkpoint)
    print("5. Initializing DQN agent...")
    agent = SimplifiedDQNAgent(
        num_nops=len(nop_list),
        embedding_dim=200,
        batch_size=BATCH_SIZE,
        memory_capacity=REPLAY_MEMORY,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=EPSILON_DECAY,
        lr=LR,
        gamma=GAMMA,
        optimizer_type='rmsprop'
    )
    print("   ✓ Agent initialized\n")
    
    # 6. Initialize trainer
    print("6. Initializing trainer...")
    trainer = SRLMalGraphTrainer(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        save_freq=20,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
    print("   ✓ Trainer ready\n")
    
    # 7. Load checkpoint
    print("7. Loading checkpoint...")
    trainer.load_checkpoint(checkpoint_path)
    print(f"   ✓ Loaded! Resuming from episode {trainer.start_episode + 1}\n")
    
    # 8. Resume training
    print("="*80)
    print(f"RESUMING TRAINING FROM EPISODE {trainer.start_episode + 1}")
    print("="*80)
    print(f"Previous training statistics:")
    print(f"  - Episodes completed: {len(trainer.episode_rewards)}")
    print(f"  - Avg reward (last 100): {sum(trainer.episode_rewards[-100:])/len(trainer.episode_rewards[-100:]) if trainer.episode_rewards else 0:.3f}")
    print(f"  - Success rate (last 100): {sum(trainer.episode_bypassed[-100:])/len(trainer.episode_bypassed[-100:]) * 100 if trainer.episode_bypassed else 0:.1f}%")
    print(f"  - Replay buffer size: {len(agent.memory)}")
    print(f"  - Current epsilon: {agent._get_epsilon():.3f}")
    print("="*80 + "\n")
    
    trainer.train(acfgs, acfgs_file_names)
    
    print("\n✓ Training resumed and completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Resume SRL-MalGraph training from checkpoint')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., ./checkpoints/checkpoint_ep500.pt)'
    )
    
    args = parser.parse_args()
    
    resume_training(args.checkpoint)


if __name__ == "__main__":
    main()
