#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRL-MalGraph Training Loop

Connects the DQN agent with the environment to train adversarial malware generation.

Author: Md Ajwad Akil
Date: December 2025
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_dqn_agent import DQNAgent
from srl_malgraph_nop_mapping import SemanticNOPMapper


class SRLMalGraphTrainer:
    """
    Training orchestrator for SRL-MalGraph attack.
    
    Handles:
        - Episode execution
        - Training statistics
        - Checkpoint saving
        - Evaluation
    """
    
    def __init__(
        self,
        env: SRLMalGraphEnvironment,
        agent: DQNAgent,
        num_episodes: int = 2500,
        max_steps_per_episode: int = 50,
        eval_freq: int = 100,
        save_freq: int = 500,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            env: SRL-MalGraph environment
            agent: DQN agent
            num_episodes: Total training episodes
            max_steps_per_episode: Maximum mutations per episode
            eval_freq: Evaluation frequency (episodes)
            save_freq: Checkpoint save frequency (episodes)
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_final_scores = []
        self.episode_bypassed = []
        self.training_losses = []
        self.eval_success_rates = []
        
        print(f"Trainer initialized:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Max steps: {max_steps_per_episode}")
        print(f"  Log dir: {log_dir}")
    
    def train_episode(self, acfg_json: Dict, episode: int) -> Dict:
        """
        Train on single malware sample for one episode.
        
        Args:
            acfg_json: ACFG of malware sample
            episode: Episode number
        
        Returns:
            Episode statistics dictionary
        """
        # Reset environment
        state = self.env.reset(acfg_json)
        
        episode_reward = 0
        episode_losses = []
        steps = 0
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action_nop_idx = self.agent.select_action(state, explore=True)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action_nop_idx)
            
            # Store experience
            self.agent.store_experience(state, action_nop_idx, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Episode statistics
        stats = {
            'episode': episode,
            'reward': episode_reward,
            'length': steps,
            'final_score': info['score'],
            'bypassed': info['bypassed'],
            'avg_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'score_delta': self.env.original_acfg['score'] - info['score'] if 'score' in self.env.original_acfg else 0.0
        }
        
        return stats
    
    def train(self, training_acfgs: List[Dict]):
        """
        Main training loop.
        
        Args:
            training_acfgs: List of ACFG dictionaries for training samples
        """
        print("\n" + "=" * 80)
        print("Starting SRL-MalGraph Training")
        print("=" * 80)
        
        num_samples = len(training_acfgs)
        print(f"Training on {num_samples} malware samples")
        print(f"Total episodes: {self.num_episodes}")
        
        start_time = time.time()
        
        for episode in tqdm(range(1, self.num_episodes + 1), desc="Training"):
            # Sample random malware
            acfg_idx = np.random.randint(0, num_samples)
            acfg = training_acfgs[acfg_idx]
            
            # Train episode
            stats = self.train_episode(acfg, episode)
            
            # Store statistics
            self.episode_rewards.append(stats['reward'])
            self.episode_lengths.append(stats['length'])
            self.episode_final_scores.append(stats['final_score'])
            self.episode_bypassed.append(1 if stats['bypassed'] else 0)
            if stats['avg_loss'] > 0:
                self.training_losses.append(stats['avg_loss'])
            
            # Logging
            if episode % 10 == 0:
                recent_rewards = np.mean(self.episode_rewards[-10:])
                recent_bypassed = np.mean(self.episode_bypassed[-10:])
                recent_loss = np.mean(self.training_losses[-10:]) if self.training_losses else 0.0
                epsilon = self.agent._get_epsilon()
                
                print(f"\nEpisode {episode}/{self.num_episodes}")
                print(f"  Avg Reward (last 10): {recent_rewards:.3f}")
                print(f"  Success Rate (last 10): {recent_bypassed*100:.1f}%")
                print(f"  Avg Loss: {recent_loss:.4f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Final Score: {stats['final_score']:.4f}")
            
            # Evaluation
            if episode % self.eval_freq == 0:
                eval_stats = self.evaluate(training_acfgs[:10])  # Eval on subset
                self.eval_success_rates.append(eval_stats['success_rate'])
                
                print(f"\n{'='*60}")
                print(f"EVALUATION at Episode {episode}")
                print(f"  Success Rate: {eval_stats['success_rate']*100:.1f}%")
                print(f"  Avg Mutations: {eval_stats['avg_mutations']:.1f}")
                print(f"  Avg Score Reduction: {eval_stats['avg_score_reduction']:.4f}")
                print(f"{'='*60}\n")
            
            # Save checkpoint
            if episode % self.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_ep{episode}.pt"
                )
                self.agent.save_checkpoint(checkpoint_path)
                self.save_training_stats()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Final success rate: {np.mean(self.episode_bypassed[-100:])*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Save final checkpoint
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self.agent.save_checkpoint(final_path)
        self.save_training_stats()
        self.plot_training_curves()
    
    def evaluate(self, eval_acfgs: List[Dict], verbose: bool = False) -> Dict:
        """
        Evaluate agent on test samples.
        
        Args:
            eval_acfgs: List of ACFG dictionaries for evaluation
            verbose: Print per-sample results
        
        Returns:
            Evaluation statistics
        """
        successes = 0
        total_mutations = []
        score_reductions = []
        
        for i, acfg in enumerate(eval_acfgs):
            state = self.env.reset(acfg)
            initial_score = state['score']
            
            done = False
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                # Greedy action (no exploration)
                action = self.agent.select_action(state, explore=False)
                state, reward, done, info = self.env.step(action)
                steps += 1
            
            final_score = info['score']
            bypassed = info['bypassed']
            
            if bypassed:
                successes += 1
            
            total_mutations.append(steps)
            score_reductions.append(initial_score - final_score)
            
            if verbose:
                print(f"Sample {i+1}: Score {initial_score:.4f} -> {final_score:.4f}, "
                      f"Mutations: {steps}, Bypassed: {bypassed}")
        
        stats = {
            'success_rate': successes / len(eval_acfgs),
            'avg_mutations': np.mean(total_mutations),
            'avg_score_reduction': np.mean(score_reductions)
        }
        
        return stats
    
    def save_training_stats(self):
        """Save training statistics to JSON."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_final_scores': self.episode_final_scores,
            'episode_bypassed': self.episode_bypassed,
            'training_losses': self.training_losses,
            'eval_success_rates': self.eval_success_rates
        }
        
        stats_path = os.path.join(self.log_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training stats saved to {stats_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Success rate (rolling average)
        window = 50
        if len(self.episode_bypassed) >= window:
            success_rate_smooth = np.convolve(
                self.episode_bypassed,
                np.ones(window)/window,
                mode='valid'
            )
            axes[0, 1].plot(success_rate_smooth)
            axes[0, 1].set_title(f'Success Rate (rolling avg, window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # Training loss
        if self.training_losses:
            axes[1, 0].plot(self.training_losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('MSE Loss')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths (Mutations)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Mutations')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to {plot_path}")
        plt.close()


def load_acfg_dataset(acfg_dir: str, max_samples: int = None) -> List[Dict]:
    """
    Load ACFG JSON files from directory.
    
    Args:
        acfg_dir: Directory containing ACFG JSON files
        max_samples: Maximum number of samples to load
    
    Returns:
        List of ACFG dictionaries
    """
    acfg_files = [f for f in os.listdir(acfg_dir) if f.endswith('.json')]
    
    if max_samples:
        acfg_files = acfg_files[:max_samples]
    
    acfgs = []
    for filename in tqdm(acfg_files, desc="Loading ACFGs"):
        filepath = os.path.join(acfg_dir, filename)
        with open(filepath, 'r') as f:
            acfg = json.load(f)
            acfgs.append(acfg)
    
    print(f"Loaded {len(acfgs)} ACFG samples")
    return acfgs


def main():
    """
    Main training script.
    
    Usage:
        python srl_malgraph_training.py
    """
    # Paths (adjust to your setup)
    ACFG_DIR = "/path/to/extracted/acfgs"  # Directory with ACFG JSON files
    LOG_DIR = "./logs/srl_malgraph"
    CHECKPOINT_DIR = "./checkpoints/srl_malgraph"
    
    # Hyperparameters
    NUM_EPISODES = 2500
    MAX_STEPS = 50
    K_TOP_BLOCKS = 10
    BATCH_SIZE = 32
    LR = 0.001
    GAMMA = 0.9
    THRESHOLD = 0.14346  # MalGraph 100fpr threshold
    
    print("Initializing SRL-MalGraph Training...")
    
    # Load dataset
    print("\n1. Loading ACFG dataset...")
    acfgs = load_acfg_dataset(ACFG_DIR, max_samples=100)  # Start with 100 samples
    
    # Initialize NOP mapper
    print("\n2. Initializing semantic NOP mapper...")
    nop_mapper = SemanticNOPMapper()
    nop_list = nop_mapper.generate_malguise_nop_list()
    print(f"   Generated {len(nop_list)} semantic NOPs")
    
    # Initialize environment (dummy classifier for now - replace with actual MalGraph)
    print("\n3. Initializing environment...")
    # env = SRLMalGraphEnvironment(
    #     malgraph_classifier=YOUR_MALGRAPH_CLASSIFIER,
    #     nop_mapper=nop_mapper,
    #     threshold=THRESHOLD,
    #     max_mutations=MAX_STEPS,
    #     top_k_blocks=K_TOP_BLOCKS,
    #     reward_type='continuous'
    # )
    print("   [NOTE: Replace with actual MalGraph classifier]")
    
    # Initialize agent
    print("\n4. Initializing DQN agent...")
    agent = DQNAgent(
        num_nops=len(nop_list),
        k=K_TOP_BLOCKS,
        input_dim=11,
        latent_dims=[32, 32, 32],
        hidden_dim=128,
        lr=LR,
        gamma=GAMMA,
        batch_size=BATCH_SIZE
    )
    
    # Initialize trainer
    print("\n5. Initializing trainer...")
    # trainer = SRLMalGraphTrainer(
    #     env=env,
    #     agent=agent,
    #     num_episodes=NUM_EPISODES,
    #     max_steps_per_episode=MAX_STEPS,
    #     log_dir=LOG_DIR,
    #     checkpoint_dir=CHECKPOINT_DIR
    # )
    
    # Start training
    print("\n6. Starting training...")
    # trainer.train(acfgs)
    
    print("\n[NOTE: Uncomment trainer.train() after setting up MalGraph classifier]")
    print("Training script ready!")


if __name__ == "__main__":
    main()
