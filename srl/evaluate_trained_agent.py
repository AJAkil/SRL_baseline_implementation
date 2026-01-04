#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRL-MalGraph Evaluation Script

Evaluates a trained DQN agent on a test set of malware samples
and generates Attack Success Rate (ASR) report.

Usage:
    python evaluate_trained_agent.py --checkpoint path/to/model.pt --test_dir path/to/test_acfgs

Author: Md Ajwad Akil
Date: December 2025
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_dqn_agent import SimplifiedDQNAgent
from srl_malgraph_nop_mapping import SemanticNOPMapper
from malgraph_classifier_adapter import SRLMalGraphClassifierAdapter


def load_test_acfgs(acfg_dir: str, max_samples: int = None) -> List[Dict]:
    """
    Load test ACFG JSON files.
    
    Args:
        acfg_dir: Directory containing ACFG JSON files
        max_samples: Maximum number to load (None = all)
    
    Returns:
        List of ACFG dictionaries
    """
    acfg_files = [f for f in os.listdir(acfg_dir) if f.endswith('.json')]
    
    if max_samples:
        acfg_files = acfg_files[:max_samples]
    
    acfgs = []
    for filename in tqdm(acfg_files, desc="Loading test ACFGs"):
        filepath = os.path.join(acfg_dir, filename)
        with open(filepath, 'r') as f:
            acfg = json.load(f)
            acfgs.append(acfg)
    
    print(f"Loaded {len(acfgs)} test samples")
    return acfgs


def evaluate_agent(
    agent: SimplifiedDQNAgent,
    env: SRLMalGraphEnvironment,
    test_acfgs: List[Dict],
    max_steps: int = 30,
    verbose: bool = True
) -> Dict:
    """
    Evaluate trained agent on test set.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        test_acfgs: Test ACFG samples
        max_steps: Max mutations per sample
        verbose: Print per-sample results
    
    Returns:
        Evaluation metrics dictionary
    """
    results = []
    
    for i, acfg_data in enumerate(tqdm(test_acfgs, desc="Evaluating")):
        # Preprocess ACFG
        if 'result' in acfg_data:
            acfg = json.loads(acfg_data['result'])
        else:
            acfg = acfg_data
        
        # Reset environment
        state = env.reset(acfg)
        initial_score = state['score']
        
        # Run episode (greedy policy - no exploration)
        done = False
        steps = 0
        mutation_log = []
        
        while not done and steps < max_steps:
            # Greedy action selection (explore=False)
            action = agent.select_action(state, explore=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Log mutation
            mutation_log.append({
                'step': steps,
                'action': action,
                'reward': reward,
                'score': info['score']
            })
            
            state = next_state
            steps += 1
        
        # Collect results
        final_score = info['score']
        bypassed = info['bypassed']
        score_reduction = initial_score - final_score
        budget_exceeded = info.get('budget_exceeded', False)
        
        sample_result = {
            'sample_idx': i,
            'initial_score': initial_score,
            'final_score': final_score,
            'score_reduction': score_reduction,
            'num_mutations': steps,
            'bypassed': bypassed,
            'budget_exceeded': budget_exceeded,
            'mutation_log': mutation_log
        }
        
        results.append(sample_result)
        
        if verbose:
            status = "✓ BYPASSED" if bypassed else "✗ FAILED"
            print(f"Sample {i+1}: {status} | Score: {initial_score:.4f} → {final_score:.4f} | "
                  f"Mutations: {steps} | Reduction: {score_reduction:.4f}")
    
    # Compute aggregate metrics
    total_samples = len(results)
    successful = sum(1 for r in results if r['bypassed'])
    
    metrics = {
        'total_samples': total_samples,
        'successful_attacks': successful,
        'attack_success_rate': successful / total_samples,
        'avg_mutations': np.mean([r['num_mutations'] for r in results]),
        'avg_score_reduction': np.mean([r['score_reduction'] for r in results]),
        'avg_final_score': np.mean([r['final_score'] for r in results]),
        'median_mutations': np.median([r['num_mutations'] for r in results]),
        'budget_exceeded_count': sum(1 for r in results if r.get('budget_exceeded', False)),
        'per_sample_results': results
    }
    
    return metrics


def save_evaluation_report(metrics: Dict, output_path: str):
    """
    Save evaluation report to JSON.
    
    Args:
        metrics: Evaluation metrics
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation report saved to: {output_path}")


def print_summary(metrics: Dict):
    """
    Print evaluation summary.
    
    Args:
        metrics: Evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Samples:           {metrics['total_samples']}")
    print(f"Successful Attacks:      {metrics['successful_attacks']}")
    print(f"Attack Success Rate:     {metrics['attack_success_rate']*100:.2f}%")
    print(f"Avg Mutations:           {metrics['avg_mutations']:.2f}")
    print(f"Median Mutations:        {metrics['median_mutations']:.0f}")
    print(f"Avg Score Reduction:     {metrics['avg_score_reduction']:.4f}")
    print(f"Avg Final Score:         {metrics['avg_final_score']:.4f}")
    print(f"Budget Exceeded:         {metrics['budget_exceeded_count']} samples")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SRL-MalGraph agent")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint (.pt file)")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test ACFG JSON files")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum test samples to evaluate (default: all)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum mutations per sample (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.14346,
                        help="MalGraph classification threshold (default: 0.14346 for 100fpr)")
    parser.add_argument("--top_k_blocks", type=int, default=1250,
                        help="Number of blocks to mutate per step (default: 1250)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("SRL-MalGraph Agent Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir:   {args.test_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)
    
    # Initialize NOP mapper
    print("\n1. Initializing NOP mapper...")
    nop_mapper = SemanticNOPMapper()
    nop_list = nop_mapper.generate_malguise_nop_list()
    print(f"   Generated {len(nop_list)} semantic NOPs")
    
    # Initialize classifier
    print("\n2. Initializing MalGraph classifier...")
    classifier = SRLMalGraphClassifierAdapter(
        use_direct_client=True,
        threshold_type='100fpr',
        device=None,
        server_port=5001
    )
    
    # Initialize environment
    print("\n3. Initializing environment...")
    env = SRLMalGraphEnvironment(
        malgraph_classifier=classifier,
        nop_mapper=nop_mapper,
        threshold=args.threshold,
        max_mutations=args.max_steps,
        top_k_blocks=args.top_k_blocks,
        reward_type='sparse',
        terminal_bonus=10.0,
        sortpooling_method='l2_norm'
    )
    
    # Initialize agent
    print("\n4. Initializing agent...")
    agent = SimplifiedDQNAgent(
        num_nops=len(nop_list),
        embedding_dim=200,
        batch_size=512,  # Not used during eval, but needed for init
        memory_capacity=3000,
        optimizer_type='rmsprop'
    )
    
    # Load checkpoint
    print(f"\n5. Loading checkpoint from {args.checkpoint}...")
    agent.load_checkpoint(args.checkpoint)
    print("   Checkpoint loaded successfully!")
    
    # Load test set
    print(f"\n6. Loading test set from {args.test_dir}...")
    test_acfgs = load_test_acfgs(args.test_dir, max_samples=args.max_samples)
    
    # Evaluate
    print(f"\n7. Evaluating on {len(test_acfgs)} test samples...")
    metrics = evaluate_agent(
        agent=agent,
        env=env,
        test_acfgs=test_acfgs,
        max_steps=args.max_steps,
        verbose=args.verbose
    )
    
    # Print summary
    print_summary(metrics)
    
    # Save detailed report
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    save_evaluation_report(metrics, report_path)
    
    # Save summary text file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SRL-MalGraph Evaluation Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Set: {args.test_dir}\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Successful Attacks: {metrics['successful_attacks']}\n")
        f.write(f"Attack Success Rate (ASR): {metrics['attack_success_rate']*100:.2f}%\n")
        f.write(f"Average Mutations: {metrics['avg_mutations']:.2f}\n")
        f.write(f"Median Mutations: {metrics['median_mutations']:.0f}\n")
        f.write(f"Average Score Reduction: {metrics['avg_score_reduction']:.4f}\n")
        f.write(f"Average Final Score: {metrics['avg_final_score']:.4f}\n")
        f.write(f"Budget Exceeded: {metrics['budget_exceeded_count']} samples\n")
        f.write("="*80 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
