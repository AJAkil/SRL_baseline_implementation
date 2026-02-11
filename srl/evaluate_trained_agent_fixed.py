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
import time
import numpy as np
import torch
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
    skipped_count = 0
    
    for i, acfg_data in enumerate(tqdm(test_acfgs, desc="Evaluating")):
        # Start timing for this sample
        sample_start_time = time.time()
        
        # Preprocess ACFG
        if 'result' in acfg_data:
            acfg = json.loads(acfg_data['result'])
        else:
            acfg = acfg_data
        
        # Skip empty ACFGs (same check as training)
        if 'acfg_list' not in acfg or len(acfg.get('acfg_list', [])) == 0:
            if verbose:
                print(f"⚠️ SKIPPING Sample {i+1}: Empty acfg_list (no functions)")
            skipped_count += 1
            continue
        
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
        
        # End timing for this sample
        sample_end_time = time.time()
        elapsed_seconds = sample_end_time - sample_start_time
        elapsed_minutes = int(elapsed_seconds // 60)
        elapsed_secs = elapsed_seconds % 60
        
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
            'elapsed_time_seconds': elapsed_seconds,
            'elapsed_time_formatted': f"{elapsed_minutes}m {elapsed_secs:.2f}s",
            'mutation_log': mutation_log
        }
        
        results.append(sample_result)
        
        if verbose:
            status = "✓ BYPASSED" if bypassed else "✗ FAILED"
            print(f"Sample {i+1}: {status} | Score: {initial_score:.4f} → {final_score:.4f} | "
                  f"Mutations: {steps} | Reduction: {score_reduction:.4f} | Time: {elapsed_minutes}m {elapsed_secs:.2f}s")
    
    # Report skipped samples
    if skipped_count > 0:
        print(f"\n⚠️ Skipped {skipped_count} samples with empty acfg_list")
    
    # Compute aggregate metrics
    total_samples = len(results)
    successful = sum(1 for r in results if r['bypassed'])
    failed = total_samples - successful
    
    # Separate timing stats for bypassed vs failed
    bypassed_times = [r['elapsed_time_seconds'] for r in results if r['bypassed']]
    failed_times = [r['elapsed_time_seconds'] for r in results if not r['bypassed']]
    all_times = [r['elapsed_time_seconds'] for r in results]
    
    # Bypassed samples timing metrics
    if bypassed_times:
        bypassed_total_hours = np.sum(bypassed_times) / 3600.0
        bypassed_mean_seconds = np.mean(bypassed_times)
        bypassed_mean_minutes = bypassed_mean_seconds / 60.0
        bypassed_median_seconds = np.median(bypassed_times)
        bypassed_median_minutes = bypassed_median_seconds / 60.0
        bypassed_throughput = successful / bypassed_total_hours if bypassed_total_hours > 0 else 0.0
    else:
        bypassed_total_hours = 0.0
        bypassed_mean_seconds = 0.0
        bypassed_mean_minutes = 0.0
        bypassed_median_seconds = 0.0
        bypassed_median_minutes = 0.0
        bypassed_throughput = 0.0
    
    # Failed samples timing metrics
    if failed_times:
        failed_total_hours = np.sum(failed_times) / 3600.0
        failed_mean_seconds = np.mean(failed_times)
        failed_mean_minutes = failed_mean_seconds / 60.0
        failed_median_seconds = np.median(failed_times)
        failed_median_minutes = failed_median_seconds / 60.0
        failed_throughput = failed / failed_total_hours if failed_total_hours > 0 else 0.0
    else:
        failed_total_hours = 0.0
        failed_mean_seconds = 0.0
        failed_mean_minutes = 0.0
        failed_median_seconds = 0.0
        failed_median_minutes = 0.0
        failed_throughput = 0.0
    
    metrics = {
        'total_samples': total_samples,
        'successful_attacks': successful,
        'failed_attacks': failed,
        'attack_success_rate': successful / total_samples,
        'avg_mutations': np.mean([r['num_mutations'] for r in results]),
        'avg_score_reduction': np.mean([r['score_reduction'] for r in results]),
        'avg_final_score': np.mean([r['final_score'] for r in results]),
        'median_mutations': np.median([r['num_mutations'] for r in results]),
        'budget_exceeded_count': sum(1 for r in results if r.get('budget_exceeded', False)),
        # Overall timing statistics
        'avg_time_seconds': np.mean(all_times),
        'median_time_seconds': np.median(all_times),
        'total_time_seconds': np.sum(all_times),
        'total_time_hours': np.sum(all_times) / 3600.0,
        'min_time_seconds': np.min(all_times),
        'max_time_seconds': np.max(all_times),
        # Bypassed samples timing
        'bypassed_total_hours': bypassed_total_hours,
        'bypassed_mean_seconds': bypassed_mean_seconds,
        'bypassed_mean_minutes': bypassed_mean_minutes,
        'bypassed_median_seconds': bypassed_median_seconds,
        'bypassed_median_minutes': bypassed_median_minutes,
        'bypassed_throughput_per_hour': bypassed_throughput,
        # Failed samples timing
        'failed_total_hours': failed_total_hours,
        'failed_mean_seconds': failed_mean_seconds,
        'failed_mean_minutes': failed_mean_minutes,
        'failed_median_seconds': failed_median_seconds,
        'failed_median_minutes': failed_median_minutes,
        'failed_throughput_per_hour': failed_throughput,
        'skipped_samples': skipped_count,
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
    # Format timing strings
    total_mins = int(metrics['total_time_seconds'] // 60)
    total_secs = metrics['total_time_seconds'] % 60
    avg_mins = int(metrics['avg_time_seconds'] // 60)
    avg_secs = metrics['avg_time_seconds'] % 60
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Samples:           {metrics['total_samples']}")
    print(f"Successful Attacks:      {metrics['successful_attacks']}")
    print(f"Failed Attacks:          {metrics['failed_attacks']}")
    print(f"Attack Success Rate:     {metrics['attack_success_rate']*100:.2f}%")
    print(f"Avg Mutations:           {metrics['avg_mutations']:.2f}")
    print(f"Median Mutations:        {metrics['median_mutations']:.0f}")
    print(f"Avg Score Reduction:     {metrics['avg_score_reduction']:.4f}")
    print(f"Avg Final Score:         {metrics['avg_final_score']:.4f}")
    print(f"Budget Exceeded:         {metrics['budget_exceeded_count']} samples")
    print(f"Skipped (empty ACFG):    {metrics.get('skipped_samples', 0)} samples")
    print("-" * 80)
    print("OVERALL TIMING STATISTICS")
    print("-" * 80)
    print(f"Total Time:              {total_mins}m {total_secs:.2f}s ({metrics['total_time_hours']:.2f} hours)")
    print(f"Avg Time per Sample:     {avg_mins}m {avg_secs:.2f}s ({metrics['avg_time_seconds']:.2f}s)")
    print(f"Median Time:             {metrics['median_time_seconds']:.2f}s")
    print(f"Min Time:                {metrics['min_time_seconds']:.2f}s")
    print(f"Max Time:                {metrics['max_time_seconds']:.2f}s")
    
    # Bypassed samples timing
    if metrics['successful_attacks'] > 0:
        print("-" * 80)
        print(f"BYPASSED SAMPLES TIMING ({metrics['successful_attacks']} samples)")
        print("-" * 80)
        print(f"Total Time:              {metrics['bypassed_total_hours']:.2f} hours")
        bypassed_mean_m = int(metrics['bypassed_mean_minutes'])
        bypassed_mean_s = (metrics['bypassed_mean_minutes'] - bypassed_mean_m) * 60
        print(f"Mean Time per Sample:    {bypassed_mean_m}m {bypassed_mean_s:.2f}s ({metrics['bypassed_mean_seconds']:.2f}s)")
        bypassed_median_m = int(metrics['bypassed_median_minutes'])
        bypassed_median_s = (metrics['bypassed_median_minutes'] - bypassed_median_m) * 60
        print(f"Median Time per Sample:  {bypassed_median_m}m {bypassed_median_s:.2f}s ({metrics['bypassed_median_seconds']:.2f}s)")
        print(f"Throughput:              {metrics['bypassed_throughput_per_hour']:.2f} samples/hour")
    
    # Failed samples timing
    if metrics['failed_attacks'] > 0:
        print("-" * 80)
        print(f"FAILED SAMPLES TIMING ({metrics['failed_attacks']} samples)")
        print("-" * 80)
        print(f"Total Time:              {metrics['failed_total_hours']:.2f} hours")
        failed_mean_m = int(metrics['failed_mean_minutes'])
        failed_mean_s = (metrics['failed_mean_minutes'] - failed_mean_m) * 60
        print(f"Mean Time per Sample:    {failed_mean_m}m {failed_mean_s:.2f}s ({metrics['failed_mean_seconds']:.2f}s)")
        failed_median_m = int(metrics['failed_median_minutes'])
        failed_median_s = (metrics['failed_median_minutes'] - failed_median_m) * 60
        print(f"Median Time per Sample:  {failed_median_m}m {failed_median_s:.2f}s ({metrics['failed_median_seconds']:.2f}s)")
        print(f"Throughput:              {metrics['failed_throughput_per_hour']:.2f} samples/hour")
    
    print("="*80)


def main():
    # Set seeds for reproducibility (same as training)
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description="Evaluate trained SRL-MalGraph agent")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint (.pt file)")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test ACFG JSON files")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum test samples to evaluate (default: all)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum mutations per sample (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.14346,
                        help="MalGraph classification threshold (default: 0.14346 for 100fpr and 0.91276 for 1000fpr)")
    parser.add_argument("--top_k_blocks", type=int, default=200,
                        help="Number of blocks to mutate per step (default: 200)")
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
        sortpooling_method='l2_norm',
        injection_budget_pct=None
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
    total_mins = int(metrics['total_time_seconds'] // 60)
    total_secs = metrics['total_time_seconds'] % 60
    avg_mins = int(metrics['avg_time_seconds'] // 60)
    avg_secs = metrics['avg_time_seconds'] % 60
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SRL-MalGraph Evaluation Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Set: {args.test_dir}\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Successful Attacks: {metrics['successful_attacks']}\n")
        f.write(f"Failed Attacks: {metrics['failed_attacks']}\n")
        f.write(f"Attack Success Rate (ASR): {metrics['attack_success_rate']*100:.2f}%\n")
        f.write(f"Average Mutations: {metrics['avg_mutations']:.2f}\n")
        f.write(f"Median Mutations: {metrics['median_mutations']:.0f}\n")
        f.write(f"Average Score Reduction: {metrics['avg_score_reduction']:.4f}\n")
        f.write(f"Average Final Score: {metrics['avg_final_score']:.4f}\n")
        f.write(f"Budget Exceeded: {metrics['budget_exceeded_count']} samples\n")
        f.write(f"Skipped (empty ACFG): {metrics.get('skipped_samples', 0)} samples\n")
        f.write("-"*80 + "\n")
        f.write("OVERALL TIMING STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Time: {total_mins}m {total_secs:.2f}s ({metrics['total_time_hours']:.2f} hours)\n")
        f.write(f"Average Time per Sample: {avg_mins}m {avg_secs:.2f}s ({metrics['avg_time_seconds']:.2f}s)\n")
        f.write(f"Median Time: {metrics['median_time_seconds']:.2f}s\n")
        f.write(f"Min Time: {metrics['min_time_seconds']:.2f}s\n")
        f.write(f"Max Time: {metrics['max_time_seconds']:.2f}s\n")
        
        # Bypassed samples timing
        if metrics['successful_attacks'] > 0:
            f.write("-"*80 + "\n")
            f.write(f"BYPASSED SAMPLES TIMING ({metrics['successful_attacks']} samples)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Time: {metrics['bypassed_total_hours']:.2f} hours\n")
            bypassed_mean_m = int(metrics['bypassed_mean_minutes'])
            bypassed_mean_s = (metrics['bypassed_mean_minutes'] - bypassed_mean_m) * 60
            f.write(f"Mean Time per Sample: {bypassed_mean_m}m {bypassed_mean_s:.2f}s ({metrics['bypassed_mean_seconds']:.2f}s)\n")
            bypassed_median_m = int(metrics['bypassed_median_minutes'])
            bypassed_median_s = (metrics['bypassed_median_minutes'] - bypassed_median_m) * 60
            f.write(f"Median Time per Sample: {bypassed_median_m}m {bypassed_median_s:.2f}s ({metrics['bypassed_median_seconds']:.2f}s)\n")
            f.write(f"Throughput: {metrics['bypassed_throughput_per_hour']:.2f} samples/hour\n")
        
        # Failed samples timing
        if metrics['failed_attacks'] > 0:
            f.write("-"*80 + "\n")
            f.write(f"FAILED SAMPLES TIMING ({metrics['failed_attacks']} samples)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Time: {metrics['failed_total_hours']:.2f} hours\n")
            failed_mean_m = int(metrics['failed_mean_minutes'])
            failed_mean_s = (metrics['failed_mean_minutes'] - failed_mean_m) * 60
            f.write(f"Mean Time per Sample: {failed_mean_m}m {failed_mean_s:.2f}s ({metrics['failed_mean_seconds']:.2f}s)\n")
            failed_median_m = int(metrics['failed_median_minutes'])
            failed_median_s = (metrics['failed_median_minutes'] - failed_median_m) * 60
            f.write(f"Median Time per Sample: {failed_median_m}m {failed_median_s:.2f}s ({metrics['failed_median_seconds']:.2f}s)\n")
            f.write(f"Throughput: {metrics['failed_throughput_per_hour']:.2f} samples/hour\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
