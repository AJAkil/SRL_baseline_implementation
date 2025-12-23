#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline Random Mutation Testing for SRL-MalGraph

Simple baseline that randomly selects mutations to test the environment
and classifier integration without any RL agent.

This is the simplest starting point for testing!

Author: Md Ajwad Akil
Date: December 2025
"""

import json
import random
import numpy as np
from pathlib import Path
import argparse

from malgraph_classifier_adapter import SRLMalGraphClassifierAdapter
from srl_malgraph_nop_mapping import SemanticNOPMapper
from srl_malgraph_environment import SRLMalGraphEnvironment


def run_random_baseline(
    acfg_path: str,
    num_iterations: int = 50,
    threshold: float = 0.14346,
    seed: int = 42,
    save_results: bool = True,
    output_dir: str = "./baseline_results"
):
    """
    Run random mutation baseline test.
    
    Args:
        acfg_path: Path to ACFG JSON file
        num_iterations: Maximum number of random mutations to try
        threshold: Classification threshold for bypass
        seed: Random seed for reproducibility
        save_results: Whether to save results to file
        output_dir: Directory to save results
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("RANDOM MUTATION BASELINE TEST")
    print("=" * 70)
    print(f"ACFG file: {acfg_path}")
    print(f"Max iterations: {num_iterations}")
    print(f"Threshold: {threshold}")
    print(f"Random seed: {seed}")
    print("=" * 70)
    
    # Initialize classifier adapter
    print("\n[1/4] Initializing MalGraph classifier...")
    classifier = SRLMalGraphClassifierAdapter(
        use_direct_client=True,
        threshold_type='100fpr',
        device=None,  # Auto-detect
        server_port=5001
    )
    
    # Initialize NOP mapper
    print("\n[2/4] Initializing semantic NOP mapper...")
    nop_mapper = SemanticNOPMapper()
    
    # Initialize environment
    print("\n[3/4] Initializing SRL-MalGraph environment...")
    env = SRLMalGraphEnvironment(
        malgraph_classifier=classifier,
        nop_mapper=nop_mapper,
        threshold=threshold,
        max_mutations=num_iterations,
        top_k_blocks=6,  # Consider top 6 important blocks
        reward_type='continuous',
        terminal_bonus=10.0,
        sortpooling_method='l2_norm'  # Non-trainable for baseline
    )
    
    # Load ACFG
    print(f"\n[4/4] Loading ACFG from {acfg_path}...")
    if isinstance(acfg_path, str):
        acfg_path = Path(acfg_path)
    
    with open(acfg_path, 'r') as f:
        acfg_data = json.load(f)
    
    # Extract result if nested JSON
    if 'result' in acfg_data:
        acfg = json.loads(acfg_data['result'])
    else:
        acfg = acfg_data
    
    print(f"  Hash: {acfg.get('hash', 'N/A')}")
    print(f"  Functions: {acfg.get('function_number', 'N/A')}")
    
    # Count total blocks
    total_blocks = sum(f['block_number'] for f in acfg['acfg_list'])
    print(f"  Total blocks: {total_blocks}")
    
    # Reset environment
    print("\n" + "=" * 70)
    print("STARTING RANDOM MUTATION EXPERIMENT")
    print("=" * 70)
    
    state = env.reset(acfg)
    initial_score = state['score']
    
    print(f"\nInitial score: {initial_score:.6f}")
    print(f"Threshold:     {threshold:.6f}")
    print(f"Status:        {'MALICIOUS' if initial_score >= threshold else 'BENIGN'}")
    print(f"\nTop {len(env.important_blocks)} important blocks identified:")
    for i, (func_idx, block_idx, importance) in enumerate(env.important_blocks[:5]):
        print(f"  {i+1}. Function {func_idx}, Block {block_idx} (importance: {importance:.4f})")
    if len(env.important_blocks) > 5:
        print(f"  ... and {len(env.important_blocks) - 5} more")
    
    # Get action space
    num_block_selections, num_nop_types = env.get_action_space_size()
    print(f"\nAction space: {num_block_selections} blocks × {num_nop_types} NOPs = {num_block_selections * num_nop_types} total actions")
    
    # Run random mutations
    print("\n" + "-" * 70)
    print("MUTATION LOG")
    print("-" * 70)
    
    iteration = 0
    done = False
    best_score = initial_score
    best_iteration = 0
    
    while not done and iteration < num_iterations:
        # Random action: randomly select block and NOP
        #block_idx = random.randint(0, num_block_selections - 1)
        # block_idx = 0
        nop_idx = random.randint(0, num_nop_types - 1)
        action = nop_idx
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        iteration += 1
        
        # Get mutation details
        mutation = env.mutation_history[-1]
        
        # Track best score
        if info['score'] < best_score:
            best_score = info['score']
            best_iteration = iteration
        
        # Print progress
        status_symbol = "✓" if info['score'] < initial_score else "✗"
        bypass_symbol = " [BYPASS!]" if info['bypassed'] else ""
        
        # Clean NOP string for display (remove newlines, truncate if needed)
        nop_display = str(mutation['nop_str']).replace('\n', '; ').strip()
        if len(nop_display) > 30:
            nop_display = nop_display[:27] + "..."
        
        # Format block info (shows all top-k blocks that were mutated)
        num_blocks_mutated = len(mutation['func_indices'])
        block_info = f"{num_blocks_mutated} blocks"
        if num_blocks_mutated <= 3:
            # Show individual blocks if few enough
            blocks_str = ", ".join([f"F{f}.B{b}" for f, b in zip(mutation['func_indices'], mutation['block_indices'])])
            block_info = f"[{blocks_str}]"
        
        print(f"Iter {iteration:3d} {status_symbol} | "
              f"Blocks: {block_info:25s} | "
              f"NOP: {nop_display:30s} | "
              f"Score: {info['score']:.6f} | "
              f"Δ: {info['score_delta']:+.6f} | "
              f"Reward: {reward:+.2f}{bypass_symbol}")
        
        # Check if bypassed
        if info['bypassed']:
            print("\n" + "=" * 70)
            print("SUCCESS! Malware bypassed the classifier!")
            print("=" * 70)
            break
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Initial score:     {initial_score:.6f}")
    print(f"Final score:       {info['score']:.6f}")
    print(f"Best score:        {best_score:.6f} (iteration {best_iteration})")
    print(f"Total mutations:   {iteration}")
    print(f"Score reduction:   {initial_score - info['score']:.6f} ({(initial_score - info['score']) / initial_score * 100:.2f}%)")
    print(f"Bypassed:          {'YES ✓' if info['bypassed'] else 'NO ✗'}")
    print(f"Threshold:         {threshold:.6f}")
    
    # Save results
    if save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mutation history
        hash_name = acfg.get('hash', 'unknown')
        history_file = output_dir / f"random_baseline_{hash_name}.json"
        
        # Convert mutation history to JSON-serializable format
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        results = {
            'experiment': 'random_baseline',
            'acfg_file': str(acfg_path),
            'hash': hash_name,
            'initial_score': float(initial_score),
            'final_score': float(info['score']),
            'best_score': float(best_score),
            'best_iteration': int(best_iteration),
            'threshold': float(threshold),
            'bypassed': bool(info['bypassed']),
            'num_mutations': int(iteration),
            'max_iterations': int(num_iterations),
            'seed': int(seed),
            'mutations': convert_to_serializable(env.mutation_history),
            'score_reduction_absolute': float(initial_score - info['score']),
            'score_reduction_percent': float((initial_score - info['score']) / initial_score * 100)
        }
        
        with open(history_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {history_file}")
        
        # Save final mutated ACFG
        mutated_acfg_file = output_dir / f"mutated_acfg_{hash_name}.json"
        with open(mutated_acfg_file, 'w') as f:
            json.dump(env.get_mutated_acfg(), f, indent=2)
        
        print(f"Mutated ACFG saved to: {mutated_acfg_file}")
    
    print("\n" + "=" * 70)
    print("BASELINE TEST COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run random mutation baseline test')
    parser.add_argument('acfg_file', type=str, 
                        help='Path to ACFG JSON file')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Maximum number of mutations (default: 30)')
    parser.add_argument('--threshold', type=float, default=0.14346,
                        help='Classification threshold (default: 0.14346 for 100fpr)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./srl_baseline_results',
                        help='Output directory for results (default: ./srl_baseline_results)')
    parser.add_argument('--no-save', action='store_true', 
                        help='Do not save results to file')
    
    args = parser.parse_args()
    
    run_random_baseline(
        acfg_path=args.acfg_file,
        num_iterations=args.iterations,
        threshold=args.threshold,
        seed=args.seed,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
