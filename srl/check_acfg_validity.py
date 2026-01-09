#!/usr/bin/env python
"""
Check ACFG dataset for samples with empty acfg_list.
This helps identify problematic samples before training.

Author: GitHub Copilot
Date: January 2026
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm


def check_acfg_validity(acfg_path):
    """
    Check if an ACFG file is valid for MalGraph.
    
    Args:
        acfg_path: Path to ACFG JSON file
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(acfg_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested result field
        if 'result' in data:
            acfg = json.loads(data['result'])
        else:
            acfg = data
        
        # Check for acfg_list
        if 'acfg_list' not in acfg:
            return False, "Missing 'acfg_list' field"
        
        acfg_list = acfg.get('acfg_list', [])
        if len(acfg_list) == 0:
            return False, f"Empty acfg_list (0 functions)"
        
        # Count blocks
        total_blocks = sum(len(cfg.get('block_features', [])) for cfg in acfg_list)
        
        return True, f"Valid: {len(acfg_list)} functions, {total_blocks} blocks"
        
    except Exception as e:
        return False, f"Exception: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_acfg_validity.py <acfg_directory_or_file>")
        print("\nExamples:")
        print("  python check_acfg_validity.py /path/to/acfg/folder")
        print("  python check_acfg_validity.py /path/to/single/file.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if input_path.is_file():
        # Single file check
        is_valid, msg = check_acfg_validity(input_path)
        if is_valid:
            print(f"✓ {input_path.name}: {msg}")
        else:
            print(f"✗ {input_path.name}: {msg}")
        sys.exit(0 if is_valid else 1)
    
    elif input_path.is_dir():
        # Directory scan
        acfg_files = list(input_path.glob("*.json"))
        
        if len(acfg_files) == 0:
            print(f"No JSON files found in {input_path}")
            sys.exit(1)
        
        print(f"Checking {len(acfg_files)} ACFG files in {input_path}...")
        print()
        
        valid_count = 0
        invalid_samples = []
        
        for acfg_file in tqdm(acfg_files, desc="Validating"):
            is_valid, msg = check_acfg_validity(acfg_file)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_samples.append((acfg_file.name, msg))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples:   {len(acfg_files)}")
        print(f"Valid samples:   {valid_count} ({valid_count/len(acfg_files)*100:.1f}%)")
        print(f"Invalid samples: {len(invalid_samples)} ({len(invalid_samples)/len(acfg_files)*100:.1f}%)")
        
        if invalid_samples:
            print(f"\n{'='*70}")
            print(f"INVALID SAMPLES ({len(invalid_samples)} total):")
            print(f"{'='*70}")
            for name, reason in invalid_samples[:20]:  # Show first 20
                print(f"  ✗ {name}: {reason}")
            
            if len(invalid_samples) > 20:
                print(f"  ... and {len(invalid_samples) - 20} more")
            
            print(f"\nRecommendation: Remove or regenerate these samples")
        else:
            print(f"\n✓ All samples are valid!")
        
        print()
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
