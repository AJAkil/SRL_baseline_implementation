# Empty ACFG Fix - Summary

## Issue
Training crashed with `TypeError: float() argument must be a string or a number, not 'NoneType'` when MalGraph classifier returned `None` for some samples.

## Root Cause
Some malware samples have **empty `acfg_list`** (no functions/basic blocks), which causes:
1. `process_one_item()` to return `None` (line 83 in ModelPredForAttack.py)
2. `MalgraphServerFeature.predict_proba()` to return `None`
3. Adapter's `float(None)` call to fail

This happens when:
- Sample has no parseable functions (packed/obfuscated)
- IDA Pro extraction failed
- Binary is corrupted or stripped

## Fixes Applied

### 1. Classifier Adapter (malgraph_classifier_adapter.py)
**Lines 140-171**: Added validation and error handling

```python
# Validate ACFG structure
if isinstance(acfg_json, dict):
    num_acfgs = len(acfg_json.get('acfg_list', []))
    if num_acfgs == 0:
        print(f"⚠️ WARNING: ACFG has 0 functions!")
        ...

# Handle None return
if score is None:
    raise ValueError(
        f"ACFG processing failed - model returned None. "
        f"Sample: {acfg_json.get('hash', 'unknown')}. "
        f"This indicates an empty acfg_list."
    )
```

### 2. Environment (srl_malgraph_environment.py)
**Lines 105-121**: Pre-validate ACFG in `reset()` method

```python
def reset(self, acfg_json: Dict) -> Dict:
    # Validate ACFG structure
    if 'acfg_list' not in acfg_json:
        raise ValueError(f"Invalid ACFG: missing 'acfg_list' field")
    if len(acfg_json.get('acfg_list', [])) == 0:
        raise ValueError(
            f"Invalid ACFG for {acfg_json.get('hash', 'unknown')}: "
            f"acfg_list is empty (no functions found)"
        )
    ...
```

### 3. Training Loop (srl_malgraph_training.py)
**Lines 180-204**: Skip invalid samples gracefully

```python
# Validate ACFG has functions before training
if 'acfg_list' not in acfg or len(acfg.get('acfg_list', [])) == 0:
    print(f"⚠️ SKIPPING Episode {episode}: Sample has no functions")
    episode += 1
    continue

# Wrap in try-catch
try:
    stats = self.train_episode(acfg, episode)
except ValueError as e:
    if "ACFG processing failed" in str(e):
        print(f"⚠️ ERROR in Episode {episode}: {e}")
        print(f"   Skipping this sample...")
        episode += 1
        continue
    else:
        raise
```

## New Utility: check_acfg_validity.py
Scan your dataset to find problematic samples **before** training:

```bash
# Check entire directory
python check_acfg_validity.py /path/to/acfg/folder

# Check single file
python check_acfg_validity.py /path/to/sample.json
```

Output example:
```
Checking 2000 ACFG files...
=======================================================================
VALIDATION SUMMARY
=======================================================================
Total samples:   2000
Valid samples:   1987 (99.4%)
Invalid samples: 13 (0.6%)

INVALID SAMPLES (13 total):
  ✗ sample1.json: Empty acfg_list (0 functions)
  ✗ sample2.json: Empty acfg_list (0 functions)
  ...
```

## Impact
- **Training continues** even if some samples are invalid
- **Clear logging** shows which samples are skipped
- **Pre-training validation** helps clean your dataset

## Recommendation
Before starting long training runs:
1. Run `check_acfg_validity.py` on your ACFG directory
2. Remove or regenerate invalid samples
3. This ensures maximum training efficiency

## Next Steps
If you continue seeing errors:
1. Check the specific sample file that fails
2. Verify IDA Pro extraction is working
3. Consider filtering samples with `min_num_functions >= 1`
