# ASR Tracking and Budget Configuration

## Attack Success Rate (ASR) Computation

### 1. **Per-Episode Tracking** 
[srl_malgraph_training.py:215](srl_malgraph_training.py#L215)
```python
self.episode_bypassed.append(1 if stats['bypassed'] else 0)
```
- Records 1 if sample bypassed classifier, 0 otherwise
- Stored in `self.episode_bypassed` list

### 2. **Rolling ASR (Last 10 Episodes)**
[srl_malgraph_training.py:222](srl_malgraph_training.py#L222)
```python
recent_bypassed = np.mean(self.episode_bypassed[-10:])
```
- Computed every 10 episodes for logging
- Shows: `Success Rate (last 10): XX.X%`

### 3. **Final Training ASR (Last 100 Episodes)**
[srl_malgraph_training.py:258](srl_malgraph_training.py#L258)
```python
print(f"Final success rate: {np.mean(self.episode_bypassed[-100:])*100:.1f}%")
```
- Displayed at end of training
- Uses last 100 episodes for stable estimate

### 4. **Evaluation ASR (Test Set)**
[srl_malgraph_training.py:314](srl_malgraph_training.py#L314)
```python
stats = {
    'success_rate': successes / len(eval_acfgs),
    'avg_mutations': np.mean(total_mutations),
    'avg_score_reduction': np.mean(score_reductions)
}
```
- Computed on separate evaluation set
- Uses greedy policy (no exploration)
- Stored in `self.eval_success_rates` every `eval_freq` episodes

## Injection Budget Configuration

### Original SRL Paper
- **Budget**: 5% of total instructions
- **Top-K Blocks**: 6 blocks per iteration
- **Works well** for small-scale mutations

### Modified for 1250 Blocks
**Problem**: With 1250 blocks mutated per iteration, 5% budget is exhausted quickly:
- Sample with 10,000 instructions → 500 instruction budget
- 1250 blocks × ~5 instructions/NOP = ~6,250 instructions needed
- **Result**: Budget exceeded in first iteration!

**Solution**: Increased budget to 10%

### Budget Settings
[srl_malgraph_environment.py:50](srl_malgraph_environment.py#L50)
```python
injection_budget_pct: float = 0.10,  # SRL paper: 5%, increased to 10% for 1250 blocks
```

### Budget Tracking
[srl_malgraph_environment.py:139-146](srl_malgraph_environment.py#L139-L146)
```python
# Track injection budget
self.original_total_ins = self._count_total_instructions(self.original_acfg)
self.budget_max_ins = int(self.injection_budget_pct * self.original_total_ins)
self.injected_ins = 0  # Track instructions added so far

print(f"  Original total instructions: {self.original_total_ins}")
print(f"  {self.injection_budget_pct*100:.0f}% budget allows: {self.budget_max_ins} additional instructions")
```

### Budget Enforcement
[srl_malgraph_environment.py:216-228](srl_malgraph_environment.py#L216-L228)
```python
# Check budget BEFORE mutation
projected_injected = self.injected_ins + nop_ins_count
if projected_injected > self.budget_max_ins:
    return (
        self._get_state(),
        0.0,
        True,  # Episode terminates
        {'score': self.current_score, 'bypassed': False, 'budget_exceeded': True}
    )
```

## Budget Calculation Example

**Sample**: 20,000 total instructions

| Budget % | Max Additional Instructions | 1250 Blocks × 5 ins/NOP | Fits? |
|----------|----------------------------|-------------------------|-------|
| 5%       | 1,000                      | 6,250                   | ❌ No  |
| 10%      | 2,000                      | 6,250                   | ❌ No  |
| 15%      | 3,000                      | 6,250                   | ❌ No  |
| 35%      | 7,000                      | 6,250                   | ✅ Yes |

**Note**: With `top_k_blocks=1250`, you may need to increase budget further or reduce K for realistic scenarios.

## Recommendations

### For Experimentation
1. **Use 10% budget** (current setting) as starting point
2. **Monitor** termination reasons in logs:
   ```
   ⚠️ Episode X ended: BUDGET EXCEEDED
   ```
3. **Adjust** if too many episodes terminate due to budget

### For SRL Paper Comparison
1. **Reset to 5%** and **reduce top_k_blocks to 6**:
   ```python
   # srl_malgraph_training.py
   K_TOP_BLOCKS = 6  # Original SRL setting
   
   # srl_malgraph_environment.py
   injection_budget_pct: float = 0.05  # Original SRL setting
   ```
2. This matches original paper exactly

### For Aggressive Attacks
1. **Increase budget to 20-30%** for 1250 blocks
2. Trade-off: More freedom to mutate vs. larger file size increase
3. Consider adversarial detectability (larger perturbations easier to detect)

## ASR Logging Locations

**Console Output**:
- Every 10 episodes: Rolling ASR (last 10)
- Every `eval_freq` episodes: Test set ASR
- End of training: Final ASR (last 100)

**Saved Files**:
- `logs/srl_malgraph/training_stats.json`: All episode results
- `checkpoints/srl_malgraph/checkpoint_epXXX_stats.json`: Stats at checkpoint time
- `logs/srl_malgraph/training_curves.png`: Plot of success rate over time

## Accessing ASR Programmatically

```python
# Load training statistics
import json

with open('logs/srl_malgraph/training_stats.json', 'r') as f:
    stats = json.load(f)

# Overall ASR
asr_overall = sum(stats['episode_bypassed']) / len(stats['episode_bypassed'])
print(f"Overall ASR: {asr_overall*100:.1f}%")

# ASR last 100 episodes
asr_final = sum(stats['episode_bypassed'][-100:]) / 100
print(f"Final ASR: {asr_final*100:.1f}%")

# Evaluation ASR over time
eval_asrs = stats['eval_success_rates']
print(f"Evaluation ASRs: {[f'{x*100:.1f}%' for x in eval_asrs]}")
```
