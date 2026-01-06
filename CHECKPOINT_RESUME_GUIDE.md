# Checkpoint & Resume Training Guide

## Overview

The SRL-MalGraph training system now supports **comprehensive checkpointing** and **resuming training** from any saved checkpoint. If training is interrupted (power loss, OOM, manual stop, etc.), you can resume exactly where you left off.

## What Gets Saved in Checkpoints

Each checkpoint includes:

1. **Q-Network weights** - Policy network parameters
2. **Target Q-Network weights** - Target network parameters  
3. **Optimizer state** - Adam/RMSprop optimizer momentum, learning rate schedule
4. **Replay buffer** - All stored experiences (states, actions, rewards, transitions)
5. **Training statistics** - Episode rewards, losses, success rates
6. **Episode counter** - Current episode number for resuming
7. **Epsilon decay state** - Current exploration rate and decay counter

## Automatic Checkpointing

Checkpoints are **automatically saved** during training:

- **Every 20 episodes** (default): `checkpoint_ep20.pt`, `checkpoint_ep40.pt`, etc.
- **After final episode**: `final_model.pt`

Files saved:
```
checkpoints/
├── checkpoint_ep20.pt         # Agent checkpoint (networks, replay buffer, optimizer)
├── checkpoint_ep20_stats.json # Training statistics (rewards, losses, etc.)
├── checkpoint_ep40.pt
├── checkpoint_ep40_stats.json
└── final_model.pt
```

## Method 1: Resume in Main Training Script

Edit `srl_malgraph_training.py`:

```python
# Around line 510
RESUME_CHECKPOINT = './checkpoints/checkpoint_ep500.pt'  # Set to checkpoint path

if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"\n⚠️  Resuming training from checkpoint: {RESUME_CHECKPOINT}")
    trainer.load_checkpoint(RESUME_CHECKPOINT)

trainer.train(acfgs, acfgs_file_names)
```

Then run:
```bash
python srl_malgraph_training.py
```

## Method 2: Use Resume Script (Recommended)

Use the dedicated resume script:

```bash
python resume_training_example.py --checkpoint ./checkpoints/checkpoint_ep500.pt
```

This script:
- ✓ Validates checkpoint exists
- ✓ Re-initializes all components
- ✓ Loads checkpoint state
- ✓ Shows resume statistics
- ✓ Continues training

## What Happens When Resuming

```
1. Loading ACFG dataset...
   Loaded 2000 samples

2. Initializing semantic NOP mapper...
   Generated 28 semantic NOPs

3. Initializing MalGraph classifier...
   ✓ Classifier ready

4. Initializing environment...
   ✓ Environment ready

5. Initializing DQN agent...
   ✓ Agent initialized

6. Initializing trainer...
   ✓ Trainer ready

7. Loading checkpoint...
   Checkpoint loaded from ./checkpoints/checkpoint_ep500.pt (restored 3000 experiences)
   Training state loaded: resuming from episode 500
     Previous episodes: 500
     Replay buffer size: 3000
   ✓ Loaded! Resuming from episode 501

================================================================================
RESUMING TRAINING FROM EPISODE 501
================================================================================
Previous training statistics:
  - Episodes completed: 500
  - Avg reward (last 100): 0.427
  - Success rate (last 100): 12.3%
  - Replay buffer size: 3000
  - Current epsilon: 0.456
================================================================================

Training: 501/10000 ██░░░░░░░░░░░░░░░░░░ 5%
```

## Manual Checkpoint Save/Load

### Save Checkpoint Manually

```python
# During training
trainer.save_full_checkpoint('./checkpoints/manual_save.pt', episode=150)
```

### Load Checkpoint Manually

```python
# Before training
trainer.load_checkpoint('./checkpoints/checkpoint_ep500.pt')

# Then start/resume training
trainer.train(acfgs, acfgs_file_names)
```

## Checkpoint Details

### Agent Checkpoint (`.pt` file)
```python
{
    'q_network': state_dict,           # Q-network parameters
    'target_q_network': state_dict,    # Target network parameters
    'optimizer': state_dict,           # Optimizer state (momentum, etc.)
    'steps_done': int,                 # Total steps for epsilon decay
    'learn_step_counter': int,         # Learning steps for target update
    'num_nops': int,                   # Action space size
    'embedding_dim': int,              # State embedding dimension
    'replay_memory': deque,            # Full replay buffer
    'epsilon_start': float,            # Epsilon decay start
    'epsilon_end': float,              # Epsilon decay end
    'epsilon_decay': int               # Epsilon decay steps
}
```

### Training Statistics (`.json` file)
```json
{
    "episode": 500,
    "episode_rewards": [0.5, 0.3, ...],
    "episode_lengths": [12, 18, ...],
    "episode_final_scores": [0.85, 0.92, ...],
    "episode_bypassed": [0, 0, 1, ...],
    "training_losses": [0.023, 0.019, ...],
    "eval_success_rates": [0.05, 0.08, 0.12]
}
```

## Example Workflow

### Normal Training
```bash
# Start fresh training
python srl_malgraph_training.py
```

### Training Interrupted at Episode 784
```
Episode 784/10000
  Avg Reward (last 10): 0.523
  Success Rate (last 10): 18.5%
  ...
KeyboardInterrupt  # Ctrl+C or system crash
```

### Resume from Last Checkpoint (Episode 780)
```bash
# Checkpoints saved every 20 episodes
python resume_training_example.py --checkpoint ./checkpoints/checkpoint_ep780.pt
```

Output:
```
RESUMING TRAINING FROM EPISODE 781
Previous training statistics:
  - Episodes completed: 780
  - Avg reward (last 100): 0.489
  - Success rate (last 100): 16.2%
  - Replay buffer size: 3000
  - Current epsilon: 0.312

Training continues from episode 781...
```

## Important Notes

1. **Replay buffer is preserved** - All 3000 experiences are saved and restored
2. **Epsilon decay continues** - Exploration rate resumes from saved state
3. **Optimizer momentum preserved** - Adam/RMSprop continues smoothly
4. **Statistics are cumulative** - Training curves show full history
5. **No data loss** - Everything needed to continue is saved

## Troubleshooting

### "Checkpoint not found"
```bash
# List available checkpoints
ls -lh checkpoints/

# Use correct path
python resume_training_example.py --checkpoint ./checkpoints/checkpoint_ep500.pt
```

### "Dimension mismatch"
Ensure configuration matches:
- `num_nops` must be 28
- `embedding_dim` must be 200
- `batch_size`, `memory_capacity` should match

### "Out of memory after resuming"
The replay buffer consumes memory. With 3000 experiences:
- Each experience: ~200-dim embeddings + metadata
- Total: ~2-5 MB (manageable)

## Best Practices

1. **Save frequently** - Set `save_freq=20` or lower for long training runs
2. **Keep multiple checkpoints** - Don't delete old checkpoints until training is complete
3. **Test resume** - Periodically test that checkpoints can be loaded
4. **Monitor disk space** - Each checkpoint ~50-100 MB with full replay buffer

## Configuration

Adjust save frequency in training script:

```python
trainer = SRLMalGraphTrainer(
    env=env,
    agent=agent,
    num_episodes=10000,
    max_steps_per_episode=30,
    save_freq=20,  # Save every 20 episodes (default)
    log_dir='./logs/srl_malgraph',
    checkpoint_dir='./checkpoints'
)
```

For very long training (100K+ episodes), increase to `save_freq=100` or `save_freq=500`.
