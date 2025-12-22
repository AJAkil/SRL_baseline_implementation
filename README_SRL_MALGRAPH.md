# SRL + MalGraph Implementation

Complete implementation of Semantics-preserving Reinforcement Learning (SRL) attack adapted for MalGraph Windows malware classifier.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SRL-MalGraph Attack Pipeline                    │
└─────────────────────────────────────────────────────────────────────┘

1. ACFG Extraction (IDA Pro)
   ├── Binary → IDA Pro → ACFG JSON
   ├── block_features: 11-dim vectors per basic block
   └── block_edges: CFG structure

2. DQN Agent (Trainable)
   ├── CFG Encoder (GNN): Extracts node embeddings
   ├── SortPooling Layer: Ranks blocks by importance ← TRAINABLE!
   ├── Q-Network: Selects (block, NOP) action
   └── Experience Replay: Stabilizes training

3. Environment (SRL-MalGraph)
   ├── State: ACFG + score + important_blocks
   ├── Action: (block_idx, nop_idx)
   ├── Mutation: Increment block_features
   └── Reward: Score reduction (continuous/sparse/binary)

4. MalGraph Classifier (Target)
   ├── Hierarchical GNN (CFG → FCG → Classification)
   └── Returns malware confidence score
```

## Files Structure

```
SRL_Implementation/srl/
├── srl_malgraph_nop_mapping.py      # Semantic NOP → Feature increment mapping
├── srl_malgraph_environment.py      # RL environment (state, action, reward)
├── srl_malgraph_dqn_agent.py       # Complete DQN agent with SortPooling
├── srl_malgraph_training.py        # Training loop and evaluation
├── DGCNN_embedding.py              # Original SRL's DGCNN (reference)
├── dqn_f.py                        # Original SRL's DQN (reference)
└── freq.py                         # Original SRL's Android environment (reference)
```

## Key Components

### 1. Semantic NOP Mapping (`srl_malgraph_nop_mapping.py`)

Maps MalGuise's semantic NOPs to MalGraph's 11-dimensional feature increments:

```python
# Example: PUSH/POP increments numIns (+2) and numDefIs (+2)
nop_str = "push eax\npop eax\n"
feature_increment = [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2]
                     #             ↑              ↑
                     #          numIns        numDefIs
```

**Feature Dimensions:**
- `[0]` numNc: Numeric constants
- `[1]` numSc: String constants
- `[2]` numAs: Arithmetic instructions
- `[3]` numCalls: Call instructions
- `[4]` numIns: Total instructions
- `[5]` numLIs: Logic instructions
- `[6]` numTIs: Transfer instructions
- `[7]` numCmpIs: Compare instructions
- `[8]` numMovIs: Move instructions
- `[9]` numTermIs: Termination instructions
- `[10]` numDefIs: Data definition instructions

### 2. Environment (`srl_malgraph_environment.py`)

Implements the RL environment following OpenAI Gym interface:

**State:**
```python
{
    'acfg': {...},                    # Current ACFG JSON
    'score': 0.95,                    # Malware confidence
    'num_mutations': 5,               # Mutations applied
    'important_blocks': [(func, blk, score), ...],  # Top-k blocks
    'terminated': False               # Episode done?
}
```

**Action:**
```python
(block_selection_idx, nop_idx)
# block_selection_idx: 0 to k-1 (which of top-k blocks)
# nop_idx: 0 to num_nops-1 (which semantic NOP)
```

**Reward:**
- **Continuous**: `reward = score_delta + (10 if bypassed else 0)`
- **Sparse**: `reward = 1 if score_decreased else 0` (+ terminal bonus)
- **Binary**: `reward = 1 if bypassed else 0`

### 3. DQN Agent (`srl_malgraph_dqn_agent.py`)

Complete DQN implementation with trainable SortPooling:

**Architecture:**
```
Input: ACFG JSON
  ↓
CFGGraphEncoder (GNN)
  - Graph convolutions: (A + I) * X * W
  - Outputs: node_embeddings [num_nodes, total_latent_dim]
  ↓
SortPoolingLayer (Trainable!)
  - Ranks nodes by embeddings[:, -1] (continuous WL color)
  - Selects top-k: [k, embedding_dim]
  ↓
QNetwork (1D Conv + MLP)
  - Outputs: Q-values for all (block, NOP) actions
  - Action space: k × num_nops
```

**Training Features:**
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration
- Gradient clipping
- Checkpoint saving/loading

### 4. Training Loop (`srl_malgraph_training.py`)

Orchestrates training with:
- Episode execution
- Evaluation every N episodes
- Checkpoint saving
- Training curve plotting
- Statistics tracking

## Usage

### Step 1: Extract ACFGs

Use MalGraph's IDA Pro extractor:

```bash
# Extract ACFGs from malware binaries
python extract_acfgs.py --input_dir /path/to/malware --output_dir /path/to/acfgs
```

Expected ACFG format:
```json
{
  "hash": "malware_hash",
  "function_number": 5,
  "function_names": ["start", "sub_401000", ...],
  "function_edges": [[1, 2], [3], ...],
  "acfg_list": [
    {
      "block_number": 3,
      "block_edges": [[0, 1], [1, 2], [2]],
      "block_features": [
        [2, 0, 1, 0, 5, 0, 1, 1, 1, 0, 0],  # Block 0: 11-dim features
        [1, 0, 0, 1, 3, 0, 0, 0, 1, 0, 0],  # Block 1
        [0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0]   # Block 2
      ]
    },
    ...
  ]
}
```

### Step 2: Initialize Components

```python
from srl_malgraph_nop_mapping import SemanticNOPMapper
from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_dqn_agent import DQNAgent
from srl_malgraph_training import SRLMalGraphTrainer

# 1. NOP mapper
nop_mapper = SemanticNOPMapper()
nop_list = nop_mapper.generate_malguise_nop_list()
print(f"Generated {len(nop_list)} NOPs")  # ~200-300 NOPs

# 2. Environment (requires MalGraph classifier)
env = SRLMalGraphEnvironment(
    malgraph_classifier=your_malgraph_classifier,
    nop_mapper=nop_mapper,
    threshold=0.14346,  # 100fpr threshold
    max_mutations=50,
    top_k_blocks=10,
    reward_type='continuous'
)

# 3. DQN Agent
agent = DQNAgent(
    num_nops=len(nop_list),
    k=10,
    input_dim=11,
    latent_dims=[32, 32, 32],
    hidden_dim=128,
    lr=0.001,
    gamma=0.9,
    batch_size=32
)

# 4. Trainer
trainer = SRLMalGraphTrainer(
    env=env,
    agent=agent,
    num_episodes=2500,
    max_steps_per_episode=50,
    log_dir='./logs',
    checkpoint_dir='./checkpoints'
)
```

### Step 3: Train

```python
# Load ACFG dataset
import json
acfgs = []
for acfg_file in os.listdir('/path/to/acfgs'):
    with open(os.path.join('/path/to/acfgs', acfg_file), 'r') as f:
        acfgs.append(json.load(f))

# Train
trainer.train(acfgs)
```

### Step 4: Evaluate

```python
# Load trained agent
agent.load_checkpoint('./checkpoints/final_model.pt')

# Evaluate on test set
test_acfgs = [...]  # Load test ACFGs
eval_stats = trainer.evaluate(test_acfgs, verbose=True)

print(f"Attack Success Rate: {eval_stats['success_rate']*100:.1f}%")
print(f"Avg Mutations: {eval_stats['avg_mutations']:.1f}")
print(f"Avg Score Reduction: {eval_stats['avg_score_reduction']:.4f}")
```

## Hyperparameters

### Agent
- `num_nops`: Number of semantic NOPs (~200-300 from MalGuise)
- `k`: Top-k blocks to consider (default: 10)
- `latent_dims`: GNN layer dimensions (default: [32, 32, 32])
- `hidden_dim`: Q-network hidden size (default: 128)
- `lr`: Learning rate (default: 0.001)
- `gamma`: Discount factor (default: 0.9)
- `epsilon_start`: Initial exploration (default: 1.0)
- `epsilon_end`: Final exploration (default: 0.1)
- `epsilon_decay`: Decay steps (default: 500)
- `batch_size`: Training batch size (default: 32)
- `memory_capacity`: Replay buffer size (default: 1000)
- `target_update_freq`: Target network update (default: 10)

### Environment
- `threshold`: Classification threshold (0.14346 for 100fpr, 0.91276 for 1000fpr)
- `max_mutations`: Maximum mutations per episode (default: 50)
- `top_k_blocks`: Number of important blocks (default: 10)
- `reward_type`: 'continuous', 'sparse', or 'binary' (default: 'continuous')
- `terminal_bonus`: Bonus for successful bypass (default: 10.0)

### Training
- `num_episodes`: Total training episodes (default: 2500)
- `eval_freq`: Evaluation frequency (default: 100 episodes)
- `save_freq`: Checkpoint save frequency (default: 500 episodes)

## Key Differences from Original SRL

| Aspect | Original SRL (Android) | SRL-MalGraph (Windows) |
|--------|------------------------|------------------------|
| **Target** | MaMaDroid (FCG-based) | MalGraph (ACFG-based) |
| **Graph** | Package/class call graph | CFG with basic blocks |
| **Features** | Package call frequencies | 11-dim block statistics |
| **Mutation** | Insert benign API calls | Increment feature counts |
| **NOPs** | Android methods | x86 semantic NOPs |
| **State** | FCG structure | ACFG with nested CFGs |

## Integration with MalGraph

You need to integrate with MalGraph's classifier. Update `srl_malgraph_environment.py`:

```python
# Replace dummy classifier with actual MalGraph
from Malguise.src.classifier.models.classifier import Classifier

malgraph_classifier = Classifier(
    clsf_params=ClassifierParams(
        base_url='/path/to/malgraph/model',
        threshold_type='1000fpr',
        clsf='malgraph'
    ),
    server_port=5001
)

env = SRLMalGraphEnvironment(
    malgraph_classifier=malgraph_classifier,
    ...
)
```

## Expected Results

Based on SRL paper:
- **Attack Success Rate (ASR)**: 60-80% on test set
- **Avg Mutations**: 10-30 per successful attack
- **Training Time**: ~6-12 hours on GPU for 2500 episodes
- **Score Reduction**: 0.3-0.7 on average

## Troubleshooting

### Issue: ACFG conversion fails
**Solution**: Ensure ACFG JSON has correct format with `acfg_list`, `block_features`, `block_edges`

### Issue: Agent doesn't learn (flat rewards)
**Solutions:**
- Reduce `epsilon_decay` for more exploration
- Increase `lr` to 0.01
- Try different `reward_type` (e.g., 'continuous' instead of 'sparse')

### Issue: Out of memory
**Solutions:**
- Reduce `batch_size` to 16
- Reduce `latent_dims` to [16, 16, 16]
- Reduce `memory_capacity` to 500

### Issue: Training unstable (loss spikes)
**Solutions:**
- Reduce `lr` to 0.0001
- Increase `target_update_freq` to 20
- Enable gradient clipping (already enabled at 10.0)

## Citation

If you use this code, please cite:

```bibtex
@article{srl_malgraph_2025,
  title={Adversarial Malware Generation via Semantics-preserving Reinforcement Learning on Control Flow Graphs},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, open a GitHub issue or contact [your email].
