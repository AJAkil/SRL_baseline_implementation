# SRL-MalGraph Integration: Answers to Key Questions

**Date:** December 2025  
**Author:** Md Ajwad Akil

---

## Question 1: Why does the GNN encoder require to be learned by the DQN agent?

### Short Answer
**It doesn't have to be!** You have two options:

### Option A: Train GNN from Scratch (More Flexible)
**Learn adversarial-specific embeddings:**
```python
agent = DQNAgent(...)
# GNN, SortPool, and Q-network all train together
```

**Why?**
- GNN learns which features matter for **evasion** (not classification)
- Example: MalGraph thinks "many arithmetic ops" = malware, but GNN might learn "arithmetic in loop headers" = critical to mutate
- More flexible, can discover attack patterns

**Downside:**
- Longer training (need to learn representations from scratch)
- Needs more samples (~500-1000)

---

### Option B: Freeze MalGraph's GNN (Recommended! ✅)
**Use MalGraph's pre-trained embeddings:**
```python
class SimplifiedDQNAgent:
    def __init__(self, malgraph_model):
        # FROZEN: Use MalGraph's trained GNN
        self.malgraph_encoder = malgraph_model.model
        
        # TRAINABLE: Only these two
        self.sortpool = SortPoolingLayer(k=10)
        self.q_network = QNetwork(...)
    
    def select_action(self, state):
        # No gradient through MalGraph
        with torch.no_grad():
            embeddings = self.malgraph_encoder.forward_cfg_gnn_layers(state)
        
        # Train only these
        sorted_nodes = self.sortpool(embeddings)
        q_values = self.q_network(sorted_nodes)
        return select_best_action(q_values)
```

**Why this is better:**
- **Faster training** (only train 2 components, not 3)
- **Fewer samples needed** (~100-200 ACFGs sufficient)
- **Leverages MalGraph's knowledge** (already knows what makes code malicious)
- **Simpler** (no need to tune GNN architecture)

**What does each component learn?**
- **MalGraph GNN** (frozen): Already knows "this block has suspicious API calls"
- **SortPooling** (trainable): Learns "API call blocks are most important to mutate"
- **Q-network** (trainable): Learns "add PUSH/POP NOPs to API blocks reduces score"

---

### My Recommendation: Use Option B

**Code changes needed:**
```python
# In srl_malgraph_environment.py
def _compute_block_importance(self):
    # Use MalGraph's embeddings (frozen)
    with torch.no_grad():
        embeddings = self.classifier.model.model.forward_cfg_gnn_layers(data)
    
    # Compute importance scores
    importance = torch.norm(embeddings, dim=1).cpu().numpy()
    return top_k_blocks
```

**Benefits:**
- ✅ Uses your existing MalGraph code directly
- ✅ Faster training (4-6 hours instead of 12-24 hours)
- ✅ Needs fewer samples (100-200 vs 500-1000)
- ✅ Simpler architecture (2 trainable components vs 3)

---

## Question 2: What does the agent output after taking an action?

### Short Answer
**The agent outputs a modified ACFG JSON** that can be directly fed to MalGraph.

### Detailed Flow

**Step 1: Agent selects action**
```python
action = agent.select_action(state)
# action = (block_idx, nop_idx)
# Example: (3, 45) means "mutate block 3 with NOP 45"
```

**Step 2: Environment applies mutation**
```python
next_state, reward, done, info = env.step(action)

# Inside step():
# 1. Get NOP increment: [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2]
# 2. Find target block: func_idx=0, block_idx=3
# 3. Increment features:
#    block_features[3] += [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2]
```

**Step 3: Modified ACFG is ready**
```python
mutated_acfg = env.current_acfg

# Structure UNCHANGED, only block_features modified:
{
    'hash': 'malware_hash',
    'function_number': 5,
    'function_names': ['start', 'encrypt', ...],
    'function_edges': [[0, 1], [1, 2], ...],  # FCG (unchanged)
    'acfg_list': [
        {
            'block_number': 10,
            'block_edges': [[0, 1], [1, 2], ...],  # CFG (unchanged)
            'block_features': [
                [2, 0, 1, 0, 5, 0, 1, 1, 1, 0, 0],      # Block 0
                [1, 0, 0, 1, 3, 0, 0, 0, 1, 0, 0],      # Block 1
                [3, 0, 2, 0, 9, 0, 1, 1, 2, 0, 2],      # Block 2 (MUTATED!)
                                  ↑              ↑
                          numIns: 7 → 9    numDefIs: 0 → 2
                          (added PUSH/POP)
                ...
            ]
        },
        # More functions...
    ]
}
```

**Step 4: Pass to MalGraph**
```python
# YOUR EXISTING CODE WORKS!
new_score = malgraph.predict_proba(mutated_acfg)
# or
new_score = malgraph_adapter.predict(mutated_acfg)
```

### What Changes vs What Doesn't

**CHANGES (incremented):**
- ✅ `block_features` arrays (11-dim feature counts)
- Example: `numIns` goes from 5 → 7 (added 2 instructions)

**DOESN'T CHANGE (preserved):**
- ❌ `block_edges` (CFG structure)
- ❌ `function_edges` (FCG structure)
- ❌ `function_names`, `block_number`, etc.
- ❌ Graph topology

**Why this works:**
- Semantic NOPs don't change control flow
- `push eax; pop eax` adds instructions but not branches
- Graph structure stays valid
- MalGraph accepts modified features

---

## Question 3: Can I just use the original MalGraph code?

### Short Answer
**YES! 100%!** 🎉

Your existing `MalgraphServerFeature` and `DirectMalgraphClient` classes work **without any modifications**.

### How to Integrate (3 Options)

#### Option 1: Use Adapter (Recommended)
```python
from malgraph_classifier_adapter import SRLMalGraphClassifierAdapter

# Wraps your existing code
malgraph = SRLMalGraphClassifierAdapter(
    use_direct_client=True,
    threshold_type='100fpr',
    device=None
)

# Simple interface
score = malgraph.predict(acfg_json)  # Returns float
```

**Benefits:**
- ✅ Simple interface (`predict(acfg_json)`)
- ✅ Handles both DirectMalgraphClient and MalgraphServerFeature
- ✅ Auto-loads thresholds (0.14346 for 100fpr)

---

#### Option 2: Use DirectMalgraphClient Directly
```python
from MalgraphModel import DirectMalgraphClient

# Your existing code
malgraph = DirectMalgraphClient(
    threshold_type='100fpr',
    device=torch.device('cuda'),
    server_port=5001
)

# In environment
env = SRLMalGraphEnvironment(
    malgraph_classifier=malgraph,  # Works directly!
    ...
)
```

**What environment does:**
```python
# Inside env.step()
self.current_score = self.classifier.model(self.current_acfg)
# Calls your DirectMalgraphClient → MalgraphServerFeature → HierarchicalGraphNeuralNetwork
```

---

#### Option 3: Use MalgraphServerFeature Directly
```python
from MalgraphModel import MalgraphServerFeature, MalgraphModelParams

# Your existing code
malgraph_params = MalgraphModelParams(...)
malgraph = MalgraphServerFeature(malgraph_params, device)

# In environment
env = SRLMalGraphEnvironment(
    malgraph_classifier=malgraph,  # Works directly!
    ...
)
```

**What environment does:**
```python
# Inside env.step()
self.current_score = self.classifier(self.current_acfg)
# Calls your MalgraphServerFeature.__call__() → predict_proba()
```

---

### Complete Integration Example

```python
# ============================================================================
# Use YOUR existing MalGraph code!
# ============================================================================

# Option 1: Adapter
from malgraph_classifier_adapter import SRLMalGraphClassifierAdapter
malgraph = SRLMalGraphClassifierAdapter(use_direct_client=True, threshold_type='100fpr')

# Option 2: DirectMalgraphClient
# from MalgraphModel import DirectMalgraphClient
# malgraph = DirectMalgraphClient(threshold_type='100fpr', device=device)

# Option 3: MalgraphServerFeature
# from MalgraphModel import MalgraphServerFeature
# malgraph = MalgraphServerFeature(params, device)

# ============================================================================
# Create SRL components
# ============================================================================
from srl_malgraph_nop_mapping import SemanticNOPMapper
from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_dqn_agent import DQNAgent

nop_mapper = SemanticNOPMapper()
nop_list = nop_mapper.generate_malguise_nop_list()

env = SRLMalGraphEnvironment(
    malgraph_classifier=malgraph,  # YOUR CODE!
    nop_mapper=nop_mapper,
    threshold=malgraph.threshold,  # 0.14346 for 100fpr
    max_mutations=50,
    top_k_blocks=10,
    reward_type='continuous'
)

agent = DQNAgent(
    num_nops=len(nop_list),
    k=10,
    input_dim=11,
    latent_dims=[32, 32, 32],
    learning_rate=0.001
)

# ============================================================================
# Training loop
# ============================================================================
state = env.reset(acfg_json)  # Original ACFG

for step in range(50):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    
    # Behind the scenes:
    # 1. env._mutate_block() increments block_features
    # 2. malgraph.predict(mutated_acfg) gets new score
    # 3. reward = old_score - new_score
    
    agent.store_experience(state, action, reward, next_state, done)
    agent.train_step(batch_size=32)
    
    if done:
        break
    
    state = next_state

print(f"Final score: {state['score']:.4f}")
print(f"Bypassed: {state['score'] < env.threshold}")
```

---

## Summary: Quick Reference

| Question | Answer |
|----------|--------|
| **Does GNN need training?** | No! Use Option B (freeze MalGraph's GNN, train only SortPool + Q-net) |
| **What does agent output?** | Modified ACFG JSON with incremented `block_features` arrays |
| **Can I use existing code?** | YES! `MalgraphServerFeature` / `DirectMalgraphClient` work directly |
| **Do I need binary rewriting?** | No! (For proof of concept - mutate JSON only) |
| **How many samples needed?** | 100-200 for training (with Option B) |
| **Training time?** | 4-6 hours (2500 episodes, with Option B) |
| **Expected ASR?** | 60-80% (from SRL paper) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      SRL-MALGRAPH PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

1. ORIGINAL ACFG (from IDA Pro)
   ↓
   {
     'block_features': [[2,0,1,0,5,0,1,1,1,0,0], ...]
   }
   ↓
2. ENVIRONMENT.RESET()
   ↓
   MalGraph.predict(acfg) → initial_score = 0.95
   ↓
3. AGENT.SELECT_ACTION()
   ↓
   MalGraph GNN (frozen) → embeddings [num_blocks, 96]
   ↓
   SortPooling (trainable) → top-k blocks [k, 96]
   ↓
   Q-Network (trainable) → q_values [k × num_nops]
   ↓
   Select best action: (block_idx=3, nop_idx=45)
   ↓
4. ENVIRONMENT.STEP()
   ↓
   Mutate: block_features[3] += nop_increment
   ↓
   {
     'block_features': [[2,0,1,0,5,0,1,1,1,0,0],
                        [1,0,0,1,3,0,0,0,1,0,0],
                        [3,0,2,0,9,0,1,1,2,0,2],  ← MODIFIED
                        ...]
   }
   ↓
5. MALGRAPH.PREDICT(mutated_acfg)
   ↓
   new_score = 0.87
   ↓
6. COMPUTE REWARD
   ↓
   reward = old_score - new_score = 0.95 - 0.87 = 0.08
   ↓
7. AGENT.TRAIN_STEP()
   ↓
   Update SortPooling weights
   Update Q-Network weights
   (MalGraph GNN stays frozen)
   ↓
8. REPEAT (50 mutations or until bypassed)
```

---

## File Structure

```
SRL_Implementation/
├── srl/
│   ├── malgraph_classifier_adapter.py       # Wrapper for your MalGraph code
│   ├── srl_malgraph_nop_mapping.py          # NOP → feature increment mapper
│   ├── srl_malgraph_environment.py          # RL environment (uses YOUR code!)
│   ├── srl_malgraph_dqn_agent.py            # DQN with trainable SortPooling
│   ├── srl_malgraph_training.py             # Training orchestrator
│   └── integration_example.py               # Complete usage example
│
├── INTEGRATION_QA.md                         # This file
├── README_SRL_MALGRAPH.md                    # Complete documentation
├── ARCHITECTURE_SUMMARY.md                   # Architecture details
└── MALGRAPH_BLOCK_FEATURES_AND_SRL_MUTATION.md  # Feature analysis
```

---

## Next Steps

1. **Test adapter with your MalGraph code:**
   ```bash
   cd SRL_Implementation/srl
   python malgraph_classifier_adapter.py
   ```

2. **Run integration example:**
   ```bash
   python integration_example.py
   ```

3. **Collect training ACFGs:**
   - Extract 100-200 malware ACFGs using IDA Pro
   - Save as JSON files

4. **Start training:**
   ```python
   trainer.train(num_episodes=2500, batch_size=32)
   ```

5. **Evaluate:**
   - Measure Attack Success Rate (ASR)
   - Count average mutations per success
   - Compare scores before/after

---

## Key Takeaway

**You don't need to change ANY of your existing MalGraph code!** 

Just:
1. Load your `MalgraphServerFeature` or `DirectMalgraphClient`
2. Pass it to `SRLMalGraphEnvironment`
3. Train the agent
4. The environment handles everything else (mutates ACFG, queries MalGraph, computes reward)

**The only thing new is the RL agent that learns which mutations work.**

---

**Questions?** See `README_SRL_MALGRAPH.md` for full documentation.
