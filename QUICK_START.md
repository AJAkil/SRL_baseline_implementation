# SRL-MalGraph Quick Start

**TL;DR:** Your existing MalGraph code works as-is. Just plug it into the SRL environment.

---

## 30-Second Setup

```python
# 1. Load YOUR existing MalGraph code
from MalgraphModel import DirectMalgraphClient
malgraph = DirectMalgraphClient(threshold_type='100fpr', device=device)

# 2. Create SRL components
from srl_malgraph_environment import SRLMalGraphEnvironment
from srl_malgraph_nop_mapping import SemanticNOPMapper
from srl_malgraph_dqn_agent import DQNAgent

nop_mapper = SemanticNOPMapper()
env = SRLMalGraphEnvironment(malgraph_classifier=malgraph, nop_mapper=nop_mapper)
agent = DQNAgent(num_nops=200, k=10)

# 3. Train
state = env.reset(acfg_json)
action = agent.select_action(state)
next_state, reward, done, info = env.step(action)
```

---

## What You Asked vs What You Get

| Your Question | Answer |
|---------------|--------|
| "Why does GNN need training?" | **It doesn't!** Use Option B (freeze MalGraph's GNN) |
| "What does agent output?" | **Modified ACFG JSON** (incremented `block_features`) |
| "Can I use my MalGraph code?" | **YES! 100%!** No changes needed |

---

## Architecture (Simplified)

```
Original ACFG → MalGraph → score = 0.95 (malware)
                    ↓
         Agent mutates block_features
                    ↓
Mutated ACFG  → MalGraph → score = 0.10 (bypassed!)
```

**What changes:** Only `block_features` arrays (instruction counts)  
**What doesn't:** Graph structure (edges), function names, etc.

---

## Training (Recommended Settings)

```python
# Option B: Freeze MalGraph's GNN (recommended)
agent = DQNAgent(
    num_nops=200,        # ~200-300 semantic NOPs
    k=10,                # Top-10 blocks
    input_dim=11,        # MalGraph's 11-dim features
    latent_dims=[32,32,32],
    learning_rate=0.001
)

trainer.train(
    num_episodes=2500,   # 4-6 hours
    batch_size=32,
    eval_interval=100
)
```

**Expected results:**
- ASR: 60-80%
- Mutations per success: 10-30
- Training time: 4-6 hours

---

## Files You Need

1. **malgraph_classifier_adapter.py** - Wrapper for your MalGraph code
2. **srl_malgraph_environment.py** - RL environment (uses your code)
3. **srl_malgraph_nop_mapping.py** - NOP → feature mapper
4. **srl_malgraph_dqn_agent.py** - DQN with trainable SortPooling
5. **integration_example.py** - Complete usage example

---

## Quick Test

```bash
cd SRL_Implementation/srl

# Test adapter
python malgraph_classifier_adapter.py

# Test integration
python integration_example.py
```

---

## Full Documentation

- **INTEGRATION_QA.md** - Answers to your 3 questions (detailed)
- **README_SRL_MALGRAPH.md** - Complete usage guide
- **ARCHITECTURE_SUMMARY.md** - Architecture deep dive

---

**Bottom line:** You're ready to train! Just provide 100-200 malware ACFGs and run `trainer.train()`.
