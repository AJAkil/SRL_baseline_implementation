# SRL (Structure Reinforcement Learning) for Android Malware Evasion - Comprehensive Analysis Report

## Executive Summary

This codebase implements a **Deep Reinforcement Learning-based adversarial attack** against Android malware classifiers using **graph-based representations**. The system uses a **DQN (Deep Q-Network)** agent to mutate function call graphs of Android applications to evade detection while maintaining functionality.

---

## 1. System Architecture Overview

### High-Level Pipeline

```
Input: Android APK Function Call Graph
    ↓
Graph Representation (NetworkX Graph + Feature Matrix)
    ↓
State Encoding (GNN + DGCNN Embedding)
    ↓
DQN Agent (Action Selection: Insert NOP-equivalent calls)
    ↓
Graph Mutation (Add benign function calls)
    ↓
Classifier Testing (Check if malware is now misclassified)
    ↓
Output: Evaded malware sample or attack failure
```

---

## 2. Input Data Format & Dimensions

### 2.1 Raw Input Format

The system processes Android applications through extracted function call information:

**File Structure:**
```
data/attack_data_test/{granularity}_{classifier}/
    └── {sample_hash}/
        ├── all_functions.txt       # List of all functions in the APK
        └── func_calls.txt           # Function call pairs (caller → callee)
```

**Example `func_calls.txt` format:**
```
0 invoke-direct 5
2 invoke-virtual 8
...
```
Where numbers are indices into `all_functions.txt`

**Example `all_functions.txt` format:**
```
Lcom/example/MainActivity;->onCreate(Landroid/os/Bundle;)V
Lcom/example/Utils;->helper()V
Landroid/app/Activity;->setContentView(I)V
...
```

### 2.2 Graph Representation (`GNNGraph` class)

The system converts function calls into a graph structure with the following dimensions:

#### Family Granularity
- **Number of nodes**: 11 (Android API families: `android/`, `java/`, `com/google/`, etc.)
- **Node features**: `[11]` dimensional one-hot + weighted degree vector
- **Adjacency matrix**: `[11 × 11]` sparse matrix (family-to-family call frequencies)
- **Call times matrix**: `[11 × 11]` (normalized by row sum to create transition probabilities)

#### Package Granularity
- **Number of nodes**: 386 (Android API packages: `com.android.`, `java.util.`, etc.)
- **Node features**: `[386]` dimensional one-hot + weighted degree vector
- **Adjacency matrix**: `[386 × 386]` sparse matrix
- **Call times matrix**: `[386 × 386]` (normalized)

#### Class Granularity
- **Number of nodes**: 2431 (Individual classes)
- **Node features**: `[2431]` dimensional one-hot + weighted degree vector
- **Adjacency matrix**: `[2431 × 2431]` sparse matrix
- **Call times matrix**: `[2431 × 2431]` (normalized)

### 2.3 State Representation

Each state `s` is a `GNNGraph` object containing:

```python
class GNNGraph:
    num_nodes: int                    # Number of nodes in graph
    num_edges: int                    # Number of edges
    edge_pairs: np.ndarray           # Shape: [2 * num_edges], flattened edge list
    node_features: np.ndarray        # Shape: [num_nodes, num_node_feats]
    node_tags: list                  # Node type identifiers (optional)
    function_calls: list             # Original function call strings
    call_times: sparse matrix        # [num_nodes × num_nodes] adjacency with weights
    degs: list                       # Degree of each node
```

**Example Dimensions for Family Granularity:**
- `num_nodes = 11`
- `node_features.shape = [11, 11]` (one-hot encoding + degree weighting)
- `call_times.shape = [11, 11]` (sparse matrix)
- `edge_pairs.shape = [2 * num_edges]` (depends on graph connectivity)

---

## 3. Deep Q-Network (DQN) Architecture

### 3.1 Network Structure

The DQN consists of two identical networks (evaluation and target):

```
Input: Graph State (GNNGraph object)
    ↓
DGCNN Embedding Layer (Graph Neural Network)
    ├── Node Feature Preparation [num_nodes, num_node_feats]
    ├── Graph Convolution Layers (3 layers: [32, 32, 32] → [1])
    ├── SortPooling (Top-k nodes, k=6)
    └── Output: [batch_size, dense_dim]
    ↓
MLP Classifier
    ├── Linear Layer: [dense_dim] → [128]
    ├── ReLU Activation
    ├── Dropout (during training)
    └── Linear Layer: [128] → [num_actions]
    ↓
Output: Q-values for each action [batch_size, num_actions]
```

### 3.2 Action Space

**Actions = Number of NOP-equivalent function calls available**

For Family Granularity:
- **Action space size**: 7 actions (from `nops_f2.txt`)
- Each action represents inserting a specific benign function call pair

**Example NOP calls:**
```
Landroid/app/Activity;->onCreate()V
Landroid/content/Context;->getResources()V
Ljava/lang/Object;-><init>()V
... (7 total)
```

**Action Encoding:**
- One-hot vector: `[0, 0, 1, 0, 0, 0, 0]` = Action 2
- `np.argmax(action)` gives action index

### 3.3 DGCNN Embedding Details

**Input Dimensions:**
- Family: `[batch_size, num_nodes=11, node_feats=11]`
- Package: `[batch_size, num_nodes=386, node_feats=386]`
- Class: `[batch_size, num_nodes=2431, node_feats=2431]`

**Graph Convolution Process:**
1. **Message Passing**: 
   ```
   Y = (A + I) × X × W
   ```
   - A: Adjacency matrix
   - I: Identity matrix
   - X: Node features
   - W: Learnable weight matrix

2. **Normalization**:
   ```
   Y = D^(-1) × Y
   ```
   - D: Degree matrix

3. **Activation**: `tanh(Y)`

4. **Multiple Layers**: 3 GCN layers with latent dimensions `[32, 32, 32]`

5. **Concatenation**: Concatenate all layer outputs → `[num_nodes, 96]`

**SortPooling Layer:**
- Select top-k=6 nodes based on the last feature dimension
- If graph has < 6 nodes, pad with zeros
- Output shape: `[batch_size, 6, 96]`

**1D Convolution:**
- Flatten: `[batch_size, 1, 6*96=576]`
- Conv1D Layer 1: `[1, 16, kernel=576]` → `[batch_size, 16, 1]`
- MaxPool1D: stride=2 → `[batch_size, 16, 0]` (depends on input)
- Conv1D Layer 2: `[16, 32, kernel=3]`
- Output: `[batch_size, dense_dim]` where `dense_dim` varies

**MLP Head:**
- Input: `[batch_size, dense_dim]`
- Hidden: `[batch_size, 128]`
- Output: `[batch_size, num_actions]` (Q-values)

---

## 4. RL Agent Workflow

### 4.1 DQN Algorithm Parameters

```python
BATCH_SIZE = 5                      # Experience replay batch size
LR = 0.01                           # Learning rate
EPSILON = 0.9                       # ε-greedy exploration rate
GAMMA = 0.6                         # Discount factor
TARGET_NETWORK_REPLACE_FREQ = 50    # Target network update frequency
MEMORY_CAPACITY = 200               # Experience replay buffer size
```

### 4.2 Action Selection (`choose_action`)

**ε-greedy Strategy:**
```python
if random() < 0.9:  # 90% exploitation
    action = argmax(Q(s))  # Choose best action
else:  # 10% exploration
    action = random_action()
```

**Output:**
- One-hot action vector: `[0, 0, 0, 1, 0, 0, 0]` (7 dimensions for family granularity)

### 4.3 Graph Mutation Process (`getReward`)

**Step 1: Action to Function Call**
```python
action_index = np.argmax(action)  # e.g., 3
nop_call = self.nops[action_index]  # Get NOP function call
```

**Step 2: Insert Call into Graph**
```python
inset_calls = "Lcom1111/qihoo/util/Configuration;-><init>()V invoke-direct " + nop_call
data.append(inset_calls)  # Add to function call list
```

**Step 3: Recompute Graph State**
```python
new_state = self.data2state(data, mama_lidu)
# This recomputes:
# - Adjacency matrix
# - Call times (transition probabilities)
# - Node features
```

**Example Mutation:**

**Before (10 function calls):**
```
Graph: 11 nodes, 10 edges
Adjacency Matrix: [11×11] with 10 non-zero entries
```

**After inserting NOP (11 function calls):**
```
Graph: 11 nodes, 11 edges
Adjacency Matrix: [11×11] with 11 non-zero entries
Edge added: Lcom1111/qihoo → {NOP target family}
```

**Key Changes:**
- `num_edges` increases by 1
- Transition probabilities change (row normalization)
- Node features update (degree changes)

### 4.4 Reward Function

```python
def compute_reward(old_state, new_state, classifier):
    old_prob = classifier.predict(old_state)  # e.g., 0.95 (malware)
    new_prob = classifier.predict(new_state)  # e.g., 0.85 (malware)
    
    if attack_method == 'SRL':  # Structure RL
        if new_prob < old_prob:
            reward = 1  # Improved (reduced malware score)
        else:
            reward = 0  # No improvement
    
    if new_prob < 0.5:  # Successfully evaded
        is_done = True
        reward = 1
    
    return reward, is_done
```

**Reward Structure:**
- **Positive reward (+1)**: Malware score decreased
- **Zero reward (0)**: No improvement
- **Terminal state**: Malware score < 0.5 (classified as benign)

### 4.5 Experience Replay

**Experience Tuple:**
```python
class Data:
    state_now: GNNGraph       # Current graph state
    state_next: GNNGraph      # Next graph state after action
    reward: float             # Reward received
    action: np.ndarray        # Action taken (one-hot)
```

**Memory Buffer:**
- Circular buffer of size 200
- Stores `(s, a, r, s')` tuples
- Randomly sample 5 experiences per training step

### 4.6 Q-Learning Update

**Bellman Equation:**
```python
Q_target(s, a) = r + γ × max_a'(Q_target(s', a'))
```

**Training Process:**
```python
# Sample batch from memory
batch = random_sample(memory, size=5)

# Compute Q-values
q_eval = eval_net(s).gather(actions)  # Q(s, a)
q_next = target_net(s').max()         # max Q(s', a')

# Target Q-value
q_target = r + 0.1 × q_next  # γ = 0.1 in code

# Loss and backprop
loss = MSE(q_eval, q_target)
loss.backward()
optimizer.step()
```

**Target Network Update:**
- Every 50 steps: `target_net.load_state_dict(eval_net.state_dict())`

---

## 5. Training & Testing Workflow

### 5.1 Training Phase (2500 episodes)

```python
for episode in range(2500):
    if episode == 0:
        env.reset(data)           # Initialize graph
        s = env.state             # Get initial state
    
    a = dqn.choose_action(s)      # ε-greedy action selection
    s', r, done = env.step(a)     # Execute action, get reward
    
    dqn.store_transition(s, a, r, s')  # Store experience
    
    if memory_full:
        dqn.learn()               # Train on batch of 5 experiences
    
    if done:
        break                     # Successfully evaded
```

### 5.2 Testing Phase (400 episodes)

```python
for episode in range(400):
    a = dqn.choose_action(s)      # Use trained policy (90% greedy)
    s', r, done, y_pred, pert = env.step(a)
    
    log_action(a)
    log_prediction(y_pred)
    
    if done:
        log_success(pert)         # Log perturbation count
        break
```

**Outputs Logged:**
- `y_pred_record.txt`: Classifier predictions at each step
- `action_record.txt`: Actions taken
- `attack_infomation.txt`: Final perturbation count and time

---

## 6. Output Format & Dimensions

### 6.1 Graph Mutation Output

**Before Mutation:**
```python
state.num_nodes = 11
state.num_edges = 50
state.function_calls = [50 function call strings]
state.call_times.shape = [11, 11]  # Transition probabilities
```

**After Mutation (1 action):**
```python
new_state.num_nodes = 11           # Unchanged
new_state.num_edges = 51           # +1 edge
new_state.function_calls = [51 function call strings]  # +1 call
new_state.call_times.shape = [11, 11]  # Updated probabilities
```

### 6.2 Perturbation Count

```python
perturbation = new_state.num_edges - initial_state.num_edges
# Example: 51 - 50 = 1 NOP call added
```

### 6.3 Attack Success Metrics

**Logged in `attack_infomation.txt`:**
```
12   45.3
```
- `12`: Number of NOP calls added (perturbation count)
- `45.3`: Time taken (seconds)

**Logged in `y_pred_record.txt`:**
```
0.95
0.92
0.88
0.75
0.48  ← Successfully evaded (< 0.5)
```

---

## 7. Key Differences Between Attack Methods

### SRL (Structure RL - Full)
- **Training**: 2500 episodes with Q-learning
- **Reward**: Positive if malware score decreases
- **Testing**: 400 episodes using trained policy

### SRL_no (Structure RL - No Reward)
- **Training**: 2500 episodes (same as SRL)
- **Reward**: Positive only if score < 0.5 (binary)
- **Testing**: 400 episodes

### SRI (Structure Random - Informed)
- **No training**: Random action selection
- **Reward**: Keep action only if score improves
- **Testing**: 400 episodes with reward-based filtering

### SRI_no (Structure Random - Uninformed)
- **No training**: Pure random action selection
- **No reward filtering**: All actions kept
- **Testing**: 400 episodes

---

## 8. Classifier Integration

### 8.1 Supported Classifiers

- **DNN**: Deep Neural Network (MaMaDroid)
- **CNN**: Convolutional Neural Network
- **RF**: Random Forest
- **AdaBoost**: Adaptive Boosting
- **1-NN**: 1-Nearest Neighbor
- **3-NN**: 3-Nearest Neighbor

### 8.2 Classification Pipeline

```python
# Convert graph to feature vector
call_times_flat = state.call_times.reshape(-1, 11*11)  # [1, 121]

# PCA dimensionality reduction
if granularity == 'family':
    x_pca = pca_model.transform(call_times_flat)  # [1, 100]

# Classifier prediction
y_pred = classifier.predict(x_pca)  # [1, 1] (probability)
```

**Threshold**: 0.5
- `y_pred >= 0.5`: Malware
- `y_pred < 0.5`: Benign (attack success)

---

## 9. Computational Complexity

### 9.1 Graph Operations

**State Encoding:** O(N²) where N = number of nodes
- Family: O(11²) = O(121)
- Package: O(386²) = O(149K)
- Class: O(2431²) = O(5.9M)

### 9.2 DGCNN Forward Pass

**Per Layer:** O(E × d) where E = edges, d = feature dimension
- 3 GCN layers
- SortPooling: O(N log k)
- 1D Convolution: O(k × d)

**Total:** O(E × d) per graph

### 9.3 Training Complexity

- **Per episode**: 1 forward pass + 1 graph mutation
- **Per training step**: 5 forward passes (batch) + backprop
- **Total training**: ~2500 episodes + ~400 test episodes

---

## 10. Summary of Data Flow

```
Raw APK Function Calls
    ↓ [Parse & Extract]
List of (caller, callee) pairs
    ↓ [Graph Construction]
NetworkX Graph [N nodes, E edges]
    ↓ [Feature Engineering]
GNNGraph [node_features: N×N, call_times: N×N]
    ↓ [DGCNN Embedding]
Graph Embedding [batch, dense_dim]
    ↓ [MLP]
Q-values [batch, num_actions]
    ↓ [ε-greedy Selection]
Action [one-hot: num_actions]
    ↓ [Graph Mutation]
New GNNGraph [N nodes, E+1 edges]
    ↓ [Classifier]
Malware Probability [0-1]
    ↓ [Reward Computation]
Reward {0, 1}
    ↓ [Experience Replay & Q-Learning]
Updated Policy Network
```

---

## 11. Critical Design Choices

1. **Graph Granularity**: Trade-off between expressiveness (class) and tractability (family)
2. **NOP Selection**: Pre-defined benign calls that don't affect app functionality
3. **Reward Sparsity**: Only positive when improvement occurs (challenging for exploration)
4. **Target Network**: Stabilizes training by providing fixed Q-targets
5. **ε-greedy**: Balances exploitation (0.9) with exploration (0.1)

---

## 12. Limitations & Considerations

1. **Scalability**: Class-level graphs (2431 nodes) are computationally expensive
2. **Stealthiness**: Adding function calls may be detectable by dynamic analysis
3. **Functionality**: NOP calls must not break app behavior
4. **Transferability**: Trained on one classifier, may not transfer to others
5. **Detection**: Graph structure changes are visible to graph-based detectors

---

## Conclusion

This codebase implements a sophisticated RL-based adversarial attack that:
- Represents Android apps as function call graphs
- Uses DGCNN to encode graph structure
- Employs DQN to learn optimal graph mutation strategies
- Achieves evasion by inserting benign function calls
- Maintains functionality while fooling classifiers

The system demonstrates the vulnerability of graph-based malware detectors to structural perturbations guided by reinforcement learning.
