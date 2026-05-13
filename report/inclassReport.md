# In-Class Report: Robot Demonstration Quality Analysis

**Team:** Alex's Group
**Date:** March 25, 2026
**Dataset:** LeRobot (HuggingFace) — 8 datasets, 251,802 frames total

---

## 1. Load Data

We loaded **8 LeRobot datasets** from HuggingFace using the `datasets` library, covering both simulation and real-robot environments.

**Data source:** `lerobot/*` on HuggingFace Hub
**Format:** Each dataset loaded via `load_dataset(name, split='train')`, then converted to Pandas DataFrame and saved as Parquet files.

| Group | Dataset | Frames | Episodes |
|-------|---------|--------|----------|
| Simulation | `lerobot/pusht` | 25,650 | 206 |
| Simulation | `lerobot/xarm_lift_medium` | 20,000 | 100 |
| Simulation | `lerobot/xarm_lift_medium_replay` | 20,000 | 100 |
| Simulation | `lerobot/xarm_push_medium` | 20,000 | 100 |
| Simulation | `lerobot/xarm_push_medium_replay` | 20,000 | 100 |
| Real Robot | `lerobot/berkeley_autolab_ur5` | 97,939 | 1,000 |
| Real Robot | `lerobot/columbia_cairlab_pusht_real` | 27,808 | 150 |
| Real Robot | `lerobot/nyu_door_opening_surprising_effectiveness` | 20,405 | 200 |
| **Total** | | **251,802** | **~1,956** |

**Loading code:**
```python
from datasets import load_dataset
ds = load_dataset('lerobot/pusht', split='train')
```

All datasets were merged into a single `robot_frames.parquet` file for unified analysis.

---

## 2. Check Data Size, Types of Features

### Data Size
- **Total frames:** 251,802
- **Total episodes:** ~1,956
- **Columns per frame:** 7

### Schema (columns and types)

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Dataset name (e.g., `pusht`, `berkeley_autolab_ur5`) |
| `episode_index` | int | Which demonstration episode |
| `frame_index` | int | Frame number within the episode |
| `observation_state` | list[float] | Robot state vector (2D–8D depending on dataset) |
| `action` | list[float] | Action command vector (2D–7D depending on dataset) |
| `next_reward` | float | Reward signal for this frame |
| `next_done` | bool | Whether this is the last frame of the episode |

### Feature Dimensions (vary by dataset)

| Dataset Type | State Dim | Action Dim | Example |
|-------------|-----------|------------|---------|
| PushT (Sim) | 2D | 2D | 2D position |
| xArm (Sim) | 4D | 3–4D | Joint angles |
| Real Robot | 8D | 7D | Full joint state + gripper |

> **Note:** Since dimensions differ across datasets, we use **L2 norm (magnitude)** to create dimension-agnostic features.

---

## 3. Check Missing Values & Imputation

### Missing Value Check

We performed a null check on all columns:

```python
null_count = robot_frames[['source','episode_index','frame_index',
                            'next_reward','next_done']].isnull().sum().sum()
# Result: 0
```

**Result: No missing values found** in any of the 5 scalar columns.

For the two list columns (`observation_state`, `action`), all entries contain valid numeric lists — no `None` or empty lists detected.

### Imputation Method

**No imputation needed.** The LeRobot datasets are well-curated research datasets with complete data. All 251,802 frames have valid values for every column.

However, there is a semantic "missing" pattern worth noting:
- **Real robot datasets** (Berkeley, Columbia, NYU) all have `max_reward = 1.0` for every episode (all demonstrations are "successful"), so reward alone cannot distinguish quality.
- **Solution:** We defined a **dual-label system** — `reward_quality` for simulation data (reward-based) and `efficiency_quality` for real robot data (shorter episode = more efficient = higher quality).

---

## 4. Preprocessing

### 4.1 L2 Norm Transformation
Since state/action dimensions vary (2D to 8D), we compute the **L2 norm** for each frame to create dimension-agnostic scalar features:

```python
frames['state_norm'] = frames['observation_state'].apply(
    lambda x: float(np.linalg.norm(x)))
frames['action_norm'] = frames['action'].apply(
    lambda x: float(np.linalg.norm(x)))
```

### 4.2 Episode-Level Aggregation
Raw data is **frame-level** (251,802 rows). We aggregate to **episode-level** (~1,956 rows) using PySpark:
- Basic statistics: mean, std, min, max of state/action norms
- Episode length (frame count per episode)
- Max/average reward per episode

### 4.3 Feature Engineering (32 Kinematic Features via PySpark)
We engineered **32 motion-based features** (no reward leakage) including:

| Category | Features | Count |
|----------|----------|-------|
| Basic statistics | state/action mean, std, min, max, range | 10 |
| Smoothness | action diffs, jerk (mean/max) | 3 |
| Path quality | path_efficiency, total_path_length | 2 |
| Consistency | action_consistency, correction_count | 2 |
| Temporal | velocity (mean/std/max), state/action trend, autocorrelation | 7 |
| Phase analysis | early/late action mean, warmup ratio | 3 |
| Final state | final_stability, state_net_change, final_deviation | 3 |
| Energy | action_effort (L2² mean) | 1 |
| Diversity | state_diversity | 1 |

### 4.4 Quality Label Definition (Dual-Label)

```python
# Simulation: reward-based
reward_quality = (max_reward > per_dataset_median).astype(int)

# Real robot: efficiency-based (all rewards = 1.0)
efficiency_quality = (episode_length < per_dataset_median).astype(int)
```

### 4.5 Tools Used
- **PySpark** for distributed aggregation and feature computation
- **Pandas** for data I/O and post-processing
- **NumPy** for vector norm calculations

---

## 5. Next Steps

1. **Classification Modeling** — Train and compare 4 models on the 32 features:
   - Decision Tree (PySpark MLlib)
   - Gradient Boosted Trees (PySpark MLlib)
   - Random Forest (PySpark MLlib)
   - MLP Neural Network (PyTorch)

2. **Model Evaluation** — Evaluate using AUC-ROC, accuracy, precision, recall

3. **Feature Importance Analysis** — Use SHAP values to identify which kinematic features best predict demonstration quality

4. **Ablation Study** — Test model performance with subsets of features to validate robustness

5. **Sim-to-Real Analysis** — Compare PushT simulation vs PushT real robot to examine whether quality patterns transfer across domains

6. **Final Report & Presentation** — Compile findings into PDF report and slides (due April 3)
