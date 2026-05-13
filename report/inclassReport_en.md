# In-Class Report: Robot Demonstration Quality Analysis

**Team:** Team Members:
Dazhi Yang
JinQing Jin
Xiao Wang

**Date:** March 25, 2026
**Dataset:** LeRobot (HuggingFace) — 8 datasets, 251,802 frames total

---

## 1. Loading Data

We loaded **8 LeRobot datasets** from HuggingFace using the `datasets` library, covering both simulation and real-robot environments.

**Source:** `lerobot/*` on HuggingFace Hub [1]
**Format:** Each dataset is loaded via `load_dataset(name, split='train')`, converted to a Pandas DataFrame, and saved as a Parquet file.

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

![Loading data code screenshot](images/01_load_data.png)

All datasets are merged into a single `robot_frames.parquet` file for unified analysis.

---

## 2. Checking Data Size and Feature Types

### Scale

- **Total frames:** 251,802
- **Total episodes:** ~1,956
- **Columns per frame:** 7

### Schema (column names and types)

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Dataset name (e.g., `pusht`, `berkeley_autolab_ur5`) |
| `episode_index` | int | Demonstration index |
| `frame_index` | int | Frame index within the episode |
| `observation_state` | list[float] | Robot state vector (2D–8D depending on dataset) |
| `action` | list[float] | Action command vector (2D–7D depending on dataset) |
| `next_reward` | float | Reward signal for the current frame |
| `next_done` | bool | Whether this is the last frame of the episode |

### Feature Dimensions (vary by dataset)

| Dataset Type | State Dim | Action Dim | Example |
|--------------|-----------|------------|---------|
| PushT (Sim) | 2D | 2D | 2D position |
| xArm (Sim) | 4D | 3–4D | Joint angles |
| Real Robot | 8D | 7D | Full joint state + gripper |

> **Note:** Because dimensions differ across datasets, we use the **L2 norm (magnitude)** to create dimension-agnostic scalar features.

---

## 3. Checking Missing Values and Imputation

### Missing Value Check

We checked all columns for null values:

![Missing value check code and results](images/03_missing_values.png)

**Result: No missing values found.**

### Imputation Method

**No imputation needed.** LeRobot is a well-curated research dataset. Every column across all 251,802 frames contains valid values.

---

## 4. Data Preprocessing

### 4.1 L2 Norm Transformation

Since state/action dimensions range from 2D to 8D, we compute the **L2 norm** for each frame to produce dimension-agnostic scalar features:

```python
frames['state_norm'] = frames['observation_state'].apply(
    lambda x: float(np.linalg.norm(x)))
frames['action_norm'] = frames['action'].apply(
    lambda x: float(np.linalg.norm(x)))
```

### 4.2 Episode-Level Aggregation

Our goal is to assess the quality of an entire trajectory, not individual frames. We therefore compress the data accordingly.

The raw data is **frame-level** (251,802 rows). We use PySpark to aggregate it to **episode-level** (~1,956 rows):
- Summary statistics: mean, std, min, max of state/action norms
- Episode length (number of frames per episode)
- Max and average reward per episode

![Episode-level aggregation code and output](images/04_episode_aggregation.png)

### 4.3 Feature Engineering (32 Kinematic Features via PySpark)

Prior work has shown that demonstration quality matters more than quantity [2], and filtering high-quality data can significantly improve policy training. Luo et al. (2024) [3] further found that `path_length` and `jerk` are the strongest predictors of learning performance.

The raw data contains only 7 columns (source, episode_index, frame_index, state, action, reward, done), which is limited in expressiveness. Drawing on the above literature, we extracted **32 kinematic features** from each trajectory to characterize it from multiple angles (smoothness, path efficiency, action consistency, etc.), enabling the model to better distinguish high- from low-quality demonstrations.

Our **32 motion-based features** are organized as follows:

| Category | Features | Count |
|----------|----------|-------|
| Basic Statistics | Mean, std, min, max, range of state/action norms | 10 |
| Smoothness | Action differences, jerk (mean/max) | 3 |
| Path Quality | Path efficiency, total path length | 2 |
| Consistency | Action consistency, correction count | 2 |
| Temporal | Velocity (mean/std/max), state/action trend, autocorrelation | 7 |
| Phase Analysis | Early/late action mean, warm-up ratio | 3 |
| Final State | Final stability, net state change, final deviation | 3 |
| Energy | Action energy expenditure (mean L2²) | 1 |
| Diversity | State diversity | 1 |

### 4.4 Quality Label Definition (Dual Labels)

```python
# Simulation: reward-based
reward_quality = (max_reward > per_dataset_median).astype(int)

# Real robot: efficiency-based (all rewards are 1.0)
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

2. **Model Evaluation** — Assess using AUC-ROC, accuracy, precision, and recall

3. **Sim-to-Real Analysis** — Compare simulation vs. real-world versions to test whether quality patterns transfer across domains

4. **Final Report & Presentation** — Compile findings into a PDF report and slide deck (deadline: April 3)

---

## References

[1] Cadene, R. et al. "LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch." HuggingFace, 2024. https://huggingface.co/lerobot

[2] Mandlekar, A. et al. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." *CoRL 2021*.

[3] Luo, J. et al. "Consistency Matters: Defining Demonstration Data Quality Metrics." 2024.
