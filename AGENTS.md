# AGENTS.md — Project Instructions for AI Coding Agents

## Project Overview
Big data course project + research: **Robot Demonstration Quality Analysis**.

Research question: Can interpretable trajectory statistics identify high-quality robot demonstrations at scale?

Based on: Mandlekar et al. (CoRL 2021), DemInf (2025), QoQ (2026).

## Setup
```bash
pip install -r requirements.txt
# or in Google Colab:
# !pip install pyspark datasets pyarrow torch -q
```

## Repository Structure
```
notebooks/
  01_data_download.ipynb        # Download lerobot datasets → parquet
  02_eda.ipynb                  # EDA and visualization
  03_feature_engineering.ipynb  # 7 quality metrics with PySpark
  04_modeling.ipynb             # Model training and comparison
data/                           # gitignored — use Google Drive
report/                         # PDF report and slides
test_data_structure.py          # 54-test validation suite
requirements.txt
```

## Data
8 HuggingFace lerobot datasets, 251,802 total frames:

**Simulation datasets:**
| Dataset | Rows | state dim | action dim |
|---------|------|-----------|------------|
| lerobot/pusht | 25,650 | 2 | 2 |
| lerobot/xarm_lift_medium | 20,000 | 4 | 4 |
| lerobot/xarm_lift_medium_replay | 20,000 | 4 | 4 |
| lerobot/xarm_push_medium | 20,000 | 4 | 3 |
| lerobot/xarm_push_medium_replay | 20,000 | 4 | 3 |

**Real robot datasets (all state=8D, action=7D — can merge directly):**
| Dataset | Rows | Institution | Task |
|---------|------|-------------|------|
| lerobot/berkeley_autolab_ur5 | 97,939 | UC Berkeley | Multi-task tabletop |
| lerobot/columbia_cairlab_pusht_real | 27,808 | Columbia | Push-T (real) |
| lerobot/nyu_door_opening_surprising_effectiveness | 20,405 | NYU | Door opening |

Note: `columbia_cairlab_pusht_real` + `lerobot/pusht` form a natural Sim-to-Real pair.

Load with:
```python
from datasets import load_dataset
ds = load_dataset('lerobot/pusht', split='train')
```

## Target Variable
Binary quality label per episode (not per frame):
```python
# Use per-dataset median — xarm_push has negative rewards
quality = (max_reward > per_dataset_median).astype(int)
```

## Features (7 quality metrics, computed per episode)
```python
smoothness          # -mean(|jerk|), higher = smoother
path_efficiency     # direct_distance / actual_path_length
action_consistency  # -std(actions)
correction_count    # direction reversals / episode_length
time_efficiency     # -episode_length
state_diversity     # std(states)
final_stability     # -std(states[-5:])
```

## ML Stack
- **PySpark MLlib** (required by course): DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
- **PyTorch** (optional comparison): MLP
- **Evaluation**: AUC-ROC, SHAP feature importance

## Key Constraints
- PySpark must be used for ML training (course requirement)
- State/action dimensions differ across datasets — use magnitude-based aggregations
- xarm_push reward is negative distance; do not use fixed threshold
- Data files go to Google Drive, not Git

## Testing
```bash
python3 test_data_structure.py
# Expected: 54/54 tests pass
```

## Coding Conventions
- Notebooks are the primary deliverable
- Use PySpark DataFrames for anything over 10k rows
- Use pandas only for visualization and small computations
- No hardcoded file paths — use relative paths or Drive mount points
- All ML must go through PySpark Pipeline API
