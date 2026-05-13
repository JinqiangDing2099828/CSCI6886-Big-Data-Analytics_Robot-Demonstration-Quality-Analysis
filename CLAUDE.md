# Big Data Project — Claude Code Instructions

## Project Context
Course project + ongoing research on robot demonstration quality analysis.
Research question: Can simple trajectory statistics identify high-quality robot demonstrations?

## Tech Stack
- Python 3.10+, PySpark 3.4+, PyTorch, pandas, scikit-learn
- Google Colab (primary) or local Jupyter
- Data: 8 lerobot datasets, 251,802 frames total

## Key Files
```
notebooks/01_data_download.ipynb      # Download all 8 datasets + save to parquet
notebooks/02_eda.ipynb                # Exploratory analysis + visualizations
notebooks/03_feature_engineering.ipynb # 7 quality metrics via PySpark
notebooks/04_modeling.ipynb           # DT / GBT / RF / MLP comparison
test_data_structure.py                # Validation script
```

## Datasets
All from HuggingFace lerobot, load with `load_dataset(name, split='train')`:

**Simulation (5 datasets, 105,650 frames):**
- `lerobot/pusht` — 25,650 rows, state=2D, action=2D
- `lerobot/xarm_lift_medium` — 20,000 rows, state=4D, action=4D
- `lerobot/xarm_lift_medium_replay` — 20,000 rows, state=4D, action=4D
- `lerobot/xarm_push_medium` — 20,000 rows, state=4D, action=3D
- `lerobot/xarm_push_medium_replay` — 20,000 rows, state=4D, action=3D

**Real Robot (3 datasets, 146,152 frames):**
- `lerobot/berkeley_autolab_ur5` — 97,939 rows, state=8D, action=7D
- `lerobot/columbia_cairlab_pusht_real` — 27,808 rows, state=8D, action=7D  ← real pusht (Sim vs Real)
- `lerobot/nyu_door_opening_surprising_effectiveness` — 20,405 rows, state=8D, action=7D

**Important:** xarm_push reward is negative (distance metric). Use per-dataset median threshold for quality labels, not a fixed value.

## Quality Label Definition
```python
# Per-dataset median to handle different reward scales
threshold = episode_df.groupby('source')['max_reward'].transform('median')
episode_df['quality'] = (episode_df['max_reward'] > threshold).astype(int)
```

## 7 Quality Metrics (feature engineering)
smoothness, path_efficiency, action_consistency, correction_count,
time_efficiency, state_diversity, final_stability
(See notebooks/03_feature_engineering.ipynb for implementation)

## ML Pipeline
- Features: 7 quality metrics (episode-level aggregation)
- Label: quality (0/1) — binary classification
- Models: DecisionTreeClassifier, GBTClassifier, RandomForestClassifier (PySpark MLlib) + MLP (PyTorch)
- Eval: AUC, feature importance (SHAP)

## Run Tests
```bash
python3 test_data_structure.py
```

## Deadlines
- Mar 21: Email professor with team + topic
- Apr 3: Submit code + PDF + slides
- Apr 5-6: Presentation (10-12 min)

## Notes
- Data too large for GitHub → Google Drive
- Different reward scales across datasets — always use per-dataset normalization
- state/action dimensions differ: pusht=2D, xarm=4D, real=8D — use magnitude-based features
- Real robot datasets share identical schema (state=8D, action=7D) → can merge directly
- pusht sim vs real: `lerobot/pusht` vs `lerobot/columbia_cairlab_pusht_real` — use for Sim-to-Real analysis
- Keep GitHub repo PRIVATE — research ideas not yet published
