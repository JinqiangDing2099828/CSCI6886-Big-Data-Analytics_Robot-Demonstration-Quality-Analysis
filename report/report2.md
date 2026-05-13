# Report 2: Robot Demonstration Quality — Model Comparison

**Team:** Alex's Group
**Date:** April 1, 2026
**Dataset:** LeRobot (HuggingFace) — 8 datasets, 5,026 episodes, 251,802 frames

---

## 1. Dataset Description

We use **8 LeRobot datasets** from HuggingFace, split into two groups:

| Type | Datasets | Episodes | State Dim | Action Dim |
|------|----------|----------|-----------|------------|
| Simulation (5) | pusht, xarm_lift/push × 2 | 3,406 | 2D–4D | 2D–4D |
| Real Robot (3) | Berkeley UR5, Columbia PushT, NYU Door | 1,620 | 8D | 7D |

### Preprocessing Steps
1. **L2 norm transformation** — Convert variable-dimension state/action vectors to scalars
2. **Episode-level aggregation** — Aggregate 251,802 frames → 5,026 episodes
3. **32 kinematic features** — Extracted via PySpark (smoothness, path efficiency, jerk, etc.)
4. **Dual quality labels:**
   - Sim: `reward_quality` = max_reward > per-dataset median
   - Real: `efficiency_quality` = episode_length < per-dataset median

---

## 2. Models

| # | Model | Framework | Key Parameters |
|---|-------|-----------|---------------|
| 1 | Decision Tree | PySpark MLlib / sklearn | max_depth=5 |
| 2 | Random Forest | PySpark MLlib / sklearn | n_estimators=100, max_depth=6 |
| 3 | GBT (Gradient Boosted Trees) | PySpark MLlib / sklearn | n_estimators=100, max_depth=5, lr=0.1 |
| 4 | MLP Neural Network | PyTorch | 128→64→32→1, BatchNorm, Dropout=0.3, 60 epochs |

All models evaluated with **5-fold Stratified Cross-Validation**.

---

## 3. Results — Simulation Experiment (n=3,406)

| Model | Accuracy | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| Decision Tree | 0.6033 | 0.5832 | 0.6140 | 0.5552 | 0.6455 |
| Random Forest | 0.6459 | 0.6504 | 0.6419 | 0.6592 | 0.7139 |
| **GBT** | **0.6483** | **0.6503** | **0.6462** | **0.6545** | **0.7134** |
| MLP (PyTorch) | 0.5531 | 0.5440 | 0.5550 | 0.5335 | 0.5947 |

**Best model:** GBT (AUC=0.7134)

### Confusion Matrices (Sim)

![Sim Confusion Matrices](../data/fig_cm_sim.png)

---

## 4. Results — Real Robot Experiment (n=1,620)

| Model | Accuracy | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| Decision Tree | 0.9123 | 0.9131 | 0.8818 | 0.9467 | 0.9413 |
| Random Forest | 0.9556 | 0.9550 | 0.9409 | 0.9695 | 0.9896 |
| **GBT** | **0.9617** | **0.9611** | **0.9515** | **0.9708** | **0.9936** |
| MLP (PyTorch) | 0.9574 | 0.9567 | 0.9455 | 0.9683 | 0.9894 |

**Best model:** GBT (AUC=0.9936)

### Confusion Matrices (Real)

![Real Confusion Matrices](../data/fig_cm_real.png)

### Model Comparison

![Model Performance Comparison](../data/fig_model_comparison_report2.png)

---

## 5. PySpark MLlib Validation (80/20 split)

| Experiment | Model | AUC | Accuracy | F1 |
|-----------|-------|-----|----------|-----|
| Sim | Decision Tree | 0.5908 | 0.6108 | 0.6108 |
| Sim | Random Forest | 0.7297 | 0.6467 | 0.6467 |
| Sim | GBT | 0.6999 | 0.6302 | 0.6301 |
| Real | Decision Tree | 0.9040 | 0.8762 | 0.8758 |
| Real | Random Forest | 0.9843 | 0.9460 | 0.9460 |
| Real | GBT | 0.9832 | 0.9302 | 0.9302 |

### Performance Heatmap

![Performance Heatmap](../data/fig_heatmap_report2.png)

---

## 6. Key Findings

1. **Real robot experiment achieves >96% accuracy** — Kinematic features strongly predict efficiency-based quality
2. **Simulation is harder (~65% accuracy)** — Reward-based quality is less predictable from kinematics alone
3. **GBT is the best model** in both experiments, followed by Random Forest
4. **MLP competitive on Real data** but weaker on Sim — tree-based models better for tabular kinematic features
5. **Practical implication:** Simple trajectory statistics CAN identify high-quality robot demonstrations, especially for real robot data

### SHAP Feature Importance (GBT)

![SHAP Feature Importance](../data/fig_shap_comparison.png)

> In Sim, `action_max` is the most important feature; in Real, `action_autocorr` dominates. The different key features suggest fundamentally different quality mechanisms between simulation and real-world settings.

---

## Environment

- Python 3.10+, PySpark 3.4+, PyTorch, scikit-learn
- Google Colab / local machine
- 5-fold Stratified Cross-Validation for all metrics

## References

[1] Cadene, R. et al. "LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch." HuggingFace, 2024.

[2] Mandlekar, A. et al. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." CoRL 2021.

[3] Luo, J. et al. "Consistency Matters: Defining Demonstration Data Quality Metrics." 2024.
