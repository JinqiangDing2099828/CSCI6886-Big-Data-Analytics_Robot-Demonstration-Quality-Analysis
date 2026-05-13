# 报告二：机器人演示质量 — 模型对比分析

**团队：** 
**日期：** 2026年4月1日
**数据集：** LeRobot (HuggingFace) — 8个数据集，5,026个Episode，251,802帧

---

## 1. 数据集描述

我们使用 HuggingFace 上的 **8个 LeRobot 数据集**，分为两组：

| 类型 | 数据集 | Episode数 | 状态维度 | 动作维度 |
|------|--------|----------|---------|---------|
| 仿真（5个） | pusht, xarm_lift/push × 2 | 3,406 | 2D–4D | 2D–4D |
| 真实机器人（3个） | Berkeley UR5, Columbia PushT, NYU Door | 1,620 | 8D | 7D |

### 预处理步骤
1. **L2范数变换** — 将不同维度的状态/动作向量转换为标量（维度无关）
2. **Episode级聚合** — 将251,802帧聚合为5,026个Episode
3. **32个运动学特征** — 通过PySpark提取（平滑度、路径效率、jerk等）
4. **双质量标签：**
   - 仿真：`reward_quality` = 最大奖励 > 该数据集中位数
   - 真实机器人：`efficiency_quality` = Episode长度 < 该数据集中位数（越短效率越高）

---

## 2. 模型

| 编号 | 模型 | 框架 | 关键参数 |
|------|------|------|---------|
| 1 | 决策树 (Decision Tree) | PySpark MLlib / sklearn | max_depth=5 |
| 2 | 随机森林 (Random Forest) | PySpark MLlib / sklearn | n_estimators=100, max_depth=6 |
| 3 | 梯度提升树 (GBT) | PySpark MLlib / sklearn | n_estimators=100, max_depth=5, lr=0.1 |
| 4 | MLP 神经网络 | PyTorch | 128→64→32→1, BatchNorm, Dropout=0.3, 60 epochs |

所有模型均使用 **5折分层交叉验证** 进行评估。

---

## 3. 结果 — 仿真实验（n=3,406）

| 模型 | 准确率 | F1 | 精确率 | 召回率 | AUC |
|------|--------|-----|--------|--------|-----|
| 决策树 | 0.6033 | 0.5832 | 0.6140 | 0.5552 | 0.6455 |
| 随机森林 | 0.6459 | 0.6504 | 0.6419 | 0.6592 | 0.7139 |
| **GBT** | **0.6483** | **0.6503** | **0.6462** | **0.6545** | **0.7134** |
| MLP (PyTorch) | 0.5531 | 0.5440 | 0.5550 | 0.5335 | 0.5947 |

**最佳模型：** GBT（AUC=0.7134）

### 混淆矩阵（仿真）

![仿真实验混淆矩阵](../data/fig_cm_sim.png)

---

## 4. 结果 — 真实机器人实验（n=1,620）

| 模型 | 准确率 | F1 | 精确率 | 召回率 | AUC |
|------|--------|-----|--------|--------|-----|
| 决策树 | 0.9123 | 0.9131 | 0.8818 | 0.9467 | 0.9413 |
| 随机森林 | 0.9556 | 0.9550 | 0.9409 | 0.9695 | 0.9896 |
| **GBT** | **0.9617** | **0.9611** | **0.9515** | **0.9708** | **0.9936** |
| MLP (PyTorch) | 0.9574 | 0.9567 | 0.9455 | 0.9683 | 0.9894 |

**最佳模型：** GBT（AUC=0.9936）

### 混淆矩阵（真实机器人）

![真实机器人混淆矩阵](../data/fig_cm_real.png)

### 模型对比柱状图

![模型性能对比](../data/fig_model_comparison_report2.png)

---

## 5. PySpark MLlib 验证（80/20划分）

| 实验 | 模型 | AUC | 准确率 | F1 |
|------|------|-----|--------|-----|
| 仿真 | 决策树 | 0.5908 | 0.6108 | 0.6108 |
| 仿真 | 随机森林 | 0.7297 | 0.6467 | 0.6467 |
| 仿真 | GBT | 0.6999 | 0.6302 | 0.6301 |
| 真实 | 决策树 | 0.9040 | 0.8762 | 0.8758 |
| 真实 | 随机森林 | 0.9843 | 0.9460 | 0.9460 |
| 真实 | GBT | 0.9832 | 0.9302 | 0.9302 |

### 性能热力图

![性能热力图](../data/fig_heatmap_report2.png)

---

## 6. 核心发现

1. **真实机器人实验准确率超过96%** — 运动学特征能够很好地预测基于效率的质量
2. **仿真实验更具挑战性（约65%准确率）** — 仅靠运动特征较难预测基于奖励的质量
3. **GBT是两个实验中的最佳模型**，其次是随机森林
4. **MLP在真实数据上表现有竞争力**，但在仿真数据上较弱 — 树模型更适合表格型运动学特征
5. **实际意义：** 简单的轨迹统计特征**能够**识别高质量的机器人演示，尤其在真实机器人数据上效果显著

### SHAP 特征重要性（GBT模型）

![SHAP特征重要性对比](../data/fig_shap_comparison.png)

> 仿真实验中 `action_max` 最重要；真实机器人实验中 `action_autocorr` 最重要。两组实验的关键特征不同，说明仿真和真实场景的质量判别机制有本质差异。

---

## 运行环境

- Python 3.10+, PySpark 3.4+, PyTorch, scikit-learn
- Google Colab / 本地机器
- 所有指标均基于5折分层交叉验证

## 参考文献

[1] Cadene, R. et al. "LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch." HuggingFace, 2024.

[2] Mandlekar, A. et al. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." CoRL 2021.

[3] Luo, J. et al. "Consistency Matters: Defining Demonstration Data Quality Metrics." 2024.
