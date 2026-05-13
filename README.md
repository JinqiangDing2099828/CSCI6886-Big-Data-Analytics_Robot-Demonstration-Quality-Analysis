# 大数据课程项目说明

> 给组员看的版本：不需要任何基础，从头读懂项目要做什么

---

## 基本信息

| 项目 | 内容 |
|------|------|
| 课程 | 大数据分析 |
| 团队 | 2-3 人（不能 1 人独做） |
| 演示时间 | 约 4 月 5-6 日 |
| 提交截止 | 演示前 **2 天**（约 4 月 3 日）提交所有材料 |
| 占分比重 | 期末成绩的 20-25% |
| 运行平台 | Google Colab（免费，有 Gmail 即可） |
| 代码仓库 | GitHub（只放代码，不放数据） |
| 数据存储 | Google Drive（数据太大，不能放 GitHub） |

---

## 我们要做什么？

### 一句话描述

> 用机器人的运动数据，判断"这条演示值不值得机器人去学习"。

### 大白话版本

机器人学技能的方式，类似人类看示范视频来学动作。但示范质量参差不齐——
有些示范动作流畅、路线直接；有些犹豫不决、走弯路、中途纠错。

**我们的问题：** 不看最终结果，只看"动作方式"，能判断这条示范质量高不高吗？

类比：就像体育教练看球员的跑动数据（跑动距离、方向变化次数、路线效率），
不看最终比分，就能判断这个球员的训练价值高不高。

### 为什么有研究价值？

- 机器人演示数据收集成本极高（需要人工遥控、硬件设备）
- 已有论文（CoRL 2021）证明：**演示质量比数量更重要**
- 2025 年最新研究发现：筛选 top 30% 高质量数据，训练效果超过全量数据
- **我们的贡献：** 用 7 个简单可解释的指标，在 10 万帧数据上大规模验证哪些特征最重要

---

## 数据集

### 来源

8 个 LeRobot 公开数据集（HuggingFace），合计 **251,802 帧**，分两组：

**仿真数据（Simulation）**

| 数据集 | 机器人 | 任务 | 帧数 |
|--------|--------|------|------|
| `lerobot/pusht` | 2D 机械臂 | 推 T 形方块（仿真）| 25,650 |
| `lerobot/xarm_lift_medium` | xArm | 抬起物体 | 20,000 |
| `lerobot/xarm_lift_medium_replay` | xArm | 抬起物体（回放）| 20,000 |
| `lerobot/xarm_push_medium` | xArm | 推物体 | 20,000 |
| `lerobot/xarm_push_medium_replay` | xArm | 推物体（回放）| 20,000 |

**真实机器人数据（Real World）**

| 数据集 | 机构 | 任务 | 帧数 |
|--------|------|------|------|
| `lerobot/berkeley_autolab_ur5` | UC Berkeley | 多任务桌面操控 | 97,939 |
| `lerobot/columbia_cairlab_pusht_real` | Columbia University | 推 T 形方块（真实）| 27,808 |
| `lerobot/nyu_door_opening_surprising_effectiveness` | NYU | 开门任务 | 20,405 |

> `pusht` 同时有仿真版和真实版 → 天然支持 **Sim vs Real 对比分析**

> 数据搜寻完整过程见 [report/data_search_log.md](report/data_search_log.md)

### 数据结构（每行 = 机器人某一时刻的状态）

| 列名 | 含义 | 例子 |
|------|------|------|
| `episode_index` | 第几次任务尝试 | 0, 1, 2... |
| `frame_index` | 该次任务的第几帧 | 0, 1, 2... |
| `observation.state` | 机器臂关节角度 | [1.31, 0.29, 0.54, ...] |
| `action` | 手臂动作指令 | [1.0, -0.87, 0.90, ...] |
| `next.reward` | 当前帧得分 | 0.574 |
| `next.done` | 是否最后一帧 | True / False |

### 标签定义

```
质量标签 = max(episode.reward) > 该数据集的中位数
高质量(1) = 表现好于平均水平
低质量(0) = 表现差于平均水平
```

---

## 我们计算的 7 个质量指标

| 指标 | 含义 | 直觉理解 |
|------|------|---------|
| `smoothness` | 动作平滑度 | 流畅 > 抖动 |
| `path_efficiency` | 路径效率（直线/实际路径）| 走直路 > 绕弯路 |
| `action_consistency` | 动作一致性 | 确定 > 犹豫 |
| `correction_count` | 纠错次数（方向反转）| 少纠错 > 多纠错 |
| `time_efficiency` | 完成速度 | 快 > 慢 |
| `state_diversity` | 状态多样性 | 覆盖面广 > 原地踏步 |
| `final_stability` | 末段稳定性 | 结尾稳 > 结尾乱 |

---

## 完整工作流程

```
Step 1: 下载 5 个数据集 → 存到 Google Drive
        ↓
Step 2: EDA —— 看数据分布，质量高/低的轨迹有什么差异？
        ↓
Step 3: 特征工程 —— 用 PySpark 计算 7 个质量指标（帧→episode 级别聚合）
        ↓
Step 4: 建模 —— Decision Tree / GBT / 神经网络 预测质量标签
        ↓
Step 5: 分析 —— 哪个指标最重要？不同任务的结论一样吗？
        ↓
Step 6: 写报告 + 做 PPT + 演示
```

---

## 团队分工

### Alex
- [ ] 下载 5 个数据集，存 Google Drive，共享链接给组员
- [ ] 特征工程：PySpark 计算 7 个质量指标
- [ ] GBT Boosting + SHAP 特征重要性分析
- [ ] 神经网络（MLP）对比实验
- [ ] 演示主讲

### 组员 2（EDA + 可视化）
- [ ] 挂载 Google Drive（Alex 给链接）
- [ ] 画质量指标分布图（高质量 vs 低质量对比）
- [ ] 画相关性热力图（7 个指标之间的关系）
- [ ] 分析不同任务（push vs lift）的质量差异

### 组员 3（Decision Tree + 报告）
- [ ] 运行 Decision Tree，可视化决策规则
- [ ] 运行 Random Forest，整理结果表格
- [ ] 写 PDF 报告（8-12 页）
- [ ] 做 PPT（10-12 分钟，每人均分时间）

---

## 时间线

| 日期 | 谁 | 要做的事 |
|------|-----|---------|
| **3 月 21 日（周六）** | Alex | 发邮件给老师，CC 组员 |
| **3 月 22-23 日** | Alex | 下载数据，共享 Drive |
| **3 月 24-25 日** | 组员 2 | EDA + 出图 |
| **3 月 24-25 日** | Alex | 完成特征工程 |
| **3 月 26-29 日** | 全体 | 各自跑模型 |
| **3 月 30-31 日** | Alex | 汇总结果 + SHAP 分析 |
| **4 月 1-2 日** | 组员 3 | 写报告 + 做 PPT |
| **4 月 3 日** | 全体 | 提交所有材料 |
| **4 月 5-6 日** | 全体 | 课堂演示（10-12 分钟）|

---

## 如何开始（3步上手）

```python
# Step 1: Colab 安装依赖（运行一次）
!pip install pyspark datasets pyarrow -q

# Step 2: 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 3: 读取数据（Alex 下载好后共享给你）
import pandas as pd
df = pd.read_parquet('/content/drive/MyDrive/bigdata-project/robot_frames.parquet')
print(f'数据加载成功: {df.shape}')
df.head()
```

---

## 项目文件结构

```
bigdata-project/              ← GitHub（只放代码）
  notebooks/
    01_data_download.ipynb    ← Alex：下载数据
    02_eda.ipynb              ← 组员 2：探索分析
    03_feature_engineering.ipynb  ← Alex：7 个质量指标
    04_modeling.ipynb         ← 全体：模型训练
  report/
    report.pdf                ← 组员 3
    slides.pptx               ← 组员 3
  requirements.txt
  README.md                   ← 你现在看的这个文件

Google Drive/bigdata-project/ ← 不在 GitHub（太大了）
  robot_frames.parquet        ← 原始帧数据
  episode_features.parquet    ← 质量特征（聚合后）
```

---

## 常见问题

**Q: 不会 PySpark 怎么办？**
A: 照着 notebook 模板改就行，关键代码 Alex 已经写好了。

**Q: 数据需要自己下载吗？**
A: 不用，Alex 下好后共享 Google Drive 给你，直接挂载读取。

**Q: "质量高"是主观判断吗？**
A: 不是，用 max(reward) 超过数据集中位数作为客观标签。

**Q: 报告要多长？**
A: 8-12 页，图表为主，文字简洁。

**Q: 演示每人要讲多少？**
A: 10-12 分钟平均分，每人约 3-4 分钟，不能一个人全讲。
