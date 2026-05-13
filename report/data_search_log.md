# 数据集搜寻过程记录

> 此文档记录完整的数据集调研和筛选过程，供 PDF 报告和 PPT 引用。

---

## 第一阶段：初始候选（Open X-Embodiment）

### 为什么首先考虑 Open X-Embodiment

Google DeepMind 于 2023 年发布的 Open X-Embodiment 数据集是目前规模最大的机器人操控数据集之一：
- 覆盖 **22 种机器人平台**、**527 种技能**
- 超过 **100 万条真实机器人轨迹**
- 子集 `fractal20220817_data`（Google RT-1）约有 **13 万条 episode、154 GB**

该数据集规模充足，且来自 Google Research，学术可信度高，是课题的自然首选。

### 遇到的问题

实际加载时发现该数据集使用自定义 Python 加载脚本（`OpenX-Embodiment.py`），而 HuggingFace `datasets` 库 **4.x 版本已废弃此格式**：

```
RuntimeError: Dataset scripts are no longer supported,
but found OpenX-Embodiment.py
```

**结论：** Open X-Embodiment 因库版本兼容性问题无法使用，需要寻找替代数据集。

---

## 第二阶段：转向 LeRobot 数据集

### 为什么选择 LeRobot

LeRobot（Hugging Face，2024）是专为机器人学习设计的开源框架，其数据集：
- 使用**标准 Parquet 格式**，完全兼容新版 datasets 库
- 提供统一的列名规范（`observation.state`、`action`、`next.reward` 等）
- 包含多种机器人平台和任务类型
- 数据质量经过官方验证

### 初步筛选结果

筛选条件：**必须有 `next.reward` 标签**（作为演示质量的客观衡量标准）

| 数据集 | 行数 | 有标签 |
|--------|------|--------|
| `lerobot/pusht` | 25,650 | ✅ |
| `lerobot/xarm_lift_medium` | 20,000 | ✅ |
| `lerobot/xarm_lift_medium_replay` | 20,000 | ✅ |
| `lerobot/xarm_push_medium` | 20,000 | ✅ |
| `lerobot/xarm_push_medium_replay` | 20,000 | ✅ |
| `lerobot/aloha_sim_*`（4个）| ~98,000 | ❌ 无 reward |
| `lerobot/unitreeh1_*`（3个）| ~37,000 | ❌ 无 reward |

**初步合计：105,650 行**（仅勉强超过 10 万要求，余量不足）

### 发现的问题

- `xarm_push_medium` 系列的 reward 为**负数**（范围 -1.3 ~ -0.04），是距离度量而非 0-1 范围，与其他数据集量纲不一致
- 105k 行数据余量不足，若后续筛选（feature engineering 后 episode 级别聚合）行数会进一步减少

---

## 第三阶段：扩大搜索范围

### 系统性搜索结果

对 HuggingFace lerobot 组织下的所有数据集逐一检测，完整结果如下：

| 数据集 | 行数 | 类型 | state 维度 | action 维度 | 有 reward |
|--------|------|------|-----------|------------|----------|
| `pusht` | 25,650 | 仿真 | 2D | 2D | ✅ |
| `xarm_lift_medium` | 20,000 | 仿真 | 4D | 4D | ✅ |
| `xarm_lift_medium_replay` | 20,000 | 仿真 | 4D | 4D | ✅ |
| `xarm_push_medium` | 20,000 | 仿真 | 4D | 3D | ✅ |
| `xarm_push_medium_replay` | 20,000 | 仿真 | 4D | 3D | ✅ |
| `xarm_lift_medium_image` | 34,675 | 仿真 | 4D | 3D | ✅ |
| `xarm_push_medium_image` | 34,675 | 仿真 | 4D | 3D | ✅ |
| `umi_cup_in_the_wild` | 699,432 | 真实 | — | — | ❌ |
| **`berkeley_autolab_ur5`** | **97,939** | **真实** | **8D** | **7D** | **✅** |
| **`columbia_cairlab_pusht_real`** | **27,808** | **真实** | **8D** | **7D** | **✅** |
| **`nyu_door_opening_surprising_effectiveness`** | **20,405** | **真实** | **8D** | **7D** | **✅** |
| `nyu_rot_dataset_...` | 440 | 真实 | 8D | 7D | ✅（太小）|
| `imperialcollege_sawyer_wrist_cam` | 7,148 | 真实 | — | — | ✅（太小）|

---

## 第四阶段：最终选择与分组

### 关键发现

1. **`berkeley_autolab_ur5`** 单个数据集就有 97,939 行，来自 UC Berkeley 自动化实验室的 UR5 真实机器人
2. **所有真实机器人数据集维度完全一致**（state=8D, action=7D），可以直接合并
3. **`columbia_cairlab_pusht_real`** 是 PushT 任务的**真实机器人版本**，与 `lerobot/pusht`（仿真版）构成天然的 **Sim vs Real 对比**

### 最终数据集组合

**分组1：仿真数据（Simulation）**

| 数据集 | 机构 | 行数 | 任务 |
|--------|------|------|------|
| `lerobot/pusht` | — | 25,650 | 推 T 形方块 |
| `lerobot/xarm_lift_medium` | — | 20,000 | xArm 抬起物体 |
| `lerobot/xarm_lift_medium_replay` | — | 20,000 | xArm 抬起（回放）|
| `lerobot/xarm_push_medium` | — | 20,000 | xArm 推物体 |
| `lerobot/xarm_push_medium_replay` | — | 20,000 | xArm 推物体（回放）|
| **小计** | | **105,650** | |

**分组2：真实机器人数据（Real World）**

| 数据集 | 机构 | 行数 | 任务 |
|--------|------|------|------|
| `lerobot/berkeley_autolab_ur5` | UC Berkeley | 97,939 | 多任务桌面操控 |
| `lerobot/columbia_cairlab_pusht_real` | Columbia University | 27,808 | 推 T 形方块（真实）|
| `lerobot/nyu_door_opening_surprising_effectiveness` | NYU | 20,405 | 开门任务 |
| **小计** | | **146,152** | |

**总计：251,802 帧**，来自 **5 所大学 / 研究机构**，覆盖**仿真和真实**两大类别。

### 为什么这个组合好

| 优点 | 说明 |
|------|------|
| 数量充足 | 251,802 行，远超课程 10 万要求 |
| 来源多样 | 仿真 + 真实，多机构，多任务类型 |
| 维度统一 | 真实机器人组全部 state=8D/action=7D，可直接合并 |
| 研究价值 | PushT 同时有仿真和真实版本 → 可分析 Sim-to-Real 质量差异 |
| 可复现 | 全部来自 HuggingFace 公开数据集，一行代码可下载 |

---

## 搜寻过程时间线

| 阶段 | 尝试方案 | 结果 |
|------|---------|------|
| 第1阶段 | Open X-Embodiment（Google RT-1）| ❌ 加载失败（库版本不兼容）|
| 第2阶段 | LeRobot 初步筛选（5个数据集）| ⚠️ 105k 行，略显不足 |
| 第3阶段 | 系统搜索所有 lerobot 数据集 | ✅ 发现 3 个真实机器人大数据集 |
| 第4阶段 | 最终组合（8个数据集）| ✅ 251k 行，仿真+真实均有 |
