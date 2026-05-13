# Teamwork: 
## Team Leader: Dazhi Yang 
## Team member: Jinqiang Ding 
## Team member: Xiao Wang


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
