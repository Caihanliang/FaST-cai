# 湖南高速公路流量数据集使用指南

## 📋 概述

本指南帮助你使用 FaST 模型对湖南省高速公路收费站点流量数据进行建模和预测。

**数据集信息：**
- **区域**：湖南省高速公路（黄兴到芙蓉镇）
- **站点数**：161 个收费站
- **时间范围**：2023 年 9 月 1 日 - 10 月 31 日（2 个月）
- **数据频率**：15 分钟间隔（如果原始数据不是这个频率，会自动重采样）

---

## 🚀 快速开始

### 第一步：准备原始数据

将你的 Excel 或 CSV 数据文件放到 `DataPipeline/` 目录下，命名为：
- `hunan_highway_traffic.xlsx` 或
- `hunan_highway_traffic.csv`

**数据格式要求：**
```
时间，站点 1，站点 2，站点 3, ..., 站点 161
2023-09-01 00:00, 100, 150, 80, ...
2023-09-01 00:15, 95, 145, 75, ...
2023-09-01 00:30, 90, 140, 70, ...
...
```

**注意事项：**
1. 第一列必须是时间列，格式为 `YYYY-MM-DD HH:MM:SS`
2. 后续列是各站点的流量数据
3. 如果数据有缺失值，会自动填充为 0
4. 如果时间间隔不是 15 分钟，会自动重采样

---

### 第二步：运行数据处理流程

在 Windows 命令行中执行：

```bash
# 方式 1：使用批处理脚本（推荐）
bash DataPipeline_HNGS.sh

# 方式 2：手动执行每一步
python DataPipeline/generate_hngs_data.py
python DataPipeline/generate_hngs_training_data.py --dataset hngs --years 2023
python DataPipeline/process_hngs_adj.py
python DataPipeline/generate_hngs_idx.py
```

**生成的文件结构：**
```
main-master/datasets/HNGS/
├── his.npz              # 历史数据（包含流量 + 时间特征）
├── adj_mx.pkl          # 邻接矩阵
├── HNGS_meta.csv       # 站点元数据
└── 96_48/             # 输入 96 步，预测 48 步
│   ├── idx_train.npy
│   ├── idx_val.npy
│   └── idx_test.npy
└── 96_96/             # 输入 96 步，预测 96 步
    ├── idx_train.npy
    ├── idx_val.npy
    └── idx_test.npy
```

---

### 第三步：训练模型

```bash
# 方式 1：使用训练脚本
bash script/HNGS.sh

# 方式 2：单独训练某个预测长度
# 预测未来 12 小时（48 个时间步）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_48.py -g 0

# 预测未来 24 小时（96 个时间步）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_96.py -g 0
```

**训练参数说明：**
- `-c`: 配置文件路径
- `-g`: 使用的 GPU 编号（0 表示第一块 GPU）

**训练过程：**
- 最大训练轮数：50 epochs
- 批量大小：32
- 学习率：0.002（每 10 个 epoch 衰减一半）
- 优化器：Adam
- 损失函数：Smooth L1 Loss

---

### 第四步：评估模型

训练完成后，模型会自动保存在 `checkpoints/FaST/` 目录下。

```bash
# 评估模型
python main-master/experiments/evaluate.py \
  -cfg FaST/HNGS_96_48.py \
  -ckpt checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt
```

**评估指标：**
- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均绝对百分比误差）

---

## 🔧 高级配置

### 调整邻接矩阵生成方法

在 `DataPipeline/generate_hngs_data.py` 中修改：

```python
adj_method = 'sequence'  # 可选：'distance', 'sequence', 'custom'
```

- **sequence**：基于站点顺序（推荐，因为高速站点是线性排列）
- **distance**：基于经纬度距离（如果有真实经纬度）
- **custom**：自定义（需要根据实际路网拓扑修改代码）

### 修改预测长度

编辑配置文件 `FaST/HNGS_96_48.py` 或 `HNGS_96_96.py`：

```python
OUTPUT_LEN = 48  # 修改为你需要的预测长度
```

然后重新生成索引：
```bash
python DataPipeline/generate_hngs_idx.py
```

### 调整模型参数

在配置文件中修改 `MODEL_PARAM`：

```python
MODEL_PARAM = {
    "num_nodes": 161,      # 站点数
    "input_len": 96,       # 输入长度
    "output_len": 48,      # 预测长度
    "layers": 3,           # 网络层数
    "num_experts": 8,      # 专家数量
    "hidden_dim": 64,      # 隐藏层维度
    "num_agent": 32,       # 代理节点数
}
```

### 使用更长的预测长度

如果你的数据量足够（2 个月数据≈5952 个时间点），可以支持更长预测：

```bash
# 修改 generate_hngs_idx.py 中的输出长度
output_time_steps = [48, 96, 192]  # 添加 192 步（2 天）预测

# 创建新的配置文件 FaST/HNGS_96_192.py
# 复制 HNGS_96_96.py 并修改 OUTPUT_LEN = 192
```

---

## 📊 数据可视化（可选）

创建一个简单的脚本来查看数据：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_hdf('DataPipeline/HNGS/hngs_his_2023.h5')

# 绘制前 7 天的流量趋势
plt.figure(figsize=(15, 5))
df.iloc[:, 0].plot(label='Station 1')
df.iloc[:, 50].plot(label='Station 51')
df.iloc[:, 100].plot(label='Station 101')
plt.title('Traffic Flow Time Series (First 7 Days)')
plt.xlabel('Time')
plt.ylabel('Flow')
plt.legend()
plt.savefig('traffic_flow_sample.png', dpi=300)
plt.show()
```

---

## ⚠️ 常见问题

### 1. 数据量太少怎么办？
- 2 个月数据勉强够用，建议至少 3-6 个月
- 可以减少预测长度（如只预测 24、48 步）
- 使用数据增强技术（时间平移、加噪等）

### 2. 显存不足怎么办？
- 减小 `BATCH_SIZE`（从 32 改为 16 或 8）
- 关闭混合精度训练：`CFG["fp16"] = False`
- 减小模型维度：`hidden_dim = 32`

### 3. 预测效果不好怎么办？
- 检查数据质量（是否有大量缺失值）
- 调整邻接矩阵（更符合实际路网结构）
- 增加训练轮数或调整学习率
- 尝试不同的预测长度

### 4. 如何加入真实的站点经纬度？
编辑 `DataPipeline/generate_hngs_data.py` 中的 `generate_meta_data` 函数：

```python
meta_entry = {
    'ID': str(i + 1),
    'StationName': station_name,
    'Lat': 真实纬度，  # 替换占位值
    'Lng': 真实经度，  # 替换占位值
    ...
}
```

然后将 `adj_method` 改为 `'distance'`。

---

## 📈 结果分析

训练完成后，查看日志文件了解训练过程：

```
checkpoints/FaST/HNGS_50_96_48/
├── FaST_best_val_MAE.pt    # 最佳模型
├── training_log.txt        # 训练日志
└── evaluation_log.txt      # 评估日志
```

---

## 🎯 下一步工作

1. **模型对比**：可以运行基线模型进行对比
   ```bash
   bash script/DLinear.sh  # 需要修改为 HNGS 配置
   ```

2. **特征工程**：添加更多特征（天气、节假日等）

3. **模型改进**：基于 FaST 架构进行创新

4. **部署应用**：将训练好的模型用于实际预测

---

## 📞 技术支持

如有问题，请检查：
1. Python 版本：3.11.8
2. PyTorch 版本：2.2.1
3. CUDA 是否可用
4. 依赖包是否安装完整

祝实验顺利！🚀
