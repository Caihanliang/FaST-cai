# 湖南高速公路流量预测 - 快速启动指南 ⚡

## 📦 前置准备

### 1. 检查环境
```bash
python --version  # 应该是 3.11.8
pip list | grep torch  # 检查 PyTorch 是否安装
```

### 2. 准备你的数据

将你的 Excel/CSV 文件重命名为 `hunan_highway_traffic.xlsx` 或 `hunan_highway_traffic.csv`，放到 `DataPipeline/` 目录下。

**数据格式示例：**
```csv
Time,Station_001,Station_002,...,Station_161
2023-09-01 00:00:00,150.5,180.2,...,120.3
2023-09-01 00:15:00,145.2,175.8,...,118.5
...
```

---

## 🚀 三步开始训练

### Step 1: 生成示例数据（仅首次）

如果你想先测试流程，可以生成一个模拟数据：

```bash
python DataPipeline/create_sample_data.py
```

这会生成 `hunan_highway_traffic_template.xlsx` 作为参考。

---

### Step 2: 数据处理

运行完整的数据处理流程：

```bash
# Windows PowerShell 或 Git Bash
bash DataPipeline_HNGS.sh

# 或者手动执行
python DataPipeline/generate_hngs_data.py
python DataPipeline/generate_hngs_training_data.py --dataset hngs --years 2023
python DataPipeline/process_hngs_adj.py
python DataPipeline/generate_hngs_idx.py
```

✅ **成功标志**：在 `main-master/datasets/HNGS/` 目录下看到生成的文件。

---

### Step 3: 开始训练

```bash
# 预测未来 12 小时（48 个时间步）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_48.py -g 0

# 或预测未来 24 小时（96 个时间步）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_96.py -g 0
```

✅ **成功标志**：看到训练进度条和损失值下降。

---

## 📊 查看结果

训练完成后：

1. **模型保存位置**：
   ```
   checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt
   ```

2. **查看训练日志**：
   ```
   checkpoints/FaST/HNGS_50_96_48/training_log.txt
   ```

3. **评估模型**：
   ```bash
   python main-master/experiments/evaluate.py \
     -cfg FaST/HNGS_96_48.py \
     -ckpt checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt
   ```

---

## ⚙️ 常见调整

### 调整预测长度

编辑 `FaST/HNGS_96_48.py`：
```python
OUTPUT_LEN = 48  # 改为 96、192 等
```

然后重新运行：
```bash
python DataPipeline/generate_hngs_idx.py
```

### 调整批量大小（显存不足时）

编辑配置文件：
```python
BATCH_SIZE = 16  # 从 32 改为 16 或 8
```

### 关闭混合精度训练（如果报错）

编辑配置文件：
```python
CFG["fp16"] = False
```

---

## 🐛 故障排查

### 问题 1：找不到 GPU
```python
# 检查 CUDA
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

### 问题 2：数据形状不匹配
检查你的数据文件：
- 第一列必须是时间
- 应该有 161 个站点列
- 时间间隔应该是 15 分钟

### 问题 3：训练中断
- 减小 `BATCH_SIZE`
- 确保有足够的磁盘空间
- 检查是否有 NaN 值

---

## 📞 需要帮助？

详细文档请查看：[`README_HNGS.md`](README_HNGS.md)

---

**预计时间**：
- 数据处理：~2 分钟
- 训练（50 epochs）：~30-60 分钟（取决于 GPU）
- 评估：~5 分钟

Good Luck! 🍀
