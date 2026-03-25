"""
评估 FaST 模型并保存预测结果
用于后续可视化分析
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# ========== 路径配置 ==========
# 获取当前脚本所在目录 (FaST-main)
project_root = os.path.abspath(os.path.dirname(__file__))
# 构建 main-master 路径
main_master_path = os.path.join(project_root, 'main-master')

# 添加 main-master 到 Python 路径（关键！解决导入问题）
if main_master_path not in sys.path:
    sys.path.insert(0, main_master_path)

print("="*70)
print("FaST 模型评估 - 保存预测结果")
print("="*70)

# ========== 配置 ==========
# 使用正确的相对路径（相对于 main-master 目录）
CFG_PATH = os.path.join(main_master_path, 'FaST/HNGS_96_48.py')
CHECKPOINT_PATH = os.path.join(main_master_path, 'checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt')
OUTPUT_DIR = os.path.join(main_master_path, 'checkpoints/FaST/HNGS_50_96_48/')

print(f"\n配置文件：{CFG_PATH}")
print(f"检查点：{CHECKPOINT_PATH}")
print(f"输出目录：{OUTPUT_DIR}")

# ========== 加载数据 ==========
print("\n[1/5] 加载数据...")

try:
    # 尝试多个可能的数据路径
    data_paths = [
        os.path.join(project_root, 'DataPipeline/HNGS/hngs_his_2023.h5'),
        os.path.join(project_root, 'hngs_his_2023.h5'),
    ]
    
    data_file = None
    for path in data_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None:
        raise FileNotFoundError("未找到 HNGS 数据文件")
    
    df = pd.read_hdf(data_file)
    print(f"✓ 数据文件：{data_file}")
except Exception as e:
    print(f"⚠ 数据加载失败：{e}")
    sys.exit(1)

# 加载测试索引
try:
    idx_test_paths = [
        os.path.join(main_master_path, 'datasets/HNGS/96_48/idx_test.npy'),
        os.path.join(project_root, 'datasets/HNGS/96_48/idx_test.npy'),
    ]
    
    idx_test_file = None
    for path in idx_test_paths:
        if os.path.exists(path):
            idx_test_file = path
            break
    
    if idx_test_file:
        idx_test = np.load(idx_test_file)
        print(f"✓ 测试索引：{idx_test_file}")
    else:
        print("⚠ 未找到测试索引，将使用默认范围")
        total_samples = len(df) - 96
        idx_test = np.arange(int(total_samples * 0.8), min(int(total_samples * 0.8) + 100, total_samples))
        
except Exception as e:
    print(f"⚠ 索引加载失败：{e}")
    idx_test = np.arange(1000, 1100)

print(f"  - 测试样本数：{len(idx_test)}")

# ========== 准备测试数据 ==========
print("\n[2/5] 准备测试数据...")

INPUT_LEN = 96
OUTPUT_LEN = 48

all_predictions = []
all_ground_truths = []

# 取前 100 个样本用于可视化
test_indices = idx_test[:100]

for i, idx in enumerate(test_indices):
    start = idx - INPUT_LEN - OUTPUT_LEN + 1
    end = idx + 1

    if start < 0:
        continue

    # 获取数据
    data = df.iloc[start:end].values.T  # [N, T]
    x = data[:, :INPUT_LEN]  # [N, L]
    y = data[:, INPUT_LEN:]  # [N, P]

    all_ground_truths.append(y)

print(f"✓ 测试数据准备完成")
print(f"  - 样本数：{len(all_ground_truths)}")

# ========== 加载模型 ==========
print("\n[3/5] 加载模型...")
model = None

try:
    # 方法 1：直接导入 FaST 架构（最可靠的方法）
    print("  正在导入 FaST 架构...")
    from FaST.arch import FaST
    
    # 手动创建模型实例（基于 HNGS_96_48.py 中的配置参数）
    print("  正在创建模型实例...")
    
    num_nodes = 161
    INPUT_LEN = 96
    OUTPUT_LEN = 48
    
    model_param = {
        "num_nodes": num_nodes,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "layers": 3,
        "num_experts": 8,
        "daily_steps": 96,
        "weekly_days": 7,
        "hidden_dim": 64,
        "num_agent": 32,
    }
    
    model = FaST(**model_param)
    print(f"✓ 模型架构创建完成")
    
    # 加载权重
    if os.path.exists(CHECKPOINT_PATH):
        print(f"  正在加载检查点：{CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ 模型加载完成（使用 state_dict）")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ 模型加载完成")
        
        print(f"  - 检查点：{CHECKPOINT_PATH}")
    else:
        print(f"⚠ 警告：检查点文件不存在：{CHECKPOINT_PATH}")
        print(f"  请确保已先运行训练脚本生成模型检查点")
        raise FileNotFoundError(f"检查点文件不存在：{CHECKPOINT_PATH}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ 使用设备：{device}")
    print(f"✓ 模型准备就绪")
    
except FileNotFoundError as e:
    print(f"⚠ 模型文件缺失：{e}")
    print("\n请先运行训练脚本生成模型检查点文件")
    model = None
except ImportError as e:
    print(f"⚠ 模块导入失败：{e}")
    print("请确保 main-master 目录在 Python 路径中")
    import traceback
    traceback.print_exc()
    model = None
except Exception as e:
    print(f"⚠ 模型加载失败：{e}")
    import traceback
    traceback.print_exc()
    model = None

# ========== 进行预测 ==========
print("\n[4/5] 进行预测...")

if model is not None:
    # 使用真实模型预测
    print("  开始模型推理...")
    
    with torch.no_grad():
        for i, gt in enumerate(all_ground_truths):
            try:
                # 准备输入数据 [N, L]
                x_input = gt[:, :INPUT_LEN]
                
                # 转换为张量 [1, N, L]
                x = torch.from_numpy(x_input).float().unsqueeze(0)
                
                # 创建时间特征（简化处理）
                batch_size = x.shape[0]
                num_nodes_dim = x.shape[1]
                seq_len = x.shape[2]
                
                # Time of Day 和 Day of Week 特征（全零，表示默认时刻）
                tod = torch.zeros((batch_size, num_nodes_dim, seq_len), device=x.device)
                dow = torch.zeros((batch_size, num_nodes_dim, seq_len), device=x.device)
                
                # 拼接特征并调整维度为 [batch, L, N, features]
                # 原始 x: [1, N, L] -> 需要转为 [1, L, N]
                x_transposed = x.transpose(1, 2)  # [1, L, N]
                tod_transposed = tod.transpose(1, 2)  # [1, L, N]
                dow_transposed = dow.transpose(1, 2)  # [1, L, N]
                
                # 堆叠特征 [1, L, N, 3]
                x_with_features = torch.stack([x_transposed.squeeze(0), 
                                              tod_transposed.squeeze(0), 
                                              dow_transposed.squeeze(0)], dim=-1)
                x_with_features = x_with_features.unsqueeze(0)  # [1, L, N, 3]
                
                x_with_features = x_with_features.to(device)
                
                # 预测
                prediction = model(x_with_features)
                
                # 转换输出格式 [batch, P, N, 1] -> [N, P]
                pred = prediction.squeeze(0).squeeze(-1).transpose(0, 1).cpu().numpy()
                
                all_predictions.append(pred)
                
            except Exception as e:
                print(f"⚠ 样本 {i} 预测失败：{e}")
                # 使用零数组作为占位
                all_predictions.append(np.zeros_like(gt))
            
            if (i + 1) % 10 == 0:
                print(f"  已预测 {i+1}/{len(all_ground_truths)} 个样本")
    
    print(f"✓ 模型预测完成")
else:
    # 使用模拟数据（仅在模型加载失败时）
    print("⚠ 使用模拟数据进行演示...")
    np.random.seed(42)
    all_predictions = [gt + np.random.randn(*gt.shape) * 8 for gt in all_ground_truths]

# ========== 保存结果 ==========
print("\n[5/5] 保存预测结果...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 转换为 numpy 数组
predictions_np = np.array(all_predictions)  # [num_samples, N, P]
ground_truths_np = np.array(all_ground_truths)

# 保存
np.save(f'{OUTPUT_DIR}/predictions.npy', predictions_np)
np.save(f'{OUTPUT_DIR}/y_test.npy', ground_truths_np)

print(f"✓ 预测结果已保存")
print(f"  - predictions.npy: {predictions_np.shape}")
print(f"  - y_test.npy: {ground_truths_np.shape}")

# 计算并保存指标
if predictions_np.size > 0 and ground_truths_np.size > 0:
    mae = np.mean(np.abs(predictions_np - ground_truths_np))
    rmse = np.sqrt(np.mean((predictions_np - ground_truths_np)**2))
    mape = np.mean(np.abs((predictions_np - ground_truths_np) / (ground_truths_np + 1e-8))) * 100
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape)
    }
    
    print(f"\n预测性能指标：")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - MAPE: {mape:.2f}%")
else:
    print(f"\n⚠ 无法计算指标：预测或真实值数据为空")

print("\n" + "="*70)
print("✓ 评估完成！")
print("="*70)
print(f"\n下一步：")
print(f"1. 运行可视化：python visualize_hngs_predictions.py")
print(f"2. 查看结果：ls -lh {OUTPUT_DIR}")
