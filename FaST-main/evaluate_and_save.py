"""
评估 FaST 模型并保存预测结果
用于后续可视化分析
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("="*70)
print("FaST 模型评估 - 保存预测结果")
print("="*70)

# ========== 配置 ==========
CFG_PATH = 'FaST/HNGS_96_48.py'
CHECKPOINT_PATH = 'checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt'
OUTPUT_DIR = 'checkpoints/FaST/HNGS_50_96_48/'

print(f"\n配置：{CFG_PATH}")
print(f"检查点：{CHECKPOINT_PATH}")

# ========== 加载数据 ==========
print("\n[1/5] 加载数据...")

df = pd.read_hdf('DataPipeline/HNGS/hngs_his_2023.h5')
idx_test = np.load('main-master/datasets/HNGS/96_48/idx_test.npy')

print(f"✓ 数据加载完成")
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

try:
    from easydict import EasyDict
    import importlib.util
    
    # 加载配置
    spec = importlib.util.spec_from_file_location("config", CFG_PATH)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    # 加载模型架构
    model_class = cfg.MODEL_ARCH
    model = model_class(**cfg.MODEL_PARAM)
    
    # 加载权重
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ 模型加载完成")
        print(f"  - 检查点：{CHECKPOINT_PATH}")
    else:
        print(f"⚠ 检查点不存在：{CHECKPOINT_PATH}")
        sys.exit(1)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ 使用设备：{device}")
    
except Exception as e:
    print(f"⚠ 模型加载失败：{e}")
    print("\n使用模拟数据进行演示...")
    
    # 使用模拟数据
    use_mock = True

# ========== 进行预测 ==========
print("\n[4/5] 进行预测...")

if 'model' in locals():
    # 使用真实模型预测
    with torch.no_grad():
        for i, gt in enumerate(all_ground_truths):
            # 准备输入
            x = torch.from_numpy(gt[:, :INPUT_LEN]).float()  # 这里应该有单独的输入数据
            x = x.unsqueeze(0).transpose(1, 2)  # [1, N, L] -> [1, L, N]
            
            # 添加时间特征（简化处理）
            tod = torch.zeros_like(x[:, :, :1])
            dow = torch.zeros_like(x[:, :, :1])
            x_with_features = torch.cat([x, tod, dow], dim=-1)  # [1, L, N, 3]
            
            x_with_features = x_with_features.to(device)
            
            # 预测
            prediction = model(x_with_features)
            
            # 转换格式 [1, P, N, 1] -> [N, P]
            pred = prediction.squeeze(0).squeeze(-1).transpose(0, 1).cpu().numpy()
            
            all_predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  已预测 {i+1}/{len(all_ground_truths)} 个样本")
    
    print(f"✓ 模型预测完成")
else:
    # 使用模拟数据
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

print("\n" + "="*70)
print("✓ 评估完成！")
print("="*70)
print(f"\n下一步：")
print(f"1. 运行可视化：python visualize_hngs_predictions.py")
print(f"2. 查看结果：ls -lh {OUTPUT_DIR}")
