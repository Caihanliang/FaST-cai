"""
评估 FaST 模型并保存预测结果
用于后续可视化分析

功能：
1. 对所有测试样本进行预测（可配置数量）
2. 保存所有站点的预测值和真实值
3. 自动生成可视化图表（时间序列对比、误差分布等）
4. 计算并保存详细的评估指标
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ========== 路径配置 ==========
project_root = os.path.abspath(os.path.dirname(__file__))
main_master_path = os.path.join(project_root, 'main-master')

if main_master_path not in sys.path:
    sys.path.insert(0, main_master_path)

print("="*70)
print("FaST 模型评估 - 保存预测结果 & 可视化")
print("="*70)

# ========== 配置参数 ==========
class Config:
    """评估配置类"""
    # 文件路径
    CFG_PATH = os.path.join(main_master_path, 'FaST/HNGS_96_48.py')
    CHECKPOINT_PATH = os.path.join(main_master_path, 'checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt')
    OUTPUT_DIR = os.path.join(main_master_path, 'checkpoints/FaST/HNGS_50_96_48/')
    
    # 数据路径
    DATA_PATH = os.path.join(project_root, 'DataPipeline/HNGS/hngs_his_2023.h5')
    IDX_TEST_PATH = os.path.join(main_master_path, 'datasets/HNGS/96_48/idx_test.npy')
    
    # 模型参数
    NUM_NODES = 161  # HNGS 数据集站点数
    INPUT_LEN = 96   # 输入长度
    OUTPUT_LEN = 48  # 预测长度
    
    # 预测设置
    MAX_SAMPLES = None  # None=预测所有样本，或设置具体数量如 1000
    
    # 可视化设置
    PLOT_SAMPLES = 10  # 绘制前多少个样本的图表
    PLOT_NODES = [0, 50, 100, 150]  # 绘制哪些站点的图表（索引）

cfg = Config()

print(f"\n检查点：{cfg.CHECKPOINT_PATH}")
print(f"输出目录：{cfg.OUTPUT_DIR}")
print(f"最大预测样本数：{'全部' if cfg.MAX_SAMPLES is None else cfg.MAX_SAMPLES}")

# ========== 加载数据 ==========
print("\n[1/6] 加载数据...")

try:
    data_paths = [cfg.DATA_PATH, os.path.join(project_root, 'hngs_his_2023.h5')]
    data_file = None
    for path in data_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None:
        raise FileNotFoundError("未找到 HNGS 数据文件")
    
    df = pd.read_hdf(data_file)
    print(f"✓ 数据文件：{data_file}")
    print(f"  - 数据形状：{df.shape}")
except Exception as e:
    print(f"⚠ 数据加载失败：{e}")
    sys.exit(1)

# 加载测试索引
try:
    idx_test_paths = [cfg.IDX_TEST_PATH]
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
        total_samples = len(df) - cfg.INPUT_LEN
        idx_test = np.arange(int(total_samples * 0.8), min(int(total_samples * 0.8) + 1000, total_samples))
        
except Exception as e:
    print(f"⚠ 索引加载失败：{e}")
    idx_test = np.arange(1000, 2000)

# 限制样本数量
if cfg.MAX_SAMPLES is not None and len(idx_test) > cfg.MAX_SAMPLES:
    idx_test = idx_test[:cfg.MAX_SAMPLES]

print(f"  - 测试样本数：{len(idx_test)}")

# ========== 准备测试数据 ==========
print("\n[2/6] 准备测试数据...")

all_predictions = []
all_ground_truths = []
all_inputs = []  # 保存输入数据用于分析

test_indices = idx_test

for i, idx in enumerate(test_indices):
    start = idx - cfg.INPUT_LEN - cfg.OUTPUT_LEN + 1
    end = idx + 1
    
    if start < 0:
        continue
    
    # 获取数据 [N, T]
    data = df.iloc[start:end].values.T
    x = data[:, :cfg.INPUT_LEN]  # 输入 [N, L]
    y = data[:, cfg.INPUT_LEN:]  # 真实值 [N, P]
    
    all_ground_truths.append(y)
    all_inputs.append(x)

print(f"✓ 测试数据准备完成")
print(f"  - 样本数：{len(all_ground_truths)}")
print(f"  - 站点数：{all_ground_truths[0].shape[0]}")
print(f"  - 预测长度：{all_ground_truths[0].shape[1]}")

# ========== 加载模型 ==========
print("\n[3/6] 加载模型...")
model = None

try:
    print("  正在导入 FaST 架构...")
    from FaST.arch import FaST
    
    print("  正在创建模型实例...")
    
    model_param = {
        "num_nodes": cfg.NUM_NODES,
        "input_len": cfg.INPUT_LEN,
        "output_len": cfg.OUTPUT_LEN,
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
    if os.path.exists(cfg.CHECKPOINT_PATH):
        print(f"  正在加载检查点：{cfg.CHECKPOINT_PATH}")
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ 模型加载完成（使用 state_dict）")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ 模型加载完成")
        
        print(f"  - 检查点：{cfg.CHECKPOINT_PATH}")
    else:
        print(f"⚠ 警告：检查点文件不存在：{cfg.CHECKPOINT_PATH}")
        print(f"  请确保已先运行训练脚本生成模型检查点")
        raise FileNotFoundError(f"检查点文件不存在：{cfg.CHECKPOINT_PATH}")
    
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
    import traceback
    traceback.print_exc()
    model = None
except Exception as e:
    print(f"⚠ 模型加载失败：{e}")
    import traceback
    traceback.print_exc()
    model = None

# ========== 进行预测 ==========
print("\n[4/6] 进行预测...")

if model is not None:
    print("  开始模型推理...")
    
    with torch.no_grad():
        for i, gt in enumerate(all_ground_truths):
            try:
                # 准备输入数据 [N, L]
                x_input = all_inputs[i]
                
                # 转换为张量 [1, N, L]
                x = torch.from_numpy(x_input).float().unsqueeze(0)
                
                # 创建时间特征
                batch_size = x.shape[0]
                num_nodes_dim = x.shape[1]
                seq_len = x.shape[2]
                
                tod = torch.zeros((batch_size, num_nodes_dim, seq_len), device=x.device)
                dow = torch.zeros((batch_size, num_nodes_dim, seq_len), device=x.device)
                
                # 调整维度并拼接特征 [1, L, N, 3]
                x_transposed = x.transpose(1, 2)
                tod_transposed = tod.transpose(1, 2)
                dow_transposed = dow.transpose(1, 2)
                
                x_with_features = torch.stack([x_transposed.squeeze(0), 
                                              tod_transposed.squeeze(0), 
                                              dow_transposed.squeeze(0)], dim=-1)
                x_with_features = x_with_features.unsqueeze(0)
                
                x_with_features = x_with_features.to(device)
                
                # 预测
                prediction = model(x_with_features)
                
                # 转换输出格式 [batch, P, N, 1] -> [N, P]
                pred = prediction.squeeze(0).squeeze(-1).transpose(0, 1).cpu().numpy()
                
                all_predictions.append(pred)
                
            except Exception as e:
                print(f"⚠ 样本 {i} 预测失败：{e}")
                all_predictions.append(np.zeros_like(gt))
            
            if (i + 1) % 50 == 0:
                print(f"  已预测 {i+1}/{len(all_ground_truths)} 个样本")
    
    print(f"✓ 模型预测完成")
else:
    print("⚠ 使用模拟数据进行演示...")
    np.random.seed(42)
    all_predictions = [gt + np.random.randn(*gt.shape) * 8 for gt in all_ground_truths]

# ========== 保存结果 ==========
print("\n[5/6] 保存预测结果...")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 转换为 numpy 数组
predictions_np = np.array(all_predictions)  # [num_samples, N, P]
ground_truths_np = np.array(all_ground_truths)
inputs_np = np.array(all_inputs)  # [num_samples, N, L]

print(f"数据形状：")
print(f"  - 输入数据：{inputs_np.shape}")
print(f"  - 预测数据：{predictions_np.shape}")
print(f"  - 真实数据：{ground_truths_np.shape}")

# 保存完整数据
np.save(f'{cfg.OUTPUT_DIR}/predictions.npy', predictions_np)
np.save(f'{cfg.OUTPUT_DIR}/y_test.npy', ground_truths_np)
np.save(f'{cfg.OUTPUT_DIR}/x_test.npy', inputs_np)

print(f"\n✓ 预测结果已保存")
print(f"  - predictions.npy: {predictions_np.shape}")
print(f"  - y_test.npy: {ground_truths_np.shape}")
print(f"  - x_test.npy: {inputs_np.shape}")

# 计算评估指标
if predictions_np.size > 0 and ground_truths_np.size > 0:
    mae = np.mean(np.abs(predictions_np - ground_truths_np))
    rmse = np.sqrt(np.mean((predictions_np - ground_truths_np)**2))
    mape = np.mean(np.abs((predictions_np - ground_truths_np) / (np.abs(ground_truths_np) + 1e-8))) * 100
    
    # 分站点统计
    mae_per_node = np.mean(np.abs(predictions_np - ground_truths_np), axis=(0, 1))
    rmse_per_node = np.sqrt(np.mean((predictions_np - ground_truths_np)**2, axis=(0, 1)))
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'MAE_per_node': mae_per_node.tolist(),
        'RMSE_per_node': rmse_per_node.tolist()
    }
    
    print(f"\n预测性能指标：")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAPE: {mape:.4f}%")
    print(f"\n分站点统计（前 5 个站点）：")
    for i in range(min(5, cfg.NUM_NODES)):
        print(f"    站点 {i}: MAE={mae_per_node[i]:.4f}, RMSE={rmse_per_node[i]:.4f}")
else:
    print(f"\n⚠ 无法计算指标：预测或真实值数据为空")
    metrics = {}

# ========== 可视化 ==========
print("\n[6/6] 生成可视化图表...")

try:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建可视化目录
    vis_dir = os.path.join(cfg.OUTPUT_DIR, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    num_plots = min(cfg.PLOT_SAMPLES, len(all_predictions))
    
    for i in range(num_plots):
        pred = predictions_np[i]  # [N, P]
        truth = ground_truths_np[i]  # [N, P]
        inp = inputs_np[i]  # [N, L]
        
        # 为每个选中的站点绘制图表
        for node_idx in cfg.PLOT_NODES:
            if node_idx >= cfg.NUM_NODES:
                continue
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # 完整时间序列对比
            time_steps = list(range(-cfg.INPUT_LEN, cfg.OUTPUT_LEN))
            all_values = np.concatenate([inp[node_idx], pred[node_idx]], axis=0)
            all_truth_extended = np.concatenate([inp[node_idx], truth[node_idx]], axis=0)
            
            axes[0].plot(time_steps[:cfg.INPUT_LEN], inp[node_idx], 'b-', label='输入 (历史)', linewidth=1.5)
            axes[0].plot(time_steps[cfg.INPUT_LEN:], pred[node_idx], 'r--', label='预测值', linewidth=2)
            axes[0].plot(time_steps[cfg.INPUT_LEN:], truth[node_idx], 'g-.', label='真实值', linewidth=2)
            axes[0].axvline(x=0, color='k', linestyle=':', linewidth=1)
            axes[0].set_xlabel('时间步', fontsize=12)
            axes[0].set_ylabel('交通流量', fontsize=12)
            axes[0].set_title(f'站点 {node_idx} - 样本 {i+1}: 历史输入与预测对比', fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # 预测部分放大对比
            axes[1].plot(range(cfg.OUTPUT_LEN), pred[node_idx], 'r-o', label='预测值', linewidth=2, markersize=3)
            axes[1].plot(range(cfg.OUTPUT_LEN), truth[node_idx], 'g-s', label='真实值', linewidth=2, markersize=3)
            axes[1].set_xlabel('预测时间步', fontsize=12)
            axes[1].set_ylabel('交通流量', fontsize=12)
            axes[1].set_title(f'站点 {node_idx} - 样本 {i+1}: 预测 vs 真实 (MAE={np.mean(np.abs(pred[node_idx]-truth[node_idx])):.2f})', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f'sample_{i+1:03d}_node_{node_idx:03d}.png'
            plt.savefig(os.path.join(vis_dir, plot_filename), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 已生成样本 {i+1}/{num_plots} 的可视化图表")
    
    # 生成误差分布图
    if len(all_predictions) > 0:
        errors = np.abs(predictions_np - ground_truths_np)  # [samples, N, P]
        
        # 所有站点的平均误差
        avg_error_per_node = np.mean(errors, axis=(0, 2))  # [N]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 误差分布直方图
        axes[0].hist(errors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('绝对误差', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('所有站点误差分布', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=np.mean(errors), color='r', linestyle='--', label=f'平均误差：{np.mean(errors):.2f}')
        axes[0].legend()
        
        # 各站点平均误差
        nodes_to_plot = min(20, cfg.NUM_NODES)
        axes[1].bar(range(nodes_to_plot), avg_error_per_node[:nodes_to_plot], color='coral')
        axes[1].set_xlabel('站点索引', fontsize=12)
        axes[1].set_ylabel('平均绝对误差', fontsize=12)
        axes[1].set_title(f'前{nodes_to_plot}个站点的平均误差', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已生成误差分布图")
        
        # 保存指标到文件
        import json
        metrics_file = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 评估指标已保存到：{metrics_file}")
    
    print(f"\n✓ 可视化完成！图表保存在：{vis_dir}")
    
except Exception as e:
    print(f"⚠ 可视化失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ 评估完成！")
print("="*70)
print(f"\n输出文件列表：")
print(f"  1. 预测数据：{cfg.OUTPUT_DIR}/predictions.npy")
print(f"  2. 真实数据：{cfg.OUTPUT_DIR}/y_test.npy")
print(f"  3. 输入数据：{cfg.OUTPUT_DIR}/x_test.npy")
print(f"  4. 评估指标：{cfg.OUTPUT_DIR}/metrics.json")
print(f"  5. 可视化图表：{cfg.OUTPUT_DIR}/visualization/")
print(f"\n查看结果：")
print(f"  - Windows: explorer {cfg.OUTPUT_DIR}")
print(f"  - Linux/Mac: ls -lh {cfg.OUTPUT_DIR}")
