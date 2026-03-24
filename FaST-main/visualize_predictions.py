import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

"""
湖南高速公路流量预测结果可视化
绘制真实值 vs 预测值的对比图
"""

def load_data_and_predictions():
    """加载测试集数据和预测结果"""
    
    # 1. 加载原始数据
    print("正在加载原始数据...")
    df = pd.read_hdf('DataPipeline/HNGS/hngs_his_2023.h5')
    
    # 2. 加载测试集索引
    print("正在加载测试集索引...")
    idx_test = np.load('main-master/datasets/HNGS/96_48/idx_test.npy')
    
    # 3. 加载预测结果（需要从检查点加载）
    # 注意：这里需要从训练时的预测结果文件中加载
    # 如果没有保存，需要重新运行预测
    
    return df, idx_test


def plot_predictions(df, idx_test, station_indices=[0, 50, 100, 150], output_dir='visualization'):
    """
    绘制预测结果对比图
    
    Args:
        df: 原始数据 DataFrame
        idx_test: 测试集索引
        station_indices: 要可视化的站点索引列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 注意：这里需要实际的预测值，需要从训练输出中获取
    # 下面是示例代码，实际使用时需要加载真实的预测值
    
    print("正在绘制预测结果...")
    
    # 创建时间索引
    time_index = df.index
    
    # 绘制几个典型站点的对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, station_idx in enumerate(station_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        station_name = df.columns[station_idx]
        
        # 真实值（示例：取最后一周的数据）
        start_idx = max(0, len(df) - 672)  # 最后一周（7 天×96 个点）
        real_values = df.iloc[start_idx:, station_idx].values
        time_range = time_index[start_idx:]
        
        # 这里需要替换为实际的预测值
        # 暂时用真实值加噪声模拟预测值
        np.random.seed(42)
        predicted_values = real_values + np.random.randn(len(real_values)) * 5
        
        # 绘制曲线
        ax.plot(time_range, real_values, 'b-', linewidth=1.5, label='Real', alpha=0.7)
        ax.plot(time_range, predicted_values, 'r--', linewidth=1.5, label='Predicted', alpha=0.7)
        
        ax.set_title(f'Station {station_idx}: {station_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Traffic Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转 x 轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 预测对比图已保存：{output_dir}/prediction_comparison.png")
    plt.close()


def plot_error_distribution(df, idx_test, output_dir='visualization'):
    """绘制误差分布图"""
    
    print("正在绘制误差分布...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 模拟误差数据（实际使用时替换为真实误差）
    np.random.seed(42)
    errors = np.random.randn(1000) * 8
    
    # 1. 误差直方图
    axes[0].hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolors='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Predicted - Real)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 误差箱线图
    error_by_hour = [np.random.randn(100) * (5 + h%5) for h in range(24)]
    axes[1].boxplot(error_by_hour, patch_artist=True)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Error by Hour')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 累积误差分布
    sorted_errors = np.sort(np.abs(errors))
    cumulative_pct = np.arange(len(sorted_errors)) / len(sorted_errors) * 100
    axes[2].plot(sorted_errors, cumulative_pct, color='#A23B72', linewidth=2)
    axes[2].axhline(y=90, color='green', linestyle='--', label='90th percentile')
    axes[2].axhline(y=95, color='orange', linestyle='--', label='95th percentile')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Cumulative Percentage (%)')
    axes[2].set_title('Cumulative Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ 误差分布图已保存：{output_dir}/error_distribution.png")
    plt.close()


def plot_heatmap(df, idx_test, output_dir='visualization'):
    """绘制预测误差热力图（所有站点×时间）"""
    
    print("正在绘制误差热力图...")
    
    plt.figure(figsize=(15, 8))
    
    # 模拟误差矩阵（实际使用时替换为真实误差）
    # 形状：[时间点数，站点数]
    np.random.seed(42)
    time_points = min(672, len(df))  # 一周
    stations = len(df.columns)
    error_matrix = np.random.randn(time_points, stations) * 10
    
    # 绘制热力图
    im = plt.imshow(error_matrix, aspect='auto', cmap='RdBu_r', 
                    extent=[0, stations, time_points, 0])
    
    plt.colorbar(im, label='Error (MAE)')
    plt.xlabel('Station Index')
    plt.ylabel('Time Step')
    plt.title('Prediction Error Heatmap (All Stations × Time)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ 误差热力图已保存：{output_dir}/error_heatmap.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("湖南高速公路流量预测结果可视化")
    print("="*60)
    
    # 加载数据
    df, idx_test = load_data_and_predictions()
    
    # 生成可视化图表
    plot_predictions(df, idx_test)
    plot_error_distribution(df, idx_test)
    plot_heatmap(df, idx_test)
    
    print("\n" + "="*60)
    print("✓ 所有可视化图表已生成完毕！")
    print("查看目录：visualization/")
    print("="*60)

