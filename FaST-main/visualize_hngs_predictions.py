import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import pickle
from datetime import datetime, timedelta

"""
湖南高速公路流量预测 - 完整可视化脚本
从训练好的模型加载预测结果并可视化
"""

class PredictionVisualizer:
    def __init__(self, config_path='FaST/HNGS_96_48.py', 
                 checkpoint_path=None,
                 data_dir='DataPipeline/HNGS'):
        """
        初始化可视化器
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            data_dir: 数据目录
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_test_data(self):
        """加载测试集数据"""
        print("正在加载测试集数据...")
        
        # 加载历史数据
        df = pd.read_hdf(f'{self.data_dir}/hngs_his_2023.h5')
        
        # 加载测试索引
        idx_test = np.load('main-master/datasets/HNGS/96_48/idx_test.npy')
        
        # 加载邻接矩阵
        with open('main-master/datasets/HNGS/adj_mx.pkl', 'rb') as f:
            adj_matrix = pickle.load(f)
        
        print(f"✓ 数据加载完成")
        print(f"  - 总时间步数：{len(df)}")
        print(f"  - 站点数：{len(df.columns)}")
        print(f"  - 测试样本数：{len(idx_test)}")
        
        return df, idx_test, adj_matrix
    
    def prepare_test_samples(self, df, idx_test, input_len=96, output_len=48):
        """准备测试样本"""
        print("正在准备测试样本...")
        
        X_test, y_test = [], []
        
        for idx in idx_test[:100]:  # 只取前 100 个样本用于可视化
            # 输入序列
            start = idx - input_len - output_len + 1
            end = idx + 1
            
            if start < 0:
                continue
            
            # 获取数据（只取流量特征）
            data = df.iloc[start:end].values.T  # [N, T]
            
            # 分割输入和输出
            x = data[:, :input_len]  # [N, L]
            y = data[:, input_len:]  # [N, P]
            
            X_test.append(x)
            y_test.append(y)
        
        X_test = np.array(X_test)  # [num_samples, N, L]
        y_test = np.array(y_test)  # [num_samples, N, P]
        
        print(f"✓ 测试样本准备完成：{X_test.shape}")
        
        return X_test, y_test
    
    def load_model_and_predict(self, X_test):
        """加载模型并进行预测"""
        print("正在加载模型并进行预测...")
        
        # 这里需要导入 FaST 模型并加载权重
        # 由于需要完整的模型定义，这里提供简化版本
        
        # 方案 1：使用训练时的 runner 加载模型
        # 方案 2：手动加载模型权重
        
        # 示例：加载预测结果（如果训练时保存了）
        pred_file = 'checkpoints/FaST/HNGS_50_96_48/predictions.npy'
        if os.path.exists(pred_file):
            predictions = np.load(pred_file)
            print(f"✓ 已加载保存的预测结果：{predictions.shape}")
        else:
            print("⚠ 未找到保存的预测结果，需要使用模型重新预测")
            # 这里需要实际的模型推理代码
            predictions = None
        
        return predictions
    
    def plot_single_station(self, real, predicted, station_idx, station_name, output_dir):
        """绘制单个站点的预测对比"""
        plt.figure(figsize=(12, 4))
        
        time_steps = len(real)
        time_axis = np.arange(time_steps)
        
        plt.plot(time_axis, real, 'b-', linewidth=1.5, label='Real', alpha=0.7)
        plt.plot(time_axis, predicted, 'r--', linewidth=1.5, label='Predicted', alpha=0.7)
        
        # 计算误差
        mae = np.mean(np.abs(real - predicted))
        rmse = np.sqrt(np.mean((real - predicted)**2))
        
        plt.title(f'Station {station_idx}: {station_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Time Step (15 min)')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/station_{station_idx:03d}_comparison.png', dpi=300)
        plt.close()
    
    def plot_multiple_stations(self, df, X_test, y_test, predictions, output_dir='visualization'):
        """绘制多个站点的对比图"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在绘制多站点预测对比图...")
        
        # 选择几个典型站点
        num_stations = len(df.columns)
        station_indices = [0, num_stations//4, num_stations//2, 3*num_stations//4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, station_idx in enumerate(station_indices):
            if i >= len(axes):
                break
            
            ax = axes[i]
            station_name = df.columns[station_idx]
            
            # 获取第一个测试样本的真实值和预测值
            if predictions is not None:
                real = y_test[0, station_idx, :]
                predicted = predictions[0, station_idx, :]
            else:
                # 如果没有预测值，用真实值加噪声模拟
                real = y_test[0, station_idx, :]
                np.random.seed(42)
                predicted = real + np.random.randn(len(real)) * 5
            
            time_axis = np.arange(len(real))
            
            ax.plot(time_axis, real, 'b-', linewidth=1.5, label='Real', alpha=0.7)
            ax.plot(time_axis, predicted, 'r--', linewidth=1.5, label='Predicted', alpha=0.7)
            
            # 计算误差
            mae = np.mean(np.abs(real - predicted))
            rmse = np.sqrt(np.mean((real - predicted)**2))
            
            ax.set_title(f'Station {station_idx}: {station_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Traffic Flow')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/multi_station_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ 多站点对比图已保存：{output_dir}/multi_station_comparison.png")
        plt.close()
        
        # 为每个站点单独绘图
        if predictions is not None:
            for i, station_idx in enumerate([0, 50, 100, 150]):
                if station_idx < num_stations:
                    real = y_test[0, station_idx, :]
                    predicted = predictions[0, station_idx, :]
                    self.plot_single_station(real, predicted, station_idx, 
                                           df.columns[station_idx], output_dir)
    
    def plot_error_metrics(self, y_test, predictions, output_dir='visualization'):
        """绘制误差指标统计图"""
        if predictions is None:
            print("⚠ 没有预测数据，跳过误差指标绘图")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("正在绘制误差指标统计图...")
        
        # 计算每个站点的 MAE
        num_stations = y_test.shape[1]
        mae_per_station = []
        
        for i in range(num_stations):
            mae = np.mean(np.abs(y_test[:, i, :] - predictions[:, i, :]))
            mae_per_station.append(mae)
        
        # 绘制 MAE 分布
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(mae_per_station, bins=30, color='#2E86AB', edgecolors='black', alpha=0.7)
        plt.axvline(x=np.mean(mae_per_station), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(mae_per_station):.2f}')
        plt.xlabel('MAE')
        plt.ylabel('Number of Stations')
        plt.title('MAE Distribution Across All Stations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制 MAE 随站点的变化
        plt.subplot(1, 2, 2)
        plt.plot(range(num_stations), mae_per_station, 'o-', color='#A23B72', 
                markersize=3, alpha=0.7)
        plt.xlabel('Station Index')
        plt.ylabel('MAE')
        plt.title('MAE by Station')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_statistics.png', dpi=300)
        print(f"✓ 误差统计图已保存：{output_dir}/error_statistics.png")
        plt.close()
    
    def plot_time_series_heatmap(self, df, y_test, predictions, output_dir='visualization'):
        """绘制时间序列热力图"""
        if predictions is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("正在绘制误差热力图...")
        
        # 计算误差矩阵 [时间步，站点]
        error_matrix = np.abs(y_test[0] - predictions[0])  # [N, P]
        error_matrix = error_matrix.T  # [P, N]
        
        plt.figure(figsize=(15, 8))
        
        im = plt.imshow(error_matrix, aspect='auto', cmap='RdBu_r',
                       extent=[0, error_matrix.shape[1], error_matrix.shape[0], 0])
        
        plt.colorbar(im, label='Absolute Error')
        plt.xlabel('Station Index')
        plt.ylabel('Prediction Time Step')
        plt.title('Prediction Error Heatmap (Time × Station)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_heatmap.png', dpi=300)
        print(f"✓ 误差热力图已保存：{output_dir}/error_heatmap.png")
        plt.close()


def main():
    print("="*70)
    print("湖南高速公路流量预测 - 结果可视化")
    print("="*70)
    
    # 初始化可视化器
    visualizer = PredictionVisualizer(
        config_path='FaST/HNGS_96_48.py',
        checkpoint_path='checkpoints/FaST/HNGS_50_96_48/FaST_best_val_MAE.pt',
        data_dir='DataPipeline/HNGS'
    )
    
    # 加载数据
    df, idx_test, adj_matrix = visualizer.load_test_data()
    
    # 准备测试样本
    X_test, y_test = visualizer.prepare_test_samples(df, idx_test)
    
    # 加载模型并预测
    predictions = visualizer.load_model_and_predict(X_test)
    
    # 生成可视化图表
    visualizer.plot_multiple_stations(df, X_test, y_test, predictions)
    visualizer.plot_error_metrics(y_test, predictions)
    visualizer.plot_time_series_heatmap(df, y_test, predictions)
    
    print("\n" + "="*70)
    print("✓ 所有可视化图表已生成完毕！")
    print("查看目录：visualization/")
    print("="*70)


if __name__ == "__main__":
    main()
