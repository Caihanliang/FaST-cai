# 湖南高速公路流量数据集训练脚本

echo "开始训练 FaST 模型 - 湖南高速公路流量预测..."

# 训练 FaST 在 HNGS 数据集上（预测 48 步，12 小时）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_48.py -g 0

# 训练 FaST 在 HNGS 数据集上（预测 96 步，24 小时）
python main-master/experiments/train_seed.py -c FaST/HNGS_96_96.py -g 0

echo "训练完成！"
echo "模型保存在：checkpoints/FaST/"
