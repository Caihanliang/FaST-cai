"""
湖南高速公路数据集配置注册
"""

# 添加 HNGS 数据集的默认配置
HNGS_CONFIG = {
    "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
    "NORM_EACH_CHANNEL": False,
    "RESCALE": True,
    "METRICS": ["MAE", "RMSE", "MAPE"],
    "NULL_VAL": 0.0,
    "INPUT_LEN": 96,
    "OUTPUT_LEN": 48,
}
