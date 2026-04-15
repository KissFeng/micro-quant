#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 资金量化系统
"""

# 股票列表（10只自选股，可替换为A股代码如 '000001.SZ'）
STOCK_LIST = {
    '000001.SZ': '平安银行',      # 银行
    '000002.SZ': '万科A',        # 房地产
    '000858.SZ': '五粮液',        # 白酒
    '600036.SH': '招商银行',      # 银行
    '600519.SH': '贵州茅台',      # 白酒
    '600887.SH': '伊利股份',      # 乳业
    '601318.SH': '中国平安',      # 保险
    '000858.SZ': '五粮液',        # 白酒
    '601888.SH': '中国中免',      # 免税
    '002594.SZ': '比亚迪',        # 新能源车
}

# 预测周期（天）
DEFAULT_HORIZON = 5

# 回测参数
INITIAL_CAPITAL = 100000  # 初始资金 10万
COMMISSION = 0.001       # 佣金千分之一
SLIPPAGE = 0.001         # 滑点千分之一
CONFIDENCE_THRESHOLD = 0.55  # 置信度阈值

# 模型参数
LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.05,
    'depth': 5,
    'random_seed': 42,
    'verbose': False
}

LSTM_PARAMS = {
    'seq_length': 20,
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001
}

# 数据参数
TRAIN_START_DATE = '2024-01-01'
TRAIN_END_DATE = '2025-12-31'
DATA_CACHE_DAYS = 7

# Email 配置（需要用户填写）
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',      # QQ邮箱SMTP服务器
    'smtp_port': 465,                   # SSL端口
    'sender': '26699525@qq.com',      # 发件人邮箱
    'password': 'akvehotucbimcajg',            # 邮箱授权码（非密码）
    'recipient': '26699525@qq.com',  # 收件人邮箱
}

# 可视化配置
CHART_CONFIG = {
    'dpi': 150,
    'figsize': (12, 6),
    'style': 'seaborn-v0_8-darkgrid'
}
