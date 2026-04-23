#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 资金量化系统
"""

# 股票列表（20只自选股，覆盖更多行业）
STOCK_LIST = {
    '000001.SZ': '平安银行',      # 银行
    '000002.SZ': '万科A',        # 房地产
    '000858.SZ': '五粮液',        # 白酒
    '600036.SH': '招商银行',      # 银行
    '600519.SH': '贵州茅台',      # 白酒
    '600887.SH': '伊利股份',      # 乳业
    '601318.SH': '中国平安',      # 保险
    '601888.SH': '中国中免',      # 免税
    '002594.SZ': '比亚迪',        # 新能源车
    '000651.SZ': '格力电器',      # 家电
    '600276.SH': '恒瑞医药',      # 医药
    '000333.SZ': '美的集团',      # 家电
    '601012.SH': '隆基绿能',      # 光伏
    '600900.SH': '长江电力',      # 电力
    '000568.SZ': '泸州老窖',      # 白酒
    '002415.SZ': '海康威视',      # 安防
    '600809.SH': '山西汾酒',      # 白酒
    '601899.SH': '紫金矿业',      # 矿业
    '002230.SZ': '科大讯飞',      # AI
    '600030.SH': '中信证券',      # 券商
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
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 4,
    'num_leaves': 15,
    'min_child_samples': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 4,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False
}

LSTM_PARAMS = {
    'seq_length': 5,
    'lstm_units_1': 16,
    'lstm_units_2': 8,
    'dropout': 0.5,
    'epochs': 30,
    'batch_size': 64,
    'learning_rate': 0.0005
}

# 数据参数
TRAIN_START_DATE = '2021-01-01'
TRAIN_END_DATE = '2025-12-31'
DATA_CACHE_DAYS = 3

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
