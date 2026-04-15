#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块 - 生成技术指标特征
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.feature_columns = []

    def calculate_technical_features(self, df):
        """
        计算所有技术指标特征

        参数:
            df: 股票数据 DataFrame，必须包含 High, Low, Close, Volume

        返回:
            DataFrame: 添加了技术指标特征的 DataFrame
        """
        if df.empty or len(df) < 30:
            return df

        df = df.copy()

        # ========== 价格收益率特征 ==========
        df['Return_1d'] = df['Close'].pct_change(1)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        df['Return_20d'] = df['Close'].pct_change(20)

        # ========== 移动平均线特征 ==========
        for window in [5, 10, 20, 60]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}'] - 1

        # 均线交叉信号
        df['MA5_20_cross'] = (df['MA_5'] > df['MA_20']).astype(int)
        df['MA10_20_cross'] = (df['MA_10'] > df['MA_20']).astype(int)

        # ========== RSI 特征 ==========
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # RSI 动量
        df['RSI_change'] = df['RSI_14'].diff()

        # ========== MACD 特征 ==========
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_change'] = df['MACD_hist'].diff()

        # ========== 布林带特征 ==========
        bb_window = 20
        df['BB_middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)

        # ========== ATR 特征 ==========
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        df['ATR_ratio'] = df['ATR_14'] / df['Close']

        # ========== 成交量特征 ==========
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_ratio'] = df['Volume'] / (df['Vol_MA20'] + 1e-10)
        df['Vol_Z_score'] = (df['Volume'] - df['Vol_MA20']) / df['Volume'].rolling(window=20).std()

        # ========== OBV 特征 ==========
        obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['OBV'] = obv
        df['OBV_MA10'] = obv.rolling(window=10).mean()
        df['OBV_ratio'] = obv / (df['OBV_MA10'] + 1e-10)

        # ========== 动量特征 ==========
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1

        # ROC (Rate of Change)
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10) * 100
        df['ROC_20'] = (df['Close'] - df['Close'].shift(20)) / (df['Close'].shift(20) + 1e-10) * 100

        # ========== 波动率特征 ==========
        df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()

        # ========== Williams %R ==========
        highest_high = df['High'].rolling(window=14).max()
        lowest_low = df['Low'].rolling(window=14).min()
        df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-10)

        # ========== Stochastic ==========
        df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # ========== 相对强弱（vs 恒生指数） ==========
        # 简化版：使用自身的成交量/价格比作为代理

        # ========== 日内特征 ==========
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = df['Close'] / df['Open'] - 1

        # ========== 支撑阻力位 ==========
        df['Support_20d'] = df['Low'].rolling(window=20).min()
        df['Resistance_20d'] = df['High'].rolling(window=20).max()
        df['Price_vs_Support'] = df['Close'] / df['Support_20d'] - 1
        df['Price_vs_Resistance'] = df['Close'] / df['Resistance_20d'] - 1

        return df

    def create_label(self, df, horizon=5, for_backtest=False):
        """
        创建标签：未来N天涨跌

        参数:
            df: 股票数据 DataFrame
            horizon: 预测周期（天）
            for_backtest: 是否为回测模式（True使用shift(-horizon)）

        返回:
            Series: 标签 (1=涨, 0=跌)
        """
        if for_backtest:
            # 回测模式：使用未来数据
            future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        else:
            # 训练模式：正常
            future_return = df['Close'].shift(-horizon) / df['Close'] - 1

        label = (future_return > 0).astype(int)
        return label

    def prepare_features(self, df, horizon=5, dropna=True):
        """
        准备训练数据：特征 + 标签

        参数:
            df: 股票数据 DataFrame
            horizon: 预测周期
            dropna: 是否删除空值

        返回:
            tuple: (features DataFrame, labels Series, feature_columns list)
        """
        # 计算技术指标
        df_features = self.calculate_technical_features(df)

        if df_features.empty or len(df_features) < 60:
            return pd.DataFrame(), pd.Series(), []

        # 创建标签
        df_features['Label'] = self.create_label(df_features, horizon)

        # 选择特征列（排除原始价格和标签）
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Label']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        # 添加一些基础价格特征
        base_features = ['Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
                        'Intraday_Range', 'Close_Open_Ratio']
        feature_cols = base_features + [col for col in feature_cols if col not in base_features]

        # 删除空值
        if dropna:
            df_clean = df_features[feature_cols + ['Label']].dropna()
        else:
            df_clean = df_features[feature_cols + ['Label']].fillna(0)

        if df_clean.empty:
            return pd.DataFrame(), pd.Series(), feature_cols

        X = df_clean[feature_cols]
        y = df_clean['Label']

        self.feature_columns = feature_cols

        return X, y, feature_cols

    def get_feature_columns(self):
        """获取特征列名"""
        return self.feature_columns


def combine_multiple_stocks(stock_data_dict, horizon=5):
    """
    合并多只股票的数据

    参数:
        stock_data_dict: {stock_code: DataFrame}
        horizon: 预测周期

    返回:
        tuple: (X, y, stock_codes)
    """
    engineer = FeatureEngineer()
    all_X = []
    all_y = []
    stock_codes = []

    for code, df in stock_data_dict.items():
        X, y, _ = engineer.prepare_features(df, horizon)
        if not X.empty:
            all_X.append(X)
            all_y.append(y)
            stock_codes.append(code)

    if not all_X:
        return pd.DataFrame(), pd.Series(), []

    X_combined = pd.concat(all_X, axis=0, ignore_index=True)
    y_combined = pd.concat(all_y, axis=0, ignore_index=True)

    return X_combined, y_combined, stock_codes


if __name__ == '__main__':
    # 测试代码
    from data_loader import load_stock_data

    print("测试特征工程模块...")

    # 加载数据
    df = load_stock_data('000001.SZ', '2022-01-01', '2024-12-31')
    print(f"原始数据: {len(df)} 条")

    # 创建特征工程师
    engineer = FeatureEngineer()
    X, y, feature_cols = engineer.prepare_features(df, horizon=5)

    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"正样本比例: {y.mean():.2%}")
    print(f"\n特征列: {feature_cols[:10]}...")
