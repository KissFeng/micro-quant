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

        # ========== 移动平均线特征（只保留ratio，去除原始MA值） ==========
        for window in [5, 10, 20, 60]:
            ma = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / ma - 1

        # 均线交叉信号
        ma5 = df['Close'].rolling(window=5).mean()
        ma10 = df['Close'].rolling(window=10).mean()
        ma20 = df['Close'].rolling(window=20).mean()
        df['MA5_20_cross'] = (ma5 > ma20).astype(int)
        df['MA10_20_cross'] = (ma10 > ma20).astype(int)

        # ========== RSI 特征 ==========
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_change'] = df['RSI_14'].diff()

        # ========== MACD 特征 ==========
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_hist'] = exp12 - exp26 - (exp12 - exp26).ewm(span=9, adjust=False).mean()
        df['MACD_hist_change'] = df['MACD_hist'].diff()

        # ========== 布林带特征 ==========
        bb_window = 20
        bb_middle = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        df['BB_width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ========== ATR 特征 ==========
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_ratio'] = tr.rolling(window=14).mean() / df['Close']

        # ========== 成交量特征 ==========
        vol_ma5 = df['Volume'].rolling(window=5).mean()
        vol_ma20 = df['Volume'].rolling(window=20).mean()
        df['Vol_ratio'] = df['Volume'] / (vol_ma20 + 1e-10)
        df['Vol_change'] = vol_ma5 / (vol_ma20 + 1e-10)

        # ========== OBV 特征（只保留ratio） ==========
        obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        obv_ma10 = obv.rolling(window=10).mean()
        df['OBV_ratio'] = obv / (obv_ma10 + 1e-10)

        # ========== 波动率特征 ==========
        df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        df['Volatility_ratio'] = df['Volatility_5d'] / (df['Volatility_20d'] + 1e-10)

        # ========== Williams %R & Stochastic ==========
        highest_high_14 = df['High'].rolling(window=14).max()
        lowest_low_14 = df['Low'].rolling(window=14).min()
        df['Williams_R'] = -100 * (highest_high_14 - df['Close']) / (highest_high_14 - lowest_low_14 + 1e-10)
        df['Stoch_K'] = 100 * (df['Close'] - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # ========== 日内特征 ==========
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = df['Close'] / df['Open'] - 1

        # ========== 支撑阻力位（只保留ratio） ==========
        support_20d = df['Low'].rolling(window=20).min()
        resistance_20d = df['High'].rolling(window=20).max()
        df['Price_vs_Support'] = df['Close'] / support_20d - 1
        df['Price_vs_Resistance'] = df['Close'] / resistance_20d - 1

        # ========== 涨跌统计特征 ==========
        df['Up_days_5'] = (df['Return_1d'] > 0).rolling(window=5).mean()
        df['Up_days_20'] = (df['Return_1d'] > 0).rolling(window=20).mean()

        return df

    def create_label(self, df, horizon=5, for_backtest=False, threshold=0.005):
        """
        创建标签：未来N天涨跌

        参数:
            df: 股票数据 DataFrame
            horizon: 预测周期（天）
            for_backtest: 是否为回测模式
            threshold: 涨跌阈值，低于此涨幅的样本标记为-1（模糊区域）

        返回:
            Series: 标签 (1=涨, 0=跌, -1=模糊/排除)
        """
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1

        label = pd.Series(-1, index=df.index)
        label[future_return > threshold] = 1    # 明确上涨
        label[future_return < -threshold] = 0    # 明确下跌
        # 中间区域保持 -1（模糊，训练时排除）

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

        # 排除模糊样本（Label == -1）
        df_clean = df_clean[df_clean['Label'] != -1]

        if df_clean.empty:
            return pd.DataFrame(), pd.Series(), feature_cols

        X = df_clean[feature_cols]
        y = df_clean['Label']

        self.feature_columns = feature_cols

        return X, y, feature_cols

    def get_feature_columns(self):
        """获取特征列名"""
        return self.feature_columns

    def add_market_features(self, df, index_df):
        """
        添加大盘指数特征

        参数:
            df: 个股数据 DataFrame
            index_df: 指数数据 DataFrame（必须包含 Close 列）

        返回:
            DataFrame: 添加了大盘特征的 DataFrame
        """
        if index_df.empty or df.empty:
            return df

        df = df.copy()

        # 对齐日期
        index_close = index_df['Close'].reindex(df.index, method='ffill')

        # 大盘收益率
        df['Market_Return_1d'] = index_close.pct_change(1)
        df['Market_Return_5d'] = index_close.pct_change(5)
        df['Market_Return_20d'] = index_close.pct_change(20)

        # 大盘波动率
        df['Market_Volatility_5d'] = df['Market_Return_1d'].rolling(window=5).std()
        df['Market_Volatility_20d'] = df['Market_Return_1d'].rolling(window=20).std()

        # 个股 vs 大盘相对强弱
        stock_return_5d = df['Close'].pct_change(5)
        df['Relative_Strength'] = stock_return_5d - df['Market_Return_5d']

        return df


def combine_multiple_stocks(stock_data_dict, horizon=5, test_ratio=0.2, index_data=None):
    """
    合并多只股票的数据，按时间顺序划分训练集和测试集

    参数:
        stock_data_dict: {stock_code: DataFrame}
        horizon: 预测周期
        test_ratio: 测试集比例
        index_data: 指数数据 DataFrame（可选）

    返回:
        tuple: (X_train, y_train, X_test, y_test, stock_codes)
    """
    engineer = FeatureEngineer()
    train_X_list = []
    train_y_list = []
    test_X_list = []
    test_y_list = []
    stock_codes = []

    for code, df in stock_data_dict.items():
        # 添加大盘特征
        if index_data is not None and not index_data.empty:
            df = engineer.add_market_features(df, index_data)

        X, y, _ = engineer.prepare_features(df, horizon)
        if X.empty or len(X) < 30:
            continue

        # 按时间顺序划分：前 80% 训练，后 20% 测试
        split_idx = int(len(X) * (1 - test_ratio))
        train_X_list.append(X.iloc[:split_idx])
        train_y_list.append(y.iloc[:split_idx])
        test_X_list.append(X.iloc[split_idx:])
        test_y_list.append(y.iloc[split_idx:])
        stock_codes.append(code)

    if not train_X_list:
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), []

    X_train = pd.concat(train_X_list, axis=0, ignore_index=True)
    y_train = pd.concat(train_y_list, axis=0, ignore_index=True)
    X_test = pd.concat(test_X_list, axis=0, ignore_index=True)
    y_test = pd.concat(test_y_list, axis=0, ignore_index=True)

    return X_train, y_train, X_test, y_test, stock_codes


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
