"""
预测数据服务模块
负责加载和管理模型预测结果
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import streamlit as st


class PredictionService:
    """预测数据服务类，管理CSV数据加载和缓存"""

    def __init__(self):
        self.predictions_file = Path('output/predictions.csv')

    @st.cache_data(ttl=300)
    def load_predictions(_self) -> pd.DataFrame:
        """
        加载预测数据（带缓存）

        返回:
            DataFrame，包含所有股票的预测结果
        """
        if not _self.predictions_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(_self.predictions_file)
            df = df.sort_values('probability', ascending=False).reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"加载预测数据失败: {str(e)}")
            return pd.DataFrame()

    def get_top_stocks(self, n: int = 10) -> pd.DataFrame:
        """
        获取Top N推荐股票

        参数:
            n: 股票数量

        返回:
            DataFrame，Top N股票
        """
        df = self.load_predictions()
        if df.empty:
            return df
        return df.head(n)

    def get_stock_by_code(self, code: str) -> Optional[pd.Series]:
        """
        根据股票代码获取预测数据

        参数:
            code: 股票代码

        返回:
            Series或None
        """
        df = self.load_predictions()
        if df.empty or code not in df['stock'].values:
            return None
        return df[df['stock'] == code].iloc[0]

    def get_statistics(self) -> Dict:
        """
        获取预测数据统计信息

        返回:
            统计信息字典
        """
        df = self.load_predictions()
        if df.empty:
            return {
                'total': 0,
                'bullish': 0,
                'bearish': 0,
                'avg_prob': 0.0
            }

        total = len(df)
        bullish = (df['probability'] > 0.5).sum()
        bearish = total - bullish
        avg_prob = df['probability'].mean()

        return {
            'total': total,
            'bullish': bullish,
            'bearish': bearish,
            'avg_prob': avg_prob
        }
