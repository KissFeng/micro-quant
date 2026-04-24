"""
上下文构建器模块
负责将预测数据格式化为LLM可理解的文本上下文
"""

import pandas as pd
from typing import Optional


class ContextBuilder:
    """上下文构建器类，格式化预测数据"""

    @staticmethod
    def build_full_context(df: pd.DataFrame) -> str:
        """
        构建完整预测数据上下文

        参数:
            df: 预测数据DataFrame

        返回:
            格式化的上下文字符串
        """
        if df.empty:
            return "当前没有可用的预测数据"

        context_parts = [
            "### 预测日期: " + str(df['date'].iloc[0]),
            "\n### 全部股票预测结果:\n"
        ]

        for idx, row in df.iterrows():
            stock_info = f"""
**{idx+1}. {row['name']} ({row['stock']})**
- 当前价格: {row['current_price']:.2f}元
- 综合预测概率: {row['probability']:.2%}
- LightGBM: {row['LightGBM']:.2%}
- CatBoost: {row['CatBoost']:.2%}
- LSTM: {row['LSTM']:.2%}
"""
            context_parts.append(stock_info)

        return "\n".join(context_parts)

    @staticmethod
    def build_summary_context(df: pd.DataFrame, top_n: int = 5) -> str:
        """
        构建摘要上下文（Top N + 整体统计）

        参数:
            df: 预测数据DataFrame
            top_n: Top N股票数量

        返回:
            格式化的摘要上下文字符串
        """
        if df.empty:
            return "当前没有可用的预测数据"

        # 统计信息
        total_stocks = len(df)
        bullish_count = (df['probability'] > 0.5).sum()
        bearish_count = total_stocks - bullish_count
        avg_prob = df['probability'].mean()

        context = f"""### 预测数据摘要
- 预测日期: {df['date'].iloc[0]}
- 股票总数: {total_stocks}只
- 看涨股票: {bullish_count}只 (概率>50%)
- 看跌股票: {bearish_count}只
- 平均预测概率: {avg_prob:.2%}

### Top {top_n} 推荐股票:
"""

        for idx, row in df.head(top_n).iterrows():
            context += f"\n{idx+1}. {row['name']}({row['stock']}): {row['probability']:.1%}, 价格{row['current_price']:.2f}元"

        return context
