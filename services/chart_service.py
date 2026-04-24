"""
图表服务模块
负责生成交互式可视化图表
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


class ChartService:
    """图表服务类，生成各类可视化图表"""

    @staticmethod
    def create_candlestick_chart(
        df: pd.DataFrame,
        stock_name: str,
        probability: Optional[float] = None
    ) -> go.Figure:
        """
        创建K线图

        参数:
            df: 股票数据DataFrame，需包含Open/High/Low/Close/Volume列
            stock_name: 股票名称
            probability: 预测概率（可选）

        返回:
            Plotly Figure对象
        """
        fig = go.Figure()

        # K线图
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green'
        ))

        # 成交量柱状图（底部）
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='成交量',
            marker_color='rgba(100, 100, 100, 0.3)',
            yaxis='y2'
        ))

        # 布局设置
        title = f'{stock_name} K线图'
        if probability is not None:
            title += f' (预测概率: {probability:.1%})'

        fig.update_layout(
            title=title,
            yaxis=dict(
                title='价格',
                domain=[0.3, 1.0]
            ),
            yaxis2=dict(
                title='成交量',
                domain=[0, 0.25]
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            height=600
        )

        return fig

    @staticmethod
    def create_probability_distribution(df: pd.DataFrame) -> go.Figure:
        """
        创建概率分布直方图

        参数:
            df: 预测数据DataFrame

        返回:
            Plotly Figure对象
        """
        fig = px.histogram(
            df,
            x='probability',
            nbins=20,
            title='综合预测概率分布',
            labels={'probability': '预测概率'},
            color_discrete_sequence=['#1f77b4']
        )

        fig.update_layout(
            xaxis_title='预测概率',
            yaxis_title='股票数量',
            showlegend=False
        )

        # 添加50%分界线
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="看涨/看跌分界"
        )

        return fig

    @staticmethod
    def create_model_comparison(df: pd.DataFrame) -> go.Figure:
        """
        创建模型对比箱线图

        参数:
            df: 预测数据DataFrame

        返回:
            Plotly Figure对象
        """
        model_data = df[['LightGBM', 'CatBoost', 'LSTM']].melt(
            var_name='Model',
            value_name='Probability'
        )

        fig = px.box(
            model_data,
            x='Model',
            y='Probability',
            title='各模型预测分布对比',
            color='Model',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )

        fig.update_layout(
            xaxis_title='模型',
            yaxis_title='预测概率',
            showlegend=False
        )

        # 添加50%分界线
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="看涨/看跌分界"
        )

        return fig
