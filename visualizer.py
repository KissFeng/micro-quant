#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块 - 生成各类图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 中文字体配置（style.use 会重置 rcParams，需在每次调用后重新设置）
def _setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Songti SC', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-darkgrid')
_setup_chinese_font()


class Visualizer:
    """可视化类"""

    def __init__(self, figsize=(12, 6), dpi=150):
        self.figsize = figsize
        self.dpi = dpi

    def plot_prediction_results(self, predictions_df, prices_series, save_path='output/predictions.png'):
        """
        绘制预测结果图

        参数:
            predictions_df: 预测结果 DataFrame [date, stock, probability, actual]
            prices_series: 价格 Series（索引为日期）
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        dates = pd.to_datetime(predictions_df['date'].unique())
        probs = predictions_df.groupby('date')['probability'].mean()
        actuals = predictions_df.groupby('date')['actual'].mean()

        # 上图：价格走势
        ax1 = axes[0]
        ax1.plot(prices_series.index, prices_series.values, 'b-', linewidth=1.5, label='价格')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('股价走势与预测信号', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 买入信号（概率 > 0.55）
        buy_signals = predictions_df[predictions_df['probability'] > 0.55]['date'].unique()
        buy_prices = prices_series.loc[prices_series.index.isin(buy_signals)]
        ax1.scatter(buy_prices.index, buy_prices.values, marker='^', color='green',
                    s=100, label='买入信号', zorder=5)

        # 卖出信号（概率 <= 0.45）
        sell_signals = predictions_df[predictions_df['probability'] <= 0.45]['date'].unique()
        sell_prices = prices_series.loc[prices_series.index.isin(sell_signals)]
        ax1.scatter(sell_prices.index, sell_prices.values, marker='v', color='red',
                   s=100, label='卖出信号', zorder=5)

        ax1.legend()

        # 下图：预测概率
        ax2 = axes[1]
        ax2.plot(dates, probs.loc[probs.index.isin(dates)].values, 'g-', linewidth=1.5, label='预测概率')
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='中性线')
        ax2.axhline(y=0.55, color='orange', linestyle='--', linewidth=1, label='买入阈值')
        ax2.axhline(y=0.45, color='red', linestyle='--', linewidth=1, label='卖出阈值')

        # 填充区域
        ax2.fill_between(dates, probs.loc[probs.index.isin(dates)].values, 0.5,
                        where=probs.loc[probs.index.isin(dates)].values > 0.5,
                        alpha=0.3, color='green', label='看涨区间')
        ax2.fill_between(dates, probs.loc[probs.index.isin(dates)].values, 0.5,
                        where=probs.loc[probs.index.isin(dates)].values <= 0.5,
                        alpha=0.3, color='red', label='看跌区间')

        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('概率', fontsize=12)
        ax2.set_title('预测概率随时间变化', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ 预测结果图已保存到 {save_path}")

    def plot_model_comparison(self, results_dict, save_path='output/model_comparison.png'):
        """
        绘制模型对比柱状图

        参数:
            results_dict: {model_name: {metric: value}}
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

        models = list(results_dict.keys())
        metrics = ['accuracy', 'auc', 'f1']
        n_metrics = len(metrics)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.25
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        metric_labels = {'accuracy': '准确率', 'auc': 'AUC', 'f1': 'F1分数'}
        for i, metric in enumerate(metrics):
            values = [results_dict[m].get(metric, 0) for m in models]
            ax.bar(x + i * width, values, width, label=metric_labels.get(metric, metric.upper()), color=colors[i])

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title('模型性能对比', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ 模型对比图已保存到 {save_path}")

    def plot_feature_importance(self, importance_dict, top_n=15, save_path='output/feature_importance.png'):
        """
        绘制特征重要性图

        参数:
            importance_dict: {feature_name: importance_value}
            top_n: 显示前 N 个特征
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

        # 排序
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('重要性 (Gain)', fontsize=12)
        ax.set_title(f'Top {top_n} 特征重要性', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ 特征重要性图已保存到 {save_path}")

    def plot_equity_curve(self, portfolio_values, benchmark_values, dates, save_path='output/equity_curve.png'):
        """
        绘制权益曲线对比图

        参数:
            portfolio_values: 策略权益列表
            benchmark_values: 基准权益列表
            dates: 日期列表
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        dates = pd.to_datetime(dates)
        ax.plot(dates, portfolio_values, label='策略', linewidth=2, color='#2ecc71')
        ax.plot(dates, benchmark_values, label='买入持有', linewidth=2, color='#3498db', alpha=0.7)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('组合价值 (¥)', fontsize=12)
        ax.set_title('权益曲线对比', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ 权益曲线图已保存到 {save_path}")

    def plot_stock_comparison(self, stock_data_dict, save_path='output/stock_comparison.png'):
        """
        绘制多只股票走势对比图

        参数:
            stock_data_dict: {stock_code: DataFrame with Close column}
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        for code, df in stock_data_dict.items():
            # 归一化到起始值
            normalized = df['Close'] / df['Close'].iloc[0]
            ax.plot(df.index, normalized, label=code, linewidth=1.5)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('归一化价格 (起始=1)', fontsize=12)
        ax.set_title('股票走势对比', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ 股票对比图已保存到 {save_path}")


if __name__ == '__main__':
    print("测试可视化模块...")

    viz = Visualizer()

    # 测试特征重要性图
    importance = {f'Feature_{i}': np.random.rand() for i in range(20)}
    importance['RSI_14'] = 10.5
    importance['MACD'] = 8.2
    importance['Vol_ratio'] = 6.8

    viz.plot_feature_importance(importance, top_n=10)
