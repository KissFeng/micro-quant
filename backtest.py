#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块 - 模拟交易并评估策略表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import INITIAL_CAPITAL, COMMISSION, SLIPPAGE


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital=INITIAL_CAPITAL, commission=COMMISSION, slippage=SLIPPAGE):
        """
        初始化回测引擎

        参数:
            initial_capital: 初始资金
            commission: 佣金费率
            slippage: 滑点费率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0  # 持仓数量
        self.trades = []   # 交易记录
        self.portfolio_values = []  # 组合价值历史
        self.buy_hold_values = []   # 买入持有基准
        self.dates = []             # 日期历史

    def run(self, predictions, prices, confidence_threshold=0.55):
        """
        运行回测

        参数:
            predictions: DataFrame，列 [date, stock, probability, actual]
            prices: Series，价格索引为日期
            confidence_threshold: 置信度阈值

        返回:
            dict: 回测结果
        """
        self.reset()

        # 按日期排序
        if 'date' in predictions.columns:
            predictions = predictions.sort_values('date')

        dates = predictions['date'].unique()

        for date in dates:
            date_preds = predictions[predictions['date'] == date]

            if len(date_preds) == 0:
                continue

            # 当日平均预测概率（简化：取平均）
            avg_prob = date_preds['probability'].mean()
            actual = date_preds['actual'].iloc[0] if 'actual' in date_preds.columns else None

            # 获取当日价格
            if date in prices.index:
                current_price = prices.loc[date]
            else:
                # 找到最接近的日期
                closest_idx = prices.index.get_indexer([date], method='ffill')[0]
                current_price = prices.iloc[closest_idx]

            # 交易逻辑
            signal = 1 if avg_prob > confidence_threshold else 0

            # 买入
            if signal == 1 and self.position == 0:
                buy_price = current_price * (1 + self.slippage)
                max_shares = int(self.capital / buy_price)
                if max_shares > 0:
                    cost = max_shares * buy_price * (1 + self.commission)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = max_shares
                        self.trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': buy_price,
                            'shares': max_shares,
                            'probability': avg_prob,
                            'actual': actual
                        })

            # 卖出
            elif signal == 0 and self.position > 0:
                sell_price = current_price * (1 - self.slippage)
                proceeds = self.position * sell_price * (1 - self.commission)
                profit = proceeds - (self.trades[-1]['shares'] *
                                     self.trades[-1]['price'] *
                                     (1 + self.commission))
                self.trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': self.position,
                    'probability': avg_prob,
                    'actual': actual,
                    'profit': profit
                })
                self.capital += proceeds
                self.position = 0

            # 记录组合价值
            portfolio_value = self.capital + self.position * current_price
            self.portfolio_values.append(portfolio_value)
            self.buy_hold_values.append(self.initial_capital * (current_price / prices.iloc[0]
                                                                 if len(self.buy_hold_values) == 0
                                                                 else self.buy_hold_values[-1] / prices.iloc[0] * current_price))
            self.dates.append(date)

        # 如果还有持仓，按最后价格计算
        if self.position > 0:
            final_price = prices.iloc[-1]
            portfolio_value = self.capital + self.position * final_price
            self.portfolio_values[-1] = portfolio_value

        return self.calculate_metrics()

    def calculate_metrics(self):
        """
        计算回测指标

        返回:
            dict: 回测指标
        """
        if len(self.portfolio_values) == 0:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        buy_hold_values = np.array(self.buy_hold_values)

        # 计算收益率
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        buy_hold_returns = np.diff(buy_hold_values) / buy_hold_values[:-1]

        # 总收益率
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        benchmark_return = (buy_hold_values[-1] - self.initial_capital) / self.initial_capital

        # 年化收益率（假设一年252个交易日）
        n_days = len(portfolio_values)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # 夏普比率
        if len(portfolio_returns) > 0 and np.std(portfolio_returns) > 0:
            excess_returns = portfolio_returns - 0.02 / 252  # 假设无风险利率2%
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 最大回撤
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)

        # 胜率
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len([t for t in self.trades if 'profit' in t]) if self.trades else 0

        # 交易次数
        total_trades = len([t for t in self.trades if 'profit' in t])

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'benchmark_return': benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'final_capital': portfolio_values[-1],
            'portfolio_values': self.portfolio_values,
            'benchmark_values': self.buy_hold_values,
            'dates': self.dates,
            'trades': self.trades
        }

    def plot_results(self, save_path='output/backtest_results.png', figsize=(14, 10)):
        """
        绘制回测结果

        参数:
            save_path: 保存路径
            figsize: 图形大小
        """
        if len(self.portfolio_values) == 0:
            print("⚠️ 无回测结果可绘图")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plt.style.use('seaborn-v0_8-darkgrid')

        dates = pd.to_datetime(self.dates)
        portfolio_values = np.array(self.portfolio_values)
        benchmark_values = np.array(self.buy_hold_values)

        # 1. 收益曲线对比
        ax1 = axes[0, 0]
        ax1.plot(dates, portfolio_values, label='Strategy', linewidth=2)
        ax1.plot(dates, benchmark_values, label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value (HK$)')
        ax1.legend()
        ax1.grid(True)

        # 2. 回撤图
        ax2 = axes[0, 1]
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max * 100
        ax2.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
        ax2.plot(dates, drawdowns, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)

        # 3. 收益分布
        ax3 = axes[1, 0]
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        ax3.hist(portfolio_returns, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)

        # 4. 关键指标
        ax4 = axes[1, 1]
        ax4.axis('off')

        metrics = self.calculate_metrics()
        metrics_text = f"""
        Backtest Summary
        ================

        Total Return:      {metrics['total_return']*100:.2f}%
        Annual Return:     {metrics['annual_return']*100:.2f}%
        Benchmark Return:   {metrics['benchmark_return']*100:.2f}%

        Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}
        Max Drawdown:      {metrics['max_drawdown']*100:.2f}%
        Win Rate:          {metrics['win_rate']*100:.2f}%

        Total Trades:      {metrics['total_trades']}
        Winning Trades:    {metrics['winning_trades']}
        Final Capital:     HK${metrics['final_capital']:,.2f}
        """

        ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 回测图表已保存到 {save_path}")


if __name__ == '__main__':
    # 简单测试
    print("测试回测引擎...")

    # 模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    prices = pd.Series(100 + np.cumsum(np.random.randn(len(dates)) * 0.5), index=dates)

    predictions = pd.DataFrame({
        'date': dates,
        'stock': ['TEST'] * len(dates),
        'probability': np.random.rand(len(dates)) * 0.3 + 0.35,
        'actual': (np.random.rand(len(dates)) > 0.5).astype(int)
    })

    # 运行回测
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(predictions, prices, confidence_threshold=0.55)

    print(f"\n回测结果:")
    print(f"总收益率: {results['total_return']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.3f}")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")

    # 绘图
    engine.plot_results()
