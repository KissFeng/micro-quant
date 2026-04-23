#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型对比评估模块
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from models import LightGBMModel, CatBoostModel, LSTMModel
from backtest import BacktestEngine


def evaluate_model(model, X_test, y_test, prices_test=None, confidence_threshold=0.55):
    """
    评估单个模型

    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        prices_test: 测试期价格（用于回测）
        confidence_threshold: 置信度阈值

    返回:
        dict: 评估指标
    """
    # 预测
    if hasattr(model, 'model_type') and model.model_type == 'lstm':
        y_pred_proba = model.predict(X_test)
        # LSTM predict 输出长度 = len(X) - seq_length，需要对齐 y_test
        seq_length = getattr(model, 'seq_length', 20)
        if len(y_pred_proba) < len(y_test):
            y_test = y_test.iloc[-len(y_pred_proba):] if hasattr(y_test, 'iloc') else y_test[-len(y_pred_proba):]
    else:
        y_pred_proba = model.predict(X_test)

    y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算分类指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics


def compare_models(model_results):
    """
    对比多个模型的结果

    参数:
        model_results: dict {model_name: metrics_dict}

    返回:
        DataFrame: 对比表格
    """
    comparison = []

    for model_name, metrics in model_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison.append(row)

    df = pd.DataFrame(comparison)
    return df


def generate_comparison_report(results_df, save_path='output/model_comparison.png'):
    """
    生成模型对比报告图表

    参数:
        results_df: 对比结果 DataFrame
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'output', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-darkgrid')

    models = results_df['Model'].tolist()
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # 1. 分类指标对比
    ax1 = axes[0]
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25

    for i, model in enumerate(models):
        values = [results_df[results_df['Model'] == model][m].values[0] for m in metrics]
        ax1.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])

    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 2. 综合评分（雷达图风格的柱状图）
    ax2 = axes[1]

    # 综合评分 = accuracy*0.3 + auc*0.3 + f1*0.2 + precision*0.1 + recall*0.1
    for i, model in enumerate(models):
        row = results_df[results_df['Model'] == model].iloc[0]
        composite = (row['accuracy'] * 0.3 + row['auc'] * 0.3 +
                    row['f1'] * 0.2 + row['precision'] * 0.1 + row['recall'] * 0.1)
        ax2.bar(model, composite, color=colors[i % len(colors)], edgecolor='black')

    ax2.set_title('Composite Score\n(Acc*0.3 + AUC*0.3 + F1*0.2 + Prec*0.1 + Rec*0.1)')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 模型对比图表已保存到 {save_path}")


def save_results_to_json(results, filepath='output/model_results.json'):
    """
    保存结果到 JSON 文件

    参数:
        results: dict 或 DataFrame
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else 'output', exist_ok=True)

    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient='records')

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"✓ 结果已保存到 {filepath}")


def generate_markdown_report(results_df, filepath='output/model_report.md'):
    """
    生成 Markdown 格式的报告

    参数:
        results_df: 对比结果 DataFrame
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else 'output', exist_ok=True)

    md = f"""# 模型对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估指标对比

| 模型 | 准确率 | AUC | 精确率 | 召回率 | F1分数 |
|------|--------|-----|--------|--------|--------|
"""

    for _, row in results_df.iterrows():
        md += f"| {row['Model']} | {row['accuracy']:.4f} | {row['auc']:.4f} | "
        md += f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n"

    # 找出最佳模型
    results_df_sorted = results_df.sort_values('auc', ascending=False)
    best_model = results_df_sorted.iloc[0]['Model']

    md += f"""
## 结论

**推荐模型**: {best_model}

根据 AUC 指标（综合考虑分类能力和排序能力），{best_model} 表现最优。

"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"✓ Markdown 报告已保存到 {filepath}")


if __name__ == '__main__':
    # 简单测试
    print("测试多模型对比...")

    # 模拟结果
    results = {
        'LightGBM': {'accuracy': 0.58, 'auc': 0.62, 'precision': 0.57, 'recall': 0.61, 'f1': 0.59},
        'CatBoost': {'accuracy': 0.60, 'auc': 0.65, 'precision': 0.59, 'recall': 0.63, 'f1': 0.61},
        'LSTM': {'accuracy': 0.56, 'auc': 0.59, 'precision': 0.55, 'recall': 0.58, 'f1': 0.56},
    }

    results_df = compare_models(results)
    print(results_df)

    generate_comparison_report(results_df)
    generate_markdown_report(results_df)
