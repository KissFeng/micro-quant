#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练入口脚本
用法: python train.py --model all --horizon 5 --start-date 2020-01-01 --end-date 2024-12-31
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCK_LIST, DEFAULT_HORIZON, TRAIN_START_DATE, TRAIN_END_DATE
from data_loader import load_multiple_stocks
from feature_engineering import combine_multiple_stocks
from models import LightGBMModel, CatBoostModel, LSTMModel
from evaluator import evaluate_model, compare_models, save_results_to_json

# 模型保存目录
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='训练量化模型')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lgbm', 'catboost', 'lstm', 'all'],
                       help='要训练的模型')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                       help='预测周期（天）')
    parser.add_argument('--start-date', type=str, default=TRAIN_START_DATE,
                       help='训练数据开始日期')
    parser.add_argument('--end-date', type=str, default=TRAIN_END_DATE,
                       help='训练数据结束日期')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='测试集比例')
    return parser.parse_args()


def train_models(stock_data, horizon, test_ratio=0.2):
    """
    训练所有模型

    参数:
        stock_data: 股票数据字典
        horizon: 预测周期
        test_ratio: 测试集比例

    返回:
        dict: {model_name: (model, metrics, feature_columns)}
    """
    print(f"\n{'='*60}")
    print(f"开始训练 - 预测周期: {horizon}天")
    print(f"{'='*60}\n")

    # 合并多只股票数据
    print("📊 准备训练数据...")
    X, y, stock_codes = combine_multiple_stocks(stock_data, horizon)

    if X.empty:
        print("⚠️ 训练数据为空")
        return {}

    print(f"✓ 样本数量: {len(X)}, 特征数量: {X.shape[1]}")
    print(f"✓ 正样本比例: {y.mean():.2%}")

    # 划分训练集和测试集（时序划分）
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"✓ 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    results = {}

    # ========== 1. 训练 LightGBM ==========
    print(f"\n{'='*40}")
    print("1. 训练 LightGBM 模型...")
    print(f"{'='*40}")

    try:
        lgbm = LightGBMModel()
        lgbm.train(X_train, y_train, X_test, y_test)
        lgbm_metrics = evaluate_model(lgbm, X_test, y_test)
        print(f"LightGBM 准确率: {lgbm_metrics['accuracy']:.4f}, AUC: {lgbm_metrics['auc']:.4f}")

        # 保存模型
        lgbm.feature_columns = X.columns.tolist()
        lgbm.save(f'{MODEL_DIR}/lgbm_model.pkl')

        results['LightGBM'] = {
            'model': lgbm,
            'metrics': lgbm_metrics,
            'feature_columns': X.columns.tolist()
        }
    except Exception as e:
        print(f"⚠️ LightGBM 训练失败: {e}")

    # ========== 2. 训练 CatBoost ==========
    print(f"\n{'='*40}")
    print("2. 训练 CatBoost 模型...")
    print(f"{'='*40}")

    try:
        catboost = CatBoostModel()
        catboost.train(X_train, y_train, X_test, y_test)
        catboost_metrics = evaluate_model(catboost, X_test, y_test)
        print(f"CatBoost 准确率: {catboost_metrics['accuracy']:.4f}, AUC: {catboost_metrics['auc']:.4f}")

        # 保存模型
        catboost.feature_columns = X.columns.tolist()
        catboost.save(f'{MODEL_DIR}/catboost_model.pkl')

        results['CatBoost'] = {
            'model': catboost,
            'metrics': catboost_metrics,
            'feature_columns': X.columns.tolist()
        }
    except Exception as e:
        print(f"⚠️ CatBoost 训练失败: {e}")

    # ========== 3. 训练 LSTM ==========
    print(f"\n{'='*40}")
    print("3. 训练 LSTM 模型...")
    print(f"{'='*40}")

    try:
        lstm = LSTMModel(seq_length=20)
        lstm.train(X_train, y_train, X_test, y_test)
        lstm_metrics = evaluate_model(lstm, X_test, y_test)
        print(f"LSTM 准确率: {lstm_metrics['accuracy']:.4f}, AUC: {lstm_metrics['auc']:.4f}")

        # 保存模型
        lstm.feature_columns = X.columns.tolist()
        lstm.save(f'{MODEL_DIR}/lstm_model.keras')

        results['LSTM'] = {
            'model': lstm,
            'metrics': lstm_metrics,
            'feature_columns': X.columns.tolist()
        }
    except Exception as e:
        print(f"⚠️ LSTM 训练失败: {e}")

    return results


def main():
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"# 量化系统 - 模型训练")
    print(f"# 模型: {args.model}")
    print(f"# 预测周期: {args.horizon}天")
    print(f"# 训练区间: {args.start_date} ~ {args.end_date}")
    print(f"{'#'*60}\n")

    # 加载数据
    print("📥 加载股票数据...")
    stock_data = load_multiple_stocks(STOCK_LIST, args.start_date, args.end_date)
    print(f"✓ 成功加载 {len(stock_data)} 只股票")

    if len(stock_data) == 0:
        print("⚠️ 没有数据，请检查网络连接")
        return

    # 训练模型
    results = train_models(stock_data, args.horizon, args.test_ratio)

    if not results:
        print("⚠️ 没有模型训练成功")
        return

    # 保存评估结果
    metrics_dict = {name: res['metrics'] for name, res in results.items()}
    save_results_to_json(metrics_dict, 'output/training_results.json')

    # 打印对比结果
    print(f"\n{'='*60}")
    print("训练完成 - 模型对比")
    print(f"{'='*60}")

    comparison_df = compare_models(metrics_dict)
    print(comparison_df.to_string(index=False))

    # 保存对比结果
    comparison_df.to_csv('output/model_comparison.csv', index=False)
    print(f"\n✓ 结果已保存到 output/model_comparison.csv")


if __name__ == '__main__':
    main()
