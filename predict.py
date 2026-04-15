#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测入口脚本
用法: python predict.py --model all --date 2026-04-15
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCK_LIST, DEFAULT_HORIZON
from data_loader import load_multiple_stocks
from feature_engineering import FeatureEngineer
from backtest import BacktestEngine
from visualizer import Visualizer
from evaluator import compare_models, generate_comparison_report, save_results_to_json

# 模型目录
MODEL_DIR = 'models'


def parse_args():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lgbm', 'catboost', 'lstm', 'all'],
                       help='要使用的模型')
    parser.add_argument('--date', type=str, default=None,
                       help='预测日期（默认今天）')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                       help='预测周期（天）')
    parser.add_argument('--send-email', action='store_true',
                       help='发送邮件报告')
    return parser.parse_args()


def load_models(model_names):
    """
    加载模型

    参数:
        model_names: 模型名称列表

    返回:
        dict: {model_name: model}
    """
    from models import LightGBMModel, CatBoostModel, LSTMModel

    models = {}
    model_classes = {
        'lgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'lstm': LSTMModel
    }

    model_files = {
        'lgbm': f'{MODEL_DIR}/lgbm_model.pkl',
        'catboost': f'{MODEL_DIR}/catboost_model.pkl',
        'lstm': f'{MODEL_DIR}/lstm_model.keras'
    }

    for name in model_names:
        if name in model_classes:
            model_file = model_files[name]
            if os.path.exists(model_file):
                try:
                    model = model_classes[name]()
                    model.load(model_file)
                    models[name.capitalize()] = model
                    print(f"✓ {name} 模型已加载")
                except Exception as e:
                    print(f"⚠️ {name} 模型加载失败: {e}")
            else:
                print(f"⚠️ {name} 模型文件不存在: {model_file}")

    return models


def make_predictions(models, stock_data, horizon):
    """
    使用多个模型进行预测

    参数:
        models: 模型字典
        stock_data: 股票数据
        horizon: 预测周期

    返回:
        DataFrame: 预测结果
    """
    engineer = FeatureEngineer()
    all_predictions = []

    for code, df in stock_data.items():
        if df.empty or len(df) < 60:
            continue

        # 计算特征
        df_features = engineer.calculate_technical_features(df)
        if df_features.empty:
            continue

        # 准备特征数据
        X, y, feature_cols = engineer.prepare_features(df, horizon, dropna=True)
        if X.empty:
            continue

        # 获取最新价格
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]

        # 各模型预测
        model_preds = {}
        for model_name, model in models.items():
            try:
                if model_name == 'Lstm' and hasattr(model, 'seq_length'):
                    # LSTM 需要特殊处理
                    X_scaled = model.scaler.transform(X.values)
                    X_seq, _ = model._create_sequences(X_scaled)
                    prob = model.model.predict(X_seq, verbose=0).flatten()[-1]
                else:
                    prob = model.predict(X.tail(1).values if hasattr(X, 'values') else X.tail(1))[0]

                model_preds[model_name] = prob
            except Exception as e:
                print(f"⚠️ {model_name} 预测失败: {e}")

        if model_preds:
            # 简单平均
            avg_prob = np.mean(list(model_preds.values()))

            all_predictions.append({
                'stock': code,
                'name': STOCK_LIST.get(code, code),
                'current_price': current_price,
                'date': current_date,
                'avg_probability': avg_prob,
                **model_preds
            })

    return pd.DataFrame(all_predictions)


def main():
    args = parse_args()

    if args.date:
        predict_date = args.date
    else:
        predict_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'#'*60}")
    print(f"# 量化系统 - 预测")
    print(f"# 日期: {predict_date}")
    print(f"# 模型: {args.model}")
    print(f"# 预测周期: {args.horizon}天")
    print(f"{'#'*60}\n")

    # 加载模型
    model_names = ['lgbm', 'catboost', 'lstm'] if args.model == 'all' else [args.model]
    models = load_models(model_names)

    if not models:
        print("⚠️ 没有可用的模型，请先运行训练: python train.py")
        return

    # 加载最新数据
    print("📥 加载最新数据...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = load_multiple_stocks(STOCK_LIST, start_date, end_date)
    print(f"✓ 成功加载 {len(stock_data)} 只股票")

    if len(stock_data) == 0:
        print("⚠️ 没有数据可用")
        return

    # 预测
    print("\n📊 生成预测...")
    predictions_df = make_predictions(models, stock_data, args.horizon)

    if predictions_df.empty:
        print("⚠️ 预测失败")
        return

    # 按概率排序
    predictions_df = predictions_df.sort_values('avg_probability', ascending=False)

    print(f"\n{'='*60}")
    print("预测结果")
    print(f"{'='*60}\n")

    # 打印预测结果
    print(f"{'代码':<12} {'名称':<10} {'当前价':<10} {'预测概率':<10} {'方向':<8}")
    print("-" * 60)

    for _, row in predictions_df.iterrows():
        direction = "📈" if row['avg_probability'] > 0.5 else "📉"
        print(f"{row['stock']:<12} {row['name']:<10} "
              f"{row['current_price']:<10.2f} {row['avg_probability']:<10.2%} {direction}")

    # 保存预测结果
    predictions_df.to_csv('output/predictions.csv', index=False)
    print(f"\n✓ 预测结果已保存到 output/predictions.csv")

    # 可视化
    viz = Visualizer()
    try:
        viz.plot_model_comparison({
            name: {'accuracy': 0, 'auc': row['avg_probability'], 'f1': 0}
            for name, row in predictions_df.iterrows()
        })
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

    # 发送邮件
    if args.send_email:
        try:
            from email_sender import EmailSender
            sender = EmailSender()
            sender.send_prediction_report(predictions_df, model_name=args.model.upper())
        except Exception as e:
            print(f"⚠️ 邮件发送失败: {e}")

    print("\n✓ 预测完成!")


if __name__ == '__main__':
    main()
