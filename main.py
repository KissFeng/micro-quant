#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析入口脚本 - 训练 + 预测 + 回测 + 可视化 + Email推送
用法: python main.py --mode full
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

from config import STOCK_LIST, DEFAULT_HORIZON, TRAIN_START_DATE, TRAIN_END_DATE
from data_loader import load_multiple_stocks, load_index_data
from feature_engineering import combine_multiple_stocks, FeatureEngineer
from models import LightGBMModel, CatBoostModel, LSTMModel
from backtest import BacktestEngine
from visualizer import Visualizer
from evaluator import (
    evaluate_model, compare_models,
    generate_comparison_report, generate_markdown_report,
    save_results_to_json
)
from email_sender import EmailSender


def parse_args():
    parser = argparse.ArgumentParser(description='量化系统综合分析')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'predict', 'backtest', 'report'],
                       help='运行模式')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                       help='预测周期（天）')
    parser.add_argument('--send-email', action='store_true',
                       help='发送邮件报告')
    parser.add_argument('--no-train', action='store_true',
                       help='跳过训练，使用已有模型')
    return parser.parse_args()


def run_training(horizon, test_ratio=0.2):
    """
    运行训练流程

    参数:
        horizon: 预测周期
        test_ratio: 测试集比例

    返回:
        dict: {model_name: (model, metrics)}
    """
    print(f"\n{'#'*60}")
    print(f"# 阶段1: 模型训练")
    print(f"{'#'*60}\n")

    # 加载数据
    print("📥 加载训练数据...")
    stock_data = load_multiple_stocks(STOCK_LIST, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"✓ 成功加载 {len(stock_data)} 只股票")

    # 加载上证指数作为大盘特征
    index_data = load_index_data('sh000001', TRAIN_START_DATE, TRAIN_END_DATE)

    # 合并数据（按股票分别划分，避免时序泄露）
    X_train, y_train, X_test, y_test, stock_codes = combine_multiple_stocks(
        stock_data, horizon, test_ratio=test_ratio, index_data=index_data
    )
    if X_train.empty:
        print("⚠️ 训练数据为空")
        return {}

    print(f"✓ 训练样本: {len(X_train)}, 测试样本: {len(X_test)}, 特征: {X_train.shape[1]}")
    print(f"✓ 正样本比例: {y_train.mean():.2%}")

    results = {}

    # LightGBM
    print("\n1️⃣ 训练 LightGBM...")
    try:
        lgbm = LightGBMModel()
        lgbm.train(X_train, y_train, X_test, y_test)
        lgbm_metrics = evaluate_model(lgbm, X_test, y_test)
        lgbm.feature_columns = X_train.columns.tolist()
        lgbm.save('models/lgbm_model.pkl')
        results['LightGBM'] = {'model': lgbm, 'metrics': lgbm_metrics}
        print(f"   准确率: {lgbm_metrics['accuracy']:.4f}, AUC: {lgbm_metrics['auc']:.4f}")
    except Exception as e:
        print(f"   ⚠️ 失败: {e}")

    # CatBoost
    print("\n2️⃣ 训练 CatBoost...")
    try:
        catboost = CatBoostModel()
        catboost.train(X_train, y_train, X_test, y_test)
        catboost_metrics = evaluate_model(catboost, X_test, y_test)
        catboost.feature_columns = X_train.columns.tolist()
        catboost.save('models/catboost_model.pkl')
        results['CatBoost'] = {'model': catboost, 'metrics': catboost_metrics}
        print(f"   准确率: {catboost_metrics['accuracy']:.4f}, AUC: {catboost_metrics['auc']:.4f}")
    except Exception as e:
        print(f"   ⚠️ 失败: {e}")

    # LSTM
    print("\n3️⃣ 训练 LSTM...")
    try:
        lstm = LSTMModel(seq_length=10)
        lstm.train(X_train, y_train, X_test, y_test)
        lstm_metrics = evaluate_model(lstm, X_test, y_test)
        lstm.feature_columns = X_train.columns.tolist()
        lstm.save('models/lstm_model.keras')
        results['LSTM'] = {'model': lstm, 'metrics': lstm_metrics}
        print(f"   准确率: {lstm_metrics['accuracy']:.4f}, AUC: {lstm_metrics['auc']:.4f}")
    except Exception as e:
        print(f"   ⚠️ 失败: {e}")

    return results


def run_prediction(models, horizon, feature_columns=None):
    """
    运行预测流程

    参数:
        models: 模型字典
        horizon: 预测周期
        feature_columns: 特征列名列表（用于确保特征一致）

    返回:
        DataFrame: 预测结果
    """
    print(f"\n{'#'*60}")
    print(f"# 阶段2: 生成预测")
    print(f"{'#'*60}\n")

    # 加载最新数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print("📥 加载最新数据...")
    stock_data = load_multiple_stocks(STOCK_LIST, start_date, end_date)
    print(f"✓ 成功加载 {len(stock_data)} 只股票")

    # 加载指数数据
    index_data = load_index_data('sh000001', start_date, end_date)

    engineer = FeatureEngineer()
    predictions_list = []

    for code, df in stock_data.items():
        if df.empty or len(df) < 60:
            continue

        # 添加大盘特征
        if not index_data.empty:
            df = engineer.add_market_features(df, index_data)

        # 计算特征
        X, y, feat_cols = engineer.prepare_features(df, horizon)
        if X.empty:
            continue

        # 如果有指定的特征列，只使用这些列
        if feature_columns:
            # 确保只使用训练时的特征
            X = X[[c for c in feature_columns if c in X.columns]]

        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]

        # 各模型预测
        model_probs = {}
        for model_name, model_res in models.items():
            try:
                # model_res 可能是嵌套字典 {'model': xxx} 或直接是模型对象
                if isinstance(model_res, dict):
                    model_obj = model_res.get('model', model_res)
                else:
                    model_obj = model_res

                if model_name == 'LSTM':
                    # LSTM 需要特殊处理
                    X_scaled = model_obj.scaler.transform(X.values)
                    X_seq, _ = model_obj._create_sequences(X_scaled)
                    prob = model_obj.model.predict(X_seq, verbose=0).flatten()[-1]
                    model_probs[model_name] = float(prob)
                else:
                    # LGBM/CatBoost
                    if hasattr(model_obj, 'feature_columns') and model_obj.feature_columns:
                        X_pred = X[[c for c in model_obj.feature_columns if c in X.columns]]
                    else:
                        X_pred = X
                    X_values = X_pred.tail(1).values if hasattr(X_pred, 'values') else X_pred.tail(1)
                    prob = model_obj.predict(X_values)
                    if hasattr(prob, '__len__'):
                        prob = prob[0]
                    model_probs[model_name] = float(prob)
            except Exception as e:
                print(f"   Warning: {model_name} prediction failed: {e}")

        if model_probs:
            avg_prob = np.mean(list(model_probs.values()))
            predictions_list.append({
                'stock': code,
                'name': STOCK_LIST.get(code, code),
                'current_price': current_price,
                'date': current_date,
                'probability': avg_prob,
                **model_probs
            })

    if not predictions_list:
        print("⚠️ 无预测结果")
        return pd.DataFrame()

    predictions_df = pd.DataFrame(predictions_list)
    predictions_df = predictions_df.sort_values('probability', ascending=False)

    print(f"\n预测结果 (共 {len(predictions_df)} 只股票):")
    print("-" * 70)
    for _, row in predictions_df.iterrows():
        direction = "📈" if row['probability'] > 0.5 else "📉"
        print(f"{row['stock']:<12} {row['name']:<10} "
              f"{row['current_price']:<10.2f} {row['probability']:<10.2%} {direction}")

    predictions_df.to_csv('output/predictions.csv', index=False)
    print(f"\n✓ 预测结果已保存到 output/predictions.csv")

    return predictions_df


def run_backtest(model_results, horizon):
    """
    运行回测流程：对回测期内每个交易日生成预测，模拟真实交易

    参数:
        model_results: 模型结果字典 {name: {'model': ..., 'metrics': ...}}
        horizon: 预测周期

    返回:
        dict: 回测结果
    """
    print(f"\n{'#'*60}")
    print(f"# 阶段3: 回测评估")
    print(f"{'#'*60}\n")

    # 加载回测期数据（最近6个月）
    print("📥 加载回测数据...")
    test_end = datetime.now().strftime('%Y-%m-%d')
    test_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = load_multiple_stocks(STOCK_LIST, test_start, test_end)

    if not stock_data:
        print("⚠️ 数据不足，跳过回测")
        return {}

    # 加载指数数据
    index_data = load_index_data('sh000001', test_start, test_end)

    # 构建每只股票的价格字典
    prices_dict = {}
    for code, df in stock_data.items():
        if not df.empty:
            prices_dict[code] = df['Close']
    if not prices_dict:
        print("⚠️ 数据不足，跳过回测")
        return {}

    # 对每只股票生成回测预测
    engineer = FeatureEngineer()
    backtest_rows = []

    for code, df in stock_data.items():
        if df.empty or len(df) < 30:
            continue

        # 添加大盘特征
        if not index_data.empty:
            df = engineer.add_market_features(df, index_data)

        # 使用与训练完全相同的 prepare_features 流程
        X, y, feat_cols = engineer.prepare_features(df, horizon, dropna=False)
        if X.empty:
            continue

        # 计算真实标签（未来 horizon 天是否上涨），与 X 的 index 对齐
        close_for_label = df['Close'].reindex(X.index)
        future_return = close_for_label.shift(-horizon) / close_for_label - 1
        actual_labels = (future_return > 0).astype(int)
        actual_labels.iloc[-horizon:] = -1

        # 对每个模型预测
        for model_name, model_res in model_results.items():
            try:
                model_obj = model_res.get('model', model_res) if isinstance(model_res, dict) else model_res

                if model_name == 'LSTM':
                    # LSTM: 需要序列输入
                    seq_length = getattr(model_obj, 'seq_length', 10)
                    fc = model_obj.feature_columns if hasattr(model_obj, 'feature_columns') and model_obj.feature_columns else feat_cols
                    X_lstm = X[[c for c in fc if c in X.columns]].fillna(0)
                    X_scaled = model_obj.scaler.transform(X_lstm.values)
                    if len(X_scaled) < seq_length:
                        continue
                    X_seq, _ = model_obj._create_sequences(X_scaled)
                    probs = model_obj.model.predict(X_seq, verbose=0).flatten()
                    # LSTM 输出比输入少 seq_length 行
                    for i in range(len(probs)):
                        data_idx = i + seq_length
                        if data_idx >= len(X) or actual_labels.iloc[data_idx] == -1:
                            continue
                        backtest_rows.append({
                            'stock': code,
                            'date': X.index[data_idx],
                            'probability': float(probs[i]),
                            'actual': int(actual_labels.iloc[data_idx]),
                            'model': model_name,
                        })
                else:
                    # Tree models: 批量预测
                    fc = model_obj.feature_columns if hasattr(model_obj, 'feature_columns') and model_obj.feature_columns else feat_cols
                    X_tree = X[[c for c in fc if c in X.columns]].fillna(0)
                    probs = model_obj.predict(X_tree.values)
                    if hasattr(probs, '__len__') and len(probs) > 1:
                        probs = probs.flatten()
                    else:
                        probs = np.full(len(X_tree), float(probs))
                    for i in range(len(probs)):
                        if actual_labels.iloc[i] == -1:
                            continue
                        backtest_rows.append({
                            'stock': code,
                            'date': X.index[i],
                            'probability': float(probs[i]),
                            'actual': int(actual_labels.iloc[i]),
                            'model': model_name,
                        })
            except Exception as e:
                print(f"   ⚠️ {model_name} 回测预测失败({code}): {type(e).__name__}: {e}")

    if not backtest_rows:
        print("⚠️ 回测数据不足，跳过回测")
        return {}

    # 合并多模型预测：同一(股票,日期)取平均
    backtest_df = pd.DataFrame(backtest_rows)
    merged = backtest_df.groupby(['stock', 'date']).agg({
        'probability': 'mean',
        'actual': 'first',
    }).reset_index()
    print(f"✓ 生成 {len(merged)} 条回测预测记录（来自 {len(backtest_rows)} 条原始预测）")

    # 运行回测
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(merged, prices_dict, confidence_threshold=0.45)

    print(f"\n回测结果:")
    print(f"  总收益率:   {results['total_return']*100:.2f}%")
    print(f"  年化收益率: {results['annual_return']*100:.2f}%")
    print(f"  夏普比率:   {results['sharpe_ratio']:.3f}")
    print(f"  最大回撤:   {results['max_drawdown']*100:.2f}%")
    print(f"  胜率:       {results['win_rate']*100:.1f}%")
    print(f"  交易次数:   {results['total_trades']}")

    # 绘图
    engine.plot_results('output/backtest_results.png')

    return results


def generate_reports(model_results, predictions_df, backtest_results):
    """
    生成报告

    参数:
        model_results: 模型结果字典
        predictions_df: 预测结果
        backtest_results: 回测结果
    """
    print(f"\n{'#'*60}")
    print(f"# 阶段4: 生成报告")
    print(f"{'#'*60}\n")

    os.makedirs('output', exist_ok=True)

    # 模型对比报告
    if model_results:
        metrics_dict = {name: res['metrics'] for name, res in model_results.items()}
        comparison_df = compare_models(metrics_dict)

        generate_comparison_report(comparison_df)
        generate_markdown_report(comparison_df)
        save_results_to_json(metrics_dict, 'output/model_results.json')

        print(f"\n模型对比:")
        print(comparison_df.to_string(index=False))

    # 特征重要性（如果有 LightGBM）
    if 'LightGBM' in model_results:
        viz = Visualizer()
        importance = model_results['LightGBM']['model'].get_feature_importance()
        if importance:
            viz.plot_feature_importance(importance, top_n=15)


def send_email_reports(predictions_df, backtest_results):
    """
    发送邮件报告

    参数:
        predictions_df: 预测结果
        backtest_results: 回测结果
    """
    print(f"\n{'#'*60}")
    print(f"# 阶段5: 发送邮件")
    print(f"{'#'*60}\n")

    try:
        sender = EmailSender()

        # 预测报告
        if not predictions_df.empty:
            sender.send_prediction_report(predictions_df, 'Multi-Model Ensemble')

        # 回测报告
        if backtest_results:
            sender.send_backtest_report(backtest_results, 'output/backtest_results.png')

        print("✓ 邮件发送完成")

    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")


def main():
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"# 量化系统 - 综合分析")
    print(f"# 模式: {args.mode}")
    print(f"# 预测周期: {args.horizon}天")
    print(f"{'#'*60}")

    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    model_results = {}

    # 训练
    if args.mode in ['full', 'train']:
        if args.no_train:
            # 加载已有模型
            print("\n📂 加载已有模型...")
            for name, filename in [('LightGBM', 'models/lgbm_model.pkl'),
                                   ('CatBoost', 'models/catboost_model.pkl'),
                                   ('LSTM', 'models/lstm_model.keras')]:
                if os.path.exists(filename):
                    try:
                        model = {'LightGBM': LightGBMModel,
                                'CatBoost': CatBoostModel,
                                'LSTM': LSTMModel}[name]()
                        model.load(filename)
                        model_results[name] = {'model': model, 'metrics': {}}
                        print(f"   ✓ {name}")
                    except:
                        pass
        else:
            model_results = run_training(args.horizon)

    # 预测
    predictions_df = pd.DataFrame()
    if args.mode in ['full', 'predict']:
        if not model_results:
            # 加载模型
            for name, filename in [('LightGBM', 'models/lgbm_model.pkl'),
                                   ('CatBoost', 'models/catboost_model.pkl'),
                                   ('LSTM', 'models/lstm_model.keras')]:
                if os.path.exists(filename):
                    try:
                        model = {'LightGBM': LightGBMModel,
                                'CatBoost': CatBoostModel,
                                'LSTM': LSTMModel}[name]()
                        model.load(filename)
                        model_results[name] = {'model': model, 'metrics': {}}
                    except:
                        pass

        if model_results:
            # 从第一个非LSTM模型获取特征列
            feature_cols = None
            for name, res in model_results.items():
                if name != 'LSTM' and hasattr(res['model'], 'feature_columns') and res['model'].feature_columns:
                    feature_cols = res['model'].feature_columns
                    break
            predictions_df = run_prediction(model_results, args.horizon, feature_cols)

    # 回测
    backtest_results = {}
    if args.mode in ['full', 'backtest']:
        if model_results:
            backtest_results = run_backtest(model_results, args.horizon)

    # 报告
    if args.mode in ['full', 'report']:
        generate_reports(model_results, predictions_df, backtest_results)

    # 邮件
    if args.send_email:
        send_email_reports(predictions_df, backtest_results)

    print(f"\n{'#'*60}")
    print(f"# 完成!")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
