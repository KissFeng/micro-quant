#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 模型模块
"""

import os
import pickle
import numpy as np
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from config import LIGHTGBM_PARAMS


class LightGBMModel:
    """LightGBM 分类模型"""

    def __init__(self, **params):
        self.model = None
        self.params = {**LIGHTGBM_PARAMS, **params}
        self.feature_columns = []
        self.scaler = None
        self.model_type = 'lgbm'
        self.metrics = {}

    def train(self, X_train, y_train, X_val=None, y_val=None, eval_set=None):
        """
        训练 LightGBM 模型

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            eval_set: 评估集元组列表

        返回:
            self
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train = X_train.values
            y_train = y_train.values if hasattr(y_train, 'values') else y_train

        # TimeSeriesSplit 交叉验证
        tscv = TimeSeriesSplit(n_splits=3, gap=5)

        # 构建数据集
        train_data = lgb.Dataset(X_train, label=y_train)

        # 回调函数
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=0)  # 静默
        ]

        # 训练
        if eval_set is None and X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            eval_set = [(X_val, y_val)]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 100),
            valid_sets=eval_set if eval_set else [train_data],
            callbacks=callbacks
        )

        # 计算训练集指标
        y_pred_proba = self.model.predict(X_train)
        y_pred = (y_pred_proba > 0.5).astype(int)

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }

        return self

    def predict(self, X):
        """
        预测概率

        参数:
            X: 特征数据

        返回:
            array: 预测概率
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_classes(self, X, threshold=0.5):
        """
        预测类别

        参数:
            X: 特征数据
            threshold: 阈值

        返回:
            array: 预测类别 (0 或 1)
        """
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            return None

        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_columns, importance))

    def save(self, path='models/lgbm_model.pkl'):
        """
        保存模型

        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else 'models', exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'params': self.params,
            'metrics': self.metrics,
            'model_type': self.model_type
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ LightGBM 模型已保存到 {path}")

    def load(self, path='models/lgbm_model.pkl'):
        """
        加载模型

        参数:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data['params']
        self.metrics = model_data.get('metrics', {})
        self.model_type = model_data.get('model_type', 'lgbm')

        print(f"✓ LightGBM 模型已从 {path} 加载")


import pandas as pd
