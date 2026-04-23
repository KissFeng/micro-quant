#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 模型模块
"""

import os
import pickle
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

from config import LIGHTGBM_PARAMS


class LightGBMModel:
    """LightGBM 分类模型"""

    def __init__(self, **params):
        self.model = None
        self.params = {**LIGHTGBM_PARAMS, **params}
        self.feature_columns = []
        self.model_type = 'lgbm'
        self.metrics = {}

    def train(self, X_train, y_train, X_val=None, y_val=None, eval_set=None):
        """训练 LightGBM 模型"""
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train = X_train.values
            y_train = y_train.values if hasattr(y_train, 'values') else y_train

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.log_evaluation(period=0)
        ]

        # 构建 eval_set
        eval_data = None
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            eval_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        valid_sets = [eval_data] if eval_data else [train_data]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 100),
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        # 计算指标
        y_pred_proba = self.model.predict(X_train)
        y_pred = (y_pred_proba > 0.5).astype(int)

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }

        return self

    def predict(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_classes(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            return None
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_columns, importance))

    def save(self, path='models/lgbm_model.pkl'):
        """保存模型"""
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
        print(f"LightGBM 模型已保存到 {path}")

    def load(self, path='models/lgbm_model.pkl'):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data['params']
        self.metrics = model_data.get('metrics', {})
        self.model_type = model_data.get('model_type', 'lgbm')
        print(f"LightGBM 模型已从 {path} 加载")
