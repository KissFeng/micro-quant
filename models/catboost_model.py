#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost 模型模块
"""

import os
import pickle
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score

from config import CATBOOST_PARAMS


class CatBoostModel:
    """CatBoost 分类模型"""

    def __init__(self, **params):
        self.model = None
        self.params = {**CATBOOST_PARAMS, **params}
        self.feature_columns = []
        self.model_type = 'catboost'
        self.metrics = {}

    def train(self, X_train, y_train, X_val=None, y_val=None, eval_set=None):
        """训练 CatBoost 模型"""
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train_np = X_train.values
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        else:
            X_train_np = X_train
            y_train_np = y_train

        train_pool = Pool(X_train_np, y_train_np)

        eval_pool = None
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            eval_pool = Pool(X_val, y_val)

        cat_params = {
            'iterations': self.params.get('iterations', 100),
            'learning_rate': self.params.get('learning_rate', 0.05),
            'depth': self.params.get('depth', 5),
            'random_seed': self.params.get('random_seed', 42),
            'verbose': self.params.get('verbose', False),
            'early_stopping_rounds': 20
        }

        if eval_pool is not None:
            self.model = CatBoostClassifier(**cat_params)
            self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        else:
            cat_params['early_stopping_rounds'] = None
            self.model = CatBoostClassifier(**cat_params)
            self.model.fit(train_pool)

        # 计算指标
        y_pred_proba = self.model.predict_proba(X_train_np)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        self.metrics = {
            'train_accuracy': accuracy_score(y_train_np, y_pred),
            'train_auc': roc_auc_score(y_train_np, y_pred_proba)
        }

        return self

    def predict(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def predict_classes(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            return None
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_columns, importance))

    def save(self, path='models/catboost_model.pkl'):
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
        print(f"CatBoost 模型已保存到 {path}")

    def load(self, path='models/catboost_model.pkl'):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data['params']
        self.metrics = model_data.get('metrics', {})
        self.model_type = model_data.get('model_type', 'catboost')
        print(f"CatBoost 模型已从 {path} 加载")
