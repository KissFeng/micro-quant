#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 深度学习模型模块
"""

import os
import pickle
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from config import LSTM_PARAMS

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class LSTMModel:
    """LSTM 时序分类模型"""

    def __init__(self, seq_length=20, **params):
        """
        初始化 LSTM 模型

        参数:
            seq_length: 序列长度（时间步数）
            **params: 其他 LSTM 参数
        """
        self.seq_length = seq_length
        self.params = {**LSTM_PARAMS, **params}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_type = 'lstm'
        self.metrics = {}
        self._input_dim = None

    def build_model(self, input_shape):
        """
        构建 LSTM 模型架构

        参数:
            input_shape: (seq_length, n_features)

        返回:
            Keras Model
        """
        model = Sequential([
            LSTM(self.params.get('lstm_units_1', 64),
                 return_sequences=True,
                 input_shape=input_shape),
            Dropout(self.params.get('dropout', 0.2)),

            LSTM(self.params.get('lstm_units_2', 32),
                 return_sequences=False),
            Dropout(self.params.get('dropout', 0.2)),

            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        self.model = model
        self._input_dim = input_shape[1]
        return model

    def _create_sequences(self, X, y=None):
        """
        创建时间序列样本

        参数:
            X: 特征数据 (n_samples, n_features)
            y: 标签数据 (n_samples,)

        返回:
            X_seq: 序列数据 (n_samples - seq_length, seq_length, n_features)
            y_seq: 标签数据 (n_samples - seq_length,)
        """
        X = np.array(X)
        if len(X) <= self.seq_length:
            raise ValueError(f"数据长度 {len(X)} 必须大于序列长度 {self.seq_length}")

        X_seq = []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])

        X_seq = np.array(X_seq)

        if y is not None:
            y = np.array(y)
            # 标签对应序列最后一个元素
            y_seq = y[self.seq_length:]
            return X_seq, y_seq

        return X_seq, None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练 LSTM 模型

        参数:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        返回:
            self
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train_vals = X_train.values
            y_train_vals = y_train.values if hasattr(y_train, 'values') else y_train
        else:
            X_train_vals = X_train
            y_train_vals = y_train

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train_vals)

        # 创建序列
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train_vals)

        # 构建模型
        input_shape = (self.seq_length, X_seq.shape[2])
        self.build_model(input_shape)

        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        # 验证集
        validation_data = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values

            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)

        # 训练
        self.model.fit(
            X_seq, y_seq,
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks if validation_data else [callbacks[0]],
            verbose=1
        )

        # 计算训练集指标
        y_pred_proba = self.model.predict(X_seq, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        self.metrics = {
            'train_accuracy': accuracy_score(y_seq, y_pred),
            'train_auc': roc_auc_score(y_seq, y_pred_proba)
        }

        return self

    def predict(self, X):
        """
        预测概率

        参数:
            X: 特征数据 (n_samples, n_features)

        返回:
            array: 预测概率
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled)

        return self.model.predict(X_seq, verbose=0).flatten()

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
        """
        获取特征重要性（通过置换重要性近似）
        注意：LSTM 本身不支持特征重要性，这里返回 None
        """
        return None

    def save(self, path='models/lstm_model.keras'):
        """
        保存模型

        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else 'models', exist_ok=True)

        # 保存 Keras 模型
        self.model.save(path)

        # 保存其他数据
        scaler_path = path.replace('.keras', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'params': self.params,
                'metrics': self.metrics,
                'model_type': self.model_type,
                'seq_length': self.seq_length
            }, f)

        print(f"✓ LSTM 模型已保存到 {path}")

    def load(self, path='models/lstm_model.keras'):
        """
        加载模型

        参数:
            path: 模型路径
        """
        self.model = load_model(path)

        scaler_path = path.replace('.keras', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)

        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.params = data['params']
        self.metrics = data.get('metrics', {})
        self.model_type = data.get('model_type', 'lstm')
        self.seq_length = data.get('seq_length', 20)

        print(f"✓ LSTM 模型已从 {path} 加载")


import pandas as pd
