#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 深度学习模型模块
"""

import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from config import LSTM_PARAMS

np.random.seed(42)
tf.random.set_seed(42)


class LSTMModel:
    """LSTM 时序分类模型"""

    def __init__(self, seq_length=20, **params):
        self.seq_length = seq_length
        self.params = {**LSTM_PARAMS, **params}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_type = 'lstm'
        self.metrics = {}
        self._input_dim = None

    def build_model(self, input_shape):
        """构建 LSTM 模型"""
        from tensorflow.keras.regularizers import l2
        model = Sequential([
            LSTM(self.params.get('lstm_units_1', 64), return_sequences=True,
                 input_shape=input_shape, kernel_regularizer=l2(1e-4)),
            Dropout(self.params.get('dropout', 0.2)),
            LSTM(self.params.get('lstm_units_2', 32), return_sequences=False,
                 kernel_regularizer=l2(1e-4)),
            Dropout(self.params.get('dropout', 0.2)),
            Dense(8, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        self._input_dim = input_shape[1]
        return model

    def _create_sequences(self, X, y=None):
        """创建时间序列样本"""
        X = np.array(X)
        if len(X) <= self.seq_length:
            raise ValueError(f"数据长度 {len(X)} 必须大于序列长度 {self.seq_length}")

        X_seq = []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
        X_seq = np.array(X_seq)

        if y is not None:
            y = np.array(y)
            if len(y) > len(X_seq):
                y = y[-len(X_seq):]
            elif len(y) < len(X_seq):
                X_seq = X_seq[-len(y):]
            y_seq = y
            return X_seq, y_seq

        return X_seq, None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练 LSTM 模型"""
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train_vals = X_train.values
            y_train_vals = y_train.values if hasattr(y_train, 'values') else y_train
        else:
            X_train_vals = X_train
            y_train_vals = y_train

        X_train_scaled = self.scaler.fit_transform(X_train_vals)
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train_vals)

        input_shape = (self.seq_length, X_seq.shape[2])
        self.build_model(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)

        # 计算类别权重以处理类别不平衡
        unique, counts = np.unique(y_seq, return_counts=True)
        class_weight = None
        if len(unique) == 2:
            total = len(y_seq)
            class_weight = {int(c): total / (2 * cnt) for c, cnt in zip(unique, counts)}

        self.model.fit(
            X_seq, y_seq,
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks if validation_data else [callbacks[0]],
            class_weight=class_weight,
            verbose=1
        )

        # 计算指标
        y_pred_proba = self.model.predict(X_seq, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        self.metrics = {
            'train_accuracy': accuracy_score(y_seq, y_pred),
            'train_auc': roc_auc_score(y_seq, y_pred_proba)
        }

        return self

    def predict(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled)
        return self.model.predict(X_seq, verbose=0).flatten()

    def predict_classes(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def save(self, path='models/lstm_model.keras'):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else 'models', exist_ok=True)
        self.model.save(path)
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
        print(f"LSTM 模型已保存到 {path}")

    def load(self, path='models/lstm_model.keras'):
        """加载模型"""
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
        print(f"LSTM 模型已从 {path} 加载")
