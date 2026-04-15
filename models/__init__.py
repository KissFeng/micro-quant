#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块 - 统一接口
"""

from .lgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .lstm_model import LSTMModel

__all__ = ['LightGBMModel', 'CatBoostModel', 'LSTMModel']
