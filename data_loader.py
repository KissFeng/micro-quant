#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取模块 - 使用 yfinance 获取股票历史数据
"""

import os
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# 缓存目录
CACHE_DIR = 'data'
STOCK_DATA_CACHE_DAYS = 7


def get_cache_path(stock_code):
    """获取股票数据缓存文件路径"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f'{stock_code.replace(".", "_")}_cache.pkl')


def is_cache_valid(cache_path):
    """检查缓存是否有效"""
    if not os.path.exists(cache_path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return (datetime.now() - file_time).days < STOCK_DATA_CACHE_DAYS


def load_stock_data(stock_code, start_date=None, end_date=None, use_cache=True):
    """
    获取单只股票历史数据

    参数:
        stock_code: 股票代码（如 '000001.SZ'）
        start_date: 开始日期，默认为3年前
        end_date: 结束日期，默认为今天
        use_cache: 是否使用缓存

    返回:
        DataFrame: 包含 Open, High, Low, Close, Volume, Adj Close
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

    cache_path = get_cache_path(stock_code)

    # 尝试从缓存加载
    if use_cache and is_cache_valid(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            # 确保数据范围足够
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df) > 10:
                return df
        except Exception:
            pass

    # 下载数据
    try:
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"⚠️ {stock_code} 无数据")
            return pd.DataFrame()

        # 确保列名一致
        df.columns = [col.capitalize() for col in df.columns]

        # 保存到缓存
        if use_cache and len(df) > 10:
            df.to_pickle(cache_path)

        return df

    except Exception as e:
        print(f"⚠️ 获取 {stock_code} 数据失败: {e}")
        return pd.DataFrame()


def load_multiple_stocks(stock_list, start_date=None, end_date=None, max_workers=5):
    """
    并行获取多只股票数据

    参数:
        stock_list: 股票代码字典或列表
        start_date: 开始日期
        end_date: 结束日期
        max_workers: 最大并行数

    返回:
        dict: {stock_code: DataFrame}
    """
    if isinstance(stock_list, dict):
        codes = list(stock_list.keys())
    else:
        codes = stock_list

    results = {}

    def download_one(code):
        df = load_stock_data(code, start_date, end_date)
        return code, df

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, code): code for code in codes}

        for future in as_completed(futures):
            code, df = future.result()
            if not df.empty:
                results[code] = df
                print(f"✓ {code} 获取成功 ({len(df)} 条数据)")
            else:
                print(f"✗ {code} 获取失败")

    return results


def load_index_data(index_code='^HSI', start_date=None, end_date=None):
    """
    获取大盘指数数据（用于相对强弱计算）

    参数:
        index_code: 指数代码（默认恒生指数 ^HSI，可改为沪深300 ^HS300）
        start_date: 开始日期
        end_date: 结束日期

    返回:
        DataFrame: 指数数据
    """
    return load_stock_data(index_code, start_date, end_date)


if __name__ == '__main__':
    # 测试代码
    print("测试数据获取模块...")

    # 测试单只股票
    df = load_stock_data('000001.SZ', '2024-01-01', '2024-12-31')
    print(f"\n000001.SZ 数据: {len(df)} 条")
    print(df.tail(3))

    # 测试多只股票
    stocks = {'000001.SZ': '平安银行', '600519.SH': '贵州茅台'}
    results = load_multiple_stocks(stocks, '2024-01-01', '2024-12-31')
    print(f"\n成功获取 {len(results)} 只股票数据")
