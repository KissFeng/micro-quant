#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
数据获取模块 - 使用腾讯财经接口获取 A 股数据
"""

import os
import time
import json
import random
import warnings
from datetime import datetime, timedelta

import requests
import pandas as pd

warnings.filterwarnings('ignore')

# 缓存目录
CACHE_DIR = 'data'
STOCK_DATA_CACHE_DAYS = 3

# 请求配置
REQUEST_DELAY = 0.3
MAX_RETRIES = 3
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Referer': 'https://stockapp.finance.qq.com/'
}


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


def format_tencent_code(stock_code):
    """
    将标准代码转换为腾讯财经格式

    参数:
        stock_code: 如 '000001.SZ', '600519.SH', 'sh000001'

    返回:
        tuple: (腾讯代码, 交易所后缀)
    """
    code_lower = stock_code.lower()

    # 已经是腾讯格式（sh/sz/bj前缀），直接返回
    if code_lower.startswith(('sh', 'sz', 'bj')):
        prefix = code_lower[:2]
        num = code_lower[2:]
        suffix = prefix.upper()
        return code_lower, suffix

    # 去除 .SZ/.SH/.sz/.sh 后缀
    code = stock_code.replace('.SZ', '').replace('.SH', '').replace('.sz', '').replace('.sh', '').upper()

    # 判断交易所
    if code.startswith('6'):
        # 上海
        return f'sh{code}', 'SH'
    elif code.startswith(('000', '001', '002', '003')):
        # 深圳
        return f'sz{code}', 'SZ'
    elif code.startswith('5'):
        # 基金
        return f'sh{code}', 'SH'
    elif code.startswith('8') or code.startswith('4'):
        # 北交所
        return f'bj{code}', 'BJ'
    else:
        # 默认按深圳处理
        return f'sz{code}', 'SZ'


def get_stock_data_tencent(stock_code, period_days=365):
    """
    通过腾讯财经接口获取股票数据

    参数:
        stock_code: 股票代码
        period_days: 获取天数

    返回:
        DataFrame: 包含 Open, High, Low, Close, Volume
    """
    tencent_code, suffix = format_tencent_code(stock_code)

    # 腾讯财经 API URL
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={tencent_code},day,,,{period_days},qfq&r=0.{random.randint(100000, 999999)}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        text_data = response.text

        # 解析 JSON
        # kline_dayqfq= 是13个字符
        if text_data.startswith("kline_dayqfq="):
            json_str = text_data[13:]
            data = json.loads(json_str)
        else:
            # 尝试直接解析
            data = json.loads(text_data)

        # 检查数据
        if data.get('code') != 0:
            print(f"⚠️ {stock_code} API错误: {data.get('msg', 'unknown')}")
            return pd.DataFrame()

        if 'data' not in data or tencent_code not in data['data']:
            print(f"⚠️ {stock_code} 无数据")
            return pd.DataFrame()

        # 提取 K 线数据（可能是 day 或 qfqday）
        stock_data = data['data'][tencent_code]
        kline_data = stock_data.get('day') or stock_data.get('qfqday')

        if not kline_data or len(kline_data) == 0:
            print(f"⚠️ {stock_code} K线为空")
            return pd.DataFrame()

        # 解析数据
        # 格式: [日期, 开盘, 收盘, 最低, 最高, 成交量]
        parsed_data = []
        for item in kline_data:
            if len(item) >= 6:
                try:
                    parsed_data.append({
                        'Date': pd.to_datetime(item[0]),
                        'Open': float(item[1]),
                        'Close': float(item[2]),
                        'Low': float(item[3]),
                        'High': float(item[4]),
                        'Volume': int(float(item[5]))
                    })
                except (ValueError, IndexError):
                    continue

        if parsed_data:
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            df = df.sort_index()
            return df

        return pd.DataFrame()

    except json.JSONDecodeError as e:
        print(f"✗ {stock_code} JSON解析失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ {stock_code} 获取失败: {e}")
        return pd.DataFrame()


def load_stock_data(stock_code, start_date=None, end_date=None, use_cache=True):
    """
    获取单只股票历史数据

    参数:
        stock_code: 股票代码（如 '000001.SZ'）
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存

    返回:
        DataFrame: 包含 Open, High, Low, Close, Volume
    """
    cache_path = get_cache_path(stock_code)

    # 尝试从缓存加载
    if use_cache and is_cache_valid(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            if len(df) > 10:
                print(f"📦 {stock_code} 从缓存加载 ({len(df)} 条)")
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                if len(df) > 10:
                    return df
        except Exception:
            pass

    # 获取数据
    for retry in range(MAX_RETRIES):
        df = get_stock_data_tencent(stock_code, period_days=730)

        if not df.empty:
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            if len(df) > 10:
                df.to_pickle(cache_path)
                print(f"✓ {stock_code} 获取成功 ({len(df)} 条)")
            return df

        if retry < MAX_RETRIES - 1:
            print(f"⚠️ {stock_code} 重试 {retry+1}/{MAX_RETRIES}...")
            time.sleep(2 ** retry)
        else:
            print(f"✗ {stock_code} 获取失败")

    return pd.DataFrame()


def load_multiple_stocks(stock_list, start_date=None, end_date=None, max_workers=2):
    """
    获取多只股票数据

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
    total = len(codes)

    print(f"📥 开始获取 {total} 只股票数据...")

    for i, code in enumerate(codes, 1):
        print(f"[{i}/{total}] ", end="", flush=True)
        df = load_stock_data(code, start_date, end_date)
        if not df.empty:
            results[code] = df
        else:
            print(f"✗ {code} 失败")

        # 每5只休息一下
        if i % 5 == 0:
            time.sleep(1)

    print(f"\n✓ 成功获取 {len(results)}/{total} 只股票")
    return results


def load_index_data(index_code='sh000001', start_date=None, end_date=None):
    """
    获取指数数据

    参数:
        index_code: 指数代码 (sh000001=上证, sz399001=深证, sh000300=沪深300)
    """
    cache_path = get_cache_path(f'idx_{index_code}')

    # 尝试从缓存加载
    if is_cache_valid(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            if len(df) > 10:
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                if len(df) > 10:
                    return df
        except Exception:
            pass

    # 直接获取
    for retry in range(MAX_RETRIES):
        df = get_stock_data_tencent(index_code, period_days=730)
        if not df.empty:
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            if len(df) > 10:
                df.to_pickle(cache_path)
                return df
        if retry < MAX_RETRIES - 1:
            print(f"⚠️ {index_code} 重试 {retry+1}/{MAX_RETRIES}...")
            time.sleep(2 ** retry)
        else:
            print(f"✗ {index_code} 获取失败")

    return pd.DataFrame()


if __name__ == '__main__':
    print("测试腾讯财经数据接口...")

    # 测试单只股票
    df = load_stock_data('000001.SZ', '2024-01-01', '2024-12-31')
    print(f"\n000001.SZ 数据: {len(df)} 条")
    if not df.empty:
        print(df.tail(3))

    # 测试上证指数
    df2 = load_index_data('sh000001', '2024-01-01', '2024-12-31')
    print(f"\n上证指数数据: {len(df2)} 条")
    if not df2.empty:
        print(df2.tail(3))
