# 资金量化系统

基于机器学习/深度学习的量化交易预测系统，支持多模型对比、回测评估和邮件推送。

## 项目结构

```
shixun1/
├── config.py              # 配置文件
├── data_loader.py         # 数据获取模块
├── feature_engineering.py  # 特征工程
├── models/               # 模型模块
│   ├── lgbm_model.py     # LightGBM 模型
│   ├── catboost_model.py # CatBoost 模型
│   └── lstm_model.py     # LSTM 深度学习模型
├── backtest.py           # 回测引擎
├── evaluator.py          # 多模型对比评估
├── visualizer.py         # 可视化模块
├── email_sender.py       # Email 推送
├── train.py              # 训练入口
├── predict.py            # 预测入口
├── main.py               # 综合分析入口
├── requirements.txt      # 依赖
└── output/               # 输出目录
    ├── predictions.csv   # 预测结果
    ├── backtest_results.png # 回测图表
    └── model_comparison.png # 模型对比图
```

## 功能特性

- **多模型支持**: LightGBM, CatBoost, LSTM
- **特征工程**: 50+ 技术指标特征
- **回测评估**: 收益率、夏普比率、最大回撤、胜率
- **可视化**: 收益曲线、特征重要性、模型对比
- **邮件推送**: 自动发送预测/回测报告

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config.py`：

```python
# 股票列表
STOCK_LIST = {
    '000001.SZ': '平安银行',
    '600519.SH': '贵州茅台',
    # ...
}

# Email 配置
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',
    'sender': 'your_email@qq.com',
    'password': 'your授权码',
    'recipient': 'recipient@example.com',
}
```

## 使用方法

### 1. 综合分析（推荐）

```bash
python main.py --mode full --horizon 5
```

流程：训练 → 预测 → 回测 → 可视化 → Email推送

### 2. 仅训练模型

```bash
python train.py --model all --horizon 5
```

### 3. 仅预测

```bash
python predict.py --model all
```

### 4. 发送邮件

```bash
python main.py --mode full --send-email
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型: lgbm/catboost/lstm/all | all |
| `--horizon` | 预测周期（天） | 5 |
| `--start-date` | 训练数据开始日期 | 2020-01-01 |
| `--end-date` | 训练数据结束日期 | 2024-12-31 |
| `--send-email` | 是否发送邮件 | False |

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/predictions.csv` | 预测结果 |
| `output/backtest_results.png` | 回测图表 |
| `output/model_comparison.png` | 模型对比图 |
| `output/model_results.json` | 模型评估指标 |
| `models/*.pkl` | 保存的模型文件 |

## 预测周期建议

| horizon | 适用场景 | 风险 |
|---------|----------|------|
| 1-5天 | 短期交易 | 高 |
| 5-20天 | 中期趋势 | 中 |
| 20+ | 长期投资 | 低 |

## 模型评估指标

- **准确率 (Accuracy)**: 预测正确率
- **AUC**: ROC曲线下面积，越大越好
- **夏普比率**: 风险调整后收益，越高越好
- **最大回撤**: 历史最大亏损，越小越好
- **胜率**: 盈利交易占比

## 注意事项

1. 模型仅供参考，不构成投资建议
2. 历史表现不代表未来收益
3. 请根据实际情况调整参数
4. 邮件发送需要正确配置 SMTP

## 依赖

- pandas, numpy
- yfinance
- lightgbm, catboost
- tensorflow
- scikit-learn
- matplotlib, seaborn
