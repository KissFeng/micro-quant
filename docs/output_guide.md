# output/ 目录内容详解

量化系统运行后，所有输出文件均保存在 `output/` 目录下。共 6 个文件，涵盖模型评估、预测结果和回测分析。

---

## 文件清单

| 文件 | 类型 | 生成模块 | 说明 |
|------|------|----------|------|
| `model_report.md` | Markdown | evaluator.py | 模型对比报告（人类可读） |
| `model_results.json` | JSON | evaluator.py | 模型评估指标（机器可读） |
| `predictions.csv` | CSV | main.py | 各股票上涨概率预测 |
| `backtest_results.png` | PNG 图片 | backtest.py | 回测结果四宫格图表 |
| `feature_importance.png` | PNG 图片 | visualizer.py | 特征重要性柱状图 |
| `model_comparison.png` | PNG 图片 | evaluator.py | 模型对比双面板图表 |

---

## 1. model_report.md — 模型对比报告

Markdown 格式的文本报告，包含：

- **生成时间**：报告创建的时间戳
- **评估指标对比表**：三个模型（LightGBM / CatBoost / LSTM）的 5 项指标对比
- **结论**：按 AUC 排名推荐最优模型

### 指标含义

| 指标 | 英文名 | 含义 |
|------|--------|------|
| 准确率 | Accuracy | 整体预测正确的比例（涨猜对 + 跌猜对 / 总数） |
| AUC | Area Under ROC Curve | ROC 曲线下面积，衡量模型区分涨跌的排序能力。0.5 = 随机猜，>0.5 = 有预测力，**这是最核心的指标** |
| 精确率 | Precision | 预测为"涨"的股票中，实际真正涨的比例 → 买进去后赚钱的概率 |
| 召回率 | Recall | 实际涨的股票中，被模型成功预测为"涨"的比例 → 漏掉了多少机会 |
| F1 分数 | F1 Score | 精确率和召回率的调和平均，综合衡量两者平衡性 |

### 当前典型结果

| 模型 | 准确率 | AUC | 精确率 | 召回率 | F1 |
|------|--------|-----|--------|--------|-----|
| LightGBM | 0.511 | 0.554 | 0.578 | 0.406 | 0.477 |
| CatBoost | 0.522 | 0.522 | 0.538 | 0.924 | 0.680 |
| LSTM | 0.531 | 0.510 | 0.558 | 0.687 | 0.616 |

> LightGBM AUC 最高（0.554），为推荐模型。

---

## 2. model_results.json — 模型评估指标

与 `model_report.md` 相同数据的 JSON 格式版本，结构：

```json
{
  "LightGBM": {
    "accuracy": 0.511,
    "auc": 0.554,
    "precision": 0.578,
    "recall": 0.406,
    "f1": 0.477
  },
  "CatBoost": { ... },
  "LSTM": { ... }
}
```

**用途**：供其他程序/脚本读取，做进一步分析或自动化流程。

---

## 3. predictions.csv — 预测结果

每只股票的最新预测，按集成概率从高到低排序。

### 列说明

| 列名 | 含义 | 示例 |
|------|------|------|
| `stock` | 股票代码 | 600036.SH |
| `name` | 股票名称 | 招商银行 |
| `current_price` | 预测时的最新收盘价 | 41.087 |
| `date` | 预测日期 | 2025-12-31 |
| `probability` | **AUC 加权集成概率** | 0.730 |
| `LightGBM` | LightGBM 单模型预测的上涨概率 | 0.950 |
| `CatBoost` | CatBoost 单模型预测的上涨概率 | 0.497 |

### 关键字段详解

- **probability（集成概率）**：多个模型按 AUC 加权后的综合上涨概率
  - 计算方式：各模型概率 × 该模型 AUC 权重，再归一化
  - AUC 越高的模型权重越大
  - > 0.5 看涨，< 0.5 看跌
  - 回测策略买入 probability 排名 top3 的股票

- **弱模型过滤**：AUC < 0.52 的模型不参与集成预测（在代码中跳过）
- **反预测修复**：若某模型 AUC < 0.5，会自动将概率取反（1 - prob）

### 当前典型排名

| 排名 | 股票 | 集成概率 | 含义 |
|------|------|----------|------|
| 1 | 招商银行 | 0.730 | 强烈看涨 |
| 2 | 格力电器 | 0.694 | 看涨 |
| 3 | 长江电力 | 0.656 | 看涨 |
| ... | ... | ... | ... |
| 20 | 山西汾酒 | 0.313 | 看跌 |

---

## 4. backtest_results.png — 回测结果图

2×2 四宫格图表，由 `backtest.py` 的 `BacktestEngine.plot_results()` 生成。

### 四个子图

| 位置 | 标题 | 内容 |
|------|------|------|
| 左上 | 组合价值对比 | 策略净值曲线 vs 买入持有基准曲线，直观比较策略是否跑赢大盘 |
| 右上 | 回撤 | 红色填充区域，表示从历史最高点回落的幅度(%)，越大说明亏损越严重 |
| 左下 | 日收益率分布 | 日收益率直方图，红色虚线=0%，看收益是否偏正、是否有厚尾风险 |
| 右下 | 回测摘要 | 关键指标文字汇总（见下表） |

### 回测摘要指标

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| 总收益率 | 整个回测期间的总盈亏比例 | > 0 即盈利 |
| 年化收益率 | 按一年 252 个交易日折算的年化收益 | 对比银行理财/通胀 |
| 基准收益率 | 同期买入持有策略的收益 | 策略应超越此值 |
| 夏普比率 | 风险调整后收益（无风险利率按 2% 计算） | > 0.5 较好，> 1.0 优秀 |
| 最大回撤 | 从历史最高点到最低点的最大跌幅 | 越小越好，< 10% 较安全 |
| 胜率 | 盈利交易占总交易的比例 | > 50% 即可 |
| 总交易次数 | 回测期间完成的买卖轮次数 | 过少可能过拟合，过多则交易成本高 |
| 盈利交易次数 | 其中赚钱的交易数 | — |
| 最终资金 | 回测结束时的组合总价值 | 初始资金 100,000 |

---

## 5. feature_importance.png — 特征重要性图

LightGBM 模型的 Top 15 特征重要性水平柱状图，由 `visualizer.py` 的 `Visualizer.plot_feature_importance()` 生成。

- **Gain 重要性**：该特征在决策树分裂时带来的信息增益总和
- 值越大 → 该特征对预测贡献越大
- 帮助理解"模型靠什么做决策"

### 典型重要特征类别

| 类别 | 示例特征 | 含义 |
|------|----------|------|
| 大盘特征 | index_momentum, index_volatility | 上证指数的动量/波动率，反映市场整体环境 |
| 动量特征 | momentum_5d, momentum_10d | 过去 N 天的涨跌幅 |
| 波动率特征 | volatility_10d, volatility_20d | 过去 N 天的收益率标准差 |
| 成交量特征 | volume_ratio, volume_ma5 | 量比、成交量均线 |
| 技术指标 | RSI_14, MACD, BBI | 相对强弱指数、MACD、多空指标 |

---

## 6. model_comparison.png — 模型对比图

1×2 双面板图表，由 `evaluator.py` 的 `generate_comparison_report()` 生成。

### 两个子图

| 位置 | 标题 | 内容 |
|------|------|------|
| 左图 | 分类指标对比 | 分组柱状图，三个模型在准确率/AUC/精确率/召回率/F1 五个指标上的对比 |
| 右图 | 综合评分 | 加权综合评分柱状图，公式：准确率×0.3 + AUC×0.3 + F1×0.2 + 精确率×0.1 + 召回率×0.1 |

---

## 数据流关系

```
训练阶段 ──→ model_results.json    (指标数据，机器可读)
          ──→ model_report.md       (指标报告，人类可读)
          ──→ model_comparison.png   (指标对比图)
          ──→ feature_importance.png (特征重要性图)

预测阶段 ──→ predictions.csv        (每只股票的上涨概率)

回测阶段 ──→ backtest_results.png   (策略表现图)
```

---

## 配置说明

输出文件路径和格式由以下代码控制：

| 文件 | 生成函数 | 默认路径 |
|------|----------|----------|
| model_report.md | `evaluator.generate_markdown_report()` | output/model_report.md |
| model_results.json | `evaluator.save_results_to_json()` | output/model_results.json |
| model_comparison.png | `evaluator.generate_comparison_report()` | output/model_comparison.png |
| feature_importance.png | `visualizer.Visualizer.plot_feature_importance()` | output/feature_importance.png |
| predictions.csv | `main.run_prediction()` | output/predictions.csv |
| backtest_results.png | `backtest.BacktestEngine.plot_results()` | output/backtest_results.png |

所有图表已配置中文字体支持，字体优先级：Arial Unicode MS → SimHei → STHeiti → PingFang SC → Microsoft YaHei。
