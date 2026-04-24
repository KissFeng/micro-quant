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
├── app.py                # Streamlit Web应用入口
├── pages/                # Streamlit多页面
│   ├── 1_📊_预测结果.py  # 预测结果展示
│   ├── 2_🤖_AI助手.py    # AI对话分析
│   └── 3_📈_股票详情.py  # 股票详情页
├── services/             # 服务层
│   ├── llm_service.py    # LLM API集成
│   ├── prediction_service.py # 预测数据服务
│   └── chart_service.py  # 图表生成
├── utils/                # 工具函数
│   ├── context_builder.py # 上下文构建
│   └── session_manager.py # 会话管理
├── requirements.txt      # 依赖
├── .env                  # 环境变量配置
├── .env.example          # 环境变量示例
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
- **🆕 Web界面**: Streamlit可视化和交互分析
- **🆕 AI助手**: 基于大语言模型的智能投资建议

## 安装

```bash
pip install -r requirements.txt
```

## 配置

### 1. 环境变量配置（必需）

创建 `.env` 文件并配置以下内容：

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件
```

**必需配置：**
```bash
# LLM API 配置（用于AI助手功能）
SILICONFLOW_API_KEY=你的API密钥    # 访问 https://siliconflow.cn/ 获取
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2
```

**可选配置：**
```bash
# Email 配置（用于发送报告）
EMAIL_SENDER=your_email@qq.com
EMAIL_PASSWORD=your_smtp授权码
EMAIL_RECIPIENT=recipient@example.com
SMTP_SERVER=smtp.qq.com
SMTP_PORT=465
```

**获取 SiliconFlow API Key：**
1. 访问 https://siliconflow.cn/ 注册账号
2. 在控制台创建 API 密钥
3. 复制密钥到 `.env` 文件


### 2. 股票池配置

编辑 `config.py`：

```python
# 股票列表
STOCK_LIST = {
    '000001.SZ': '平安银行',
    '600519.SH': '贵州茅台',
    # ...
}
```

## 使用方法

### 1. 启动 Web 应用（推荐）

```bash
# 在 conda 环境中启动
conda activate ai_3.10
streamlit run app.py

# 或后台运行
streamlit run app.py --server.headless=true
```

访问 http://localhost:8501 使用 Web 界面

**Web 功能：**
- 📊 **预测结果**: 查看所有股票预测数据、筛选和可视化
- 🤖 **AI助手**: 基于预测数据的智能对话和投资建议
- 📈 **股票详情**: K线图、模型预测详情
- 💡 **投资组合**: 一键生成投资配置建议

### 2. 命令行使用

#### 综合分析（完整流程）

```bash
python main.py --mode full --horizon 5
```

流程：训练 → 预测 → 回测 → 可视化

#### 仅训练模型

```bash
python train.py --model all --horizon 5
```

#### 仅预测

```bash
python predict.py --model all
```

#### 发送邮件

```bash
python main.py --mode full --horizon 5 --send-email
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

**核心依赖：**
- pandas, numpy
- yfinance, akshare
- lightgbm, catboost
- tensorflow
- scikit-learn
- matplotlib, seaborn

**Web 应用依赖：**
- streamlit>=1.32.0
- openai>=1.0.0
- plotly>=5.18.0
- python-dotenv>=1.0.0

安装所有依赖：
```bash
pip install -r requirements.txt
```

## AI助手功能说明

### 功能特点
- **智能对话**: 基于预测数据回答投资相关问题
- **投资建议**: 分析股票投资价值和风险
- **组合推荐**: 自动生成投资组合配置方案
- **上下文理解**: 支持多轮对话和追问

### 使用示例
- "请分析招商银行的投资价值"
- "为什么LightGBM和LSTM预测结果差异这么大？"
- "当前市场环境下的投资建议是什么？"
- 点击"生成投资组合建议"获取配置方案

### 注意事项
- AI建议仅供参考，不构成实际投资建议
- 使用AI功能需要配置 SILICONFLOW_API_KEY
- 建议在摘要模式下使用以节省 Token 消耗
