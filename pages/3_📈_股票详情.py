"""
股票详情页
显示股票K线图、基本信息和预测详情
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from services.prediction_service import PredictionService
from services.chart_service import ChartService
from data_loader import load_stock_data
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="股票详情 - 量化交易AI助手", page_icon="📈")

# 初始化服务
@st.cache_resource
def get_services():
    return {
        'prediction': PredictionService(),
        'chart': ChartService()
    }

services = get_services()

# 页面标题
st.title("📈 股票详情分析")
st.markdown("查看股票K线图、预测详情和历史数据")

# 加载预测数据
predictions = services['prediction'].load_predictions()

if predictions.empty:
    st.warning("暂无预测数据")
    st.info("请先运行 `python main.py --mode predict` 生成预测数据")
    st.stop()

# 股票选择
st.markdown("### 选择股票")
col1, col2 = st.columns([2, 3])
with col1:
    # 构建选项
    stock_options = [f"{row['name']} ({row['stock']})" for _, row in predictions.iterrows()]
    selected_option = st.selectbox("选择股票", stock_options, label_visibility="collapsed")

# 提取股票代码
stock_code = selected_option.split('(')[1].rstrip(')')

# 获取预测数据
stock_data = services['prediction'].get_stock_by_code(stock_code)

if stock_data is None:
    st.error("未找到该股票的预测数据")
    st.stop()

# 显示基本信息
st.markdown("---")
st.markdown(f"### {stock_data['name']} ({stock_data['stock']})")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("当前价格", f"{stock_data['current_price']:.2f}元")
with col2:
    prob_color = "normal" if stock_data['probability'] > 0.5 else "inverse"
    st.metric("综合预测概率", f"{stock_data['probability']:.2%}")
with col3:
    st.metric("LightGBM", f"{stock_data['LightGBM']:.2%}")
with col4:
    st.metric("CatBoost", f"{stock_data['CatBoost']:.2%}")

col1, col2 = st.columns(2)
with col1:
    st.metric("LSTM", f"{stock_data['LSTM']:.2%}")
with col2:
    prediction = "看涨 📈" if stock_data['probability'] > 0.5 else "看跌 📉"
    st.metric("预测结果", prediction)

# K线图
st.markdown("---")
st.markdown("### K线图")

try:
    # 加载股票历史数据
    stock_df = load_stock_data(stock_code)

    if stock_df is not None and not stock_df.empty:
        # 只显示最近60天
        stock_df_recent = stock_df.tail(60)

        # 创建K线图
        fig = services['chart'].create_candlestick_chart(
            stock_df_recent,
            stock_data['name'],
            stock_data['probability']
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示数据范围
        st.caption(f"数据显示范围: {stock_df_recent.index[0].strftime('%Y-%m-%d')} 至 {stock_df_recent.index[-1].strftime('%Y-%m-%d')}")

    else:
        st.warning("未找到股票历史数据")
        st.info(f"股票数据文件应该在: data/{stock_code}_cache.pkl")

except Exception as e:
    st.error(f"加载K线数据失败: {str(e)}")
    with st.expander("查看错误详情"):
        st.exception(e)

# 模型预测详情
st.markdown("---")
st.markdown("### 模型预测详情")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### LightGBM")
    st.markdown(f"**预测概率**: {stock_data['LightGBM']:.2%}")
    st.progress(float(stock_data['LightGBM']))
    st.caption("梯度提升树模型，擅长处理结构化特征")

with col2:
    st.markdown("#### CatBoost")
    st.markdown(f"**预测概率**: {stock_data['CatBoost']:.2%}")
    st.progress(float(stock_data['CatBoost']))
    st.caption("处理类别特征的增强模型")

with col3:
    st.markdown("#### LSTM")
    st.markdown(f"**预测概率**: {stock_data['LSTM']:.2%}")
    st.progress(float(stock_data['LSTM']))
    st.caption("长短期记忆网络，捕捉时序依赖")

# 投资建议
st.markdown("---")
st.markdown("### 简要分析")

analysis_parts = []

if stock_data['probability'] > 0.6:
    analysis_parts.append("✅ **强烈看涨**: 综合概率超过60%，多个模型给出积极信号")
elif stock_data['probability'] > 0.5:
    analysis_parts.append("📈 **看涨**: 综合概率超过50%，但需谨慎观察")
elif stock_data['probability'] > 0.4:
    analysis_parts.append("📊 **观望**: 综合概率接近50%，建议继续观察")
else:
    analysis_parts.append("⚠️ **看跌**: 综合概率低于50%，建议谨慎操作")

# 模型一致性分析
model_probs = [stock_data['LightGBM'], stock_data['CatBoost'], stock_data['LSTM']]
if max(model_probs) - min(model_probs) < 0.1:
    analysis_parts.append("✅ **模型一致性高**: 三个模型预测结果接近，可信度较高")
elif max(model_probs) - min(model_probs) < 0.2:
    analysis_parts.append("⚠️ **模型存在分歧**: 不同模型预测差异中等，建议结合其他因素判断")
else:
    analysis_parts.append("⚠️ **模型分歧较大**: 不同模型预测差异显著，建议谨慎决策")

for analysis in analysis_parts:
    st.markdown(f"- {analysis}")

# 风险提示
st.warning("""
⚠️ **风险提示**
- 以上分析仅基于模型预测结果，不构成投资建议
- 股票投资存在风险，请结合市场环境、公司基本面综合判断
- 建议进行充分的风险管理和仓位控制
""")
