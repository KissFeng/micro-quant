"""
预测结果展示页
显示模型预测数据表格、筛选功能和可视化图表
"""

import streamlit as st
import pandas as pd
from services.prediction_service import PredictionService
from services.chart_service import ChartService
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="预测结果 - 量化交易AI助手", page_icon="📊")

# 初始化服务
@st.cache_resource
def get_services():
    return {
        'prediction': PredictionService(),
        'chart': ChartService()
    }

services = get_services()

# 页面标题
st.title("📊 模型预测结果")
st.markdown("查看LightGBM、CatBoost、LSTM三模型的预测数据")

# 加载数据
predictions = services['prediction'].load_predictions()

if predictions.empty:
    st.warning("暂无预测数据")
    st.info("请先运行 `python main.py --mode predict` 生成预测数据")
    st.stop()

# 筛选器
st.markdown("### 筛选条件")
col1, col2, col3 = st.columns(3)
with col1:
    min_prob = st.slider("最小概率阈值", 0.0, 1.0, 0.0, 0.05)
with col2:
    show_bullish = st.checkbox("仅显示看涨股票 (概率>50%)")
with col3:
    sort_by = st.selectbox("排序依据", ['probability', 'LightGBM', 'CatBoost', 'LSTM'])

# 过滤数据
filtered_df = predictions.copy()
if min_prob > 0:
    filtered_df = filtered_df[filtered_df['probability'] >= min_prob]
if show_bullish:
    filtered_df = filtered_df[filtered_df['probability'] > 0.5]

filtered_df = filtered_df.sort_values(sort_by, ascending=False)

# 数据表格
st.markdown(f"### 预测结果 ({len(filtered_df)}只股票)")
display_df = filtered_df[['name', 'stock', 'current_price', 'probability', 'LightGBM', 'CatBoost', 'LSTM']].copy()
display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
display_df['LightGBM'] = display_df['LightGBM'].apply(lambda x: f"{x:.2%}")
display_df['CatBoost'] = display_df['CatBoost'].apply(lambda x: f"{x:.2%}")
display_df['LSTM'] = display_df['LSTM'].apply(lambda x: f"{x:.2%}")
display_df.columns = ['股票名称', '代码', '当前价格', '综合概率', 'LightGBM', 'CatBoost', 'LSTM']
st.dataframe(display_df, use_container_width=True, hide_index=True)

# 导出CSV
col1, col2 = st.columns([1, 4])
with col1:
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 导出CSV",
        csv,
        "predictions_filtered.csv",
        "text/csv",
        use_container_width=True
    )

# 可视化图表
st.markdown("---")
st.markdown("### 数据可视化")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 概率分布")
    fig_dist = services['chart'].create_probability_distribution(predictions)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.markdown("#### 模型对比")
    fig_compare = services['chart'].create_model_comparison(predictions)
    st.plotly_chart(fig_compare, use_container_width=True)

# 统计信息
st.markdown("---")
st.markdown("### 统计信息")
stats = services['prediction'].get_statistics()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("平均综合概率", f"{stats['avg_prob']:.2%}")
with col2:
    st.metric("最高概率", f"{predictions['probability'].max():.2%}")
with col3:
    st.metric("最低概率", f"{predictions['probability'].min():.2%}")
