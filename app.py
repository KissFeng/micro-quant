"""
Streamlit量化交易AI助手 - 主应用入口
提供预测结果展示、AI对话分析、股票详情查询等功能
"""

import streamlit as st
from services.prediction_service import PredictionService
from services.llm_service import LLMService
from services.chart_service import ChartService
from utils.session_manager import SessionManager
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="量化交易AI助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化服务（单例模式）
@st.cache_resource
def init_services():
    """初始化所有服务实例"""
    try:
        llm_service = LLMService()
    except Exception as e:
        st.error(f"LLM服务初始化失败: {str(e)}")
        llm_service = None

    return {
        'prediction': PredictionService(),
        'llm': llm_service,
        'chart': ChartService(),
        'session': SessionManager()
    }

services = init_services()

# 侧边栏
with st.sidebar:
    st.title("📊 量化交易AI助手")
    st.markdown("---")
    st.markdown("""
    ### 功能导航
    - 📊 **预测结果**: 查看模型预测
    - 🤖 **AI助手**: 智能投资建议
    - 📈 **股票详情**: K线图分析

    ### 快速操作
    - 刷新预测数据
    - 导出分析报告
    """)

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 💡 使用说明
    1. 首次使用需运行模型预测生成数据
    2. AI助手基于预测数据提供分析
    3. 所有建议仅供参考，不构成投资建议
    """)

# 主页内容
st.title("🚀 欢迎使用量化交易AI助手")
st.markdown("基于LightGBM、CatBoost、LSTM三模型融合的智能量化分析平台")

# 快速概览
predictions = services['prediction'].load_predictions()

if not predictions.empty:
    # 统计指标
    stats = services['prediction'].get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("股票总数", stats['total'])
    with col2:
        st.metric("看涨股票", f"{stats['bullish']}只", f"{stats['bullish']/stats['total']:.1%}")
    with col3:
        st.metric("看跌股票", f"{stats['bearish']}只", f"{stats['bearish']/stats['total']:.1%}")
    with col4:
        st.metric("平均概率", f"{stats['avg_prob']:.2%}")

    st.markdown("---")

    # Top 5 推荐
    st.subheader("🏆 Top 5 推荐股票")
    top_5 = predictions.head(5)[['name', 'stock', 'current_price', 'probability', 'LightGBM', 'CatBoost', 'LSTM']]
    top_5_display = top_5.copy()
    top_5_display['probability'] = top_5_display['probability'].apply(lambda x: f"{x:.2%}")
    top_5_display['LightGBM'] = top_5_display['LightGBM'].apply(lambda x: f"{x:.2%}")
    top_5_display['CatBoost'] = top_5_display['CatBoost'].apply(lambda x: f"{x:.2%}")
    top_5_display['LSTM'] = top_5_display['LSTM'].apply(lambda x: f"{x:.2%}")
    top_5_display.columns = ['股票名称', '代码', '当前价格', '综合概率', 'LightGBM', 'CatBoost', 'LSTM']
    st.dataframe(top_5_display, use_container_width=True, hide_index=True)

    # 快速入口
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("pages/1_📊_预测结果.py", label="📊 查看完整预测结果", icon="📊")
    with col2:
        if services['llm']:
            st.page_link("pages/2_🤖_AI助手.py", label="🤖 AI投资助手", icon="🤖")
        else:
            st.warning("LLM服务未初始化")
    with col3:
        st.page_link("pages/3_📈_股票详情.py", label="📈 股票详情分析", icon="📈")

else:
    st.warning("⚠️ 暂无预测数据")
    st.info("请先运行 `python main.py --mode predict` 生成预测数据")

    with st.expander("📖 使用指南"):
        st.markdown("""
        ### 使用步骤：
        1. **训练模型**（首次使用）：
           ```bash
           python main.py --mode train
           ```

        2. **生成预测**：
           ```bash
           python main.py --mode predict
           ```

        3. **查看结果**：
           - 预测结果保存在 `output/predictions.csv`
           - 刷新本页面即可看到数据

        4. **AI分析**：
           - 点击"AI助手"页面，询问投资建议
           - 基于预测数据进行智能分析
        """)
