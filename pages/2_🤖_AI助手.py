"""
AI助手页面 - 核心功能
提供基于预测数据的智能对话分析，支持投资组合建议
"""

import streamlit as st
from services.llm_service import LLMService
from services.prediction_service import PredictionService
from utils.context_builder import ContextBuilder
from utils.session_manager import SessionManager
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="AI助手 - 量化交易AI助手", page_icon="🤖")

# 初始化服务
@st.cache_resource
def get_services():
    try:
        return {
            'llm': LLMService(),
            'prediction': PredictionService(),
            'session': SessionManager()
        }
    except Exception as e:
        st.error(f"服务初始化失败: {str(e)}")
        return None

services = get_services()

if not services or not services['llm']:
    st.error("LLM服务未正确初始化，请检查API配置")
    st.info("请确保.env文件中配置了SILICONFLOW_API_KEY和SILICONFLOW_MODEL")
    st.stop()

# 页面标题
st.title("🤖 AI投资助手")
st.markdown("基于模型预测结果的智能问答系统，支持多轮对话和投资组合建议")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 对话配置")

    # 上下文注入选项
    inject_context = st.checkbox("注入预测数据上下文", value=True, help="将预测数据作为上下文提供给AI")
    context_mode = st.radio(
        "上下文模式",
        ["摘要模式", "完整模式"],
        horizontal=True,
        help="摘要模式节省Token，完整模式提供所有股票详细数据"
    )

    top_n = st.slider("Top N股票数量", 3, 20, 5, help="摘要模式下显示的推荐股票数量")

    st.markdown("---")

    # 投资组合建议按钮
    if st.button("💡 生成投资组合建议", use_container_width=True, type="primary"):
        st.session_state.generate_portfolio = True

    if st.button("🗑️ 清空聊天历史", use_container_width=True):
        services['session'].clear_history()
        if 'generate_portfolio' in st.session_state:
            del st.session_state.generate_portfolio
        st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 💡 使用提示
    - 开启上下文注入可获得更精准的建议
    - 可以询问具体股票的分析
    - 支持追问和多轮对话
    - 点击"生成投资组合建议"获取配置方案
    """)

# 检查是否需要生成投资组合建议
if 'generate_portfolio' in st.session_state and st.session_state.generate_portfolio:
    # 准备上下文
    predictions = services['prediction'].load_predictions()
    if not predictions.empty:
        context = ContextBuilder.build_summary_context(predictions, top_n)

        # AI响应
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("⏳ 正在生成投资组合建议...")

            try:
                full_response = services['llm'].generate_portfolio_recommendation(context)
                response_placeholder.markdown(full_response)

                # 保存到历史
                services['session'].add_message("assistant", full_response)
            except Exception as e:
                error_msg = f"⚠️ 生成失败: {str(e)}"
                response_placeholder.markdown(error_msg)
                services['session'].add_message("assistant", error_msg)

    del st.session_state.generate_portfolio

# 聊天历史
chat_history = services['session'].get_history()

# 显示历史消息
for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("输入您的问题... 例如：请分析招商银行的投资价值"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 保存用户消息
    services['session'].add_message("user", prompt)

    # 准备上下文
    context = ""
    if inject_context:
        predictions = services['prediction'].load_predictions()
        if not predictions.empty:
            if context_mode == "摘要模式":
                context = ContextBuilder.build_summary_context(predictions, top_n)
            else:
                context = ContextBuilder.build_full_context(predictions)

    # AI响应（流式输出）
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in services['llm'].chat_with_context(prompt, context, chat_history):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # 保存到历史
            services['session'].add_message("assistant", full_response)

        except Exception as e:
            error_msg = f"⚠️ API调用失败: {str(e)}"
            response_placeholder.markdown(error_msg)
            services['session'].add_message("assistant", error_msg)

# 欢迎消息（首次进入）
if not chat_history:
    with st.chat_message("assistant"):
        welcome_msg = """您好！我是量化交易AI助手，可以为您分析股票预测数据并提供投资建议。

**我能帮您：**
- 📊 分析具体股票的投资价值
- 💡 推荐投资组合配置方案
- 📈 解读模型预测结果
- ⚠️ 提醒投资风险

**快速开始：**
1. 在侧边栏点击"生成投资组合建议"
2. 或直接输入问题，例如："请分析招商银行的投资价值"

请问您想了解什么？"""
        st.markdown(welcome_msg)
