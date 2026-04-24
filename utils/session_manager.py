"""
会话管理器模块
负责管理聊天历史和用户会话状态
"""

import streamlit as st
from typing import List, Dict


class SessionManager:
    """会话管理器类，管理聊天历史"""

    def __init__(self):
        """初始化会话状态"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def get_history(self) -> List[Dict]:
        """
        获取聊天历史

        返回:
            聊天历史列表
        """
        return st.session_state.chat_history

    def add_message(self, role: str, content: str):
        """
        添加消息到历史

        参数:
            role: 消息角色 ('user' 或 'assistant')
            content: 消息内容
        """
        st.session_state.chat_history.append({
            "role": role,
            "content": content
        })

    def clear_history(self):
        """清空聊天历史"""
        st.session_state.chat_history = []
