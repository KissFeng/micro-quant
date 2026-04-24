"""
LLM服务模块
负责与大语言模型API交互，支持流式输出和上下文注入
"""

import os
from openai import OpenAI
from typing import List, Dict, Generator


class LLMService:
    """大语言模型服务类，封装API调用逻辑"""

    def __init__(self):
        """初始化LLM客户端"""
        self.api_key = os.environ.get('SILICONFLOW_API_KEY')
        self.model = os.environ.get('SILICONFLOW_MODEL', 'deepseek-ai/DeepSeek-V3.2')
        self.base_url = 'https://api.siliconflow.cn/v1'

        if not self.api_key:
            raise ValueError("未找到SILICONFLOW_API_KEY环境变量，请检查.env配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat_with_context(
        self,
        user_message: str,
        context: str,
        history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        带上下文的对话（流式输出）

        参数:
            user_message: 用户问题
            context: 注入的预测数据上下文
            history: 聊天历史

        返回:
            生成器，yield文本片段
        """
        system_prompt = self._build_system_prompt(context)
        messages = self._build_messages(system_prompt, history or [], user_message)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2000
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"⚠️ API调用失败: {str(e)}"

    def generate_portfolio_recommendation(self, context: str) -> str:
        """
        生成投资组合建议

        参数:
            context: 预测数据上下文

        返回:
            投资组合建议文本
        """
        prompt = """基于当前预测数据，请推荐一个投资组合配置方案。

请按照以下结构回答：
1. **推荐股票池**（3-5只）：说明选择理由和每只股票的配置比例
2. **风险等级**：低/中/高，并说明判断依据
3. **预期收益**：基于模型预测概率的合理预期
4. **风险提示**：需要关注的风险点
5. **操作建议**：建仓时机、止盈止损策略

请结合模型预测概率、行业分散度、个股波动性等因素综合分析。"""

        full_response = ""
        for chunk in self.chat_with_context(prompt, context, []):
            full_response += chunk

        return full_response

    def _build_system_prompt(self, context: str) -> str:
        """构建系统提示词，注入预测数据"""
        return f"""你是一位专业的量化交易分析师，擅长基于机器学习模型进行股票预测分析。

## 当前预测数据
{context}

## 你的职责
1. 基于提供的预测数据，给出专业的投资建议
2. 解释不同模型的预测结果及其差异
3. 分析股票的技术面和趋势
4. 提醒风险和注意事项
5. 回答用户关于股票投资的问题

## 分析要点
- LightGBM: 梯度提升树模型，擅长处理结构化特征
- CatBoost: 处理类别特征的增强模型
- LSTM: 长短期记忆网络，捕捉时序依赖
- 综合概率: 多模型加权平均结果

## 注意事项
- 所有预测仅供参考，不构成投资建议
- 需要结合市场环境、公司基本面综合判断
- 强调风险管理和仓位控制"""

    def _build_messages(
        self,
        system_prompt: str,
        history: List[Dict],
        user_message: str
    ) -> List[Dict]:
        """构建完整消息列表"""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages
