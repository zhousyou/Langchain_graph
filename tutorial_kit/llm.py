"""加载真实的大模型。

通过项目根目录下的 .env 文件配置（参考 .env.example）。
任何兼容 OpenAI 接口的服务都可以使用：OpenAI、DeepSeek、通义千问（DashScope）、Moonshot、
本地的 Ollama / vLLM 等。注意：第 6 章之后需要模型支持“工具调用（function calling）”。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def has_real_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_llm(temperature: float = 0.0):
    """返回一个真实的聊天模型（ChatOpenAI）。未配置 API Key 时给出清晰的报错。"""
    if not has_real_llm():
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写你的模型配置。\n"
            "（自动测试不需要 API Key；只有运行 demo 和最终的 Agent 应用时才需要。）"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=temperature,
    )
