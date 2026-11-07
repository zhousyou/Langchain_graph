"""教程公共工具包：提供真实模型加载、离线假模型等辅助功能。

你不需要修改这个包里的代码，但可以阅读它来理解测试是如何工作的。
"""

from tutorial_kit.fake import FakeToolChatModel, ai_text, ai_tool_call
from tutorial_kit.llm import get_llm, has_real_llm

__all__ = ["FakeToolChatModel", "ai_text", "ai_tool_call", "get_llm", "has_real_llm"]
