"""离线假模型。

真实大模型的输出不可预测、需要 API Key、还要花钱，不适合做自动化验证。
LangChain 自带 GenericFakeChatModel：你预先给它一串“剧本”回复，它就按顺序返回。
这里在它的基础上增加了两个能力，方便测试检查你的代码：

- 支持 bind_tools（真实模型绑定工具后才能发起工具调用）
- 记录每次收到的消息（received），以及绑定了哪些工具（bound_tools）
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


class FakeToolChatModel(GenericFakeChatModel):
    """按剧本回复、可绑定工具、会记录输入的假聊天模型。

    用法：
        model = FakeToolChatModel.from_responses([
            ai_tool_call("get_weather", {"city": "北京"}),  # 第 1 次调用：要求调用工具
            "北京今天晴。",                                   # 第 2 次调用：给出最终回答
        ])
    """

    received: list[list[Any]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @classmethod
    def from_responses(cls, responses: list[AIMessage | str]) -> "FakeToolChatModel":
        msgs = [AIMessage(content=r) if isinstance(r, str) else r for r in responses]
        return cls(messages=iter(msgs))

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FakeToolChatModel":
        self.bound_tools = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def ai_tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """构造一条“模型要求调用工具”的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}],
    )


def ai_text(text: str) -> AIMessage:
    """构造一条普通文本回复的 AIMessage。"""
    return AIMessage(content=text)
