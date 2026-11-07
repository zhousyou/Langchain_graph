from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.conftest import load
from tutorial_kit import FakeToolChatModel

ch = load("ch01_models_messages")


def test_1_1_build_messages():
    msgs = ch.build_messages("你是翻译官", "hello")
    assert len(msgs) == 2, "应该返回两条消息"
    assert isinstance(msgs[0], SystemMessage), "第 1 条应该是 SystemMessage"
    assert isinstance(msgs[1], HumanMessage), "第 2 条应该是 HumanMessage"
    assert msgs[0].content == "你是翻译官"
    assert msgs[1].content == "hello"


def test_1_2_ask_returns_text():
    model = FakeToolChatModel.from_responses(["LangChain 是一个 LLM 应用开发框架。"])
    answer = ch.ask(model, "什么是 LangChain？")
    assert isinstance(answer, str), "ask 应该返回字符串（AIMessage.content），而不是 AIMessage 对象"
    assert answer == "LangChain 是一个 LLM 应用开发框架。"


def test_1_2_ask_sends_system_and_human():
    model = FakeToolChatModel.from_responses(["ok"])
    ch.ask(model, "你好")
    sent = model.received[0]
    assert isinstance(sent[0], SystemMessage) and sent[0].content == ch.DEFAULT_SYSTEM_PROMPT
    assert isinstance(sent[-1], HumanMessage) and sent[-1].content == "你好"


def test_1_3_build_conversation():
    msgs = ch.build_conversation([("system", "s"), ("human", "你好"), ("ai", "你好！")])
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]
    assert [m.content for m in msgs] == ["s", "你好", "你好！"]


def test_1_3_build_conversation_unknown_role():
    import pytest

    with pytest.raises(ValueError):
        ch.build_conversation([("robot", "beep")])


def test_1_4_stream_answer():
    model = FakeToolChatModel.from_responses(["LangGraph 用 图 来 编排 Agent"])
    chunks = ch.stream_answer(model, "介绍 LangGraph")
    assert isinstance(chunks, list) and len(chunks) > 1, "流式输出应该得到多个片段"
    assert all(isinstance(c, str) for c in chunks), "列表里的每一项应该是 chunk.content 字符串"
    assert "".join(chunks) == "LangGraph 用 图 来 编排 Agent"
