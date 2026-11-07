from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.conftest import load
from tutorial_kit import FakeToolChatModel

ch = load("ch07_memory_hitl")


def test_7_2_make_config():
    assert ch.make_config("abc") == {"configurable": {"thread_id": "abc"}}


def test_7_1_and_7_3_memory_within_thread():
    model = FakeToolChatModel.from_responses(["你好小明！", "你叫小明。"])
    app = ch.build_memory_chatbot(model, InMemorySaver())
    assert ch.chat(app, "u1", "我叫小明") == "你好小明！"
    assert ch.chat(app, "u1", "我叫什么？") == "你叫小明。"
    second_input = model.received[1]
    assert [m.content for m in second_input] == ["我叫小明", "你好小明！", "我叫什么？"], (
        "第二轮调用模型时，应该自动带上同一会话的历史消息（checkpointer 的作用）"
    )


def test_7_3_threads_are_isolated():
    model = FakeToolChatModel.from_responses(["a", "b"])
    app = ch.build_memory_chatbot(model, InMemorySaver())
    ch.chat(app, "u1", "我是用户1")
    ch.chat(app, "u2", "我是用户2")
    assert [m.content for m in model.received[1]] == ["我是用户2"], "不同 thread_id 的记忆应互相隔离"


def test_7_4_get_history():
    model = FakeToolChatModel.from_responses(["收到"])
    app = ch.build_memory_chatbot(model, InMemorySaver())
    assert ch.get_history(app, "nobody") == []
    ch.chat(app, "u1", "记住数字 42")
    history = ch.get_history(app, "u1")
    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert history[0].content == "记住数字 42"


def _start_transfer(thread_id="t1"):
    app = ch.build_transfer_graph(InMemorySaver())
    result = app.invoke({"to": "张三", "amount": 100, "status": "pending"}, ch.make_config(thread_id))
    return app, result


def test_7_5_interrupt_pauses_graph():
    app, result = _start_transfer()
    assert "__interrupt__" in result, "approval 节点应调用 interrupt() 暂停图"
    question = result["__interrupt__"][0].value["question"]
    assert "张三" in question and "100" in question
    assert result["status"] == "pending", "暂停时还没有审批，状态应保持 pending"
    assert app.get_state(ch.make_config("t1")).next == ("approval",), "图应停在 approval 节点"


def test_7_6_and_7_7_approve():
    app, _ = _start_transfer()
    final = ch.resume(app, "t1", "yes")
    assert final["status"] == "done", "批准后应执行 execute 节点"


def test_7_6_and_7_7_reject():
    app, _ = _start_transfer()
    final = ch.resume(app, "t1", "no")
    assert final["status"] == "rejected", "拒绝后应直接结束，不执行转账"
