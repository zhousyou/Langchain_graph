from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from tests.conftest import load
from tutorial_kit import FakeToolChatModel, ai_tool_call

ch = load("ch06_react_agent")


def _multi_step_model():
    return FakeToolChatModel.from_responses(
        [
            ai_tool_call("get_stock_price", {"symbol": "NVDA"}, "c1"),
            ai_tool_call("calculate", {"expression": "180.0 * 10"}, "c2"),
            "10 股 NVDA 价值 1800 美元。",
        ]
    )


def test_6_1_call_model_node():
    model = FakeToolChatModel.from_responses(["你好"])
    node = ch.make_call_model_node(model, ch.TOOLS, "你是测试助手")
    assert callable(node), "make_call_model_node 应该返回一个函数"
    out = node({"messages": [HumanMessage("hi")]})
    assert isinstance(out, dict) and len(out["messages"]) == 1, "节点应返回 {'messages': [模型回复]}"
    assert isinstance(out["messages"][0], AIMessage)
    sent = model.received[0]
    assert isinstance(sent[0], SystemMessage) and sent[0].content == "你是测试助手", "第一条应是 SystemMessage"
    assert sent[1].content == "hi"
    assert model.bound_tools == ["get_stock_price", "calculate"], "需要 model.bind_tools(tools)"


def test_6_2_should_continue():
    assert ch.should_continue({"messages": [ai_tool_call("calculate", {"expression": "1+1"})]}) == "tools"
    assert ch.should_continue({"messages": [AIMessage("完成")]}) == END


def test_6_3_agent_graph_multi_step():
    model = _multi_step_model()
    graph = ch.build_agent_graph(model, ch.TOOLS)
    result = graph.invoke({"messages": [HumanMessage("10 股 NVDA 值多少？")]})
    types = [type(m).__name__ for m in result["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"], (
        f"消息序列不符合预期：{types}。Agent 应该循环：模型 -> 工具 -> 模型 -> 工具 -> 模型"
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == "NVDA 当前价格 180.0 美元"
    assert tool_msgs[1].content == "1800.0"
    assert result["messages"][-1].content == "10 股 NVDA 价值 1800 美元。"
    assert not any(isinstance(m, SystemMessage) for m in result["messages"]), "system 提示词不应存入状态"


def test_6_3_agent_graph_no_tool():
    model = FakeToolChatModel.from_responses(["你好，我是股票助手。"])
    graph = ch.build_agent_graph(model, ch.TOOLS)
    result = graph.invoke({"messages": [HumanMessage("你好")]})
    assert len(result["messages"]) == 2


def test_6_4_prebuilt_agent():
    model = _multi_step_model()
    agent = ch.build_prebuilt_agent(model, ch.TOOLS)
    result = agent.invoke({"messages": [HumanMessage("10 股 NVDA 值多少？")]})
    assert result["messages"][-1].content == "10 股 NVDA 价值 1800 美元。"
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == 2
    assert isinstance(model.received[0][0], SystemMessage), "别忘了把 system_prompt 传给 create_agent"
