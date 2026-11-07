from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from tests.conftest import load
from tutorial_kit import FakeToolChatModel, ai_tool_call

ch = load("ch03_tools")


def _get_tool(name):
    t = getattr(ch, name, None)
    assert t is not None, f"没有找到工具 {name}，请先定义它"
    assert isinstance(t, BaseTool), f"{name} 应该用 @tool 装饰，变成一个 Tool 对象"
    return t


def test_3_1_multiply_tool():
    t = _get_tool("multiply")
    assert t.name == "multiply"
    assert t.description, "工具必须有 docstring（description），模型靠它理解工具用途"
    assert set(t.args) == {"a", "b"}
    assert t.invoke({"a": 3, "b": 4}) == 12


def test_3_2_weather_tool():
    t = _get_tool("get_weather")
    assert t.description, "工具必须有 docstring"
    assert set(t.args) == {"city"}
    assert t.invoke({"city": "北京"}) == "晴，25°C"
    assert t.invoke({"city": "火星"}) == "未知城市: 火星"


def test_3_3_tools_list():
    assert [t.name for t in ch.TOOLS] == ["multiply", "get_weather"]


def test_3_4_execute_tool_calls():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "上海"}, "id": "c1", "type": "tool_call"},
            {"name": "multiply", "args": {"a": 2, "b": 5}, "id": "c2", "type": "tool_call"},
        ],
    )
    results = ch.execute_tool_calls(ai, ch.TOOLS)
    assert len(results) == 2
    assert all(isinstance(r, ToolMessage) for r in results), "返回值应该是 ToolMessage 列表"
    assert results[0].tool_call_id == "c1" and results[0].content == "小雨，22°C"
    assert results[1].tool_call_id == "c2" and float(results[1].content) == 10


def test_3_4_execute_unknown_tool():
    ai = AIMessage(content="", tool_calls=[{"name": "fly", "args": {}, "id": "c9", "type": "tool_call"}])
    results = ch.execute_tool_calls(ai, ch.TOOLS)
    assert len(results) == 1 and isinstance(results[0], ToolMessage)
    assert results[0].tool_call_id == "c9"
    assert "fly" in results[0].content, "错误信息里应包含不存在的工具名"


def test_3_5_run_one_round_with_tool():
    model = FakeToolChatModel.from_responses(
        [ai_tool_call("get_weather", {"city": "广州"}, "c1"), "广州今天多云，30°C。"]
    )
    answer = ch.run_one_round(model, ch.TOOLS, "广州天气？")
    assert answer == "广州今天多云，30°C。"
    assert model.bound_tools == ["multiply", "get_weather"], "需要先 model.bind_tools(tools)"
    second_call = model.received[1]
    assert isinstance(second_call[0], HumanMessage), "第二次调用应带上完整历史：Human"
    assert isinstance(second_call[1], AIMessage) and second_call[1].tool_calls, "…然后是带 tool_calls 的 AIMessage"
    assert isinstance(second_call[2], ToolMessage) and second_call[2].content == "多云，30°C", "…最后是工具结果"


def test_3_5_run_one_round_without_tool():
    model = FakeToolChatModel.from_responses(["你好！我能帮你查天气和做乘法。"])
    answer = ch.run_one_round(model, ch.TOOLS, "你好")
    assert answer == "你好！我能帮你查天气和做乘法。"
    assert len(model.received) == 1, "模型没有要求调用工具时，不应该再调用第二次"
