import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from tests.conftest import load
from tutorial_kit import FakeToolChatModel, ai_tool_call

tools_mod = load("ch08_final_agent.tools")
graph_mod = load("ch08_final_agent.graph")
app_mod = load("ch08_final_agent.app")


@pytest.fixture(autouse=True)
def _clear_notes():
    tools_mod.NOTES.clear()
    yield
    tools_mod.NOTES.clear()


def _tool(name):
    t = getattr(tools_mod, name, None)
    assert isinstance(t, BaseTool), f"tools.py 中需要用 @tool 定义 {name}"
    assert t.description, f"{name} 需要 docstring"
    return t


def _cfg(tid="t1"):
    return {"configurable": {"thread_id": tid}}


# ---------------- 工具 ----------------

def test_8_1_calculator():
    calc = _tool("calculator")
    assert calc.invoke({"expression": "(3 + 5) * 2"}) == "16"
    assert calc.invoke({"expression": "2 ** 10"}) == "1024"
    assert calc.invoke({"expression": "1 / 0"}).startswith("计算出错"), "异常应被捕获并返回“计算出错: ...”"
    assert calc.invoke({"expression": "__import__('os')"}).startswith("计算出错")


def test_8_2_search_knowledge():
    search = _tool("search_knowledge")
    result = search.invoke({"query": "checkpointer"})
    assert result.startswith("【Checkpointer】"), "应忽略大小写，并按【标题】内容 的格式返回"
    multi = search.invoke({"query": "LCEL interrupt"})
    assert "【LCEL】" in multi and "【interrupt】" in multi, "多个关键词任一命中即可"
    assert len(search.invoke({"query": "LangGraph"}).split("\n")) <= 3, "最多返回 3 条"
    assert search.invoke({"query": "量子力学"}) == "知识库中没有找到相关内容"


def test_8_3_and_8_4_notes():
    save, ls = _tool("save_note"), _tool("list_notes")
    assert ls.invoke({}) == "暂无笔记"
    assert save.invoke({"title": "LCEL", "content": "用 | 串联"}) == "已保存笔记《LCEL》"
    assert tools_mod.NOTES == {"LCEL": "用 | 串联"}
    save.invoke({"title": "State", "content": "共享数据"})
    assert ls.invoke({}) == "- LCEL\n- State"


def test_8_5_all_tools():
    assert [t.name for t in tools_mod.ALL_TOOLS] == ["calculator", "search_knowledge", "save_note", "list_notes"]


# ---------------- 图的组成部分 ----------------

def test_8_6_agent_node():
    model = FakeToolChatModel.from_responses(["hi"])
    node = graph_mod.make_agent_node(model, tools_mod.ALL_TOOLS, "SYS")
    out = node({"messages": [HumanMessage("你好")]})
    assert out["messages"][0].content == "hi"
    assert isinstance(model.received[0][0], SystemMessage) and model.received[0][0].content == "SYS"
    assert model.bound_tools == ["calculator", "search_knowledge", "save_note", "list_notes"]


def test_8_7_route_after_agent():
    route = graph_mod.route_after_agent
    assert route({"messages": [AIMessage("答案")]}) == END
    assert route({"messages": [ai_tool_call("search_knowledge", {"query": "x"})]}) == "tools"
    assert route({"messages": [ai_tool_call("save_note", {"title": "a", "content": "b"})]}) == "human_review"


# ---------------- 端到端场景 ----------------

def test_8_9_direct_answer():
    model = FakeToolChatModel.from_responses(["你好！我是学习助理。"])
    app = graph_mod.build_study_agent(model, InMemorySaver())
    result = app.invoke({"messages": [HumanMessage("你好")]}, _cfg())
    assert result["messages"][-1].content == "你好！我是学习助理。"


def test_8_9_search_then_answer():
    model = FakeToolChatModel.from_responses(
        [ai_tool_call("search_knowledge", {"query": "LCEL"}, "c1"), "LCEL 用 | 把组件串成链。"]
    )
    app = graph_mod.build_study_agent(model, InMemorySaver())
    result = app.invoke({"messages": [HumanMessage("什么是 LCEL？")]}, _cfg())
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and "【LCEL】" in tool_msgs[0].content
    assert result["messages"][-1].content == "LCEL 用 | 把组件串成链。"
    assert "__interrupt__" not in result, "普通工具不需要审批"


def _note_model():
    return FakeToolChatModel.from_responses(
        [ai_tool_call("save_note", {"title": "ReAct", "content": "思考-行动-观察"}, "n1"), "好的，已处理。"]
    )


def test_8_8_save_note_interrupts():
    app = graph_mod.build_study_agent(_note_model(), InMemorySaver())
    result = app.invoke({"messages": [HumanMessage("帮我记一条 ReAct 的笔记")]}, _cfg())
    assert "__interrupt__" in result, "调用敏感工具 save_note 前应该暂停等待审批"
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "approval"
    assert payload["tool_calls"] == [{"name": "save_note", "args": {"title": "ReAct", "content": "思考-行动-观察"}}]
    assert tools_mod.NOTES == {}, "审批之前不能真的保存笔记"


def test_8_8_approve():
    from langgraph.types import Command

    app = graph_mod.build_study_agent(_note_model(), InMemorySaver())
    app.invoke({"messages": [HumanMessage("记笔记")]}, _cfg())
    result = app.invoke(Command(resume="yes"), _cfg())
    assert tools_mod.NOTES == {"ReAct": "思考-行动-观察"}, "批准后应执行 save_note"
    assert result["messages"][-1].content == "好的，已处理。"


def test_8_8_reject():
    from langgraph.types import Command

    model = _note_model()
    app = graph_mod.build_study_agent(model, InMemorySaver())
    app.invoke({"messages": [HumanMessage("记笔记")]}, _cfg())
    result = app.invoke(Command(resume="no"), _cfg())
    assert tools_mod.NOTES == {}, "拒绝后不能保存笔记"
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "n1", "拒绝时要为每个 tool_call 回一条 ToolMessage"
    assert "拒绝" in tool_msgs[0].content
    assert result["messages"][-1].content == "好的，已处理。", "拒绝后应回到 agent，由模型向用户解释"


def test_8_9_memory_across_turns():
    model = FakeToolChatModel.from_responses(["记住了，你叫小红。", "你叫小红。"])
    app = graph_mod.build_study_agent(model, InMemorySaver())
    app.invoke({"messages": [HumanMessage("我叫小红")]}, _cfg("m"))
    app.invoke({"messages": [HumanMessage("我叫什么？")]}, _cfg("m"))
    contents = [m.content for m in model.received[1][1:]]
    assert contents == ["我叫小红", "记住了，你叫小红。", "我叫什么？"], "同一 thread 的第二轮应带上历史"


# ---------------- run_turn ----------------

def test_8_10_run_turn_with_approval():
    app = graph_mod.build_study_agent(_note_model(), InMemorySaver())
    seen = []

    def approve(payload):
        seen.append(payload)
        return True

    answer = app_mod.run_turn(app, "r1", "记笔记", approve)
    assert answer == "好的，已处理。"
    assert len(seen) == 1 and seen[0]["tool_calls"][0]["name"] == "save_note", "应把 interrupt 的内容交给 ask_approval"
    assert "ReAct" in tools_mod.NOTES


def test_8_10_run_turn_multiple_interrupts():
    model = FakeToolChatModel.from_responses(
        [
            ai_tool_call("save_note", {"title": "A", "content": "1"}, "a"),
            ai_tool_call("save_note", {"title": "B", "content": "2"}, "b"),
            "两条都处理完了。",
        ]
    )
    app = graph_mod.build_study_agent(model, InMemorySaver())
    decisions = iter([True, False])
    answer = app_mod.run_turn(app, "r2", "记两条笔记", lambda p: next(decisions))
    assert answer == "两条都处理完了。", "恢复后可能再次暂停，需要用 while 循环处理"
    assert tools_mod.NOTES == {"A": "1"}


def test_8_10_run_turn_no_interrupt():
    model = FakeToolChatModel.from_responses(["直接回答"])
    app = graph_mod.build_study_agent(model, InMemorySaver())

    def never(payload):
        raise AssertionError("没有敏感操作时不应请求审批")

    assert app_mod.run_turn(app, "r3", "你好", never) == "直接回答"
