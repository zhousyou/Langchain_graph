import typing

from tests.conftest import load

ch = load("ch04_graph_basics")


def test_4_1_counter_state():
    hints = typing.get_type_hints(ch.CounterState, include_extras=True)
    assert {"count", "logs"} <= set(hints), "CounterState 需要 count 和 logs 两个字段"
    assert typing.get_origin(hints["logs"]) is typing.Annotated, "logs 需要用 Annotated[list[str], operator.add]"


def test_4_2_increment_node():
    assert ch.increment({"count": 5, "logs": ["x"]}) == {"count": 6, "logs": ["increment"]}, (
        "节点只返回要更新的字段；logs 只返回新增条目，由 reducer 负责追加"
    )


def test_4_3_double_node():
    assert ch.double({"count": 5, "logs": []}) == {"count": 10, "logs": ["double"]}


def test_4_4_linear_graph():
    graph = ch.build_linear_graph()
    assert hasattr(graph, "invoke"), "别忘了 builder.compile()"
    assert graph.invoke({"count": 1, "logs": []}) == {"count": 4, "logs": ["increment", "double"]}
    assert graph.invoke({"count": 0, "logs": ["start"]}) == {"count": 2, "logs": ["start", "increment", "double"]}
    nodes = set(graph.get_graph().nodes)
    assert {"increment", "double"} <= nodes


def test_4_5_fanout_nodes():
    assert ch.fetch_news({"topic": "上海", "reports": []}) == {"reports": ["新闻:上海"]}
    assert ch.fetch_weather({"topic": "上海", "reports": []}) == {"reports": ["天气:上海"]}
    assert ch.summarize({"topic": "x", "reports": ["天气:x", "新闻:x"], "summary": ""}) == {"summary": "天气:x | 新闻:x"}


def test_4_6_fanout_graph():
    graph = ch.build_fanout_graph()
    result = graph.invoke({"topic": "北京", "reports": []})
    assert sorted(result["reports"]) == ["天气:北京", "新闻:北京"]
    assert result["summary"] == "天气:北京 | 新闻:北京"
    edges = graph.get_graph().edges
    from_start = {e.target for e in edges if e.source == "__start__"}
    assert from_start == {"fetch_news", "fetch_weather"}, "START 应该同时连到 fetch_news 和 fetch_weather（并行）"
    into_summary = {e.source for e in edges if e.target == "summarize"}
    assert into_summary == {"fetch_news", "fetch_weather"}, "两个分支都应汇合到 summarize"
