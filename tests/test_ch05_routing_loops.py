import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END

from tests.conftest import load

ch = load("ch05_routing_loops")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我要申请退款", "billing"),
        ("这个月被重复扣费了", "billing"),
        ("能开发票吗", "billing"),
        ("登录时报错 500", "tech"),
        ("App 崩溃了", "tech"),
        ("发现一个 BUG", "tech"),
        ("你们周末上班吗", "general"),
    ],
)
def test_5_1_classify(text, expected):
    assert ch.classify({"text": text}) == {"category": expected}


def test_5_2_route():
    assert ch.route_by_category({"category": "billing"}) == "billing_agent"
    assert ch.route_by_category({"category": "tech"}) == "tech_agent"
    assert ch.route_by_category({"category": "general"}) == "general_agent"
    assert ch.route_by_category({"category": "whatever"}) == "general_agent"


def test_5_3_router_graph():
    graph = ch.build_router_graph()
    r = graph.invoke({"text": "我要退款"})
    assert r["category"] == "billing" and r["reply"].startswith("【财务】")
    r = graph.invoke({"text": "页面报错了"})
    assert r["category"] == "tech" and r["reply"].startswith("【技术】")
    r = graph.invoke({"text": "你好"})
    assert r["category"] == "general" and r["reply"].startswith("【客服】")
    visited = [list(c)[0] for c in graph.stream({"text": "我要退款"}, stream_mode="updates")]
    assert visited == ["classify", "billing_agent"], "应该只经过 classify 和被路由到的那一个节点"


def test_5_4_revise():
    assert ch.revise({"draft": "初稿", "revisions": 0, "max_revisions": 3}) == {
        "draft": "初稿[第1次修订]",
        "revisions": 1,
    }


def test_5_5_should_continue():
    assert ch.should_continue({"draft": "", "revisions": 1, "max_revisions": 3}) == "revise"
    assert ch.should_continue({"draft": "", "revisions": 3, "max_revisions": 3}) == END


def test_5_6_loop_graph():
    graph = ch.build_loop_graph()
    r = graph.invoke({"draft": "初稿", "revisions": 0, "max_revisions": 3})
    assert r["revisions"] == 3
    assert r["draft"] == "初稿[第1次修订][第2次修订][第3次修订]"


def test_5_6_recursion_limit():
    graph = ch.build_loop_graph()
    with pytest.raises(GraphRecursionError):
        graph.invoke({"draft": "", "revisions": 0, "max_revisions": 100}, {"recursion_limit": 10})
