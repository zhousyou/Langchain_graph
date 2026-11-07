"""第 4 章 参考答案：LangGraph 基础"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    count: int
    logs: Annotated[list[str], operator.add]


def increment(state: CounterState) -> dict:
    return {"count": state["count"] + 1, "logs": ["increment"]}


def double(state: CounterState) -> dict:
    return {"count": state["count"] * 2, "logs": ["double"]}


def build_linear_graph():
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_node("double", double)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile()


class ReportState(TypedDict):
    topic: str
    reports: Annotated[list[str], operator.add]
    summary: str


def fetch_news(state: ReportState) -> dict:
    return {"reports": [f"新闻:{state['topic']}"]}


def fetch_weather(state: ReportState) -> dict:
    return {"reports": [f"天气:{state['topic']}"]}


def summarize(state: ReportState) -> dict:
    return {"summary": " | ".join(sorted(state["reports"]))}


def build_fanout_graph():
    builder = StateGraph(ReportState)
    builder.add_node("fetch_news", fetch_news)
    builder.add_node("fetch_weather", fetch_weather)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "fetch_news")
    builder.add_edge(START, "fetch_weather")
    builder.add_edge(["fetch_news", "fetch_weather"], "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_linear_graph()
    print(graph.invoke({"count": 1, "logs": []}))
    print(graph.get_graph().draw_mermaid())
    print(build_fanout_graph().invoke({"topic": "北京", "reports": []}))
