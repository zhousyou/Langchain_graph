"""第 4 章 练习：LangGraph 基础 —— State / Node / Edge / Reducer

讲义：docs/04_LangGraph基础.md
验证：uv run python check.py 4

目标：
  1. 用 TypedDict 定义图的“状态”（State）——所有节点共享的数据
  2. 节点（Node）就是普通函数：接收当前 state，返回“要更新的字段”组成的字典
  3. 用边（Edge）连接节点，START / END 是特殊的起点和终点
  4. 理解 Reducer：默认是“覆盖”，用 Annotated[list, operator.add] 可以改为“追加”
  5. 让两个节点并行执行，再汇合（fan-out / fan-in）

本章不需要大模型 —— LangGraph 本身是一个通用的“有状态流程编排”框架。
"""

import operator  # noqa: F401
from typing import Annotated, TypedDict  # noqa: F401

from langgraph.graph import END, START, StateGraph  # noqa: F401


# 练习 4.1：定义状态 CounterState（继承 TypedDict），包含两个字段：
#   count: int                                   —— 默认 reducer：新值直接覆盖旧值
#   logs:  Annotated[list[str], operator.add]    —— 自定义 reducer：新列表追加到旧列表后面
class CounterState(TypedDict):
    # TODO: 在这里定义字段
    pass


def increment(state: CounterState) -> dict:
    """练习 4.2：节点 increment —— count 加 1，并记录日志 "increment"。

    返回：{"count": 原值 + 1, "logs": ["increment"]}
    注意：只返回要更新的字段；logs 返回的是“本次新增的条目”，reducer 会负责追加。
    """
    # TODO: 在这里实现
    raise NotImplementedError


def double(state: CounterState) -> dict:
    """练习 4.3：节点 double —— count 乘 2，并记录日志 "double"。"""
    # TODO: 在这里实现
    raise NotImplementedError


def build_linear_graph():
    """练习 4.4：构建并编译一个线性图：START -> increment -> double -> END

    步骤：
      1. builder = StateGraph(CounterState)
      2. builder.add_node("increment", increment)，同理添加 double
      3. builder.add_edge(START, "increment") ... 依次连边，最后连到 END
      4. return builder.compile()

    用法：graph.invoke({"count": 1, "logs": []})
          -> {"count": 4, "logs": ["increment", "double"]}
    """
    # TODO: 在这里实现
    raise NotImplementedError


# ---------------- 并行分支（fan-out / fan-in） ----------------

class ReportState(TypedDict):
    topic: str
    reports: Annotated[list[str], operator.add]  # 两个并行节点都会往这里写，所以必须用追加型 reducer
    summary: str


def fetch_news(state: ReportState) -> dict:
    """练习 4.5：返回 {"reports": [f"新闻:{topic}"]}"""
    # TODO: 在这里实现
    raise NotImplementedError


def fetch_weather(state: ReportState) -> dict:
    """练习 4.5：返回 {"reports": [f"天气:{topic}"]}"""
    # TODO: 在这里实现
    raise NotImplementedError


def summarize(state: ReportState) -> dict:
    """练习 4.5：把 reports 排序后用 " | " 连接，写入 summary 字段。"""
    # TODO: 在这里实现
    raise NotImplementedError


def build_fanout_graph():
    """练习 4.6：构建并行图

            ┌─> fetch_news ──┐
      START ┤                ├─> summarize -> END
            └─> fetch_weather┘

    提示：
      - 从 START 分别连一条边到 fetch_news 和 fetch_weather，它们会在同一步并行执行
      - builder.add_edge(["fetch_news", "fetch_weather"], "summarize")
        表示“两个节点都完成后”才执行 summarize
    思考：如果 reports 不用 operator.add 会发生什么？（试试看，LangGraph 会报错 InvalidUpdateError）
    """
    # TODO: 在这里实现
    raise NotImplementedError


if __name__ == "__main__":
    # 运行：uv run python -m exercises.ch04_graph_basics
    graph = build_linear_graph()
    print(graph.invoke({"count": 1, "logs": []}))
    print(graph.get_graph().draw_mermaid())  # 把输出粘贴到 https://mermaid.live 可以看到图
    print(build_fanout_graph().invoke({"topic": "北京", "reports": []}))
