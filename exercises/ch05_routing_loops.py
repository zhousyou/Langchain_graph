"""第 5 章 练习：条件边（路由）与循环（LangGraph）

讲义：docs/05_条件边与循环.md
验证：uv run python check.py 5

目标：
  1. 用 add_conditional_edges 根据状态决定“下一步去哪个节点”（路由）
  2. 用条件边指回自身/上游节点，形成“循环”，并设置退出条件
     —— Agent 的本质就是一个“思考 -> 行动 -> 观察 -> 再思考”的循环

场景一：客服工单路由。先分类，再交给对应的处理节点。
场景二：草稿反复修订，直到达到修订次数上限。
"""

import stat
from typing import TypedDict

from langgraph.graph import END, START, StateGraph  # noqa: F401


class TicketState(TypedDict):
    text: str       # 用户的工单内容
    category: str   # 分类结果：billing / tech / general
    reply: str      # 最终回复


def classify(state: TicketState) -> dict:
    """练习 5.1：基于关键词给工单分类，返回 {"category": ...}

    规则（按顺序判断）：
      - 包含 "退款"、"扣费"、"发票" 任一 -> "billing"
      - 包含 "报错"、"崩溃"、"bug"（大小写均可）任一 -> "tech"
      - 其它 -> "general"
    （真实项目里这一步通常交给大模型 + 结构化输出来做，原理完全一样。）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    if any(k in state["text"] for k in ["退款", "扣费", "发票"]):
        return {"category": "billing"}
    elif any(k in state["text"].lower() for k in ["报错", "崩溃", "bug"]):
        return {"category": "tech"}
    else:
        return {"category": "general"}


def route_by_category(state: TicketState) -> str:
    """练习 5.2：路由函数 —— 根据 category 返回下一个节点的名字。

      "billing" -> "billing_agent"
      "tech"    -> "tech_agent"
      其它      -> "general_agent"

    注意：路由函数不修改状态，只返回“去哪”。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    if state["category"] == "billing":
        return "billing_agent"
    elif state["category"] == "tech":
        return "tech_agent"
    else:
        return "general_agent"


# 下面三个处理节点已经写好
def billing_agent(state: TicketState) -> dict:
    return {"reply": "【财务】已收到您的账务问题，将在 24 小时内处理。"}


def tech_agent(state: TicketState) -> dict:
    return {"reply": "【技术】请提供报错截图和版本号，我们会尽快排查。"}


def general_agent(state: TicketState) -> dict:
    return {"reply": "【客服】感谢您的反馈，我们会尽快回复。"}


def build_router_graph():
    """练习 5.3：构建路由图

      START -> classify --(route_by_category)--> billing_agent / tech_agent / general_agent -> END

    提示：
      builder.add_conditional_edges(
          "classify",                                         # 从哪个节点出发
          route_by_category,                                  # 路由函数
          ["billing_agent", "tech_agent", "general_agent"],   # 可能的目标（可选，但能让画图更准确）
      )
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("billing_agent", billing_agent)
    builder.add_node("tech_agent", tech_agent)
    builder.add_node("general_agent", general_agent)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        ["billing_agent", "tech_agent", "general_agent"]
    )
    builder.add_edge(["billing_agent", "tech_agent", "general_agent"], END)

    return builder.compile()


# ---------------- 循环 ----------------

class DraftState(TypedDict):
    draft: str
    revisions: int       # 已修订次数
    max_revisions: int   # 最多修订次数


def revise(state: DraftState) -> dict:
    """练习 5.4：修订节点 —— 修订次数 +1，并在草稿末尾追加 "[第N次修订]"（N 为新的次数）。

    例：{"draft": "初稿", "revisions": 0} -> {"draft": "初稿[第1次修订]", "revisions": 1}
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    return {
        "draft" : state["draft"]+f"[第{state['revisions']+1}次修订]",
        "revisions" : state["revisions"]+1
    }


def should_continue(state: DraftState) -> str:
    """练习 5.5：循环的退出条件。

    revisions >= max_revisions 时返回 END（结束），否则返回 "revise"（继续循环）。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    return END if state["revisions"] >= state["max_revisions"] else "revise"


def build_loop_graph():
    """练习 5.6：构建循环图

      START -> revise --(should_continue)--> revise（回到自己）
                                         └─> END

    思考：如果退出条件永远不满足会怎样？LangGraph 有 recursion_limit（默认 25 步）兜底，
    超出会抛出 GraphRecursionError。测试里会验证这一点。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    builder = StateGraph(DraftState)
    builder.add_node("revise", revise)

    builder.add_edge(START, "revise")
    builder.add_conditional_edges(
        "revise",
        should_continue,
        ["revise", END]
    )

    return builder.compile()


if __name__ == "__main__":
    # 运行：uv run python -m exercises.ch05_routing_loops
    router = build_router_graph()
    for t in ["我要退款", "App 一打开就崩溃", "你们周末上班吗"]:
        print(t, "->", router.invoke({"text": t}))
    print(build_loop_graph().invoke({"draft": "初稿", "revisions": 0, "max_revisions": 2}))
