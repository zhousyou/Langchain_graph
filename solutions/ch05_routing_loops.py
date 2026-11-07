"""第 5 章 参考答案：条件边与循环"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    text: str
    category: str
    reply: str


def classify(state: TicketState) -> dict:
    text = state["text"]
    if any(k in text for k in ["退款", "扣费", "发票"]):
        category = "billing"
    elif any(k in text for k in ["报错", "崩溃", "bug", "Bug", "BUG"]):
        category = "tech"
    else:
        category = "general"
    return {"category": category}


def route_by_category(state: TicketState) -> str:
    return {"billing": "billing_agent", "tech": "tech_agent"}.get(state["category"], "general_agent")


def billing_agent(state: TicketState) -> dict:
    return {"reply": "【财务】已收到您的账务问题，将在 24 小时内处理。"}


def tech_agent(state: TicketState) -> dict:
    return {"reply": "【技术】请提供报错截图和版本号，我们会尽快排查。"}


def general_agent(state: TicketState) -> dict:
    return {"reply": "【客服】感谢您的反馈，我们会尽快回复。"}


def build_router_graph():
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("billing_agent", billing_agent)
    builder.add_node("tech_agent", tech_agent)
    builder.add_node("general_agent", general_agent)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_category, ["billing_agent", "tech_agent", "general_agent"])
    for name in ["billing_agent", "tech_agent", "general_agent"]:
        builder.add_edge(name, END)
    return builder.compile()


class DraftState(TypedDict):
    draft: str
    revisions: int
    max_revisions: int


def revise(state: DraftState) -> dict:
    n = state["revisions"] + 1
    return {"draft": f"{state['draft']}[第{n}次修订]", "revisions": n}


def should_continue(state: DraftState) -> str:
    if state["revisions"] >= state["max_revisions"]:
        return END
    return "revise"


def build_loop_graph():
    builder = StateGraph(DraftState)
    builder.add_node("revise", revise)
    builder.add_edge(START, "revise")
    builder.add_conditional_edges("revise", should_continue, ["revise", END])
    return builder.compile()


if __name__ == "__main__":
    router = build_router_graph()
    for t in ["我要退款", "App 一打开就崩溃", "你们周末上班吗"]:
        print(t, "->", router.invoke({"text": t}))
    print(build_loop_graph().invoke({"draft": "初稿", "revisions": 0, "max_revisions": 3}))
