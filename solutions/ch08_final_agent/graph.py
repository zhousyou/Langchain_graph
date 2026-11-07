"""第 8 章 参考答案：学习助理的图"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .tools import ALL_TOOLS, SENSITIVE_TOOLS

SYSTEM_PROMPT = (
    "你是一个 LangChain / LangGraph 学习助理。\n"
    "- 回答概念问题前，先用 search_knowledge 查询知识库，并基于查询结果回答；\n"
    "- 需要计算时使用 calculator，不要心算；\n"
    "- 用户要求记笔记时使用 save_note，查看笔记时使用 list_notes；\n"
    "- 回答使用简洁的中文。"
)


def make_agent_node(model: BaseChatModel, tools: list, system_prompt: str = SYSTEM_PROMPT):
    model_with_tools = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = model_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        return {"messages": [response]}

    return agent


def route_after_agent(state: MessagesState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return END
    if any(c["name"] in SENSITIVE_TOOLS for c in tool_calls):
        return "human_review"
    return "tools"


def human_review(state: MessagesState) -> Command:
    last = state["messages"][-1]
    sensitive = [{"name": c["name"], "args": c["args"]} for c in last.tool_calls if c["name"] in SENSITIVE_TOOLS]
    decision = interrupt({"type": "approval", "tool_calls": sensitive})
    if decision == "yes":
        return Command(goto="tools")
    rejected = [ToolMessage(content="用户拒绝了该操作，请不要重试，并告知用户。", tool_call_id=c["id"]) for c in last.tool_calls]
    return Command(goto="agent", update={"messages": rejected})


def build_study_agent(model: BaseChatModel, checkpointer, tools: list | None = None):
    tools = tools or ALL_TOOLS
    builder = StateGraph(MessagesState)
    builder.add_node("agent", make_agent_node(model, tools))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_review", human_review, destinations=("tools", "agent"))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "human_review", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
