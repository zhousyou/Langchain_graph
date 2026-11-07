"""第 8 章 综合项目（2/3）：学习助理的图

目标结构：

    START -> agent --route_after_agent--> END                （没有工具调用：回答完毕）
               ^              ├──────────> tools ──┐          （普通工具：直接执行）
               │              └──> human_review    │          （敏感工具：先暂停等人工审批）
               │                     ├─批准─> tools │
               └──────拒绝────────────┘            │
               └───────────────────────────────────┘

讲义：docs/08_综合项目_学习助理Agent.md
验证：uv run python check.py 8
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage  # noqa: F401
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401
from langgraph.prebuilt import ToolNode  # noqa: F401
from langgraph.types import Command, interrupt  # noqa: F401

from .tools import ALL_TOOLS, SENSITIVE_TOOLS  # noqa: F401

# 系统提示词决定了 Agent 的“性格”和工作方式。写好之后，用真实模型运行时可以反复调整它。
SYSTEM_PROMPT = (
    "你是一个 LangChain / LangGraph 学习助理。\n"
    "- 回答概念问题前，先用 search_knowledge 查询知识库，并基于查询结果回答；\n"
    "- 需要计算时使用 calculator，不要心算；\n"
    "- 用户要求记笔记时使用 save_note，查看笔记时使用 list_notes；\n"
    "- 回答使用简洁的中文。"
)


def make_agent_node(model: BaseChatModel, tools: list, system_prompt: str = SYSTEM_PROMPT):
    """练习 8.6：返回 agent 节点函数（与第 6 章的 make_call_model_node 相同）。

    - 外层：绑定工具
    - 内层 agent(state)：把 SystemMessage(system_prompt) 拼在 state["messages"] 前面调用模型，
      返回 {"messages": [response]}
    """
    # TODO: 在这里实现
    raise NotImplementedError


def route_after_agent(state: MessagesState) -> str:
    """练习 8.7：agent 之后的路由。

    看最后一条消息的 tool_calls：
      - 没有工具调用                                -> END
      - 任一工具调用的 name 属于 SENSITIVE_TOOLS     -> "human_review"
      - 其它                                        -> "tools"
    """
    # TODO: 在这里实现
    raise NotImplementedError


def human_review(state: MessagesState) -> Command:
    """练习 8.8：人工审批节点。

    步骤：
      1. last = state["messages"][-1]（带 tool_calls 的 AIMessage）
      2. 找出其中属于 SENSITIVE_TOOLS 的调用，整理成 [{"name": ..., "args": ...}, ...]
      3. decision = interrupt({"type": "approval", "tool_calls": 上一步的列表})
      4. decision == "yes"：return Command(goto="tools")   —— 放行，由 ToolNode 执行
      5. 否则：为 last.tool_calls 中的【每一个】调用都构造一条
             ToolMessage(content="用户拒绝了该操作，请不要重试，并告知用户。", tool_call_id=调用的 id)
         然后 return Command(goto="agent", update={"messages": 这些 ToolMessage})

    为什么第 5 步要给每个调用都回一条 ToolMessage？
      —— 主流模型 API 要求：每个 tool_call 后面都必须跟着一条对应 tool_call_id 的 ToolMessage，
         否则下一次调用模型会报错。把“拒绝”作为工具结果告诉模型，模型就能向用户解释。
    """
    # TODO: 在这里实现
    raise NotImplementedError


def build_study_agent(model: BaseChatModel, checkpointer, tools: list | None = None):
    """练习 8.9：组装学习助理图（结构见文件开头的示意图）。

    - tools = tools or ALL_TOOLS
    - 节点："agent"（make_agent_node）、"tools"（ToolNode）、
            "human_review"（human_review，destinations=("tools", "agent")）
    - 边：START -> agent；agent 用 route_after_agent 做条件边（目标 "tools"/"human_review"/END）；
          tools -> agent
    - 编译时传入 checkpointer（多轮记忆 + interrupt 都需要它）
    """
    # TODO: 在这里实现
    raise NotImplementedError
