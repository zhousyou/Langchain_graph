"""第 7 章 练习：记忆（Checkpointer）与人在回路（Human-in-the-loop）

讲义：docs/07_记忆与人在回路.md
验证：uv run python check.py 7

目标：
  1. 编译图时传入 checkpointer，让图在每一步之后自动保存状态（“存档”）
  2. 用 thread_id 区分不同的会话：同一个 thread_id 共享记忆，不同的互不干扰
  3. 用 get_state 读取某个会话当前的状态
  4. 用 interrupt() 让图在关键步骤暂停，等待人类审批；用 Command(resume=...) 恢复执行
  5. 用 Command(goto=..., update=...) 在节点里同时“更新状态 + 决定下一步”
"""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage  # noqa: F401
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401
from langgraph.types import Command, interrupt  # noqa: F401


def build_memory_chatbot(model: BaseChatModel, checkpointer):
    """练习 7.1：构建一个带记忆的聊天机器人。

      - 状态用 MessagesState
      - 只有一个节点 "chatbot"：用 state["messages"] 调用 model，返回 {"messages": [回复]}
      - START -> chatbot -> END
      - 编译时传入：builder.compile(checkpointer=checkpointer)

    有了 checkpointer，每次 invoke 只需要传“新的一条消息”，历史消息会从存档里自动恢复。
    """
    # TODO: 在这里实现
    raise NotImplementedError


def make_config(thread_id: str) -> dict:
    """练习 7.2：构造运行配置，返回 {"configurable": {"thread_id": thread_id}}"""
    # TODO: 在这里实现
    raise NotImplementedError


def chat(app, thread_id: str, text: str) -> str:
    """练习 7.3：在指定会话中发送一条消息，返回机器人回复的文本。

    提示：
      result = app.invoke({"messages": [HumanMessage(text)]}, make_config(thread_id))
      result["messages"] 是该会话的完整历史，最后一条就是最新回复
    """
    # TODO: 在这里实现
    raise NotImplementedError


def get_history(app, thread_id: str) -> list[BaseMessage]:
    """练习 7.4：读取指定会话的全部历史消息。

    提示：app.get_state(config) 返回 StateSnapshot，它的 .values 就是当前状态字典。
    会话不存在时返回空列表（.values 是空字典）。
    """
    # TODO: 在这里实现
    raise NotImplementedError


# ---------------- 人在回路：转账审批 ----------------

class TransferState(TypedDict):
    to: str
    amount: float
    status: str   # pending -> approved -> done，或 pending -> rejected


def approval(state: TransferState) -> Command:
    """练习 7.5：审批节点。

    步骤：
      1. decision = interrupt({"question": f"确认向 {to} 转账 {amount} 元吗？(yes/no)"})
         —— 执行到这里图会暂停；invoke 的返回值里会出现 "__interrupt__" 字段。
         —— 之后调用 app.invoke(Command(resume="yes"), config) 恢复时，
            本节点会【从头重新执行】，这一次 interrupt(...) 直接返回 "yes"。
      2. decision == "yes"：return Command(goto="execute", update={"status": "approved"})
         否则：            return Command(goto=END,       update={"status": "rejected"})
    """
    # TODO: 在这里实现
    raise NotImplementedError


def execute(state: TransferState) -> dict:
    """执行转账（已写好）。"""
    return {"status": "done"}


def build_transfer_graph(checkpointer):
    """练习 7.6：构建审批图：START -> approval --(Command 决定)--> execute -> END / 直接 END

    提示：
      - builder.add_node("approval", approval, destinations=("execute", END))
        destinations 用来声明 Command 可能跳转的目标（便于画图），approval 后面不需要再 add_edge
      - interrupt 依赖 checkpointer 保存暂停时的现场，所以编译时必须传入 checkpointer
    """
    # TODO: 在这里实现
    raise NotImplementedError


def resume(app, thread_id: str, decision: str) -> dict:
    """练习 7.7：用人类的决定恢复被暂停的图，返回最终状态。

    提示：app.invoke(Command(resume=decision), make_config(thread_id))
    """
    # TODO: 在这里实现
    raise NotImplementedError


if __name__ == "__main__":
    # 配置好 .env 后运行：uv run python -m exercises.ch07_memory_hitl
    from langgraph.checkpoint.memory import InMemorySaver

    from tutorial_kit import get_llm

    bot = build_memory_chatbot(get_llm(), InMemorySaver())
    print(chat(bot, "u1", "我叫小明，喜欢打篮球"))
    print(chat(bot, "u1", "我叫什么？喜欢什么？"))
    print(chat(bot, "u2", "我叫什么？"))  # 另一个会话，应该不知道

    app = build_transfer_graph(InMemorySaver())
    r = app.invoke({"to": "张三", "amount": 100, "status": "pending"}, make_config("t1"))
    print(r["__interrupt__"][0].value["question"])
    print(resume(app, "t1", input("> ").strip()))
