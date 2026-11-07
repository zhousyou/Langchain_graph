"""第 7 章 参考答案：记忆与人在回路"""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


def build_memory_chatbot(model: BaseChatModel, checkpointer):
    def chatbot(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def chat(app, thread_id: str, text: str) -> str:
    result = app.invoke({"messages": [HumanMessage(content=text)]}, make_config(thread_id))
    return result["messages"][-1].content


def get_history(app, thread_id: str) -> list[BaseMessage]:
    return app.get_state(make_config(thread_id)).values.get("messages", [])


class TransferState(TypedDict):
    to: str
    amount: float
    status: str


def approval(state: TransferState) -> Command:
    decision = interrupt({"question": f"确认向 {state['to']} 转账 {state['amount']} 元吗？(yes/no)"})
    if decision == "yes":
        return Command(goto="execute", update={"status": "approved"})
    return Command(goto=END, update={"status": "rejected"})


def execute(state: TransferState) -> dict:
    return {"status": "done"}


def build_transfer_graph(checkpointer):
    builder = StateGraph(TransferState)
    builder.add_node("approval", approval, destinations=("execute", END))
    builder.add_node("execute", execute)
    builder.add_edge(START, "approval")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=checkpointer)


def resume(app, thread_id: str, decision: str) -> dict:
    return app.invoke(Command(resume=decision), make_config(thread_id))


if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver

    from tutorial_kit import get_llm

    bot = build_memory_chatbot(get_llm(), InMemorySaver())
    print(chat(bot, "u1", "我叫小明，喜欢打篮球"))
    print(chat(bot, "u1", "我叫什么？喜欢什么？"))
    print(chat(bot, "u2", "我叫什么？"))

    app = build_transfer_graph(InMemorySaver())
    r = app.invoke({"to": "张三", "amount": 100, "status": "pending"}, make_config("t1"))
    print(r["__interrupt__"][0].value["question"])
    print(resume(app, "t1", input("> ").strip()))
