"""第 8 章 参考答案：命令行聊天程序

运行：uv run python -m solutions.ch08_final_agent.app
"""

import uuid
from collections.abc import Callable

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command


def run_turn(app, thread_id: str, user_input: str, ask_approval: Callable[[dict], bool]) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = "yes" if ask_approval(payload) else "no"
        result = app.invoke(Command(resume=decision), config)
    return result["messages"][-1].content


def console_approval(payload: dict) -> bool:
    print("\n⚠️  Agent 请求执行敏感操作：")
    for call in payload["tool_calls"]:
        print(f"   - {call['name']}({call['args']})")
    return input("   是否批准？(y/n) > ").strip().lower() in {"y", "yes"}


def _message_count(app, thread_id: str) -> int:
    return len(app.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", []))


def _print_tool_steps(app, thread_id: str, start: int) -> None:
    messages = app.get_state({"configurable": {"thread_id": thread_id}}).values["messages"][start:]
    for m in messages:
        if isinstance(m, ToolMessage):
            print(f"   🔧 [{m.name}] {m.content.replace(chr(10), ' ')[:80]}")


def main() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from tutorial_kit import get_llm

    from .graph import build_study_agent
    from .tools import NOTES

    app = build_study_agent(get_llm(), InMemorySaver())
    thread_id = uuid.uuid4().hex[:8]
    print("📚 学习助理已启动。输入问题开始对话；/new 新会话，/notes 查看笔记，/quit 退出。")
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"（已开启新会话 {thread_id}，之前的对话不再记得）")
            continue
        if user_input == "/notes":
            print(NOTES or "暂无笔记")
            continue
        start = _message_count(app, thread_id)
        answer = run_turn(app, thread_id, user_input, console_approval)
        _print_tool_steps(app, thread_id, start)
        print("助理 >", answer)


if __name__ == "__main__":
    main()
