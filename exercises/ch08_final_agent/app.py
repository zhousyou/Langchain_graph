"""第 8 章 综合项目（3/3）：命令行聊天程序

完成 tools.py、graph.py 和下面的 run_turn 并通过测试后，配置好 .env 运行：
    uv run python -m exercises.ch08_final_agent.app
"""

import uuid
from collections.abc import Callable

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command  # noqa: F401


def run_turn(app, thread_id: str, user_input: str, ask_approval: Callable[[dict], bool]) -> str:
    """练习 8.10：处理用户的一轮输入（包括可能出现的多次审批），返回助理的最终回复文本。

    步骤：
      1. config = {"configurable": {"thread_id": thread_id}}
      2. result = app.invoke({"messages": [HumanMessage(user_input)]}, config)
      3. 只要 result 中包含 "__interrupt__"（图暂停了，等待审批）就循环：
           payload = result["__interrupt__"][0].value      # 即 human_review 里 interrupt(...) 的参数
           approved = ask_approval(payload)                 # 由调用方决定是否批准（命令行里是问用户）
           result = app.invoke(Command(resume="yes" if approved else "no"), config)
      4. 返回 result["messages"][-1].content

    为什么是“循环”而不是“if”？—— 恢复后 Agent 可能又调用了一次敏感工具，于是再次暂停。
    """
    # TODO: 在这里实现
    raise NotImplementedError


# ---------------- 以下为命令行界面，已写好 ----------------

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
