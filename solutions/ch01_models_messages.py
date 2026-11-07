"""第 1 章 参考答案：模型与消息"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的助手，请用简洁的中文回答。"


def build_messages(system_prompt: str, user_input: str) -> list[BaseMessage]:
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]


def ask(model: BaseChatModel, question: str) -> str:
    response = model.invoke(build_messages(DEFAULT_SYSTEM_PROMPT, question))
    return response.content


def build_conversation(history: list[tuple[str, str]]) -> list[BaseMessage]:
    role_to_cls = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
    messages = []
    for role, content in history:
        if role not in role_to_cls:
            raise ValueError(f"未知角色: {role}")
        messages.append(role_to_cls[role](content=content))
    return messages


def stream_answer(model: BaseChatModel, question: str) -> list[str]:
    chunks = []
    for chunk in model.stream(build_messages(DEFAULT_SYSTEM_PROMPT, question)):
        chunks.append(chunk.content)
    return chunks


if __name__ == "__main__":
    from tutorial_kit import get_llm

    llm = get_llm()
    print(ask(llm, "用一句话介绍 LangChain"))
    for piece in stream_answer(llm, "用三句话介绍 LangGraph"):
        print(piece, end="", flush=True)
    print()
