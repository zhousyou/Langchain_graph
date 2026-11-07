"""第 1 章 练习：模型与消息（LangChain）

讲义：docs/01_模型与消息.md
验证：uv run python check.py 1

目标：
  1. 理解 SystemMessage / HumanMessage / AIMessage 三种消息的含义
  2. 学会用 model.invoke() 调用模型，并从返回的 AIMessage 中取出文本
  3. 学会用 model.stream() 进行流式输出

说明：所有函数接收的 model 参数都是一个“聊天模型”（BaseChatModel）。
测试时传入的是假模型，运行 demo 时传入的是真实模型——你的代码不需要关心区别，
这正是 LangChain 统一模型接口的价值。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的助手，请用简洁的中文回答。"
DEFAULT_SYSTEM_PROMPT = "你是一个说话像海盗的助手。"


def build_messages(system_prompt: str, user_input: str) -> list[BaseMessage]:
    """练习 1.1：构造发给模型的消息列表。

    返回一个包含两条消息的列表：
      第 1 条：SystemMessage，内容是 system_prompt（设定模型的角色/规则）
      第 2 条：HumanMessage，内容是 user_input（用户说的话）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]



def ask(model: BaseChatModel, question: str) -> str:
    """练习 1.2：向模型提问并返回回答的文本。

    步骤：
      1. 用 build_messages(DEFAULT_SYSTEM_PROMPT, question) 构造消息
      2. 调用 model.invoke(消息列表)，它会返回一个 AIMessage
      3. 返回 AIMessage 的 .content 属性（字符串）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    messages = build_messages(DEFAULT_SYSTEM_PROMPT, question)
    ai_message = model.invoke(messages)
    # print(f"====\n完整的AImessage回复：{ai_message}\n======")
    return ai_message.content


def build_conversation(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """练习 1.3：把 (角色, 内容) 形式的对话历史转换成消息对象列表。

    角色映射：
      "system" -> SystemMessage
      "human"  -> HumanMessage
      "ai"     -> AIMessage
    遇到其它角色时抛出 ValueError。

    示例：
      build_conversation([("human", "你好"), ("ai", "你好！")])
      == [HumanMessage("你好"), AIMessage("你好！")]

    思考：模型本身是“无状态”的，多轮对话就是把历史消息一起发给它。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    messages_list = []
    for message in history:
      if message[0] == "system":
        messages_list.append(SystemMessage(content=message[1]))
      elif message[0] == "human":
        messages_list.append(HumanMessage(content=message[1]))
      elif message[0] == "ai":
        messages_list.append(AIMessage(content=message[1]))
      else:
        raise ValueError(f"错误的角色映射{message[0]}")
    return messages_list



def stream_answer(model: BaseChatModel, question: str) -> list[str]:
    """练习 1.4：流式调用模型，返回所有片段（chunk）的文本组成的列表。

    提示：
      - model.stream(消息列表) 返回一个迭代器，每次产出一个 AIMessageChunk
      - 每个 chunk 的 .content 是一小段文本
      - 消息同样用 build_messages(DEFAULT_SYSTEM_PROMPT, question) 构造
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    chunks = []
    messages = build_messages(DEFAULT_SYSTEM_PROMPT, question)
    for answer in model.stream(messages):
      chunks.append(answer.content)
    return chunks

if __name__ == "__main__":
    # 配置好 .env 后运行：uv run python -m exercises.ch01_models_messages
    from tutorial_kit import get_llm

    llm = get_llm()
    print(ask(llm, "用一句话介绍 LangChain"))
    for piece in stream_answer(llm, "用三句话介绍 LangGraph"):
        print(piece, end="", flush=True)
    print()
