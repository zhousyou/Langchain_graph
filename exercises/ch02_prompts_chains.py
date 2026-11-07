"""第 2 章 练习：提示词模板与 LCEL 链（LangChain）

讲义：docs/02_提示词与LCEL链.md
验证：uv run python check.py 2

目标：
  1. 用 ChatPromptTemplate 把“提示词”变成可复用、可填参数的模板
  2. 用 LCEL 的管道符 `|` 把 提示词 -> 模型 -> 输出解析器 串成一条链
  3. 用 MessagesPlaceholder 给模板预留“对话历史”的位置
  4. 用 Pydantic + PydanticOutputParser 让模型输出结构化数据
  5. 用 RunnableLambda 把普通 Python 函数接入链
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: F401
from langchain_core.runnables import Runnable, RunnableLambda  # noqa: F401
from pydantic import BaseModel, Field  # noqa: F401


def make_translate_prompt() -> ChatPromptTemplate:
    """练习 2.1：创建一个翻译提示词模板。

    要求：
      - 使用 ChatPromptTemplate.from_messages([...])
      - 一条 system 消息，内容里包含变量 {language}，例如 "你是一名专业翻译，请翻译成{language}"
      - 一条 human 消息，内容就是变量 {text}
    所以模板的输入变量正好是 language 和 text 两个。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    messages = [
      ("system", "你是一名专业翻译，请翻译成{language}"),
      ("human", "{text}")
    ]
    return ChatPromptTemplate.from_messages(messages)


def make_translate_chain(model: BaseChatModel) -> Runnable:
    """练习 2.2：用 LCEL 组装翻译链：提示词模板 | 模型 | 字符串输出解析器。

    组装好的链用法：
      chain.invoke({"language": "英文", "text": "你好"})  ->  "Hello"（字符串）
    提示：StrOutputParser() 会把 AIMessage 转成它的 .content 字符串。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    chain = make_translate_prompt()| model | StrOutputParser()
    return chain


def make_chat_prompt() -> ChatPromptTemplate:
    """练习 2.3：创建一个支持对话历史的提示词模板。

    消息顺序：
      1. system 消息：任意内容，例如 "你是一个友好的助手。"
      2. MessagesPlaceholder("history")：调用时用一个消息列表填充
      3. human 消息："{question}"
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    messages = [
      ("system", "你是一个友好的助手，用中文回答问题"),
      MessagesPlaceholder("history"),
      ("human", "{question}")
    ]
    return ChatPromptTemplate.from_messages(messages)


# 练习 2.4：定义结构化输出的数据模型 MovieReview（继承 pydantic.BaseModel）
#   字段：
#     title:   str  电影名称
#     rating:  int  评分，必须在 1~10 之间（提示：Field(ge=1, le=10, description=...)）
#     summary: str  一句话影评
#   给每个字段写上 description —— 它会被写进给模型的格式说明里，帮助模型理解字段含义。
class MovieReview(BaseModel):
    # TODO: 在这里定义字段
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分", ge=1, le=10)
    summary: str = Field(description="一句话影评")


def make_review_chain(model: BaseChatModel) -> Runnable:
    """练习 2.5：组装一个输出 MovieReview 对象的影评链。

    步骤：
      1. parser = PydanticOutputParser(pydantic_object=MovieReview)
      2. 提示词模板：system 消息中包含 {format_instructions}，human 消息中包含 {movie}
      3. 用 .partial(format_instructions=parser.get_format_instructions()) 预先填好格式说明
      4. 返回 prompt | model | parser

    用法：chain.invoke({"movie": "流浪地球"}) -> MovieReview(title=..., rating=..., summary=...)

    补充：支持工具调用的真实模型还可以直接用 model.with_structured_output(MovieReview)，
    讲义里有介绍；这里用解析器的方式，是为了让你看清“格式说明 + 解析”的原理。
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    messages = [
      ("system", "从文本中提取电影的信息。\n{fomat_instruction}"),
      ("human", "点评下电影{movie}。")
    ]
    prompt = ChatPromptTemplate.from_messages(messages).partial(fomat_instruction=parser.get_format_instructions())
    # # 打印出prompt的模版输入
    # for m in prompt.format_messages(movie="{movie}"):
    #   m.pretty_print()
    chain = prompt | model | parser
    return chain

    


def make_word_count_chain() -> Runnable:
    """练习 2.6：用 RunnableLambda 组装一个不需要模型的“文本处理链”。

    链由两步组成（用 | 连接）：
      第 1 步：去掉首尾空白并转成小写       "  Hello World  " -> "hello world"
      第 2 步：返回字典 {"text": 文本, "word_count": 按空白切分后的单词数}

    用法：
      chain.invoke("  Hello World  ") -> {"text": "hello world", "word_count": 2}
      chain.batch(["a b", "c"])       -> [{...}, {...}]   （所有 Runnable 都自带 batch）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    to_lower = RunnableLambda(lambda x: x.strip().lower())
    to_dict = RunnableLambda(lambda x: {"text": x, "word_count": x.count(' ')+1})
    chain = to_lower | to_dict
    return chain 


if __name__ == "__main__":
    # 配置好 .env 后运行：uv run python -m exercises.ch02_prompts_chains
    from tutorial_kit import get_llm

    llm = get_llm()
    print(make_translate_chain(llm).invoke({"language": "日语", "text": "我爱学习 LangChain"}))
    print(make_review_chain(llm).invoke({"movie": "流浪地球"}))
