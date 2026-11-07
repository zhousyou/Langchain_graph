"""第 2 章 参考答案：提示词模板与 LCEL 链"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


def make_translate_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名专业翻译，请把用户的内容翻译成{language}，只输出译文。"),
            ("human", "{text}"),
        ]
    )


def make_translate_chain(model: BaseChatModel) -> Runnable:
    return make_translate_prompt() | model | StrOutputParser()


def make_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的助手。"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )


class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: int = Field(ge=1, le=10, description="评分，1~10 的整数")
    summary: str = Field(description="一句话影评")


def make_review_chain(model: BaseChatModel) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是影评人。请按要求的格式输出。\n{format_instructions}"),
            ("human", "请评价电影：{movie}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | model | parser


def make_word_count_chain() -> Runnable:
    clean = RunnableLambda(lambda text: text.strip().lower())
    count = RunnableLambda(lambda text: {"text": text, "word_count": len(text.split())})
    return clean | count


if __name__ == "__main__":
    from tutorial_kit import get_llm

    llm = get_llm()
    print(make_translate_chain(llm).invoke({"language": "英文", "text": "我爱学习 LangChain"}))
    print(make_review_chain(llm).invoke({"movie": "流浪地球"}))
