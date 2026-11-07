import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from tests.conftest import load
from tutorial_kit import FakeToolChatModel

ch = load("ch02_prompts_chains")


def test_2_1_translate_prompt():
    prompt = ch.make_translate_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert set(prompt.input_variables) == {"language", "text"}, "模板变量应该正好是 language 和 text"
    msgs = prompt.invoke({"language": "日语", "text": "早上好"}).to_messages()
    assert isinstance(msgs[0], SystemMessage) and "日语" in msgs[0].content
    assert isinstance(msgs[-1], HumanMessage) and msgs[-1].content == "早上好"


def test_2_2_translate_chain():
    model = FakeToolChatModel.from_responses(["Good morning"])
    chain = ch.make_translate_chain(model)
    result = chain.invoke({"language": "英文", "text": "早上好"})
    assert result == "Good morning", "链的输出应该是字符串，别忘了 StrOutputParser"
    assert "英文" in model.received[0][0].content, "模型应该收到填充好 language 的 system 消息"


def test_2_3_chat_prompt_with_history():
    prompt = ch.make_chat_prompt()
    history = [HumanMessage("我叫小明"), AIMessage("你好，小明！")]
    msgs = prompt.invoke({"history": history, "question": "我叫什么？"}).to_messages()
    assert len(msgs) == 4, "应为 system + 2 条历史 + 1 条问题"
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[1].content == "我叫小明" and msgs[2].content == "你好，小明！"
    assert isinstance(msgs[3], HumanMessage) and msgs[3].content == "我叫什么？"


def test_2_4_movie_review_model():
    fields = ch.MovieReview.model_fields
    assert {"title", "rating", "summary"} <= set(fields), "MovieReview 需要 title/rating/summary 三个字段"
    ok = ch.MovieReview(title="流浪地球", rating=9, summary="好看")
    assert ok.rating == 9
    with pytest.raises(ValidationError):
        ch.MovieReview(title="x", rating=11, summary="rating 超过 10 应该校验失败")
    with pytest.raises(ValidationError):
        ch.MovieReview(title="x", rating=0, summary="rating 小于 1 应该校验失败")
    assert all(f.description for f in fields.values()), "每个字段都应该写 description"


def test_2_5_review_chain():
    model = FakeToolChatModel.from_responses(['{"title": "流浪地球", "rating": 9, "summary": "中国科幻里程碑"}'])
    chain = ch.make_review_chain(model)
    review = chain.invoke({"movie": "流浪地球"})
    assert isinstance(review, ch.MovieReview), "链的输出应该是 MovieReview 对象"
    assert review.title == "流浪地球" and review.rating == 9
    sent_text = " ".join(m.content for m in model.received[0])
    assert "流浪地球" in sent_text, "human 消息里应该包含 {movie}"
    assert "rating" in sent_text, "system 消息里应该包含格式说明 format_instructions"


def test_2_6_word_count_chain():
    chain = ch.make_word_count_chain()
    assert chain.invoke("  Hello World  ") == {"text": "hello world", "word_count": 2}
    assert chain.batch(["A B C", "x"]) == [
        {"text": "a b c", "word_count": 3},
        {"text": "x", "word_count": 1},
    ]
