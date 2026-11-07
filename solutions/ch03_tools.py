"""第 3 章 参考答案：工具与工具调用"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

FAKE_WEATHER = {"北京": "晴，25°C", "上海": "小雨，22°C", "广州": "多云，30°C"}


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数的乘积。"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询指定城市今天的天气。参数 city 是中文城市名，例如 北京。"""
    return FAKE_WEATHER.get(city, f"未知城市: {city}")


TOOLS = [multiply, get_weather]


def execute_tool_calls(ai_message: AIMessage, tools: list[BaseTool]) -> list[ToolMessage]:
    tools_by_name = {t.name: t for t in tools}
    results = []
    for call in ai_message.tool_calls:
        selected = tools_by_name.get(call["name"])
        if selected is None:
            results.append(ToolMessage(content=f"错误：没有名为 {call['name']} 的工具", tool_call_id=call["id"]))
            continue
        results.append(selected.invoke(call))
    return results


def run_one_round(model: BaseChatModel, tools: list[BaseTool], question: str) -> str:
    model_with_tools = model.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    if not ai_msg.tool_calls:
        return ai_msg.content
    messages.extend(execute_tool_calls(ai_msg, tools))
    final = model_with_tools.invoke(messages)
    return final.content


if __name__ == "__main__":
    from tutorial_kit import get_llm

    print(run_one_round(get_llm(), TOOLS, "北京今天天气怎么样？另外 12.5 乘以 4 等于多少？"))
