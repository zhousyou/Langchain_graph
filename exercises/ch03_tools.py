"""第 3 章 练习：工具与工具调用（LangChain）

讲义：docs/03_工具与工具调用.md
验证：uv run python check.py 3

目标：
  1. 用 @tool 装饰器把 Python 函数变成模型可调用的“工具”
  2. 理解工具调用的完整流程：
        模型返回 AIMessage(tool_calls=[...])  ->  我们执行工具  ->  把结果包装成 ToolMessage
        ->  连同历史一起再发给模型  ->  模型给出最终回答
  3. 亲手写出“一轮”工具调用——这就是 Agent 的核心原理，后面 LangGraph 会把它变成循环

关键提醒：模型自己【不会】执行工具！它只会说“我想调用 X 工具，参数是 Y”，执行是我们代码的事。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: F401
from langchain_core.tools import BaseTool, tool  # noqa: F401

FAKE_WEATHER = {"北京": "晴，25°C", "上海": "小雨，22°C", "广州": "多云，30°C"}


# 练习 3.1：定义工具 multiply
#   - 用 @tool 装饰
#   - 函数签名：multiply(a: float, b: float) -> float，返回 a * b
#   - 必须写 docstring！LangChain 会把 函数名 / docstring / 参数类型 转成工具说明发给模型，
#     模型就是靠这些信息决定“什么时候、用什么参数”调用工具的。
# TODO: 在这里定义 multiply


# 练习 3.2：定义工具 get_weather
#   - 签名：get_weather(city: str) -> str
#   - 从 FAKE_WEATHER 字典中查询天气；查不到时返回 f"未知城市: {city}"
#   - 同样要写清楚 docstring
# TODO: 在这里定义 get_weather


# 练习 3.3：把上面两个工具放进列表
TOOLS: list = []  # TODO: 改成 [multiply, get_weather]


def execute_tool_calls(ai_message: AIMessage, tools: list[BaseTool]) -> list[ToolMessage]:
    """练习 3.4：执行 AIMessage 中的所有工具调用，返回 ToolMessage 列表。

    ai_message.tool_calls 是一个列表，每一项形如：
        {"name": "get_weather", "args": {"city": "北京"}, "id": "call_123", "type": "tool_call"}

    步骤：
      1. 建立 {工具名: 工具对象} 的字典（工具对象有 .name 属性）
      2. 遍历每个 tool_call：
         - 找到对应工具，调用 工具.invoke(tool_call)
           （小技巧：直接把整个 tool_call 字典传给 invoke，返回的就是 ToolMessage，
             并且 tool_call_id 已自动填好）
         - 若工具不存在，不要抛异常，而是返回
           ToolMessage(content="错误：没有名为 xxx 的工具", tool_call_id=tool_call["id"])
           —— 把错误告诉模型，模型往往能自己纠正
      3. 按顺序返回所有 ToolMessage
    """
    # TODO: 在这里实现
    raise NotImplementedError


def run_one_round(model: BaseChatModel, tools: list[BaseTool], question: str) -> str:
    """练习 3.5：手动完成“一轮”工具调用，返回模型的最终回答文本。

    步骤：
      1. model_with_tools = model.bind_tools(tools)   # 告诉模型有哪些工具可用
      2. messages = [HumanMessage(question)]
      3. ai_msg = model_with_tools.invoke(messages)，并把 ai_msg 追加到 messages
      4. 如果 ai_msg.tool_calls 为空：模型直接回答了，返回 ai_msg.content
      5. 否则：用 execute_tool_calls 执行工具，把得到的 ToolMessage 全部追加到 messages
      6. 再调用一次 model_with_tools.invoke(messages)，返回其 .content

    思考：如果第 6 步模型又要求调用工具怎么办？——这就需要“循环”，第 6 章用 LangGraph 解决。
    """
    # TODO: 在这里实现
    raise NotImplementedError


if __name__ == "__main__":
    # 配置好 .env 后运行：uv run python -m exercises.ch03_tools
    from tutorial_kit import get_llm

    print(run_one_round(get_llm(), TOOLS, "北京今天天气怎么样？另外 12.5 乘以 4 等于多少？"))
