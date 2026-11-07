"""第 6 章 参考答案：手写 ReAct Agent"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

FAKE_STOCK = {"AAPL": 230.5, "TSLA": 410.2, "NVDA": 180.0}


@tool
def get_stock_price(symbol: str) -> str:
    """查询股票的最新价格（美元）。symbol 为股票代码，例如 AAPL。"""
    price = FAKE_STOCK.get(symbol.upper())
    return f"{symbol.upper()} 当前价格 {price} 美元" if price is not None else f"未找到股票 {symbol}"


@tool
def calculate(expression: str) -> str:
    """计算一个只包含数字和 + - * / ( ) 的数学表达式，例如 (230.5 - 180) * 10。"""
    if not set(expression) <= set("0123456789.+-*/() "):
        return "表达式包含非法字符"
    return str(eval(expression))


TOOLS = [get_stock_price, calculate]
SYSTEM_PROMPT = "你是一个股票助手。需要数据时调用工具，需要计算时调用 calculate 工具，不要心算。"


def make_call_model_node(model: BaseChatModel, tools: list[BaseTool], system_prompt: str):
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return call_model


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph(model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", make_call_model_node(model, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


def build_prebuilt_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT):
    return create_agent(model, tools=tools, system_prompt=system_prompt)


if __name__ == "__main__":
    from tutorial_kit import get_llm

    agent = build_agent_graph(get_llm(), TOOLS)
    question = "我有 10 股 NVDA 和 2 股 AAPL，一共值多少美元？"
    for step in agent.stream({"messages": [("human", question)]}, stream_mode="values"):
        step["messages"][-1].pretty_print()
