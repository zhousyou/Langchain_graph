"""第 6 章 练习：用 LangGraph 手写一个 ReAct Agent

讲义：docs/06_手写ReAct_Agent.md
验证：uv run python check.py 6

目标：
  把第 3 章“一轮工具调用”和第 5 章“循环”结合起来，得到真正的 Agent：

        START -> agent(调用模型) --有工具调用?--> tools(执行工具) --> agent ...（循环）
                                  └─没有──> END

  1. 使用 MessagesState：内置 messages 字段，reducer 为 add_messages（追加消息）
  2. 自己实现“调用模型”节点与“是否继续”的路由函数
  3. 使用 LangGraph 预置的 ToolNode 执行工具
  4. 对比 LangChain 的 create_agent：一行代码得到同样结构的 Agent
"""

from langchain.agents import create_agent  # noqa: F401
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage  # noqa: F401
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401
from langgraph.prebuilt import ToolNode  # noqa: F401

FAKE_STOCK = {"AAPL": 230.5, "TSLA": 410.2, "NVDA": 180.0}


# 工具已经写好（第 3 章你已经练过了）
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
SYSTEM_PROMPT = "你是一个股票助手。需要数据时调用工具，需要计算时调用 calculate 工具。"


def make_call_model_node(model: BaseChatModel, tools: list[BaseTool], system_prompt: str):
    """练习 6.1：返回一个“调用模型”的节点函数 call_model(state) -> dict。

    这是一个“工厂函数”：外层负责准备（绑定工具），内层才是真正的节点。
      1. 外层：model_with_tools = model.bind_tools(tools)
      2. 内层 call_model(state: MessagesState)：
           - messages = [SystemMessage(system_prompt)] + state["messages"]
             （system 提示词每次调用时临时拼在最前面，不存进状态里）
           - response = model_with_tools.invoke(messages)
           - return {"messages": [response]}   # add_messages reducer 会把它追加到历史
      3. return call_model
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    model_with_tools = model.bind_tools(tools)
    def call_model(state: MessagesState) -> dict:
      messages = [SystemMessage(content=system_prompt)] + state["messages"]
      response = model_with_tools.invoke(messages)
      return {
        "messages": [response]
      }
    return call_model


def should_continue(state: MessagesState) -> str:
    """练习 6.2：路由函数 —— 看最后一条消息是否包含工具调用。

      最后一条消息有 tool_calls（非空） -> 返回 "tools"
      否则                              -> 返回 END
    （LangGraph 预置的 tools_condition 做的就是这件事，这里自己实现一遍加深理解。）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    messages = state["messages"]
    return "tools" if messages[-1].tool_calls else END


def build_agent_graph(model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT):
    """练习 6.3：组装 ReAct Agent 图。

      - 状态：MessagesState
      - 节点 "agent"：make_call_model_node(model, tools, system_prompt)
      - 节点 "tools"：ToolNode(tools)  —— 它会执行最后一条 AIMessage 里的所有工具调用，
                                          并返回对应的 ToolMessage
      - 边：START -> agent；agent --should_continue--> tools / END；tools -> agent
      - 返回编译后的图

    用法：graph.invoke({"messages": [("human", "NVDA 多少钱？")]})
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    builder = StateGraph(MessagesState)
    builder.add_node("agent", make_call_model_node(model, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
      "agent",
      should_continue,
      ["tools", END]
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def build_prebuilt_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT):
    """练习 6.4：用 LangChain 的 create_agent 一行创建同样功能的 Agent。

    提示：create_agent(model, tools=tools, system_prompt=system_prompt)
    它返回的也是一个编译好的 LangGraph 图，用法与 build_agent_graph 完全一致。
    思考：既然有一行的写法，为什么还要学手写？（答案见讲义）
    """
    # TODO: 在这里实现
    # raise NotImplementedError
    return  create_agent(model, tools, system_prompt=system_prompt)


if __name__ == "__main__":
    # 配置好 .env 后运行：uv run python -m exercises.ch06_react_agent
    from tutorial_kit import get_llm

    # agent = build_agent_graph(get_llm(), TOOLS)
    question = "我有 10 股 NVDA 和 2 股 AAPL，一共值多少美元？"
    # for step in agent.stream({"messages": [("human", question)]}, stream_mode="values"):
    #     step["messages"][-1].pretty_print()
    # print(agent.get_graph().draw_mermaid())
    agent_new = build_prebuilt_agent(get_llm(), TOOLS)
    res = agent_new.invoke({"messages":[("human",question)]})
    print(res)
    print("====================")
    print(res["messages"][-1].content)
    print("====================")
    for m in res["messages"]:
      m.pretty_print()
