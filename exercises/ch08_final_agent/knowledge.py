"""学习助理的本地知识库（已写好）。真实项目中这里通常是向量数据库 + 检索（RAG）。"""

KNOWLEDGE_BASE: dict[str, str] = {
    "LangChain": "LangChain 是构建大模型应用的框架，提供统一的模型接口、提示词模板、输出解析器、工具和 Agent 等组件。",
    "LCEL": "LCEL（LangChain Expression Language）用管道符 | 把 Runnable 组件串成链，所有链都支持 invoke、batch、stream。",
    "Tool": "Tool（工具）是模型可以调用的函数。用 @tool 装饰器定义，函数名、docstring 和参数类型会被转换成工具说明发给模型。",
    "LangGraph": "LangGraph 是用“图”编排有状态、多步骤 LLM 应用的框架，核心概念是 State、Node、Edge，支持循环、持久化和人在回路。",
    "State": "State 是图中所有节点共享的数据，通常用 TypedDict 定义；Reducer 决定节点返回的更新如何合并进状态。",
    "Checkpointer": "Checkpointer 会在图的每一步之后保存状态快照，配合 thread_id 实现多轮对话记忆、断点恢复和人在回路。",
    "interrupt": "interrupt() 可以让图在某个节点暂停并等待人类输入，之后用 Command(resume=...) 恢复执行。需要配合 checkpointer 使用。",
    "ReAct": "ReAct（Reasoning + Acting）是一种 Agent 模式：模型思考后决定调用工具，观察工具结果后继续思考，直到给出最终答案。",
}
