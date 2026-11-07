"""第 8 章 综合项目（1/3）：学习助理的工具

讲义：docs/08_综合项目_学习助理Agent.md
验证：uv run python check.py 8
"""

import ast
import operator

from langchain_core.tools import tool  # noqa: F401

from .knowledge import KNOWLEDGE_BASE  # noqa: F401

NOTES: dict[str, str] = {}          # 保存的笔记：{标题: 内容}
SENSITIVE_TOOLS = {"save_note"}     # 敏感工具：执行前必须经过人工审批

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """安全地计算数学表达式（只允许数字和 + - * / ** %），比 eval 安全。已写好，直接使用。"""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式: {expression}")

    return _eval(ast.parse(expression, mode="eval"))


# 练习 8.1：工具 calculator(expression: str) -> str
#   - 用 safe_eval 计算，返回 str(结果)，例如 "16"
#   - 出现任何异常时不要抛出，返回 f"计算出错: {异常}"（让模型看到错误、自行修正）
#   - 写好 docstring：说明支持哪些运算，并给一个例子
# TODO: 在这里定义 calculator


# 练习 8.2：工具 search_knowledge(query: str) -> str
#   - 把 query 按空白切分成多个关键词，统一转小写
#   - 遍历 KNOWLEDGE_BASE，若任一关键词出现在 (标题 + 内容).lower() 中，则命中
#   - 每条命中格式化为 f"【{标题}】{内容}"，最多返回前 3 条，用 "\n" 连接
#   - 一条都没命中时返回 "知识库中没有找到相关内容"
# TODO: 在这里定义 search_knowledge


# 练习 8.3：工具 save_note(title: str, content: str) -> str
#   - 写入 NOTES[title] = content
#   - 返回 f"已保存笔记《{title}》"
# TODO: 在这里定义 save_note


# 练习 8.4：工具 list_notes() -> str（无参数）
#   - NOTES 为空时返回 "暂无笔记"
#   - 否则每个标题一行，格式 "- 标题"，用 "\n" 连接
# TODO: 在这里定义 list_notes


# 练习 8.5：按顺序放入四个工具
ALL_TOOLS: list = []  # TODO: [calculator, search_knowledge, save_note, list_notes]
