"""第 8 章 参考答案：学习助理的工具"""

import ast
import operator

from langchain_core.tools import tool

from .knowledge import KNOWLEDGE_BASE

NOTES: dict[str, str] = {}
SENSITIVE_TOOLS = {"save_note"}

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
    """安全地计算数学表达式（只允许数字和 + - * / ** %），比 eval 安全。"""

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


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持 + - * / ** % 和括号，例如 (3 + 5) * 2。"""
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"计算出错: {e}"


@tool
def search_knowledge(query: str) -> str:
    """在 LangChain / LangGraph 学习知识库中搜索概念解释。query 为关键词，多个关键词用空格分隔。"""
    keywords = [k.lower() for k in query.split()]
    hits = [
        f"【{title}】{content}"
        for title, content in KNOWLEDGE_BASE.items()
        if any(k in (title + content).lower() for k in keywords)
    ]
    if not hits:
        return "知识库中没有找到相关内容"
    return "\n".join(hits[:3])


@tool
def save_note(title: str, content: str) -> str:
    """把一条学习笔记保存下来。title 是笔记标题，content 是笔记内容。"""
    NOTES[title] = content
    return f"已保存笔记《{title}》"


@tool
def list_notes() -> str:
    """列出已经保存的所有学习笔记的标题。"""
    if not NOTES:
        return "暂无笔记"
    return "\n".join(f"- {t}" for t in NOTES)


ALL_TOOLS = [calculator, search_knowledge, save_note, list_notes]
