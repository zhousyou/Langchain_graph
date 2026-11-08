"""测试公共配置。

默认测试 exercises/ 下你自己写的代码；
设置环境变量 TUTORIAL_TARGET=solutions 可以改为测试参考答案。
"""

import importlib
import os
import traceback
from pathlib import Path

import pytest

TARGET = os.getenv("TUTORIAL_TARGET", "exercises")


def load(module_name: str):
    """加载被测模块，例如 load("ch01_models_messages")。"""
    return importlib.import_module(f"{TARGET}.{module_name}")


def _is_todo_placeholder(exc: NotImplementedError) -> bool:
    """只有练习文件里原样保留的 `raise NotImplementedError` 才算“未实现”；
    LangChain 等库内部抛出的 NotImplementedError 是真实错误，需要原样展示。"""
    frames = traceback.extract_tb(exc.__traceback__)
    return not exc.args and bool(frames) and TARGET in Path(frames[-1].filename).parts


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """把 TODO 占位的 NotImplementedError 转换成更友好的提示。"""
    try:
        return (yield)
    except NotImplementedError as exc:
        if not _is_todo_placeholder(exc):
            raise
    pytest.fail("这个练习还没有实现（仍是 raise NotImplementedError），去完成 TODO 吧！", pytrace=False)
