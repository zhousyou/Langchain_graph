"""测试公共配置。

默认测试 exercises/ 下你自己写的代码；
设置环境变量 TUTORIAL_TARGET=solutions 可以改为测试参考答案。
"""

import importlib
import os

import pytest

TARGET = os.getenv("TUTORIAL_TARGET", "exercises")


def load(module_name: str):
    """加载被测模块，例如 load("ch01_models_messages")。"""
    return importlib.import_module(f"{TARGET}.{module_name}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """把 NotImplementedError 转换成更友好的提示。"""
    try:
        return (yield)
    except NotImplementedError:
        pass
    pytest.fail("这个练习还没有实现（仍是 raise NotImplementedError），去完成 TODO 吧！", pytrace=False)
