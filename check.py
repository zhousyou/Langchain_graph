"""练习进度检查。

用法：
  uv run python check.py          # 查看所有章节的完成进度
  uv run python check.py 3        # 详细检查第 3 章（显示每个测试和失败原因）
  uv run python check.py 3 --solutions   # 用参考答案运行第 3 章测试（确认环境正常）
"""

import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
CHAPTERS = {
    1: ("模型与消息", "test_ch01_models_messages.py"),
    2: ("提示词与 LCEL 链", "test_ch02_prompts_chains.py"),
    3: ("工具与工具调用", "test_ch03_tools.py"),
    4: ("LangGraph 基础", "test_ch04_graph_basics.py"),
    5: ("条件边与循环", "test_ch05_routing_loops.py"),
    6: ("手写 ReAct Agent", "test_ch06_react_agent.py"),
    7: ("记忆与人在回路", "test_ch07_memory_hitl.py"),
    8: ("综合项目：学习助理", "test_ch08_final_agent.py"),
}


def pad(text: str, width: int) -> str:
    """按终端显示宽度补齐空格（中文字符占两格）。"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def run_pytest(test_file: str, verbose: bool, env: dict) -> subprocess.CompletedProcess:
    args = [sys.executable, "-m", "pytest", f"tests/{test_file}", "--no-header", "-rN"]
    args += ["-v", "--tb=short"] if verbose else ["-q", "--tb=no"]
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=not verbose, text=True)


def parse_counts(output: str) -> tuple[int, int]:
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", output))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", output))
    return passed, failed


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    env = dict(os.environ)
    if "--solutions" in sys.argv:
        env["TUTORIAL_TARGET"] = "solutions"
        print("（正在测试 solutions/ 参考答案）\n")

    if args:
        num = int(args[0])
        title, test_file = CHAPTERS[num]
        print(f"===== 第 {num} 章：{title} =====", flush=True)
        result = run_pytest(test_file, verbose=True, env=env)
        print("\n🎉 本章全部通过！可以进入下一章。" if result.returncode == 0 else "\n还有测试没通过，看看上面的失败原因吧。")
        return

    total_passed = total = 0
    print(pad("章节", 8) + pad("内容", 24) + pad("进度", 10) + "状态")
    print("-" * 56)
    for num, (title, test_file) in CHAPTERS.items():
        result = run_pytest(test_file, verbose=False, env=env)
        passed, failed = parse_counts(result.stdout + result.stderr)
        count = passed + failed
        total_passed += passed
        total += count
        status = "✅ 完成" if count and failed == 0 else ("⏳ 进行中" if passed else "⬜ 未开始")
        if count == 0:
            status = "❗ 无法运行（语法错误？用 check.py %d 查看）" % num
        print(pad(f"第{num}章", 8) + pad(title, 24) + pad(f"{passed}/{count}", 10) + status)
    print("-" * 56)
    print(f"总进度：{total_passed}/{total}")
    print("\n提示：uv run python check.py <章节号> 查看某一章的详细结果。")


if __name__ == "__main__":
    main()
