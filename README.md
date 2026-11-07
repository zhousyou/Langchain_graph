# LangChain + LangGraph 实战教程

面向 Agent 开发初学者的动手教程：8 章讲义 + 65 个自动测试，从调用模型开始，最终亲手完成一个**带知识库检索、工具调用、人工审批和多轮记忆的学习助理 Agent**。

- 每章 = **讲义**（`docs/`）+ **练习**（`exercises/`，你来写）+ **测试**（`tests/`，自动判题）
- 测试使用离线“假模型”，**不需要 API Key、不花钱**就能验证你的代码
- 配置 API Key 后，每章都可以用真实大模型运行 demo

---

## 一、环境准备（约 5 分钟）

### 1. 安装依赖

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境（你的电脑上已安装）。在项目根目录执行：

```bash
cd ~/Desktop/Langchain_graph
uv sync
```

### 2. 验证环境

```bash
uv run python check.py --solutions
```

这会用参考答案跑一遍所有测试，看到 `总进度：65/65` 就说明环境正常。

再运行：

```bash
uv run python check.py
```

你会看到所有章节都是 `⬜ 未开始` —— 这是正常的，接下来就由你来完成。

### 3. 配置真实模型（可选，第 8 章运行应用时需要）

```bash
cp .env.example .env
```

编辑 `.env`，按文件中的示例选一个模型服务（OpenAI / DeepSeek / 通义千问 / 本地 Ollama 都可以），填入 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MODEL_NAME`。模型需要支持**工具调用（function calling）**。

---

## 二、目录结构

```
Langchain_graph/
├── README.md              ← 你正在看的文件
├── check.py               ← 进度检查脚本
├── docs/                  ← 讲义（先读这里）
├── exercises/             ← 练习文件（你要修改的地方，搜索 TODO）
│   ├── ch01_models_messages.py
│   ├── ...
│   └── ch08_final_agent/  ← 综合项目
├── tests/                 ← 自动测试（不需要修改）
├── solutions/             ← 参考答案（卡住时再看）
└── tutorial_kit/          ← 公共工具：真实模型加载、离线假模型
```

---

## 三、学习路线

| 章节 | 讲义 | 练习文件 | 主题 | 测试数 |
| --- | --- | --- | --- | --- |
| 0 | `docs/00_概览.md` | — | LangChain 与 LangGraph 的定位、Agent 的本质 | — |
| 1 | `docs/01_模型与消息.md` | `exercises/ch01_models_messages.py` | 消息类型、invoke、stream | 6 |
| 2 | `docs/02_提示词与LCEL链.md` | `exercises/ch02_prompts_chains.py` | 提示词模板、LCEL、结构化输出 | 6 |
| 3 | `docs/03_工具与工具调用.md` | `exercises/ch03_tools.py` | @tool、tool_calls、手写一轮工具调用 | 7 |
| 4 | `docs/04_LangGraph基础.md` | `exercises/ch04_graph_basics.py` | State、Node、Edge、Reducer、并行 | 6 |
| 5 | `docs/05_条件边与循环.md` | `exercises/ch05_routing_loops.py` | 路由、循环、递归上限 | 13 |
| 6 | `docs/06_手写ReAct_Agent.md` | `exercises/ch06_react_agent.py` | 手写 Agent 图 vs `create_agent` | 5 |
| 7 | `docs/07_记忆与人在回路.md` | `exercises/ch07_memory_hitl.py` | Checkpointer、thread_id、interrupt、Command | 7 |
| 8 | `docs/08_综合项目_学习助理Agent.md` | `exercises/ch08_final_agent/` | 综合项目：学习助理 Agent | 15 |

第 1–3 章打 LangChain 基础，第 4–5 章打 LangGraph 基础，第 6–7 章把两者结合，第 8 章做出完整应用。**请按顺序学习**，后面的章节会用到前面的知识。

---

## 四、每一章的练习步骤

对每一章，重复下面 5 步：

**第 1 步：读讲义。** 打开 `docs/0X_xxx.md`，理解关键概念和 API 用法。讲义里的示例代码与练习不同，需要你举一反三。

**第 2 步：打开练习文件，找到 `TODO`。** 每个 TODO 上方的 docstring 写明了要求、步骤和提示。把 `raise NotImplementedError` 替换成你的实现。

**第 3 步：运行本章测试。**

```bash
uv run python check.py 1          # 把 1 换成当前章节号
```

失败的测试会显示原因，例如：

```
FAILED test_1_2_ask_returns_text - AssertionError: ask 应该返回字符串（AIMessage.content），而不是 AIMessage 对象
```

只想跑某一个练习的测试时：

```bash
uv run pytest tests/test_ch01_models_messages.py -k "1_2" -v
```

**第 4 步：反复修改直到全部通过**，看到 `🎉 本章全部通过！`。

**第 5 步（推荐）：用真实模型运行本章 demo**（需要 `.env`），观察真实模型的行为：

```bash
uv run python -m exercises.ch01_models_messages   # 模块名就是练习文件名去掉 .py
```

> 第 4、5 章不涉及模型，不需要 `.env` 也能运行 demo。

随时查看总进度：

```bash
uv run python check.py
```

---

## 五、最终项目：运行你的学习助理

第 8 章全部测试通过后：

```bash
uv run python -m exercises.ch08_final_agent.app
```

```
📚 学习助理已启动。输入问题开始对话；/new 新会话，/notes 查看笔记，/quit 退出。

你 > 什么是 Checkpointer？
   🔧 [search_knowledge] 【Checkpointer】Checkpointer 会在图的每一步之后保存状态快照……
助理 > Checkpointer 是 LangGraph 中负责保存状态快照的组件……

你 > 把它记成一条笔记

⚠️  Agent 请求执行敏感操作：
   - save_note({'title': 'Checkpointer', 'content': '……'})
   是否批准？(y/n) > y
   🔧 [save_note] 已保存笔记《Checkpointer》
助理 > 已经帮你保存好了笔记。
```

完整的验收对话脚本和进阶挑战见 `docs/08_综合项目_学习助理Agent.md`。

---

## 六、卡住了怎么办

1. **重读 docstring 和讲义**：大部分答案的线索都在里面。
2. **读测试代码**：`tests/test_chXX_*.py` 会告诉你测试具体检查了什么。
3. **打印调试**：在函数里 `print(...)`，然后用 `uv run pytest tests/xxx.py -k "名字" -s` 运行（`-s` 会显示 print 输出）。
4. **看参考答案**：`solutions/` 下有同名文件。建议先独立尝试 15 分钟以上再看，看懂后**关掉答案自己重写一遍**。

## 七、常见问题

**Q：测试没有 API Key 是怎么运行的？**
测试使用 `tutorial_kit/fake.py` 中的 `FakeToolChatModel`，它按预设的“剧本”回复，并记录收到的消息。因为 LangChain 对所有模型提供统一接口，你的代码不需要区分真假模型。

**Q：运行 demo 报错 `未找到 OPENAI_API_KEY`？**
没有配置 `.env`，见“环境准备”第 3 步。

**Q：真实模型不调用工具 / 报错不支持 tools？**
换一个支持 function calling 的模型（如 `gpt-4o-mini`、`deepseek-chat`、`qwen-plus`）。本地 Ollama 需选择支持工具的模型（如 `qwen2.5`）。

**Q：想从头重做某一章？**
如果使用 git：`git checkout exercises/ch0X_xxx.py`。

## 八、参考资料

- LangChain 文档：<https://docs.langchain.com/oss/python/langchain/overview>
- LangGraph 文档：<https://docs.langchain.com/oss/python/langgraph/overview>
- 本教程依赖版本：langchain 1.4、langchain-core 1.7、langgraph 1.2（见 `uv.lock`）
