# 第 4 章：LangGraph 基础 —— State / Node / Edge / Reducer

> 练习：`exercises/ch04_graph_basics.py`　验证：`uv run python check.py 4`

## 1. LangGraph 是什么

LangGraph 把一个应用描述成一张**有向图**：

- **State（状态）**：一份在整张图里流动、被所有节点共享的数据
- **Node（节点）**：一个函数，读取状态、做一件事、返回对状态的更新
- **Edge（边）**：决定一个节点完成后，下一个执行哪个节点

LangGraph 本身与大模型无关，它是通用的“有状态流程编排”引擎；只不过节点里通常会调用 LangChain 的模型和工具。本章完全不涉及模型，专注理解图本身。

## 2. 关键概念

### 2.1 定义 State

```python
from typing import TypedDict

class OrderState(TypedDict):
    order_id: str
    price: float
    status: str
```

### 2.2 Node：返回“部分更新”

```python
def apply_discount(state: OrderState) -> dict:
    return {"price": state["price"] * 0.9}    # 只返回要改的字段，其他字段保持不变
```

节点**不要**直接修改传入的 state，而是返回一个字典，由 LangGraph 合并到状态里。

### 2.3 构建、编译、运行

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OrderState)
builder.add_node("apply_discount", apply_discount)
builder.add_node("confirm", lambda s: {"status": "confirmed"})
builder.add_edge(START, "apply_discount")
builder.add_edge("apply_discount", "confirm")
builder.add_edge("confirm", END)

graph = builder.compile()        # 编译后得到可运行的图（它也是一个 Runnable）
graph.invoke({"order_id": "A1", "price": 100, "status": "new"})
# {'order_id': 'A1', 'price': 90.0, 'status': 'confirmed'}
```

查看每一步的执行过程：

```python
for update in graph.stream(inputs, stream_mode="updates"):
    print(update)     # {'apply_discount': {'price': 90.0}} ...
```

可视化：`print(graph.get_graph().draw_mermaid())`，把输出粘贴到 <https://mermaid.live> 即可看到图。

### 2.4 Reducer：状态如何合并

默认情况下，节点返回的新值会**覆盖**旧值。若希望“追加”，就用 `Annotated[类型, reducer函数]`：

```python
import operator
from typing import Annotated

class State(TypedDict):
    total: int                                  # 覆盖
    history: Annotated[list[str], operator.add] # 追加：旧列表 + 新列表
```

节点返回 `{"history": ["step1"]}` 时，LangGraph 执行的是 `旧history + ["step1"]`。

后面会大量使用的 `MessagesState`，本质就是：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # add_messages：追加消息（同 id 则替换）
```

### 2.5 并行：fan-out / fan-in

从同一个节点出发连多条边，这些目标节点会在**同一步（super-step）并行执行**：

```python
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_db")
builder.add_edge(["search_web", "search_db"], "merge")   # 两者都完成后才执行 merge
```

并行节点如果写同一个字段，该字段**必须有 reducer**，否则 LangGraph 无法决定保留谁，会抛出 `InvalidUpdateError`。

## 3. 使用场景

- 把多步骤业务流程（审核、数据处理流水线）显式地画成图，便于理解和调试
- 多个互相独立的子任务并行执行（同时查多个数据源）

## 4. 练习清单

| 编号 | 内容 | 要点 |
| --- | --- | --- |
| 4.1 | `CounterState` | 普通字段 + `Annotated[list, operator.add]` |
| 4.2-4.3 | `increment` / `double` | 只返回部分更新 |
| 4.4 | `build_linear_graph` | add_node / add_edge / compile |
| 4.5 | 三个报告节点 | |
| 4.6 | `build_fanout_graph` | 并行分支 + 汇合 |

## 5. 动手实验

完成后运行 `uv run python -m exercises.ch04_graph_basics`，把打印的 mermaid 代码贴到 mermaid.live 看看你画的图。
然后试着把 `ReportState.reports` 的 `Annotated[..., operator.add]` 去掉，再运行一次，观察报错信息。
