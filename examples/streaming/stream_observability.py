"""可观测性流式输出（Stream Observability: checkpoints / tasks / debug）。

合并演示 graph.stream 的三种「可观测性」流模式（均需 checkpointer，version="v2"）：
  - checkpoints：每一步落盘后的检查点快照（含 checkpoint_id / next 等）
  - tasks      ：节点的「开始」与「结束」事件（任务级追踪）
  - debug=True ：最详细的执行日志（在终端打印 [values]/[updates] 文本）

对应文档：
  - https://docs.langchain.com/oss/python/langgraph/streaming#checkpoints
  - https://docs.langchain.com/oss/python/langgraph/streaming#tasks
  - https://docs.langchain.com/oss/python/langgraph/streaming#debug

运行：
    uv run python examples/streaming/stream_observability.py
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ── 定义状态与节点 ────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


def n1(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(content="第一步完成")]}


def n2(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(content="第二步完成")]}


# 注意：checkpoints / tasks 模式要求编译时传入 checkpointer
builder = StateGraph(State)
builder.add_node("n1", n1)
builder.add_node("n2", n2)
builder.add_edge(START, "n1")
builder.add_edge("n1", "n2")
builder.add_edge("n2", END)
graph = builder.compile(checkpointer=MemorySaver())

INPUT = {"messages": [HumanMessage(content="开始")]}


# ── 演示 1：checkpoints —— 每步的检查点快照 ──────────────────────────────────

print("=== 演示 1：checkpoints 模式（每步落盘的检查点）===\n")

for chunk in graph.stream(
    INPUT,
    {"configurable": {"thread_id": "demo-1"}},  # checkpointer 需要 thread_id
    stream_mode="checkpoints",
    version="v2",
):
    # if chunk["type"] == "checkpoints":
    #     print(chunk["data"])

    print("--------------------------------------------------------------------------------")
    data = chunk["data"]
    checkpoint_id = data["config"]["configurable"].get("checkpoint_id", "?")
    next_nodes = data.get("next", [])
    print(f"  └─ checkpoint_id: {checkpoint_id}")
    print(f"     下一步: {next_nodes}")

print("\n说明：每个 checkpoints 事件对应一次状态落盘，可用于断点续跑/时间旅行。\n")


# ── 演示 2：tasks —— 任务级开始/结束事件 ────────────────────────────────────

print("=== 演示 2：tasks 模式（任务的开始与结束）===\n")

for chunk in graph.stream(
    INPUT,
    {"configurable": {"thread_id": "demo-2"}},
    stream_mode="tasks",
    version="v2",
):
    # if chunk["type"] == "tasks":
    #     print(chunk["data"])

    print("--------------------------------------------------------------------------------")
    task = chunk["data"]  # 任务字典
    if "input" in task:
        # 任务「开始」事件：含 input / triggers
        print(f"  ▶ 任务开始: {task['name']}  (triggers={list(task['triggers'])})")
    else:
        # 任务「结束」事件：含 result / error / interrupts
        status = "出错" if task["error"] else "成功"
        has_interrupt = bool(task["interrupts"])
        print(f"  ■ 任务结束: {task['name']}  [{status}]"
              + ("  ⚠含中断" if has_interrupt else ""))

print("\n说明：tasks 模式以「任务」为粒度，成对出现开始/结束事件，适合做追踪面板。\n")

# ── 演示 3:stream_mode="debug" —— 结构化调试事件 ─────────────────────────────
# 与 debug=True 不同,stream_mode="debug" 将调试信息作为结构化事件返回,便于程序化处理。

print("=== 演示 3:stream_mode='debug'(结构化调试事件)===\n")

for chunk in graph.stream(
    INPUT,
    {"configurable": {"thread_id": "demo-3"}},
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        print(chunk["data"])

print(
    "\n说明:stream_mode='debug' 将调试信息以结构化字典形式返回,包含 type='debug',"
    "适合需要程序化处理调试信息的场景(如日志收集、监控告警等)。"
)

# ── 演示 4：debug=True —— 最详细的执行日志 ───────────────────────────────────
# debug 模式会在终端直接打印 [values]/[updates] 文本日志,同时仍返回 stream_mode 的事件。

print("=== 演示 4:debug=True(详细执行日志 + updates 事件)===\n")

event_count = 0
for chunk in graph.stream(
    INPUT,
    {"configurable": {"thread_id": "demo-4"}},
    stream_mode="updates",  # 仍可指定 stream_mode 收集结构化事件
    debug=True,             # 额外打印详细的执行日志到终端
    version="v2",
):
    event_count += 1
    node = next(iter(chunk.get("data", {})), "?")
    print(f"  [updates 事件 {event_count}] 节点 {node}")

print(
    "\n说明:debug=True 除了返回结构化事件,还会在终端打印 [values]/[updates] 文本日志,"
    "是排障/理解执行流程的利器(注意:上方带 [values]/[updates] 的行就是 debug 输出)。\n"
)
