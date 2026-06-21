"""状态流式输出（Stream Graph State: updates / values）。

对比 graph.stream 的两种状态流模式（version="v2"）：
  - updates：只返回每个节点产出的「增量」，未变的键不出现
  - values ：每步执行后返回完整的状态快照（累加视角）

对应文档：https://docs.langchain.com/oss/python/langgraph/streaming#graph-state

运行：
    uv run python examples/streaming/stream_state.py
"""

from typing import Annotated, Any, TypedDict
from operator import add

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph


# ── 定义状态与节点 ────────────────────────────────────────────────────────────

class State(TypedDict, total=False):
    """多键状态，便于对比 updates / values 的差异。"""
    messages: Annotated[list, add]   # 消息累积
    steps: Annotated[list[str], add]  # 执行步骤记录
    final: str                        # 最终结论


def step_one(state: State) -> dict[str, Any]:
    return {"steps": ["第一步：理解问题"], "messages": [AIMessage(content="已理解")]}


def step_two(state: State) -> dict[str, Any]:
    return {"steps": ["第二步：分析思路"], "messages": [AIMessage(content="已分析")]}


def step_three(state: State) -> dict[str, Any]:
    return {"steps": ["第三步：得出结论"], "final": "这就是答案"}


workflow = StateGraph(State)
workflow.add_node("step_one", step_one)
workflow.add_node("step_two", step_two)
workflow.add_node("step_three", step_three)
workflow.add_edge(START, "step_one")
workflow.add_edge("step_one", "step_two")
workflow.add_edge("step_two", "step_three")
workflow.add_edge("step_three", END)
graph = workflow.compile()

INPUT = {"messages": [HumanMessage(content="解释一下流式输出")], "steps": []}


# ── 演示 1：updates —— 只看增量 ──────────────────────────────────────────────

print("=== 演示 1：updates 模式（只看每步的增量）===\n")

for chunk in graph.stream(INPUT, stream_mode="updates", version="v2"):
    # v2 下 chunk 是完整信封；data 形如 {节点名: 该节点返回的更新字典}
    for node_name, state_update in chunk["data"].items():
        print(f"[{node_name}] 产出键: {list(state_update.keys())}")
        for key, value in state_update.items():
            print(f"    └─ {key}: {value}")

print("\n说明：updates 只包含「这一步新产生/改变的键」，比如 final 到第三步才出现。\n")


# ── 演示 2：values —— 看完整快照 ─────────────────────────────────────────────

print("=== 演示 2：values 模式（每步的完整状态快照）===\n")

step = 0
for chunk in graph.stream(INPUT, stream_mode="values", version="v2"):
    step += 1
    # v2 下 chunk 是完整信封；data 是当前完整状态（包含历史累积的所有键）
    snapshot = chunk["data"]
    print(f"[第 {step} 步后] 完整快照：")
    print(f"    └─ steps : {snapshot.get('steps', [])}")
    print(f"    └─ final : {snapshot.get('final', '<尚未产出>')}")
    print(f"    └─ 消息数: {len(snapshot.get('messages', []))}")

print("\n说明：values 每步都返回「累积到此刻」的完整状态，key 越来越齐全。")


# ── 演示 3：两种模式一起消费，直观对比 ───────────────────────────────────────

print("=== 演示 3：[updates, values] 并排对比 ===\n")

for chunk in graph.stream(INPUT, stream_mode=["updates", "values"], version="v2"):
    match chunk["type"]:
        case "updates":
            node = next(iter(chunk["data"]), "?")
            keys = list(chunk["data"][node].keys())
            print(f"updates → {node} 新增键: {keys}")
        case "values":
            snapshot_keys = list(chunk["data"].keys())
            print(f"values  → 快照键: {snapshot_keys}")
