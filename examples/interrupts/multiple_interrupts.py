"""处理多个中断（Handling Multiple Interrupts）。

演示并行分支同时触发中断时，如何在一次恢复中处理所有中断：
  - 两个并行节点各自调用 interrupt() 暂停
  - stream.interrupts 包含所有待处理的中断负载（Interrupt 对象列表）
  - 使用 {interrupt.id: resume_value} 映射一次性恢复所有中断
  - 确保每个中断的响应与其对应的中断正确配对

运行：
    uv run python examples/interrupts/multiple_interrupts.py
"""

import operator
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class State(TypedDict):
    """包含字符串列表的状态，使用 operator.add 进行归约（自动追加）。"""
    vals: Annotated[list[str], operator.add]


# ── 定义两个并行节点 ──────────────────────────────────────────────────────────

def node_a(state: State) -> dict:
    """节点 A：暂停并询问 question_a。"""
    print("  [node_a] 触发 interrupt，等待回答...")
    answer = interrupt("question_a")
    print(f"  [node_a] 收到回答：{answer}")
    return {"vals": [f"a:{answer}"]}


def node_b(state: State) -> dict:
    """节点 B：暂停并询问 question_b。"""
    print("  [node_b] 触发 interrupt，等待回答...")
    answer = interrupt("question_b")
    print(f"  [node_b] 收到回答：{answer}")
    return {"vals": [f"b:{answer}"]}


# ── 构建图：两个节点从 START 并行启动 ──────────────────────────────────────────

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")    # START 同时触发 a 和 b（并行）
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    # ── 第 1 步：启动图，两个并行节点都会触发 interrupt() 暂停 ──
    print("=== 第 1 步：启动图，两个并行节点同时中断 ===\n")

    stream = graph.stream_events({"vals": []}, config, version="v3")
    _ = stream.output  # 驱动流执行完毕（两个节点均暂停）

    # stream.interrupts 包含两个 Interrupt 对象，每个都有 id 和 value
    print(f"\n[中断] 检测到 {len(stream.interrupts)} 个待处理中断：")
    for i in stream.interrupts:
        print(f"  - id={i.id}, value={i.value!r}")

    # ── 第 2 步：构建 resume_map，一次性恢复所有中断 ──
    print("\n=== 第 2 步：构建映射并一次性恢复所有中断 ===\n")

    # 关键：使用 {interrupt.id: 对应恢复值} 的字典，确保每个中断得到正确的响应
    resume_map = {
        i.id: f"answer for {i.value}" for i in stream.interrupts
    }
    print(f"[恢复映射] {resume_map}")

    resumed = graph.stream_events(
        Command(resume=resume_map),  # 传入字典，键为 interrupt id
        config=config,
        version="v3",
    )

    final = resumed.output
    print(f"\n[最终状态] vals={final['vals']}")
    # 输出示例：['a:answer for question_a', 'b:answer for question_b']
