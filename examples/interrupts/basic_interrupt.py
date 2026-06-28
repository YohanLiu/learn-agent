"""基础中断（Basic Interrupt）—— 使用 interrupt() 暂停图执行并通过 Command(resume=...) 恢复。

演示内容：
  1. 在节点中调用 interrupt() 暂停执行，将负载（payload）返回给调用方
  2. 使用 stream_events(version="v3") 驱动图，通过 stream.interrupted / stream.interrupts 检测中断
  3. 通过 Command(resume=...) 传入恢复值，恢复图执行
  4. 恢复值会成为 interrupt() 调用的返回值

运行：
    uv run python examples/interrupts/basic_interrupt.py
"""

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class State(TypedDict):
    """简单审批状态。"""
    input_data: str
    approved: Optional[bool]


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def approval_node(state: State) -> dict:
    """暂停图执行，等待外部审批决策。

    interrupt() 被调用时：
      1. 图执行立即暂停
      2. 当前状态通过 checkpointer 保存
      3. 传入 interrupt() 的值作为负载返回给调用方（stream.interrupts）
      4. 图无限期等待，直到通过 Command(resume=...) 恢复
      5. 恢复值成为 interrupt() 的返回值
    """
    print(f"  [approval_node] 等待审批，输入数据：{state['input_data']}")

    # 暂停并返回问题负载，恢复时 resume 值会赋给 approved
    approved = interrupt("Do you approve this action?")

    print(f"  [approval_node] 收到审批结果：{approved}")
    return {"approved": approved}


# ── 构建图 ────────────────────────────────────────────────────────────────────

workflow = StateGraph(State)
workflow.add_node("approval", approval_node)
workflow.add_edge(START, "approval")
workflow.add_edge("approval", END)

# 必须配置 checkpointer + thread_id，interrupt 才能正常工作
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "thread-1"}}

    # ── 第 1 步：初始运行，图在 interrupt 处暂停 ──
    print("=== 第 1 步：提交数据，等待中断 ===\n")

    stream = graph.stream_events(
        {"input_data": "data", "approved": None},
        config=config,
        version="v3",
    )

    # stream.output 会等待流结束或中断，驱动流执行完毕
    _ = stream.output

    # 检查是否发生中断
    if stream.interrupted:
        print(f"\n[中断] 图已暂停！")
        print(f"[中断负载] {stream.interrupts}")
        # 输出类似：(Interrupt(value='Do you approve this action?'),)

    # ── 第 2 步：用 Command(resume=...) 恢复执行 ──
    print("\n=== 第 2 步：批准并恢复执行 ===\n")

    # resume 值会成为 interrupt() 调用的返回值
    resumed = graph.stream_events(
        Command(resume=True),  # True 传递给 approved = interrupt(...)
        config=config,
        version="v3",
    )
    final_state = resumed.output

    print(f"\n[最终状态] approved={final_state['approved']}")
    # 输出：approved=True
