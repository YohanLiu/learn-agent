"""审核并编辑状态（Review and Edit State）。

演示内容（完整示例）：
  - 让审核者在 interrupt 中查看 LLM 生成的内容
  - 审核者可以修改内容后通过 resume 传回
  - interrupt() 的负载包含当前内容和编辑指引
  - 恢复后更新状态中的 generated_text 为编辑后的版本

运行：
    uv run python examples/interrupts/review_and_edit.py
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class ReviewState(TypedDict):
    """审核流程状态。"""
    generated_text: str  # LLM 生成的文本，可由审核者编辑


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def review_node(state: ReviewState) -> dict:
    """审核节点：暂停并将当前内容展示给审核者，等待编辑后的版本。

    interrupt() 的负载包含：
      - instruction：审核指引，告知审核者该做什么
      - content：当前生成的内容

    恢复时 resume 值（编辑后的文本）会成为 interrupt() 的返回值，
    然后更新到状态的 generated_text 字段。
    """
    print(f"  [review_node] 当前内容：{state['generated_text']!r}")
    print("  [review_node] 等待审核者编辑...")

    # 暂停，将内容和指引暴露给审核者
    updated = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"],
    })

    print(f"  [review_node] 收到编辑后的内容：{updated!r}")
    return {"generated_text": updated}


# ── 构建图 ────────────────────────────────────────────────────────────────────

builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-42"}}

    # ── 第 1 步：提交初始生成的文本，触发中断 ──
    print("=== 第 1 步：提交初始草稿，等待审核 ===\n")

    initial = graph.stream_events(
        {"generated_text": "Initial draft"},
        config=config,
        version="v3",
    )
    _ = initial.output  # 驱动流执行完毕
    print(f"\n[中断] {initial.interrupts}")
    # 输出：(Interrupt(value={'instruction': 'Review and edit this content',
    #                           'content': 'Initial draft'}),)

    # ── 第 2 步：审核者修改内容后恢复 ──
    print("\n=== 第 2 步：审核者提交编辑后的内容 ===\n")

    # resume 值（编辑后的文本）会成为 interrupt() 的返回值
    final = graph.stream_events(
        Command(resume="Improved draft after review"),
        config=config,
        version="v3",
    )

    result = final.output
    print(f"\n[最终状态] generated_text={result['generated_text']!r}")
    # 输出：'Improved draft after review'
