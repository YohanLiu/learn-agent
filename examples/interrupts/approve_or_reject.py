"""审批或拒绝（Approve or Reject）—— 使用 interrupt + Command(goto=...) 实现审批工作流。

演示内容（完整示例）：
  - interrupt() 暂停在审批节点，将操作详情暴露给调用方
  - 根据恢复值（True/False）通过 Command(goto=...) 路由到不同节点
  - proceed 节点：审批通过时执行
  - cancel 节点：审批拒绝时执行
  - 使用 stream_events(version="v3") 驱动并检测中断

运行：
    uv run python examples/interrupts/approve_or_reject.py
"""

from typing import Literal, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class ApprovalState(TypedDict):
    """审批流程状态。"""
    action_details: str                              # 待审批的操作详情
    status: Optional[Literal["pending", "approved", "rejected"]]  # 审批状态


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    """审批节点：暂停执行，等待人工决策，然后路由到对应节点。

    使用 Command(goto=...) 在恢复后路由到 proceed 或 cancel 节点。
    interrupt() 的负载会包含操作详情，方便调用方在 UI 中展示。
    """
    print(f"  [approval_node] 等待审批：{state['action_details']}")

    # 暂停并暴露问题及操作详情，调用方可渲染到 UI
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })

    print(f"  [approval_node] 收到决策：{decision}")

    # 根据恢复值路由：True → proceed，False → cancel
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState) -> dict:
    """审批通过后执行。"""
    print(f"  [proceed_node] 操作已批准：{state['action_details']}")
    return {"status": "approved"}


def cancel_node(state: ApprovalState) -> dict:
    """审批拒绝时执行。"""
    print(f"  [cancel_node] 操作已拒绝：{state['action_details']}")
    return {"status": "rejected"}


# ── 构建图 ────────────────────────────────────────────────────────────────────

builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# 生产环境应使用持久化 checkpointer（如数据库支持的实现）
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 场景 1：批准操作 ──
    print("=== 场景 1：批准转账操作 ===\n")

    config = {"configurable": {"thread_id": "approval-123"}}

    # 第 1 步：提交审批请求，图在 interrupt 处暂停
    initial = graph.stream_events(
        {"action_details": "Transfer $500", "status": "pending"},
        config=config,
        version="v3",
    )
    _ = initial.output  # 驱动流执行完毕
    print(f"\n[中断] {initial.interrupts}")
    # 输出：(Interrupt(value={'question': 'Approve this action?', 'details': 'Transfer $500'}),)

    # 第 2 步：批准（resume=True），路由到 proceed 节点
    print("\n--- 批准操作 ---")
    resumed = graph.stream_events(
        Command(resume=True),  # True → proceed
        config=config,
        version="v3",
    )
    print(f"[结果] status={resumed.output['status']}")
    # 输出：status=approved

    # ── 场景 2：拒绝操作 ──
    print("\n\n=== 场景 2：拒绝删除操作 ===\n")

    config2 = {"configurable": {"thread_id": "approval-456"}}

    initial2 = graph.stream_events(
        {"action_details": "Delete all records", "status": "pending"},
        config=config2,
        version="v3",
    )
    _ = initial2.output
    print(f"\n[中断] {initial2.interrupts}")

    # 拒绝（resume=False），路由到 cancel 节点
    print("\n--- 拒绝操作 ---")
    resumed2 = graph.stream_events(
        Command(resume=False),  # False → cancel
        config=config2,
        version="v3",
    )
    print(f"[结果] status={resumed2.output['status']}")
    # 输出：status=rejected
