"""中断恢复（Resume After Interrupt）。

演示 human-in-the-loop 场景下的中断与恢复：
  - 图在执行过程中通过 interrupt() 暂停，等待人类输入
  - 使用 stream.interrupted 和 stream.interrupts 检查中断状态
  - 通过 Command(resume=...) 恢复执行
  - 需要 checkpointer + thread_id 配置

运行：
    uv run python examples/event_streaming/resume_after_interrupt.py
"""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class ApprovalState(TypedDict):
    """审批流程状态。"""
    request: str           # 用户的请求内容
    risk_level: str        # 风险等级评估结果
    approved: bool         # 是否审批通过
    result: str            # 最终执行结果


# ── 构建带中断的审批流程 ─────────────────────────────────────────────────────

def assess_risk(state: ApprovalState) -> dict[str, Any]:
    """第 1 步：评估风险等级。"""
    request = state["request"]
    # 简单规则：包含"大额"或"删除"的请求视为高风险
    if any(kw in request for kw in ["大额", "删除", "转账"]):
        risk = "high"
    else:
        risk = "low"
    print(f"  [assess_risk] 请求「{request}」→ 风险等级: {risk}")
    return {"risk_level": risk}


def human_approval(state: ApprovalState) -> dict[str, Any]:
    """第 2 步：人工审批（通过 interrupt 中断等待人类输入）。"""
    risk = state["risk_level"]
    print(f"  [human_approval] 风险等级={risk}，等待人工审批...")

    # interrupt 会暂停图的执行，等待外部传入审批决策
    decision = interrupt({
        "question": f"请求「{state['request']}」风险评估为 {risk}，是否批准？",
        "options": ["approve", "reject"],
    })

    approved = decision.get("action") == "approve"
    print(f"  [human_approval] 人工决策: {decision} → approved={approved}")
    return {"approved": approved}


def execute_request(state: ApprovalState) -> dict[str, Any]:
    """第 3 步：执行请求（仅在审批通过后执行）。"""
    if state["approved"]:
        result = f"已执行：{state['request']}"
        print(f"  [execute_request] {result}")
    else:
        result = f"已拒绝：{state['request']}"
        print(f"  [execute_request] {result}")
    return {"result": result}


workflow = StateGraph(ApprovalState)
workflow.add_node("assess_risk", assess_risk)
workflow.add_node("human_approval", human_approval)
workflow.add_node("execute_request", execute_request)

workflow.add_edge(START, "assess_risk")
workflow.add_edge("assess_risk", "human_approval")
workflow.add_edge("human_approval", "execute_request")
workflow.add_edge("execute_request", END)


# ── 编译时配置 MemorySaver checkpointer ───────────────────────────────────────

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


# ── 演示：中断 → 检查 → 恢复 ──────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "interrupt-demo-1"}}

    # ── 第 1 阶段：提交请求，图执行到 interrupt 处暂停 ──
    print("=== 第 1 阶段：提交高风险请求 ===\n")

    stream = graph.stream_events(
        {
            "request": "大额转账 100 万元",
            "risk_level": "",
            "approved": False,
            "result": "",
        },
        version="v3",
        config=config,
    )

    # 消费流式消息
    for message in stream.messages:
        text = str(message.text)
        if text.strip():
            print(f"  [消息] {text}")

    # 检查是否被中断
    if stream.interrupted:
        print(f"\n[中断] 图已暂停！")
        print(f"[中断信息] {stream.interrupts}")
    else:
        print("\n[完成] 图未被中断，直接完成")

    # ── 第 2 阶段：人工审批后恢复执行 ──
    print("\n=== 第 2 阶段：批准并恢复执行 ===\n")

    stream2 = graph.stream_events(
        Command(resume={"action": "approve"}),
        version="v3",
        config=config,
    )

    # 消费恢复后的流
    for message in stream2.messages:
        text = str(message.text)
        if text.strip():
            print(f"  [消息] {text}")

    # 获取最终状态
    final = stream2.output
    print(f"\n[最终状态]")
    print(f"  request: {final['request']}")
    print(f"  risk_level: {final['risk_level']}")
    print(f"  approved: {final['approved']}")
    print(f"  result: {final['result']}")

    # ── 第 3 阶段：演示拒绝场景 ──
    print("\n\n=== 第 3 阶段：提交新请求并拒绝 ===\n")

    config2 = {"configurable": {"thread_id": "interrupt-demo-2"}}

    stream3 = graph.stream_events(
        {
            "request": "删除所有数据",
            "risk_level": "",
            "approved": False,
            "result": "",
        },
        version="v3",
        config=config2,
    )

    for _ in stream3.messages:
        pass

    if stream3.interrupted:
        print(f"[中断] 图已暂停，中断信息: {stream3.interrupts}")

        # 拒绝
        stream4 = graph.stream_events(
            Command(resume={"action": "reject"}),
            version="v3",
            config=config2,
        )
        final2 = stream4.output
        print(f"[最终状态] approved={final2['approved']}, result={final2['result']}")
