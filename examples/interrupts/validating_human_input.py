"""验证人类输入（Validating Human Input）。

演示内容（完整示例）：
  - 使用 interrupt() 收集用户输入，并通过条件边（conditional edge）验证
  - 关键原则：每个节点每次调用 interrupt() 恰好一次（不要在节点内 while 循环）
  - 若输入无效：将错误问题存入状态 pending_question，条件边路由回同一节点
  - 若输入有效：条件边路由到 END，图正常结束
  - 每次恢复只执行一次 interrupt()，避免指数级重复执行

注意：
  ❌ 不要在节点内用 while True + interrupt() 循环验证
  ✅ 使用条件边 + pending_question 状态字段实现重试

运行：
    uv run python examples/interrupts/validating_human_input.py
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class FormState(TypedDict):
    """表单收集状态。"""
    age: int | None               # 收集到的年龄，None 表示尚未收集
    pending_question: str | None  # 待提问的问题（含错误提示），None 表示使用默认问题


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def get_age_node(state: FormState) -> dict:
    """收集年龄节点：每次调用恰好 interrupt() 一次。

    - 首次执行时使用默认问题 "What is your age?"
    - 重试时使用 pending_question 中的错误提示问题
    - 若回答无效（非正整数），返回更新的 pending_question，由条件边路由回本节点
    - 若回答有效，返回 age 值，条件边路由到 END
    """
    # 使用 pending_question 或默认问题
    question = state.get("pending_question") or "What is your age?"
    print(f"  [get_age_node] 提问：{question!r}")

    # 每次节点调用恰好 interrupt() 一次（关键！）
    answer = interrupt(question)

    print(f"  [get_age_node] 收到回答：{answer!r}")

    # 验证回答
    if isinstance(answer, int) and answer > 0:
        print(f"  [get_age_node] ✓ 有效年龄：{answer}")
        return {"age": answer, "pending_question": None}

    # 无效：更新 pending_question，条件边会把我们路由回这里重新提问
    error_msg = f"'{answer}' is not a valid age. Please enter a positive number."
    print(f"  [get_age_node] ✗ 无效回答，将重新提问")
    return {"pending_question": error_msg}


def route(state: FormState) -> str:
    """条件路由：年龄有效则结束，否则回到 collect_age 节点重新收集。"""
    if state.get("age") is not None:
        return END          # 年龄有效，图结束
    return "collect_age"    # 年龄无效，回到收集节点


# ── 构建图 ────────────────────────────────────────────────────────────────────

builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_conditional_edges("collect_age", route)  # 动态路由

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "form-1"}}

    # ── 第 1 步：初始提交，触发第一次 interrupt ──
    print("=== 第 1 步：启动表单，询问年龄 ===\n")

    first = graph.stream_events(
        {"age": None, "pending_question": None},
        config=config,
        version="v3",
    )
    _ = first.output
    print(f"\n[中断] {first.interrupts}")
    # 输出：(Interrupt(value='What is your age?'),)

    # ── 第 2 步：提供无效数据（字符串），触发重新提问 ──
    print("\n=== 第 2 步：提供无效回答 'thirty' ===\n")

    retry = graph.stream_events(
        Command(resume="thirty"),  # 字符串，不是有效年龄
        config=config,
        version="v3",
    )
    _ = retry.output
    print(f"\n[中断] {retry.interrupts}")
    # 输出：(Interrupt(value="'thirty' is not a valid age. Please enter a positive number."),)

    # ── 第 3 步：提供有效数据（整数），图正常结束 ──
    print("\n=== 第 3 步：提供有效回答 30 ===\n")

    final = graph.stream_events(
        Command(resume=30),  # 正整数，有效
        config=config,
        version="v3",
    )

    result = final.output
    print(f"\n[最终状态] age={result['age']}")
    # 输出：age=30
