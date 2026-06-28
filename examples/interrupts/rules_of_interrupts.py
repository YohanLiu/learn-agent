"""中断使用规则（Rules of Interrupts）。

演示 LangGraph interrupt() 的重要使用规则与最佳实践：
  1. 不要用 try/except 包裹 interrupt() 调用（会捕获内部异常导致中断失效）
  2. 不要在节点内条件跳过或重排 interrupt() 调用（顺序必须一致）
  3. 不要向 interrupt() 传入复杂对象（函数、类实例等不可序列化的值）
  4. interrupt() 之前的副作用必须是幂等的（节点恢复时会从头重新执行）

每个规则都包含正确用法（✅）和错误用法（❌）的对比说明。

运行：
    uv run python examples/interrupts/rules_of_interrupts.py
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 公共状态定义 ──────────────────────────────────────────────────────────────

class State(TypedDict):
    name: str | None
    age: int | None
    city: str | None
    result: str | None


# ═══════════════════════════════════════════════════════════════════════════════
# 规则 1：不要用 try/except 包裹 interrupt() 调用
# ═══════════════════════════════════════════════════════════════════════════════

def correct_try_except(state: State) -> dict:
    """✅ 正确：捕获特定异常类型，不会捕获 interrupt 的内部异常。"""
    try:
        name = interrupt("What's your name?")
        # 可能抛出特定异常的操作
        data = f"processed:{name}"
    except ValueError as e:
        # 只捕获 ValueError，不影响 interrupt 机制
        print(f"  ValueError: {e}")
        data = "error"
    return {"name": name if "name" in dir() else None, "result": data}


def correct_separate(state: State) -> dict:
    """✅ 正确：先 interrupt，再单独处理可能出错的逻辑。"""
    name = interrupt("What's your name?")
    try:
        # 可能出错的操作单独放在 try 块
        data = f"processed:{name}"
    except Exception as e:
        print(f"  Error: {e}")
        data = "error"
    return {"name": name, "result": data}


# ❌ 错误示例（仅展示，不实际运行）：
# def wrong_try_except(state: State) -> dict:
#     """❌ 错误：裸 except 会捕获 interrupt 的内部异常，导致中断失效。"""
#     try:
#         interrupt("What's your name?")  # interrupt 通过异常机制暂停
#     except Exception as e:              # ← 会捕获 interrupt 异常！
#         print(e)
#     return state


# ═══════════════════════════════════════════════════════════════════════════════
# 规则 2：不要在节点内条件跳过或重排 interrupt() 调用
# ═══════════════════════════════════════════════════════════════════════════════

def correct_consistent_order(state: State) -> dict:
    """✅ 正确：interrupt 调用顺序每次都保持一致。"""
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")
    return {"name": name, "age": age, "city": city}


# ❌ 错误示例（仅展示）：
# def wrong_skip_interrupt(state: State) -> dict:
#     """❌ 错误：条件跳过 interrupt 会导致顺序错乱，恢复时索引不匹配。"""
#     name = interrupt("What's your name?")
#     if state.get("needs_age"):          # ← 可能有时跳过有时不跳过
#         age = interrupt("What's your age?")
#     city = interrupt("What's your city?")
#     return {"name": name, "city": city}
#
# def wrong_loop_interrupt(state: State) -> dict:
#     """❌ 错误：循环次数不确定，导致 interrupt 调用次数变化。"""
#     results = []
#     for item in state.get("dynamic_list", []):  # ← 列表可能变化
#         result = interrupt(f"Approve {item}?")
#         results.append(result)
#     return {"result": str(results)}


# ═══════════════════════════════════════════════════════════════════════════════
# 规则 3：不要向 interrupt() 传入复杂对象
# ═══════════════════════════════════════════════════════════════════════════════

def correct_simple_values(state: State) -> dict:
    """✅ 正确：传入简单 JSON 可序列化类型。"""
    name = interrupt("What's your name?")   # 字符串
    count = interrupt(42)                    # 整数
    approved = interrupt(True)               # 布尔值
    return {"name": name, "age": count, "result": str(approved)}


def correct_structured_data(state: State) -> dict:
    """✅ 正确：传入包含简单值的字典。"""
    response = interrupt({
        "question": "Enter user details",
        "fields": ["name", "email", "age"],
        "current_values": {"name": "Alice"},
    })
    return {"result": str(response)}


# ❌ 错误示例（仅展示）：
# def wrong_function_value(state: State) -> dict:
#     """❌ 错误：函数不可序列化。"""
#     def validate(v): return len(v) > 0
#     response = interrupt({
#         "question": "What's your name?",
#         "validator": validate,  # ← 函数不可序列化！
#     })
#     return {"name": response}
#
# class DataProcessor:
#     def __init__(self, config): self.config = config
#
# def wrong_class_instance(state: State) -> dict:
#     """❌ 错误：类实例不可序列化。"""
#     processor = DataProcessor({"mode": "strict"})
#     response = interrupt({
#         "question": "Enter data",
#         "processor": processor,  # ← 类实例不可序列化！
#     })
#     return {"result": response}


# ═══════════════════════════════════════════════════════════════════════════════
# 规则 4：interrupt() 之前的副作用必须是幂等的
# ═══════════════════════════════════════════════════════════════════════════════

def correct_idempotent(state: State) -> dict:
    """✅ 正确：使用幂等操作（upsert）或将副作用放在 interrupt() 之后。"""
    # 幂等操作：upsert（插入或更新），多次执行结果相同
    # db.upsert_user(user_id=state["user_id"], status="pending_approval")

    approved = interrupt("Approve this change?")

    # 副作用放在 interrupt() 之后，只在批准后执行一次
    if approved:
        print("  [side_effect] 创建审计日志（仅执行一次）")
        # db.create_audit_log(user_id=..., action="approved")

    return {"result": "approved" if approved else "rejected"}


def correct_separate_nodes(state: State) -> dict:
    """✅ 正确：将副作用分离到单独节点，interrupt 在专用节点中处理。"""
    approved = interrupt("Approve this change?")
    return {"result": "approved" if approved else "rejected"}


# ❌ 错误示例（仅展示）：
# def wrong_non_idempotent(state: State) -> dict:
#     """❌ 错误：非幂等操作在 interrupt 之前执行，恢复时会重复创建记录。"""
#     audit_id = db.create_audit_log({      # ← 每次恢复都会创建新记录！
#         "user_id": state["user_id"],
#         "action": "pending_approval",
#     })
#     approved = interrupt("Approve?")
#     return {"result": str(audit_id)}


# ═══════════════════════════════════════════════════════════════════════════════
# 演示：规则 1 —— 正确的 try/except 用法
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("规则 1：正确的异常处理（捕获特定异常，不影响 interrupt）")
    print("=" * 60)

    builder1 = StateGraph(State)
    builder1.add_node("node", correct_separate)
    builder1.add_edge(START, "node")
    builder1.add_edge("node", END)
    graph1 = builder1.compile(checkpointer=InMemorySaver())

    config1 = {"configurable": {"thread_id": "rule1"}}

    stream = graph1.stream_events(
        {"name": None, "age": None, "city": None, "result": None},
        config=config1, version="v3",
    )
    _ = stream.output
    print(f"\n[中断] {stream.interrupts}")

    resumed = graph1.stream_events(
        Command(resume="Alice"),
        config=config1, version="v3",
    )
    print(f"[结果] name={resumed.output['name']}, result={resumed.output['result']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 演示：规则 2 —— 一致的 interrupt 调用顺序
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("规则 2：一致的 interrupt 调用顺序（三次 interrupt 依次恢复）")
    print("=" * 60)

    builder2 = StateGraph(State)
    builder2.add_node("node", correct_consistent_order)
    builder2.add_edge(START, "node")
    builder2.add_edge("node", END)
    graph2 = builder2.compile(checkpointer=InMemorySaver())

    config2 = {"configurable": {"thread_id": "rule2"}}

    # 第 1 次 interrupt
    s1 = graph2.stream_events(
        {"name": None, "age": None, "city": None, "result": None},
        config=config2, version="v3",
    )
    _ = s1.output
    print(f"\n[第 1 次中断] {s1.interrupts[0].value}")

    # 第 2 次 interrupt
    s2 = graph2.stream_events(Command(resume="Alice"), config=config2, version="v3")
    _ = s2.output
    print(f"[第 2 次中断] {s2.interrupts[0].value}")

    # 第 3 次 interrupt
    s3 = graph2.stream_events(Command(resume=30), config=config2, version="v3")
    _ = s3.output
    print(f"[第 3 次中断] {s3.interrupts[0].value}")

    # 最终恢复
    s4 = graph2.stream_events(Command(resume="Beijing"), config=config2, version="v3")
    result = s4.output
    print(f"\n[最终结果] name={result['name']}, age={result['age']}, city={result['city']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 演示：规则 3 —— 使用简单可序列化的值
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("规则 3：传入简单 JSON 可序列化类型（字符串、整数、布尔值、字典）")
    print("=" * 60)

    builder3 = StateGraph(State)
    builder3.add_node("node", correct_simple_values)
    builder3.add_edge(START, "node")
    builder3.add_edge("node", END)
    graph3 = builder3.compile(checkpointer=InMemorySaver())

    config3 = {"configurable": {"thread_id": "rule3"}}

    s1 = graph3.stream_events(
        {"name": None, "age": None, "city": None, "result": None},
        config=config3, version="v3",
    )
    _ = s1.output
    print(f"\n[中断 1 - 字符串] {s1.interrupts[0].value}")

    s2 = graph3.stream_events(Command(resume="Bob"), config=config3, version="v3")
    _ = s2.output
    print(f"[中断 2 - 整数] {s2.interrupts[0].value}")

    s3 = graph3.stream_events(Command(resume=25), config=config3, version="v3")
    _ = s3.output
    print(f"[中断 3 - 布尔值] {s3.interrupts[0].value}")

    s4 = graph3.stream_events(Command(resume=True), config=config3, version="v3")
    result = s4.output
    print(f"\n[最终结果] name={result['name']}, age={result['age']}, result={result['result']}")

    print("\n✅ 所有规则演示完毕！")
