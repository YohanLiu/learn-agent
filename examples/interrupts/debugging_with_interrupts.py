"""使用中断调试（Debugging with Interrupts）。

演示内容：
  - 静态中断（Static Interrupts）：在编译时或运行时设置断点，逐步调试图执行
  - interrupt_before：在指定节点执行之前暂停
  - interrupt_after：在指定节点执行完之后暂停
  - 通过 graph.invoke(None, config=config) 恢复到下一个断点

注意：
  - 静态中断不推荐用于 human-in-the-loop 工作流
  - 生产环境请使用 interrupt() 函数（动态中断）
  - 静态中断主要用于调试和测试，类似 IDE 断点

运行：
    uv run python examples/interrupts/debugging_with_interrupts.py
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class DebugState(TypedDict):
    """调试演示状态。"""
    step: str          # 当前步骤描述
    counter: int       # 计数器


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def node_a(state: DebugState) -> dict:
    """节点 A：第一步处理。"""
    print(f"  [node_a] 执行，当前 counter={state['counter']}")
    return {"step": "A executed", "counter": state["counter"] + 1}


def node_b(state: DebugState) -> dict:
    """节点 B：第二步处理。"""
    print(f"  [node_b] 执行，当前 counter={state['counter']}")
    return {"step": "B executed", "counter": state["counter"] + 1}


def node_c(state: DebugState) -> dict:
    """节点 C：第三步处理。"""
    print(f"  [node_c] 执行，当前 counter={state['counter']}")
    return {"step": "C executed", "counter": state["counter"] + 1}


# ── 构建图 ────────────────────────────────────────────────────────────────────

builder = StateGraph(DebugState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)


# ═══════════════════════════════════════════════════════════════════════════════
# 演示 1：在编译时设置静态中断（interrupt_before / interrupt_after）
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("演示 1：编译时设置静态中断（interrupt_before + interrupt_after）")
    print("=" * 60)

    checkpointer = InMemorySaver()

    # 编译时指定断点：
    #   interrupt_before=["node_a"]  → 在 node_a 执行前暂停
    #   interrupt_after=["node_b"]   → 在 node_b 执行完后暂停
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["node_a"],
        interrupt_after=["node_b"],
    )

    config = {"configurable": {"thread_id": "debug-1"}}

    # 第 1 步：启动图，在 node_a 之前暂停（interrupt_before）
    print("\n--- 第 1 步：启动图（应在 node_a 之前暂停）---")
    result = graph.invoke(
        {"step": "initial", "counter": 0},
        config=config,
    )
    print(f"[状态] step={result['step']}, counter={result['counter']}")
    print("  → node_a 尚未执行，counter 仍为 0")

    # 第 2 步：恢复，执行 node_a，然后在 node_b 之后暂停（interrupt_after）
    print("\n--- 第 2 步：恢复（执行 node_a + node_b，在 node_b 之后暂停）---")
    result = graph.invoke(None, config=config)  # 传入 None 恢复执行
    print(f"[状态] step={result['step']}, counter={result['counter']}")
    print("  → node_a 和 node_b 已执行，counter 应为 2")

    # 第 3 步：恢复，执行 node_c，图结束
    print("\n--- 第 3 步：恢复（执行 node_c，图结束）---")
    result = graph.invoke(None, config=config)
    print(f"[最终状态] step={result['step']}, counter={result['counter']}")
    print("  → 所有节点执行完毕，counter 应为 3")


# ═══════════════════════════════════════════════════════════════════════════════
# 演示 2：在运行时设置静态中断（通过 invoke 参数传入）
# ═══════════════════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 60)
    print("演示 2：运行时设置静态中断（通过 invoke 参数动态指定）")
    print("=" * 60)

    checkpointer2 = InMemorySaver()
    graph2 = builder.compile(checkpointer=checkpointer2)

    config2 = {"configurable": {"thread_id": "debug-2"}}

    # 第 1 步：在运行时指定 interrupt_before=["node_b"]，在 node_b 之前暂停
    print("\n--- 第 1 步：启动图（在 node_b 之前暂停）---")
    result = graph2.invoke(
        {"step": "initial", "counter": 0},
        config=config2,
        interrupt_before=["node_b"],  # 运行时指定断点
    )
    print(f"[状态] step={result['step']}, counter={result['counter']}")
    print("  → node_a 已执行，node_b 尚未执行，counter 应为 1")

    # 第 2 步：恢复执行 node_b，然后继续到 node_c（无更多断点）
    print("\n--- 第 2 步：恢复（执行 node_b + node_c，图结束）---")
    result = graph2.invoke(None, config=config2)
    print(f"[最终状态] step={result['step']}, counter={result['counter']}")
    print("  → 所有节点执行完毕，counter 应为 3")

    print("\n✅ 静态中断调试演示完毕！")
    print("\n提示：可使用 LangSmith Studio 在 UI 中可视化设置静态断点。")
