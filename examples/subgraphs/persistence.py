"""子图持久化模式：per-invocation、per-thread、stateless。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

子图的 checkpointer 参数控制其内部数据在调用之间的行为：
  - checkpointer=None（默认，per-invocation）：每次调用从零开始，
    但在单次调用内支持 interrupt 和状态检查
  - checkpointer=True（per-thread）：状态跨调用累积，
    子图记住之前的对话历史
  - checkpointer=False（stateless）：完全无状态，
    不支持 interrupt、不支持状态检查、不支持恢复
"""

from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 通用子图定义 ───────────────────────────────────────────────────────────────

def add(a: list[str], b: list[str]) -> list[str]:
    """列表追加 reducer：将新值追加到已有列表。"""
    return a + b


class SubgraphState(TypedDict):
    """子图状态。

    foo: 共享字段（与父图通信）
    history: 使用 add reducer，节点返回的增量会自动追加到已有列表
    """
    foo: str
    history: Annotated[list[str], add]


def subgraph_step_a(state: SubgraphState) -> dict[str, Any]:
    """子图步骤 A：记录到 history。"""
    return {"history": ["step_a"]}


def subgraph_step_b(state: SubgraphState) -> dict[str, Any]:
    """子图步骤 B：记录到 history。"""
    return {"history": ["step_b"]}


def build_subgraph(checkpointer=None):
    """构建子图，根据参数决定 checkpointer。

    Args:
        checkpointer: None=per-invocation, True=per-thread, False=stateless
    """
    builder = StateGraph(SubgraphState)
    builder.add_node("step_a", subgraph_step_a)
    builder.add_node("step_b", subgraph_step_b)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    return builder.compile(checkpointer=checkpointer)


# ── 父图状态 ──────────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    """父图状态。

    foo: 与子图共享的字段
    history: 从子图传回的 history，用于演示中打印实际值
    """
    foo: str
    history: list[str]


# ============================================================================
# 演示 1：Per-invocation 模式（checkpointer=None）
# ============================================================================
# 特征：每次调用从零开始，但单次调用内支持 interrupt 和状态检查。

def call_subgraph_node(state: ParentState) -> dict[str, Any]:
    """在节点函数内调用子图（per-invocation 用）。"""
    subgraph = _per_invocation_subgraph
    output = subgraph.invoke({"foo": state["foo"], "history": []})
    return {"foo": output["foo"], "history": output["history"]}


_per_invocation_subgraph = None


class PerInvocationDemo:
    """Per-invocation 模式：每次调用从零开始，但单次调用内可检查状态。"""

    def run(self) -> None:
        print("=" * 70)
        print("演示 1：Per-invocation 模式（checkpointer=None，默认）")
        print("=" * 70)

        subgraph = build_subgraph(checkpointer=None)
        global _per_invocation_subgraph
        _per_invocation_subgraph = subgraph

        builder = StateGraph(ParentState)
        builder.add_node("call_subgraph", call_subgraph_node)
        builder.add_edge(START, "call_subgraph")
        builder.add_edge("call_subgraph", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config = {"configurable": {"thread_id": "1"}}

        # 第一次调用
        result = graph.invoke({"foo": "第一次", "history": []}, config)
        print(f"  第一次调用: foo='{result['foo']}'")
        print(f"  子图 history: {result['history']}")

        # 第二次调用 —— 子图每次从零开始，不会累积
        result = graph.invoke({"foo": "第二次", "history": []}, config)
        print(f"  第二次调用: foo='{result['foo']}'")
        print(f"  子图 history: {result['history']}")

        # 关键特征：父图有 checkpointer，可以查看调用历史
        history = list(graph.get_state_history(config))
        print(f"  父图检查点数量: {len(history)}（可以回溯父图状态）")
        print("  说明：每次调用子图从零开始，但父图的检查点记录完整\n")


# ============================================================================
# 演示 2：Per-thread 模式（checkpointer=True）
# ============================================================================
# 特征：状态跨调用累积，子图记住之前的对话历史。

class PerThreadDemo:
    """Per-thread 模式：子图状态跨调用累积。

    关键：子图直接添加为节点，LangGraph 正确管理子图的检查点状态。
    """

    def run(self) -> None:
        print("=" * 70)
        print("演示 2：Per-thread 模式（checkpointer=True）")
        print("=" * 70)

        subgraph = build_subgraph(checkpointer=True)

        # 父图：子图直接添加为节点，共享 foo 和 history 字段
        builder = StateGraph(ParentState)
        builder.add_node("subgraph_node", subgraph)
        builder.add_edge(START, "subgraph_node")
        builder.add_edge("subgraph_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config = {"configurable": {"thread_id": "2"}}

        # 第一次调用
        result = graph.invoke({"foo": "第一次", "history": []}, config)
        print(f"  第一次调用: foo='{result['foo']}'")
        print(f"  子图 history: {result['history']}")

        # 第二次调用 —— 子图状态跨调用累积
        result = graph.invoke({"foo": "第二次", "history": []}, config)
        print(f"  第二次调用: foo='{result['foo']}'")
        print(f"  子图 history: {result['history']}（累积！）")

        # 关键特征：可以检查子图的完整历史
        history = list(graph.get_state_history(config))
        print(f"  父图检查点数量: {len(history)}")
        print("  说明：子图状态跨调用累积，第二次包含第一次的 history\n")


# ============================================================================
# 演示 3：Stateless 模式（checkpointer=False）
# ============================================================================
# 特征：完全无状态，不支持 interrupt、状态检查、恢复。

def call_subgraph_stateless(state: ParentState) -> dict[str, Any]:
    """在节点函数内调用子图（stateless 用）。"""
    subgraph = _stateless_subgraph
    output = subgraph.invoke({"foo": state["foo"], "history": []})
    return {"foo": output["foo"], "history": output["history"]}


_stateless_subgraph = None


class StatelessDemo:
    """Stateless 模式：完全无状态。

    与 Per-invocation 的区别：
    - Per-invocation：父图有 checkpointer，可以检查/恢复父图状态
    - Stateless：父图也没有 checkpointer，无法检查任何状态，无法恢复
    """

    def run(self) -> None:
        print("=" * 70)
        print("演示 3：Stateless 模式（checkpointer=False）")
        print("=" * 70)

        subgraph = build_subgraph(checkpointer=False)
        global _stateless_subgraph
        _stateless_subgraph = subgraph

        builder = StateGraph(ParentState)
        builder.add_node("call_subgraph", call_subgraph_stateless)
        builder.add_edge(START, "call_subgraph")
        builder.add_edge("call_subgraph", END)
        # 关键：父图也没有 checkpointer
        graph = builder.compile()

        # 无需 config（没有 checkpointer）
        result = graph.invoke({"foo": "测试", "history": []})
        print(f"  调用结果: foo='{result['foo']}'")
        print(f"  子图 history: {result['history']}")

        # 关键区别：无法检查状态
        try:
            list(graph.get_state_history({"configurable": {"thread_id": "1"}}))
            print("  状态检查: 成功")
        except Exception as e:
            print(f"  状态检查: 失败（{type(e).__name__}）")

        print("  说明：无 checkpointer，无法检查/恢复状态，像普通函数调用")
        print("  对比：Per-invocation 模式的父图有 checkpointer，可以检查状态\n")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    PerInvocationDemo().run()
    PerThreadDemo().run()
    StatelessDemo().run()
