"""多层子图嵌套：父图 → 子图 → 孙图。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

演示三层子图嵌套时状态如何在不同层级之间传递。
每一层只能访问自己的状态 key，无法直接访问其他层级的 key。
状态转换需要在每层的节点函数中手动完成。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 孙图（最内层）──────────────────────────────────────────────────────────────

class GrandChildState(TypedDict):
    """孙图状态：只能访问自己的 key。"""
    my_grandchild_key: str


def grandchild_1(state: GrandChildState) -> dict[str, Any]:
    """孙图节点：追加问候语。"""
    return {"my_grandchild_key": state["my_grandchild_key"] + "，你好吗"}


# 构建孙图
grandchild_builder = StateGraph(GrandChildState)
grandchild_builder.add_node("grandchild_1", grandchild_1)
grandchild_builder.add_edge(START, "grandchild_1")
grandchild_builder.add_edge("grandchild_1", END)
grandchild_graph = grandchild_builder.compile()


# ── 子图（中间层）──────────────────────────────────────────────────────────────

class ChildState(TypedDict):
    """子图状态：只能访问自己的 key，无法访问父图或孙图的 key。"""
    my_child_key: str


def call_grandchild_graph(state: ChildState) -> dict[str, Any]:
    """子图节点：调用孙图，手动完成状态转换。

    注意：这里只能访问子图的 my_child_key，
    父图的 my_key 和孙图的 my_grandchild_key 都不可见。
    """
    # 子图状态 → 孙图输入
    grandchild_input = {"my_grandchild_key": state["my_child_key"]}
    # 调用孙图
    grandchild_output = grandchild_graph.invoke(grandchild_input)
    # 孙图输出 → 子图状态
    return {"my_child_key": grandchild_output["my_grandchild_key"] + " 今天？"}


# 构建子图
child_builder = StateGraph(ChildState)
child_builder.add_node("child_1", call_grandchild_graph)
child_builder.add_edge(START, "child_1")
child_builder.add_edge("child_1", END)
child_graph = child_builder.compile()


# ── 父图（最外层）──────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    """父图状态：只能访问自己的 key。"""
    my_key: str


def parent_1(state: ParentState) -> dict[str, Any]:
    """父图前置节点：添加前缀。"""
    return {"my_key": "嗨 " + state["my_key"]}


def parent_2(state: ParentState) -> dict[str, Any]:
    """父图后置节点：添加后缀。"""
    return {"my_key": state["my_key"] + " 再见！"}


def call_child_graph(state: ParentState) -> dict[str, Any]:
    """父图节点：调用子图，手动完成状态转换。"""
    # 父图状态 → 子图输入
    child_input = {"my_child_key": state["my_key"]}
    # 调用子图
    child_output = child_graph.invoke(child_input)
    # 子图输出 → 父图状态
    return {"my_key": child_output["my_child_key"]}


class MultiLevelDemo:
    """演示多层子图嵌套（父 → 子 → 孙）。"""

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> StateGraph:
        builder = StateGraph(ParentState)
        builder.add_node("parent_1", parent_1)
        builder.add_node("child", call_child_graph)
        builder.add_node("parent_2", parent_2)
        builder.add_edge(START, "parent_1")
        builder.add_edge("parent_1", "child")
        builder.add_edge("child", "parent_2")
        builder.add_edge("parent_2", END)
        return builder.compile()

    def run(self) -> None:
        print("=== 多层子图嵌套（父 → 子 → 孙）===\n")

        result = self._graph.invoke({"my_key": "小明"})
        print(f"最终结果: my_key = {result['my_key']}")
        print("\n执行路径:")
        print("  父图 parent_1: '嗨 小明'")
        print("  → 子图 child_1: 调用孙图")
        print("    → 孙图 grandchild_1: '嗨 小明，你好吗'")
        print("    ← 子图拼接: '嗨 小明，你好吗 今天？'")
        print("  ← 父图接收: my_key = '嗨 小明，你好吗 今天？'")
        print("  父图 parent_2: '嗨 小明，你好吗 今天？ 再见！'")


if __name__ == "__main__":
    MultiLevelDemo().run()
