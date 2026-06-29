"""在节点内调用子图：父图与子图使用不同的状态模式。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

当父图和子图的状态没有共享的 key 时，需要在节点函数中手动完成状态转换：
  1. 将父图状态映射为子图输入
  2. 调用子图 invoke
  3. 将子图输出映射回父图状态

这是多智能体系统中最常用的模式——每个智能体保持私有状态。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 定义子图 ──────────────────────────────────────────────────────────────────

class SubgraphState(TypedDict):
    """子图状态：与父图完全独立的 key。"""
    bar: str
    baz: str


def subgraph_node_1(state: SubgraphState) -> dict[str, Any]:
    """子图节点 1：设置 baz 字段。"""
    return {"baz": "世界"}


def subgraph_node_2(state: SubgraphState) -> dict[str, Any]:
    """子图节点 2：拼接 bar 和 baz。"""
    return {"bar": state["bar"] + state["baz"]}


# 构建子图
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# ── 定义父图 ──────────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    """父图状态：与子图没有共享的 key。"""
    foo: str


def node_1(state: ParentState) -> dict[str, Any]:
    """父图节点 1：在 foo 前加前缀。"""
    return {"foo": "你好，" + state["foo"]}


def call_subgraph(state: ParentState) -> dict[str, Any]:
    """在节点函数内调用子图，手动完成状态转换。

    关键步骤：
    1. 将父图 state["foo"] 映射为子图的 {"bar": ...}
    2. 调用 subgraph.invoke(...)
    3. 将子图返回的 {"bar": ...} 映射回父图的 {"foo": ...}
    """
    # 父图状态 → 子图输入
    subgraph_input = {"bar": state["foo"]}
    # 调用子图
    subgraph_output = subgraph.invoke(subgraph_input)
    # 子图输出 → 父图状态
    return {"foo": subgraph_output["bar"]}


class CallInNodeDemo:
    """演示在节点函数内调用子图（不同状态模式）。"""

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> StateGraph:
        builder = StateGraph(ParentState)
        builder.add_node("node_1", node_1)
        builder.add_node("call_subgraph", call_subgraph)
        builder.add_edge(START, "node_1")
        builder.add_edge("node_1", "call_subgraph")
        builder.add_edge("call_subgraph", END)
        return builder.compile()

    def run(self) -> None:
        print("=== 在节点内调用子图（不同状态模式）===\n")

        result = self._graph.invoke({"foo": "小明"})
        print(f"最终结果: foo = {result['foo']}")
        print("执行路径: foo='小明' → node_1: '你好，小明' → 子图: bar='你好，小明' + baz='世界' → '你好，小明世界'")


if __name__ == "__main__":
    CallInNodeDemo().run()
