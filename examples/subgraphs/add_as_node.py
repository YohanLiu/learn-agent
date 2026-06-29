"""将子图直接添加为节点：父图与子图共享状态 key。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

当父图和子图共享部分状态 key 时，可以直接将编译后的子图传给 add_node()，
无需包装函数——子图会自动读写父图的同名 channel。

子图可以拥有父图中不存在的私有 key（如 bar），这些 key 仅在子图内部可见。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 定义子图 ──────────────────────────────────────────────────────────────────

class SubgraphState(TypedDict):
    """子图状态：foo 与父图共享，bar 是子图私有字段。"""
    foo: str   # 共享字段
    bar: str   # 子图私有字段


def subgraph_node_1(state: SubgraphState) -> dict[str, Any]:
    """子图节点 1：设置私有字段 bar。"""
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState) -> dict[str, Any]:
    """子图节点 2：读取私有字段 bar，更新共享字段 foo。"""
    return {"foo": state["foo"] + state["bar"]}


# 构建子图
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# ── 定义父图 ──────────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    """父图状态：只有共享字段 foo。"""
    foo: str


def node_1(state: ParentState) -> dict[str, Any]:
    """父图节点 1：在 foo 前加前缀。"""
    return {"foo": "你好，" + state["foo"]}


class AddAsNodeDemo:
    """演示将子图直接添加为节点（共享状态模式）。

    关键区别：
    - 不需要包装函数，直接将编译后的子图传给 add_node()
    - 子图自动读写父图的共享 channel（foo）
    - 子图私有字段（bar）在父图中不可见
    """

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> StateGraph:
        builder = StateGraph(ParentState)
        builder.add_node("node_1", node_1)
        # 关键：直接传入编译后的子图，无需包装函数
        builder.add_node("node_2", subgraph)
        builder.add_edge(START, "node_1")
        builder.add_edge("node_1", "node_2")
        builder.add_edge("node_2", END)
        return builder.compile()

    def run(self) -> None:
        print("=== 将子图直接添加为节点（共享状态模式）===\n")

        result = self._graph.invoke({"foo": "小明"})
        print(f"最终结果: foo = {result['foo']}")
        print("执行路径: foo='小明' → node_1: '你好，小明' → 子图: bar='bar', foo='你好，小明' + 'bar' → '你好，小明bar'")


if __name__ == "__main__":
    AddAsNodeDemo().run()
