"""流式输出子图：观察子图内部的执行过程。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

默认情况下，父图流式输出时子图被「折叠」为一个黑盒节点。
通过 stream_events(version="v3") 可以看到子图内部每个节点的执行过程，
以及通过 namespace 区分事件来自根图还是子图。

注意：examples/streaming/stream_subgraphs.py 已演示了 stream() 的 subgraphs 参数，
本文件侧重演示 stream_events() 方式。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 定义子图 ──────────────────────────────────────────────────────────────────

class SubgraphState(TypedDict):
    """子图状态：foo 与父图共享，bar 是子图私有字段。"""
    foo: str
    bar: str


def subgraph_node_1(state: SubgraphState) -> dict[str, Any]:
    """子图节点 1：设置私有字段 bar。"""
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState) -> dict[str, Any]:
    """子图节点 2：使用私有字段 bar 更新共享字段 foo。"""
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
    """父图状态。"""
    foo: str


def node_1(state: ParentState) -> dict[str, Any]:
    """父图节点 1：添加前缀。"""
    return {"foo": "你好，" + state["foo"]}


class StreamSubgraphsDemo:
    """演示通过 stream_events 观察子图内部执行过程。

    使用 stream_events(version="v3")：
    - event["method"] == "values" 表示状态更新事件
    - event["method"] == "lifecycle" 表示生命周期事件（started/completed）
    - event["params"]["namespace"] 为空列表 → 根图事件
    - event["params"]["namespace"] 非空 → 子图内部事件，
      格式如 ["node_2:uuid"]
    """

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> StateGraph:
        builder = StateGraph(ParentState)
        builder.add_node("node_1", node_1)
        # 将子图直接添加为节点
        builder.add_node("node_2", subgraph)
        builder.add_edge(START, "node_1")
        builder.add_edge("node_1", "node_2")
        builder.add_edge("node_2", END)
        return builder.compile()

    def run(self) -> None:
        print("=== 流式输出子图（stream_events）===\n")

        # 使用 stream_events 观察所有事件
        print("--- 所有事件（含子图内部）---")
        stream = self._graph.stream_events({"foo": "小明"}, version="v3")
        for event in stream:
            if event["method"] == "values":
                namespace = event["params"]["namespace"]
                data = event["params"]["data"]
                if not namespace:
                    scope = "根图"
                else:
                    scope = f"子图({namespace[0].split(':')[0]})"
                print(f"  [{scope}] {data}")
            elif event["method"] == "lifecycle":
                namespace = event["params"]["namespace"]
                lifecycle_data = event["params"]["data"]
                event_type = lifecycle_data["event"]
                if not namespace:
                    scope = "根图"
                else:
                    scope = f"子图({lifecycle_data.get('graph_name', namespace[0].split(':')[0])})"
                print(f"  [{scope}] 生命周期: {event_type}")

        print("\n--- 只看根图事件（过滤子图）---")
        stream = self._graph.stream_events({"foo": "小明"}, version="v3")
        for event in stream:
            if event["method"] == "values" and not event["params"]["namespace"]:
                data = event["params"]["data"]
                print(f"  [根图] {data}")


if __name__ == "__main__":
    StreamSubgraphsDemo().run()
