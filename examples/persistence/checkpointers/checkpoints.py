"""Checkpoints 示例：thread 内 super-step 快照。

对应文档：https://docs.langchain.com/oss/python/langgraph/checkpointers#checkpoints
"""

from operator import add
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


class CheckpointsDemo:
    """顺序图 START → node_a → node_b → END，展示 4 个 checkpoint。"""

    def __init__(self):
        self._graph = self._build()

    def _build(self):
        workflow = StateGraph(State)
        workflow.add_node(node_a)
        workflow.add_node(node_b)
        workflow.add_edge(START, "node_a")
        workflow.add_edge("node_a", "node_b")
        workflow.add_edge("node_b", END)

        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    def run(self, thread_id: str = "1") -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self._graph.invoke({"foo": "", "bar": []}, config)

        history = list(self._graph.get_state_history(config))
        print(f"共 {len(history)} 个 checkpoint（期望 4 个）")
        for snap in reversed(history):
            step = snap.metadata.get("step")
            print(f"  step={step:>2}  next={snap.next!r}  values={snap.values}")

        latest = self._graph.get_state(config)
        print(f"\n最终状态: {latest.values}")
        assert latest.values == {"foo": "b", "bar": ["a", "b"]}


class CheckpointNamespaceDemo:
    """演示从节点内读取 checkpoint_ns。"""

    @staticmethod
    def my_node(state: State, config: RunnableConfig) -> dict[str, Any]:
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        # "" 表示父图，"node_name:uuid" 表示子图
        print(f"checkpoint_ns = {checkpoint_ns!r}")
        return {"foo": "ns-demo"}

    def run(self) -> None:
        workflow = StateGraph(State)
        workflow.add_node(self.my_node)
        workflow.add_edge(START, "my_node")
        workflow.add_edge("my_node", END)

        graph = workflow.compile(checkpointer=InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "ns-demo"}}
        graph.invoke({"foo": "", "bar": []}, config)


if __name__ == "__main__":
    print("=== Checkpoints ===")
    CheckpointsDemo().run()

    print("\n=== Checkpoint namespace ===")
    CheckpointNamespaceDemo().run()
