"""Get / Update state：读取、筛选、修改与重放 checkpoint。

对应文档：https://docs.langchain.com/oss/python/langgraph/checkpointers#get-and-update-state
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


def _build_demo_graph():
    def node_a(state: State) -> dict[str, Any]:
        return {"foo": "a", "bar": ["a"]}

    def node_b(state: State) -> dict[str, Any]:
        return {"foo": "b", "bar": ["b"]}

    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)
    return workflow.compile(checkpointer=InMemorySaver())


class GetStateDemo:
    """演示 get_state 与 get_state_history。"""

    def __init__(self):
        self._graph = _build_demo_graph()

    def run(self, thread_id: str = "1") -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self._graph.invoke({"foo": "", "bar": []}, config)

        # get the latest state snapshot
        latest = self._graph.get_state(config)
        print("最新 StateSnapshot:")
        print(f"  values = {latest.values}")
        print(f"  next   = {latest.next}")
        print(f"  step   = {latest.metadata.get('step')}")

        # get a state snapshot for a specific checkpoint_id
        checkpoint_id = latest.config["configurable"]["checkpoint_id"]
        specific_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        }
        specific = self._graph.get_state(specific_config)
        print(f"\n按 checkpoint_id 查询: values = {specific.values}")

        history = list(self._graph.get_state_history(config))
        print(f"\n历史共 {len(history)} 条（最新在前）:")
        for snap in history:
            print(
                f"  step={snap.metadata.get('step'):>2}  "
                f"source={snap.metadata.get('source')!r}  "
                f"next={snap.next!r}"
            )


class FindCheckpointDemo:
    """演示从历史中筛选特定 checkpoint。"""

    def __init__(self):
        self._graph = _build_demo_graph()

    def run(self, thread_id: str = "1") -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self._graph.invoke({"foo": "", "bar": []}, config)
        history = list(self._graph.get_state_history(config))

        before_node_b = next(s for s in history if s.next == ("node_b",))
        print(f"node_b 执行前的 checkpoint: values={before_node_b.values}")

        step_2 = next(s for s in history if s.metadata["step"] == 2)
        print(f"step=2 的 checkpoint: values={step_2.values}")

        forks = [s for s in history if s.metadata["source"] == "update"]
        print(f"由 update_state 产生的 checkpoint 数量: {len(forks)}")

        interrupted = next(
            (
                s
                for s in history
                if s.tasks and any(t.interrupts for t in s.tasks)
            ),
            None,
        )
        print(f"含 interrupt 的 checkpoint: {'有' if interrupted else '无'}")


class UpdateStateDemo:
    """演示 update_state：创建新 checkpoint，不修改原始记录。"""

    def __init__(self):
        self._graph = _build_demo_graph()

    def run(self, thread_id: str = "1") -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self._graph.invoke({"foo": "", "bar": []}, config)

        before = self._graph.get_state(config)
        print(f"更新前: {before.values}")

        new_config = self._graph.update_state(config, {"foo": "updated"})
        after = self._graph.get_state(new_config)
        print(f"更新后: {after.values}")
        print(f"metadata.source = {after.metadata.get('source')!r}")


class ReplayDemo:
    """演示 Replay：从指定 checkpoint 重新执行后续节点。"""

    def __init__(self):
        self._graph = _build_demo_graph()

    def run(self, thread_id: str = "1") -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self._graph.invoke({"foo": "", "bar": []}, config)

        history = list(self._graph.get_state_history(config))
        replay_from = next(s for s in history if s.next == ("node_b",))
        replay_config = replay_from.config

        print(f"从 checkpoint step={replay_from.metadata['step']} 重放")
        self._graph.invoke(None, replay_config)

        latest = self._graph.get_state(config)
        print(f"重放后最终状态: {latest.values}")


if __name__ == "__main__":
    print("=== Get state ===")
    GetStateDemo().run()

    print("\n=== Find checkpoint ===")
    FindCheckpointDemo().run()

    print("\n=== Update state ===")
    UpdateStateDemo().run()

    print("\n=== Replay ===")
    ReplayDemo().run()
