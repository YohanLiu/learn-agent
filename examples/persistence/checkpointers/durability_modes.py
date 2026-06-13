"""Durability modes：checkpoint 持久化时机。

对应文档：https://docs.langchain.com/oss/python/langgraph/checkpointers#durability-modes

三种模式（由快到慢）：exit → async → sync
"""

from operator import add
from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class DurabilityInput(TypedDict):
    input: str
    steps: Annotated[list[str], add]


class DurabilityModesDemo:
    """演示 durability 参数。"""

    def __init__(self):
        self._graph = self._build()

    @staticmethod
    def _process(state: DurabilityInput) -> dict[str, Any]:
        return {"steps": [f"processed:{state['input']}"]}

    def _build(self):
        workflow = StateGraph(DurabilityInput)
        workflow.add_node("process", self._process)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        return workflow.compile(checkpointer=InMemorySaver())

    def run(self, durability: str = "sync") -> None:
        config = {"configurable": {"thread_id": f"durability-{durability}"}}
        events = list(
            self._graph.stream(
                {"input": "test", "steps": []},
                config,
                durability=durability,
            )
        )
        print(f"durability={durability!r}  事件数={len(events)}")
        print(f"  最终状态: {self._graph.get_state(config).values}")


if __name__ == "__main__":
    demo = DurabilityModesDemo()
    for mode in ("exit", "async", "sync"):
        demo.run(durability=mode)
        print()
