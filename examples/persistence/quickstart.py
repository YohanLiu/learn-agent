"""Persistence Quickstart：checkpointer + store 最小示例。

对应文档：https://docs.langchain.com/oss/python/langgraph/persistence#quickstart
"""

import operator
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]


def passthrough(_state: MessagesState) -> dict:
    """占位节点，演示 compile 时同时传入 checkpointer 与 store。"""
    return {}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("passthrough", passthrough)
    builder.add_edge(START, "passthrough")
    builder.add_edge("passthrough", END)

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    return builder.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Hi, my name is Bob."}]},
        {"configurable": {"thread_id": "thread-1"}},
    )

    print("invoke 结果:")
    for msg in result["messages"]:
        print(f"  {msg}")

    snap = graph.get_state({"configurable": {"thread_id": "thread-1"}})
    print(f"\ncheckpoint 已保存，messages 数量: {len(snap.values['messages'])}")
