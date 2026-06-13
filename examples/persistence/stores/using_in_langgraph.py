"""Stores + LangGraph：checkpointer 与 store 协同。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#using-in-langgraph

文档示例使用 graph.stream；async 节点在当前 LangGraph 需通过 astream 调用。
"""

import asyncio
import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


@dataclass
class Context:
    user_id: str


def _last_message_content(messages: list) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return last.get("content", "")
    return last.content


async def update_memory(state: MessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memory = f"User said: {_last_message_content(state['messages'])}"
    await runtime.store.aput(namespace, memory_id, {"memory": memory})
    return {}


async def call_model(state: MessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    query = _last_message_content(state["messages"])
    memories = await runtime.store.asearch(namespace, query=query, limit=3)
    info = "\n".join(d.value["memory"] for d in memories)

    reply = f"Known memories:\n{info}" if info else "No memories yet."
    return {"messages": [AIMessage(content=reply)]}


def build_graph():
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("update_memory", update_memory)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "update_memory")
    builder.add_edge("update_memory", "call_model")
    builder.add_edge("call_model", END)

    return builder.compile(checkpointer=checkpointer, store=store)


class StoreLangGraphDemo:
    """演示 Runtime 注入 store，跨 thread 共享记忆。"""

    def __init__(self):
        self._graph = build_graph()

    async def stream_thread(
        self, thread_id: str, user_message: str, user_id: str = "1"
    ) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n--- thread_id={thread_id} ---")
        async for update in self._graph.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            config,
            stream_mode="updates",
            context=Context(user_id=user_id),
        ):
            print(update)

    async def run(self) -> None:
        await self.stream_thread("1", "hi")
        await self.stream_thread("2", "hi, tell me about my memories")


if __name__ == "__main__":
    asyncio.run(StoreLangGraphDemo().run())
