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


# ... existing code ...
async def update_memory(state: MessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memory = f"User said: {_last_message_content(state['messages'])}"
    print(f"[DEBUG update_memory] Saving: {memory}")
    await runtime.store.aput(namespace, memory_id, {"memory": memory})
    print(f"[DEBUG update_memory] Saved with ID: {memory_id}")
    return {}


async def call_model(state: MessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    query = _last_message_content(state["messages"])
    print(f"[DEBUG call_model] Query: '{query}'")
    memories = await runtime.store.asearch(namespace, query=query, limit=3)
    print(f"[DEBUG call_model] Found {len(memories)} memories")
    for i, m in enumerate(memories):
        print(f"  [{i}] {m.value['memory']}")

    info = "\n".join(d.value["memory"] for d in memories)

    reply = f"Known memories:\n{info}" if info else "No memories yet."
    print(f"[DEBUG call_model] Reply: {reply}")
    return {"messages": [AIMessage(content=reply)]}


# ... existing code ...


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
        # 不同的 user_id 不可以访问相同的记忆
        await self.stream_thread("1", "hi-1", "1")
        await self.stream_thread("2", "hi-2, tell me about my memories", "2")

        # 如果你创建了一个新线程，只要 user_id 相同，你仍然可以访问相同的记忆
        await self.stream_thread("3", "hi-3", )
        await self.stream_thread("4", "hi-4, tell me about my memories",)


if __name__ == "__main__":
    asyncio.run(StoreLangGraphDemo().run())
