"""长期记忆：使用 InMemoryStore 跨会话存储用户信息。

对应文档：https://docs.langchain.com/oss/python/langgraph/add-memory#add-long-term-memory

长期记忆（long-term memory）通过 store 实现，可以跨会话保留用户偏好等信息。
使用 Runtime 模式注入 store，通过 context_schema 传递 user_id。
"""

import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from models.model_factory import ModelFactory


@dataclass
class UserContext:
    """用户上下文，用于在节点中获取 user_id。"""
    user_id: str


class LongTermMemoryDemo:
    """演示通过 store 添加长期记忆，跨会话保留用户信息。

    核心步骤：
    1. 创建 InMemoryStore
    2. 定义 Context dataclass，用于传递 user_id
    3. compile 时传入 store
    4. 节点函数通过 runtime: Runtime[UserContext] 访问 store 和 context
    5. invoke 时通过 context=UserContext(user_id=...) 传入用户信息
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        factory = model_factory or ModelFactory()
        self._model = factory.create_dashscope_chat_model()
        self._store = InMemoryStore()
        self._graph = self._build()

    def _build(self):
        """构建带长期记忆的对话图。"""

        def call_model(
            state: MessagesState,
            runtime: Runtime[UserContext],
        ) -> dict[str, Any]:
            # 从 runtime.context 中获取 user_id，用于隔离不同用户的记忆
            user_id = runtime.context.user_id
            namespace = (user_id, "memories")

            # 搜索相关记忆
            memories = runtime.store.search(
                namespace, query=state["messages"][-1].content
            )
            info = "\n".join([d.value["data"] for d in memories])
            system_msg = (
                f"你是一个乐于助人的助手，正在与用户交谈。"
                f"用户信息：{info}"
                if info
                else "你是一个乐于助人的助手，正在与用户交谈。"
            )

            # 如果用户要求记住某些信息，则存入长期记忆
            last_message = state["messages"][-1]
            if "记住" in last_message.content:
                # 提取需要记忆的内容（简化处理，直接存储整条消息）
                memory = last_message.content
                runtime.store.put(
                    namespace, str(uuid.uuid4()), {"data": memory}
                )

            response = self._model.invoke(
                [{"role": "system", "content": system_msg}] + state["messages"]
            )
            return {"messages": response}

        builder = StateGraph(MessagesState, context_schema=UserContext)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        # 关键：compile 时同时传入 checkpointer 和 store
        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer, store=self._store)

    def run(self) -> None:
        # 会话 1：让用户记住一些信息
        print("--- 会话 1：存储记忆 ---")
        config1: RunnableConfig = {"configurable": {"thread_id": "1"}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="你好！请记住：我最喜欢的颜色是蓝色。")]},
            config1,
            context=UserContext(user_id="user-bob"),
        )
        print(f"AI: {result['messages'][-1].content}")

        # 会话 2：不同 thread，但同一用户 → 能访问之前的长期记忆
        print("\n--- 会话 2：跨会话读取记忆 ---")
        config2: RunnableConfig = {"configurable": {"thread_id": "2"}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="我最喜欢的颜色是什么？")]},
            config2,
            context=UserContext(user_id="user-bob"),
        )
        print(f"AI: {result['messages'][-1].content}")

        # 会话 3：不同用户 → 无法访问之前的记忆
        print("\n--- 会话 3：不同用户（隔离） ---")
        config3: RunnableConfig = {"configurable": {"thread_id": "3"}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="我最喜欢的颜色是什么？")]},
            config3,
            context=UserContext(user_id="user-alice"),
        )
        print(f"AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    LongTermMemoryDemo().run()
