"""删除消息：使用 RemoveMessage 从状态中删除消息。

对应文档：https://docs.langchain.com/oss/python/langgraph/add-memory#delete-messages

通过 RemoveMessage 可以从图状态中永久删除消息，控制消息历史长度。
与裁剪不同，删除操作会真正从 checkpoint 中移除消息。
"""

from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from models.model_factory import ModelFactory


class DeleteMessagesDemo:
    """演示使用 RemoveMessage 从状态中删除消息，控制消息历史长度。

    当消息数超过阈值时，删除最早的两条消息。
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        factory = model_factory or ModelFactory()
        self._model = factory.create_dashscope_chat_model()
        self._graph = self._build()

    def _build(self):
        """构建带消息删除的对话图。"""

        def delete_messages(state: MessagesState) -> dict[str, Any]:
            """当消息数超过 2 条时，删除最早的两条。"""
            messages = state["messages"]
            if len(messages) > 2:
                # 使用 RemoveMessage 从状态中移除消息
                return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
            return {}

        def call_model(state: MessagesState) -> dict[str, Any]:
            response = self._model.invoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        # 按顺序执行：call_model → delete_messages
        builder.add_sequence([call_model, delete_messages])
        builder.add_edge(START, "call_model")

        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def run(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}

        conversations = [
            "你好！我是小明。",
            "我叫什么名字？",
            "给我讲个笑话吧。",
        ]

        for i, text in enumerate(conversations, 1):
            print(f"\n--- 第 {i} 轮: {text} ---")
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=text)]},
                config,
            )
            msgs = result["messages"]
            print(f"  当前消息数: {len(msgs)}")
            for msg in msgs:
                role = msg.type
                content = msg.content[:60]
                print(f"    [{role}] {content}")


if __name__ == "__main__":
    DeleteMessagesDemo().run()
