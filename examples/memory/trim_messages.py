"""裁剪消息：使用 trim_messages 管理消息历史长度。

对应文档：https://docs.langchain.com/oss/python/langgraph/add-memory#trim-messages

当对话轮数增多，消息历史可能超出 LLM 的上下文窗口限制。
trim_messages 会保留最近的 N 个 token 的消息，丢弃较早的消息。
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from models.model_factory import ModelFactory


class TrimMessagesDemo:
    """演示使用 trim_messages 裁剪消息历史，防止超出 LLM 上下文窗口。

    trim_messages 会保留最近的 N 个 token 的消息，丢弃较早的消息。
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        factory = model_factory or ModelFactory()
        self._model = factory.create_dashscope_chat_model()
        self._graph = self._build()

    def _build(self):
        """构建带消息裁剪的对话图。"""

        def call_model(state: MessagesState) -> dict[str, Any]:
            # 裁剪消息：保留最近 128 个 token，从 human 消息开始
            messages = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=128,
                start_on="human",
                end_on=("human", "tool"),
            )
            response = self._model.invoke(messages)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def run(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}

        # 多轮对话，消息会逐步累积
        conversations = [
            "你好，我叫小明。",
            "写一首关于猫的小诗。",
            "再写一首关于狗的。",
            "我叫什么名字？",
        ]

        for i, text in enumerate(conversations, 1):
            print(f"\n--- 第 {i} 轮: {text} ---")
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=text)]},
                config,
            )
            # 打印裁剪后实际发送给模型的消息数量
            all_msgs = result["messages"]
            print(f"  历史消息总数: {len(all_msgs)}")
            print(f"  AI: {all_msgs[-1].content}")


if __name__ == "__main__":
    TrimMessagesDemo().run()
