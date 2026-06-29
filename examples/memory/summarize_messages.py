"""摘要消息：使用摘要替代裁剪/删除，保留对话要点信息。

对应文档：https://docs.langchain.com/oss/python/langgraph/add-memory#summarize-messages

裁剪或删除消息会丢失信息，而摘要可以保留对话要点。
通过扩展 MessagesState 增加 summary 字段，在消息过多时生成摘要并删除旧消息。
"""

from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from models.model_factory import ModelFactory


class SummarizeMessagesDemo:
    """演示使用摘要替代裁剪/删除，保留对话要点信息。

    扩展 MessagesState 增加 summary 字段，在消息过多时生成摘要并删除旧消息。
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        factory = model_factory or ModelFactory()
        self._model = factory.create_dashscope_chat_model()
        self._graph = self._build()

    def _build(self):
        """构建带消息摘要的对话图。"""

        # 扩展 MessagesState，增加 summary 字段
        class State(MessagesState):
            summary: str

        def summarize_conversation(state: State) -> dict[str, Any]:
            """当消息超过 3 条时，生成摘要并删除旧消息（保留最近 2 条）。"""
            if len(state["messages"]) <= 3:
                return {}

            # 获取已有摘要（如果有）
            summary = state.get("summary", "")

            # 构建摘要提示
            if summary:
                summary_message = (
                    f"以下是迄今为止的对话摘要：{summary}\n\n"
                    "请结合上面的新消息扩展这个摘要："
                )
            else:
                summary_message = "请为上面的对话创建一个摘要："

            # 调用模型生成摘要
            messages = state["messages"] + [HumanMessage(content=summary_message)]
            response = self._model.invoke(messages)

            # 删除除最近 2 条之外的所有消息
            delete_msgs = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
            return {"summary": response.content, "messages": delete_msgs}

        def call_model(state: State) -> dict[str, Any]:
            # 如果有摘要，将其作为系统消息注入
            messages = state["messages"]
            summary = state.get("summary", "")
            if summary:
                system_msg = f"迄今为止的对话摘要：{summary}"
                messages = [HumanMessage(content=system_msg)] + messages
            response = self._model.invoke(messages)
            return {"messages": [response]}

        builder = StateGraph(State)
        builder.add_node("summarize", summarize_conversation)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "call_model")

        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def run(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}

        conversations = [
            "你好，我叫小明。",
            "我喜欢意大利菜和寿司。",
            "我是一名在金融公司工作的软件工程师。",
            "你对我了解多少？",
        ]

        for i, text in enumerate(conversations, 1):
            print(f"\n--- 第 {i} 轮: {text} ---")
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=text)]},
                config,
            )
            msgs = result["messages"]
            print(f"  当前消息数: {len(msgs)}")
            print(f"  AI: {msgs[-1].content}")
            # 打印摘要（如果存在）
            summary = result.get("summary", "")
            if summary:
                print(f"  [摘要] {summary[:100]}...")


if __name__ == "__main__":
    SummarizeMessagesDemo().run()
