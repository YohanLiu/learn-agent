"""短期记忆：使用 InMemorySaver 保持多轮对话上下文。

对应文档：https://docs.langchain.com/oss/python/langgraph/add-memory#add-short-term-memory

短期记忆（short-term memory）通过 checkpointer 实现，同一 thread_id 内的
多轮对话会自动保留上下文，不同 thread_id 之间互相隔离。
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from models.model_factory import ModelFactory


class ShortTermMemoryDemo:
    """演示通过 checkpointer 添加短期记忆，实现多轮对话。

    核心步骤：
    1. 创建 InMemorySaver 作为 checkpointer
    2. compile 时传入 checkpointer
    3. invoke 时通过 thread_id 区分不同对话线程
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        factory = model_factory or ModelFactory()
        self._model = factory.create_dashscope_chat_model()
        self._graph = self._build()

    def _build(self):
        """构建带短期记忆的对话图。"""

        def call_model(state: MessagesState) -> dict[str, Any]:
            response = self._model.invoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        # 关键：compile 时传入 checkpointer，启用短期记忆
        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def run(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}

        # 第一轮：用户自我介绍
        print("--- 第一轮对话 ---")
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="你好！我叫小明。")]},
            config,
        )
        print(f"AI: {result['messages'][-1].content}")

        # 第二轮：AI 能记住用户的名字（因为同一 thread_id）
        print("\n--- 第二轮对话 ---")
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="我叫什么名字？")]},
            config,
        )
        print(f"AI: {result['messages'][-1].content}")

        # 不同 thread_id 之间互相隔离
        print("\n--- 新线程（隔离） ---")
        new_config: RunnableConfig = {"configurable": {"thread_id": "2"}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content="我叫什么名字？")]},
            new_config,
        )
        print(f"AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    ShortTermMemoryDemo().run()
