"""Checkpointer Serializer：pickle fallback 与加密。

对应文档：https://docs.langchain.com/oss/python/langgraph/checkpointers#serializer
"""

import os
from operator import add
from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class SerdeState(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def _build_graph(checkpointer: InMemorySaver):
    def node_a(state: SerdeState) -> dict[str, Any]:
        return {"foo": "a", "bar": ["a"]}

    workflow = StateGraph(SerdeState)
    workflow.add_node(node_a)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", END)
    return workflow.compile(checkpointer=checkpointer)


class PickleFallbackDemo:
    """JsonPlusSerializer(pickle_fallback=True)。"""

    def run(self) -> None:
        graph = _build_graph(
            InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
        )
        config = {"configurable": {"thread_id": "pickle-demo"}}
        graph.invoke({"foo": "", "bar": []}, config)
        print(f"pickle_fallback 运行成功: {graph.get_state(config).values}")


class EncryptionDemo:
    """EncryptedSerializer + InMemorySaver（文档示例为 SqliteSaver / PostgresSaver）。"""

    def run(self) -> None:
        try:
            from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
        except ImportError:
            print("跳过加密演示：EncryptedSerializer 不可用")
            return

        if not os.environ.get("LANGGRAPH_AES_KEY"):
            os.environ["LANGGRAPH_AES_KEY"] = "0123456789abcdef0123456789abcdef"

        try:
            serde = EncryptedSerializer.from_pycryptodome_aes()
        except ImportError as exc:
            print(f"跳过加密演示：{exc}")
            return

        graph = _build_graph(InMemorySaver(serde=serde))
        config = {"configurable": {"thread_id": "encrypted-demo"}}
        graph.invoke({"foo": "", "bar": []}, config)
        print(f"加密 checkpointer 运行成功: {graph.get_state(config).values}")


def _show_production_snippets() -> None:
    print("\n--- SqliteSaver（需 langgraph-checkpoint-sqlite）---")
    print("""
import sqlite3
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
""")

    print("--- PostgresSaver（需 langgraph-checkpoint-postgres）---")
    print("""
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
""")


if __name__ == "__main__":
    print("=== Pickle fallback ===")
    PickleFallbackDemo().run()

    print("\n=== Encryption ===")
    EncryptionDemo().run()

    _show_production_snippets()
