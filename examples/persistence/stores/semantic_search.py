"""Stores 语义搜索：基于 embedding 检索记忆。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#semantic-search

需要 OPENAI_API_KEY。
"""

import os
import uuid

from langgraph.store.memory import InMemoryStore


class StoreSemanticSearchDemo:
    """演示带 embedding 索引的 InMemoryStore。"""

    def __init__(self):
        from langchain.embeddings import init_embeddings

        self.namespace = ("1", "memories")
        self.store = InMemoryStore(
            index={
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "dims": 1536,
                "fields": ["food_preference", "$"],
            }
        )

    def seed_memories(self) -> None:
        self.store.put(
            self.namespace,
            str(uuid.uuid4()),
            {
                "food_preference": "I love Italian cuisine",
                "context": "Discussing dinner plans",
            },
            index=["food_preference"],
        )
        self.store.put(
            self.namespace,
            str(uuid.uuid4()),
            {"system_info": "Last updated: 2024-01-01"},
            index=False,
        )

    def search(self, query: str = "What does the user like to eat?") -> None:
        memories = self.store.search(self.namespace, query=query, limit=3)
        print(f"查询: {query!r}")
        print(f"命中 {len(memories)} 条:")
        for item in memories:
            print(f"  - {item.value}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("跳过语义搜索演示：请设置 OPENAI_API_KEY")
    else:
        demo = StoreSemanticSearchDemo()
        demo.seed_memories()
        demo.search()
