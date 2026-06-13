"""Stores 基础用法：InMemoryStore put / search。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#basic-usage
"""

import uuid
from typing import Any

from langgraph.store.memory import InMemoryStore


class StoreBasicDemo:
    """演示 namespace、put、search 与 Item 字段。"""

    def __init__(self):
        self.store = InMemoryStore()

    def run(self, user_id: str = "1") -> dict[str, Any]:
        namespace_for_memory = (user_id, "memories")

        memory_id = str(uuid.uuid4())
        memory = {"food_preference": "I like pizza"}
        self.store.put(namespace_for_memory, memory_id, memory)

        memories = self.store.search(namespace_for_memory)
        item = memories[-1]

        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        else:
            item_dict = item.dict()

        print("最新记忆:")
        print(f"  value      = {item_dict['value']}")
        print(f"  key        = {item_dict['key']}")
        print(f"  namespace  = {item_dict['namespace']}")
        print(f"  created_at = {item_dict['created_at']}")
        print(f"  updated_at = {item_dict['updated_at']}")

        return item_dict


if __name__ == "__main__":
    StoreBasicDemo().run()
