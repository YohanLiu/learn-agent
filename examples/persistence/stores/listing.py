"""Stores 列表与分页：枚举 namespace 中的条目。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#listing-items-in-a-namespace
"""

import uuid

from langgraph.store.memory import InMemoryStore


class StoreListingDemo:
    """演示 search 列表、分页与 list_namespaces。"""

    def __init__(self):
        self.store = InMemoryStore()
        self._seed_data()

    def _seed_data(self) -> None:
        for i in range(5):
            self.store.put(
                ("alice", "memories"),
                str(uuid.uuid4()),
                {"note": f"memory-{i}"},
            )
        self.store.put(
            ("alice", "preferences"),
            str(uuid.uuid4()),
            {"theme": "dark"},
        )

    def list_namespace(self) -> None:
        items = self.store.search(("alice", "memories"), limit=100)
        print(f"('alice', 'memories') 共 {len(items)} 条")
        items = self.store.search(("alice", "preferences"), limit=100)
        print(f"('alice', 'preferences') 共 {len(items)} 条")

    def paginate(self) -> None:
        page_size = 2
        offset = 0
        page_num = 0
        while True:
            page = self.store.search(
                ("alice", "memories"), limit=page_size, offset=offset
            )
            if not page:
                break
            page_num += 1
            print(f"  第 {page_num} 页: {len(page)} 条")
            offset += page_size

    def list_namespaces(self) -> None:
        namespaces = self.store.list_namespaces(prefix=("alice",), max_depth=2)
        print(f"alice 相关 namespace: {namespaces}")

    def run(self) -> None:
        print("=== 列表查询 ===")
        self.list_namespace()

        print("\n=== 分页遍历 ===")
        self.paginate()

        print("\n=== list_namespaces ===")
        self.list_namespaces()


if __name__ == "__main__":
    StoreListingDemo().run()
