"""Stores 语义搜索：基于 embedding 检索记忆。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#semantic-search

通过 ModelFactory 创建 embedding 模型，需在 .zshrc 中配置 EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL。

注意：index=False 的条目仍会出现在 search() 结果中，但 score 为 None（不参与向量排序）。
"""

import uuid

from langgraph.store.memory import InMemoryStore
from models.model_factory import ModelFactory


class StoreSemanticSearchDemo:
    """演示带 embedding 索引的 InMemoryStore 语义搜索。

    InMemoryStore 特点：
    - 数据存储在内存中，进程退出后数据丢失
    - 无需额外依赖，适合开发/测试
    - 不需要提前调用 setup()
    - index=False 的条目仍会出现在 search() 结果中，score=None
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        """初始化 InMemoryStore 并配置 embedding 索引。

        Args:
            model_factory: ModelFactory 实例，为 None 时自动创建（会从 .zshrc 加载环境变量）。
        """
        factory = model_factory or ModelFactory()
        # namespace 是层级路径，用于隔离不同用户/场景的记忆
        self.namespace = ("1", "memories")
        self.store = InMemoryStore(
            index={
                # embed: embedding 函数，用于将文本转为向量
                "embed": factory.create_embedding_model(),
                # dims: 向量维度，必须与 embedding 模型输出一致
                "dims": 4096,  # Qwen3-Embedding-8B 输出 4096 维
                # fields: 需要 embedding 的字段路径
                # "food_preference" 表示只 embed 该字段
                # "$" 表示 embed 整个 JSON 文档
                "fields": ["food_preference", "$"],
            }
        )

    def seed_memories(self) -> None:
        """向 store 写入两条示例记忆，演示不同 index 策略。"""
        # index=["food_preference"]: 只对 food_preference 字段建立向量索引
        self.store.put(
            self.namespace,
            str(uuid.uuid4()),
            {
                "food_preference": "I love Italian cuisine",
                "context": "Discussing dinner plans",
            },
            index=["food_preference"],
        )
        # index=False: 不建立向量索引，search() 仍会返回该条目，但 score 为 None
        self.store.put(
            self.namespace,
            str(uuid.uuid4()),
            {"system_info": "Last updated: 2024-01-01"},
            index=False,
        )

    def search(self, query: str = "What does the user like to eat?") -> None:
        """用自然语言查询记忆，打印结果及 score 信息。

        Args:
            query: 自然语言查询文本，会被 embedding 后做向量相似度搜索。
        """
        memories = self.store.search(self.namespace, query=query, limit=3)
        print(f"查询: {query!r}")
        print(f"命中 {len(memories)} 条:")
        for item in memories:
            # item.score: 有索引的条目返回相似度分数，无索引的条目为 None
            score_info = f"score={item.score:.4f}" if item.score is not None else "未索引(score=None)"
            print(f"  - [{score_info}] {item.value}")
        print("\n注意: index=False 的条目仍会出现在搜索结果中，但没有相似度分数(score=None)，")
        print("      即不参与向量排序。如需排除，可通过 score 是否为 None 来过滤。")


if __name__ == "__main__":
    demo = StoreSemanticSearchDemo()
    demo.seed_memories()
    demo.search()
