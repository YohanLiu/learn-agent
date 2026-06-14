"""RedisStore 语义搜索：基于 Redis + embedding 检索记忆。

对应文档：https://docs.langchain.com/oss/python/langgraph/stores#semantic-search

使用 RedisStore 替代 InMemoryStore，数据持久化到 Redis，支持语义搜索。
依赖：langgraph-checkpoint-redis（已含 RedisStore）
通过 ModelFactory 创建 embedding 模型，需在 .zshrc 中配置 EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL。
"""

import uuid

import redis
from langgraph.store.redis import RedisStore
from models.model_factory import ModelFactory

# 本地 Redis 连接地址，默认端口 6379
REDIS_URL = "redis://localhost:6379"


class RedisStoreSemanticSearchDemo:
    """演示带 embedding 索引的 RedisStore 语义搜索。

    RedisStore 特点：
    - 数据持久化到 Redis，进程退出后数据不丢失
    - 依赖 Redis Stack（需要 RediSearch 模块支持向量搜索）
    - 必须先调用 setup() 创建索引，再写入数据，否则新数据不会被索引
    - dims 必须与 embedding 模型输出维度严格匹配，否则文档无法被索引
    - index=False 的条目不会出现在语义搜索结果中（与 InMemoryStore 不同）
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        model_factory: ModelFactory | None = None,
    ):
        """初始化 RedisStore 连接并配置 embedding 索引。

        Args:
            redis_url: Redis 连接字符串，默认 redis://localhost:6379。
            model_factory: ModelFactory 实例，为 None 时自动创建（会从 .zshrc 加载环境变量）。
        """
        factory = model_factory or ModelFactory()
        self._redis_url = redis_url  # 保存连接字符串，供 clear_namespace 使用
        # namespace 是层级路径，用于隔离不同用户/场景的记忆
        self.namespace = ("1", "memories")
        # from_conn_string 返回上下文管理器，管理 Redis 连接的生命周期
        self._ctx = RedisStore.from_conn_string(
            redis_url,
            index={
                # embed: embedding 函数，用于将文本转为向量
                "embed": factory.create_embedding_model(),
                # dims: 向量维度，必须与 embedding 模型输出严格一致
                # 与 InMemoryStore 不同，RedisStore 严格按此值建向量索引，不匹配则无法索引
                "dims": 4096,  # Qwen3-Embedding-8B 输出 4096 维
                # fields: 需要 embedding 的字段路径
                # "food_preference" 表示只 embed 该字段
                # "$" 表示 embed 整个 JSON 文档
                "fields": ["food_preference", "$"],
            },
        )
        self.store: RedisStore = self._ctx.__enter__()

    def setup(self) -> None:
        """创建 RediSearch 索引（若已存在则跳过）。

        必须在 put() 之前调用！
        RediSearch 索引是先建后写入的模式，如果先写入数据再 setup()，
        已写入的数据不会被索引，导致 search() 返回空结果。
        """
        self.store.setup()

    def close(self) -> None:
        """关闭 Redis 连接，释放上下文管理器资源。"""
        self._ctx.__exit__(None, None, None)

    def clear_namespace(self) -> None:
        """清除当前 namespace 下的所有 store 数据，避免历史数据干扰演示结果。

        直接用 redis 客户端删除 store:* 开头的 key，
        比 store.search() + store.delete() 更稳定（后者可能触发 RediSearch 语法错误）。
        """
        r = redis.from_url(self._redis_url)
        # 匹配所有 store 数据 key 和向量 key
        keys = r.keys("store:*") + r.keys("store_vectors:*")
        if keys:
            r.delete(*keys)
            print(f"已清除 {len(keys)} 条 Redis key")

    def seed_memories(self) -> None:
        """向 Redis 写入两条示例记忆，演示不同 index 策略。"""
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
        # index=False: 不建立向量索引，RedisStore 的语义搜索不会返回该条目
        # （与 InMemoryStore 不同，后者仍会返回但 score=None）
        self.store.put(
            self.namespace,
            str(uuid.uuid4()),
            {"system_info": "Last updated: 2024-01-01"},
            index=False,
        )
        print("已写入 2 条记忆到 Redis")

    def search(self, query: str = "What does the user like to eat?") -> None:
        """用自然语言查询记忆，打印结果及 score 信息。

        Args:
            query: 自然语言查询文本，会被 embedding 后做向量相似度搜索。
        """
        memories = self.store.search(self.namespace, query=query, limit=3)
        print(f"\n查询: {query!r}")
        print(f"命中 {len(memories)} 条:")
        for item in memories:
            score_info = (
                f"score={item.score:.4f}" if item.score is not None else "未索引(score=None)"
            )
            print(f"  - [{score_info}] {item.value}")


if __name__ == "__main__":
    demo = RedisStoreSemanticSearchDemo()
    try:
        # setup() 必须在写入数据前调用，确保索引已创建，否则新写入的数据不会被索引
        demo.setup()
        demo.clear_namespace()
        demo.seed_memories()
        demo.search()
    finally:
        demo.close()
