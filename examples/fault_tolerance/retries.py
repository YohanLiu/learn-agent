"""Retries：自动重试失败节点。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#retries

演示内容：
1. 基本 RetryPolicy 用法
2. 自定义重试逻辑（retry_on）
3. 通过 execution_info 检查重试状态
"""

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, default_retry_on
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 示例 1：基本 RetryPolicy
# ---------------------------------------------------------------------------
class ApiState(TypedDict):
    result: str


class RetriesDemo:
    """演示基本重试策略：节点前两次调用抛异常，第三次成功。"""

    def __init__(self):
        self._call_count = 0
        self._graph = self._build()

    def _call_api(self, state: ApiState) -> dict:
        self._call_count += 1
        print(f"  [call_api] 第 {self._call_count} 次调用")
        if self._call_count < 3:
            raise ConnectionError("网络连接超时")
        return {"result": "success"}

    def _build(self):
        builder = StateGraph(ApiState)
        builder.add_node(
            "call_api",
            self._call_api,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        builder.add_edge(START, "call_api")
        builder.add_edge("call_api", END)
        return builder.compile()

    def run(self) -> None:
        self._call_count = 0
        result = self._graph.invoke({"result": ""})
        print(f"  最终结果: {result}")


# ---------------------------------------------------------------------------
# 示例 2：自定义重试逻辑
# ---------------------------------------------------------------------------
class CustomRetryError(Exception):
    """不应重试的自定义异常。"""


def custom_retry_on(exc: BaseException) -> bool:
    """遇到 CustomRetryError 不重试，其他走默认逻辑。"""
    if isinstance(exc, CustomRetryError):
        return False
    return default_retry_on(exc)


class CustomRetryDemo:
    """演示 retry_on 参数：自定义哪些异常可重试。"""

    def __init__(self):
        self._call_count = 0
        self._graph = self._build()

    def _call_api(self, state: ApiState) -> dict:
        self._call_count += 1
        print(f"  [call_api] 第 {self._call_count} 次调用")
        # 抛出一个不可重试的自定义异常，应该只尝试 1 次
        raise CustomRetryError("业务校验失败，不应重试")

    def _build(self):
        builder = StateGraph(ApiState)
        builder.add_node(
            "call_api",
            self._call_api,
            retry_policy=RetryPolicy(max_attempts=3, retry_on=custom_retry_on),
        )
        builder.add_edge(START, "call_api")
        builder.add_edge("call_api", END)
        return builder.compile()

    def run(self) -> None:
        self._call_count = 0
        try:
            self._graph.invoke({"result": ""})
        except CustomRetryError as exc:
            print(f"  捕获不可重试异常：{exc}（共尝试 {self._call_count} 次）")


# ---------------------------------------------------------------------------
# 示例 3：通过 execution_info 检查重试状态
# ---------------------------------------------------------------------------
class InspectRetryStateDemo:
    """演示在节点内读取 runtime.execution_info.node_attempt，实现降级逻辑。"""

    def __init__(self):
        self._primary_calls = 0
        self._graph = self._build()

    def _call_primary(self) -> str:
        self._primary_calls += 1
        raise ConnectionError("主服务不可用")

    def _call_fallback(self) -> str:
        return "fallback 结果"

    def _my_node(self, state: ApiState, runtime: Runtime) -> dict:
        attempt = runtime.execution_info.node_attempt
        print(f"  [my_node] 当前 attempt={attempt}")
        if attempt > 1:
            # 第二次及以后尝试：切换到备用服务
            print("  -> 切换到 fallback")
            return {"result": self._call_fallback()}
        # 首次尝试：调用主服务（会抛异常触发重试）
        print("  -> 调用主服务")
        return {"result": self._call_primary()}

    def _build(self):
        builder = StateGraph(ApiState)
        builder.add_node(
            "my_node",
            self._my_node,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        return builder.compile()

    def run(self) -> None:
        self._primary_calls = 0
        result = self._graph.invoke({"result": ""})
        print(f"  最终结果: {result}")


if __name__ == "__main__":
    print("=== 1. 基本 RetryPolicy ===")
    RetriesDemo().run()

    print("\n=== 2. 自定义重试逻辑（retry_on） ===")
    CustomRetryDemo().run()

    print("\n=== 3. 检查重试状态（execution_info） ===")
    InspectRetryStateDemo().run()
