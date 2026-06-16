"""Graph defaults：图级统一配置重试、超时和错误处理。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#graph-defaults

演示内容：
1. set_node_defaults 一次配置所有节点默认值
2. 优先级：add_node 参数覆盖默认值
3. 全局默认 error_handler
"""

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


class WorkflowState(TypedDict):
    status: str


# ---------------------------------------------------------------------------
# 示例 1：set_node_defaults 统一配置
# ---------------------------------------------------------------------------
class GraphDefaultsDemo:
    """使用 set_node_defaults 为所有节点设置相同的重试、超时和错误处理策略。"""

    @staticmethod
    def step_a(state: WorkflowState) -> dict:
        print("  [step_a] 执行中...")
        return {"status": "step_a done"}

    @staticmethod
    def step_b(state: WorkflowState) -> dict:
        print("  [step_b] 执行中...")
        raise ConnectionError("step_b 网络错误")

    @staticmethod
    def default_error_handler(state: WorkflowState, error: NodeError) -> dict:
        print(f"  [default_error_handler] 节点 '{error.node}' 失败：{error.error}")
        return {"status": f"handled: {error.error}"}

    def run(self) -> None:
        # 注意：timeout 也可在 set_node_defaults 中配置，但要求所有节点为 async
        # 此处仅演示 retry_policy + error_handler（超时示例见 timeouts.py）
        graph = (
            StateGraph(WorkflowState)
            .set_node_defaults(
                retry_policy=RetryPolicy(max_attempts=3),
                error_handler=self.default_error_handler,
            )
            .add_node("step_a", self.step_a)
            .add_node("step_b", self.step_b)
            .add_edge(START, "step_a")
            .add_edge("step_a", "step_b")
            .add_edge("step_b", END)
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


# ---------------------------------------------------------------------------
# 示例 2：优先级——per-node 参数覆盖默认值
# ---------------------------------------------------------------------------
class PrecedenceDemo:
    """add_node 的参数会覆盖 set_node_defaults 中的同名配置。"""

    @staticmethod
    def step_a(state: WorkflowState) -> dict:
        print("  [step_a] 使用默认 error_handler")
        #raise ConnectionError("step_a 网络错误")

    @staticmethod
    def step_b(state: WorkflowState) -> dict:
        print("  [step_b] 使用自定义 error_handler")
        raise ConnectionError("step_b 网络错误")

    @staticmethod
    def default_handler(state: WorkflowState, error: NodeError) -> dict:
        print(f"  [default_handler] 捕获 '{error.node}'：{error.error}")
        return {"status": f"default handled: {error.node}"}

    @staticmethod
    def custom_handler(state: WorkflowState, error: NodeError) -> dict:
        print(f"  [custom_handler] 自定义处理 '{error.node}'：{error.error}")
        return {"status": f"custom handled: {error.node}"}

    def run(self) -> None:
        graph = (
            StateGraph(WorkflowState)
            .set_node_defaults(
                retry_policy=RetryPolicy(max_attempts=2, retry_on=ConnectionError),
                error_handler=self.default_handler,
            )
            .add_node("step_a", self.step_a)  # 使用 default_handler
            .add_node("step_b", self.step_b, error_handler=self.custom_handler)  # 覆盖
            .add_edge(START, "step_a")
            .add_edge("step_a", "step_b")
            .add_edge("step_b", END)
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


# ---------------------------------------------------------------------------
# 示例 3：全局默认 error_handler（所有节点共享）
# ---------------------------------------------------------------------------
class GlobalHandlerDemo:
    """一个永远失败的节点，通过全局默认 error_handler 兜底恢复。"""

    @staticmethod
    def always_failing(state: WorkflowState) -> dict:
        print("  [always_failing] 即将抛出异常...")
        raise ValueError("something went wrong")

    @staticmethod
    def default_handler(state: WorkflowState, error: NodeError) -> dict:
        print(f"  [default_handler] 从 '{error.node}' 恢复：{error.error}")
        return {"status": f"recovered from {error.node}: {error.error}"}

    def run(self) -> None:
        graph = (
            StateGraph(WorkflowState)
            .set_node_defaults(
                retry_policy=RetryPolicy(max_attempts=2),
                error_handler=self.default_handler,
            )
            .add_node("always_failing", self.always_failing)
            .add_edge(START, "always_failing")
            .add_edge("always_failing", END)
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


if __name__ == "__main__":
    print("=== 1. set_node_defaults 统一配置 ===")
    GraphDefaultsDemo().run()

    print("\n=== 2. 优先级（per-node 覆盖默认值） ===")
    PrecedenceDemo().run()

    print("\n=== 3. 全局默认 error_handler ===")
    GlobalHandlerDemo().run()
