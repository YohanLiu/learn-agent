"""Error handling：重试耗尽后的错误恢复。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#error-handling

演示内容：
1. 基本 error_handler：捕获 NodeError 并用 Command 路由
2. Saga / 补偿模式：多步流程中的失败回滚
"""

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 示例 1：基本 error_handler + Command 路由
# ---------------------------------------------------------------------------
class PaymentState(TypedDict):
    status: str


class BasicErrorHandlerDemo:
    """支付节点失败后，error_handler 用 Command 更新状态并路由到 finalize。"""

    @staticmethod
    def charge_payment(state: PaymentState) -> dict:
        print("  [charge_payment] 尝试扣款...")
        raise RuntimeError("支付网关超时")

    @staticmethod
    def payment_error_handler(state: PaymentState, error: NodeError) -> Command:
        print(f"  [error_handler] 节点 '{error.node}' 失败：{error.error}")
        return Command(
            update={"status": f"compensated: {error.error}"},
            goto="finalize",
        )

    @staticmethod
    def finalize(state: PaymentState) -> dict:
        print(f"  [finalize] 当前状态：{state['status']}")
        return state

    def run(self) -> None:
        graph = (
            StateGraph(PaymentState)
            .add_node(
                "charge_payment",
                self.charge_payment,
                retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
                error_handler=self.payment_error_handler,
            )
            .add_node("finalize", self.finalize)
            .add_edge(START, "charge_payment")
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


# ---------------------------------------------------------------------------
# 示例 2：Saga / 补偿模式
# ---------------------------------------------------------------------------
class OrderState(TypedDict):
    status: str


class SagaPatternDemo:
    """多步订单流程：预留库存 -> 扣款 -> 完成。

    扣款失败时通过 error_handler 触发补偿逻辑，跳转到 finalize 节点。
    """

    @staticmethod
    def reserve_inventory(state: OrderState) -> dict:
        print("  [reserve_inventory] 库存已预留")
        return {"status": "reserved"}

    @staticmethod
    def charge_payment(state: OrderState) -> dict:
        print("  [charge_payment] 尝试扣款...")
        raise RuntimeError("支付超时")

    @staticmethod
    def payment_error_handler(state: OrderState, error: NodeError) -> Command:
        print(f"  [compensate] 补偿 '{error.node}' 的失败：{error.error}")
        return Command(
            update={"status": f"compensated_after_{error.node}: {error.error}"},
            goto="finalize",
        )

    @staticmethod
    def finalize(state: OrderState) -> dict:
        print(f"  [finalize] 订单最终状态：{state['status']}")
        return state

    def run(self) -> None:
        graph = (
            StateGraph(OrderState)
            .add_node("reserve_inventory", self.reserve_inventory)
            .add_node(
                "charge_payment",
                self.charge_payment,
                retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
                error_handler=self.payment_error_handler,
            )
            .add_node("finalize", self.finalize)
            .add_edge(START, "reserve_inventory")
            .add_edge("reserve_inventory", "charge_payment")
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


# ---------------------------------------------------------------------------
# 示例 3：error_handler 不接收 NodeError（简化签名）
# ---------------------------------------------------------------------------
class SimpleHandlerDemo:
    """error_handler 可以只用 (state) 签名，不需要错误上下文。"""

    @staticmethod
    def failing_node(state: PaymentState) -> dict:
        print("  [failing_node] 即将失败...")
        raise ValueError("数据校验失败")

    @staticmethod
    def simple_handler(state: PaymentState) -> dict:
        print("  [simple_handler] 执行简单恢复逻辑")
        return {"status": "recovered"}

    def run(self) -> None:
        graph = (
            StateGraph(PaymentState)
            .add_node(
                "failing_node",
                self.failing_node,
                error_handler=self.simple_handler,
            )
            .add_edge(START, "failing_node")
            .add_edge("failing_node", END)
            .compile()
        )

        result = graph.invoke({"status": ""})
        print(f"  最终状态: {result}")


if __name__ == "__main__":
    print("=== 1. 基本 error_handler + Command 路由 ===")
    BasicErrorHandlerDemo().run()

    print("\n=== 2. Saga / 补偿模式 ===")
    SagaPatternDemo().run()

    print("\n=== 3. 简化签名的 error_handler ===")
    SimpleHandlerDemo().run()
