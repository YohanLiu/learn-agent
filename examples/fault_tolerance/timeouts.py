"""Timeouts：限制单次节点执行时长。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#timeouts

注意：节点超时仅适用于 **async** 节点，且需通过 ainvoke / astream 异步执行。
同步节点在 compile 时会报错；使用 invoke 运行 async 节点时 timeout 也会被拒绝。

演示内容：
1. 简单超时（数字秒 / timedelta）
2. Run timeout：硬性墙上时钟上限
3. Idle timeout：空闲超时（无进度信号时触发）
4. Progress signals：auto 模式下 yield 输出自动重置空闲计时器
5. Manual heartbeats：heartbeat 模式手动重置空闲计时器
6. 超时 + 重试组合
"""

import asyncio
from datetime import timedelta

from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy
from typing_extensions import TypedDict


class TimeoutState(TypedDict):
    result: str


# ---------------------------------------------------------------------------
# 示例 1：简单超时（数字秒 / timedelta）
# ---------------------------------------------------------------------------
async def slow_node(state: TimeoutState) -> dict:
    print("  [slow_node] 开始执行，将 sleep 3 秒...")
    await asyncio.sleep(3)
    return {"result": "完成"}


class SimpleTimeoutDemo:
    """演示数字秒和 timedelta 两种超时写法。"""

    def run(self) -> None:
        # 写法 1：直接传数字（秒）
        builder = StateGraph(TimeoutState)
        builder.add_node("slow_node", slow_node, timeout=2)  # 2 秒超时
        builder.add_edge(START, "slow_node")
        builder.add_edge("slow_node", END)
        graph = builder.compile()

        try:
            asyncio.run(graph.ainvoke({"result": ""}))
        except NodeTimeoutError as exc:
            print(f"  超时！节点={exc.node}, 耗时={exc.elapsed:.2f}s, 类型={exc.kind}")

        # 写法 2：传 timedelta
        print("  --- timedelta 写法 ---")
        builder2 = StateGraph(TimeoutState)
        builder2.add_node("slow_node", slow_node, timeout=timedelta(seconds=2))
        builder2.add_edge(START, "slow_node")
        builder2.add_edge("slow_node", END)
        graph2 = builder2.compile()

        try:
            asyncio.run(graph2.ainvoke({"result": ""}))
        except NodeTimeoutError as exc:
            print(f"  timedelta 超时！节点={exc.node}, 耗时={exc.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 示例 2：TimeoutPolicy - run_timeout
# ---------------------------------------------------------------------------
async def call_model(state: TimeoutState) -> dict:
    print("  [call_model] 开始执行（run_timeout=1.5s）...")
    await asyncio.sleep(3)
    return {"result": "不应到达这里"}


class RunTimeoutDemo:
    """run_timeout 是硬性墙上时钟上限，无论节点活动如何都不会刷新。"""

    def run(self) -> None:
        builder = StateGraph(TimeoutState)
        builder.add_node(
            "call_model",
            call_model,
            timeout=TimeoutPolicy(run_timeout=1.5),
        )
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)
        graph = builder.compile()

        try:
            asyncio.run(graph.ainvoke({"result": ""}))
        except NodeTimeoutError as exc:
            print(f"  run_timeout 触发！节点={exc.node}, kind={exc.kind}, 耗时={exc.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 示例 3：Idle timeout - 空闲超时（无进度信号时触发）
# ---------------------------------------------------------------------------
async def stalling_node(state: TimeoutState) -> dict:
    """节点长时间无输出，触发 idle_timeout。"""
    print("  [stalling_node] 开始执行，将静默 sleep 3 秒（无进度信号）...")
    await asyncio.sleep(3)
    return {"result": "不应到达这里"}


class IdleTimeoutDemo:
    """idle_timeout 是进度重置型超时：节点停止产出可观测进度时触发。

    与 run_timeout 不同，idle 时钟会在节点产出进度信号时重置。
    """

    def run(self) -> None:
        builder = StateGraph(TimeoutState)
        builder.add_node(
            "stalling_node",
            stalling_node,
            timeout=TimeoutPolicy(idle_timeout=1.0),
        )
        builder.add_edge(START, "stalling_node")
        builder.add_edge("stalling_node", END)
        graph = builder.compile()

        try:
            asyncio.run(graph.ainvoke({"result": ""}))
        except NodeTimeoutError as exc:
            print(f"  idle_timeout 触发！节点={exc.node}, kind={exc.kind}, 耗时={exc.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 示例 4：Progress signals - auto 模式下 stream_writer 自动重置空闲计时器
# ---------------------------------------------------------------------------
async def streaming_node(state: TimeoutState, runtime: Runtime) -> dict:
    """节点通过 runtime.stream_writer 产出进度信号，每次写入都会重置 auto 模式的 idle 时钟。"""
    writer = runtime.stream_writer
    chunks = ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    for chunk in chunks:
        await asyncio.sleep(0.8)  # 间隔 0.4s < idle_timeout 1.0s
        writer(chunk)  # 产出进度信号，重置 idle 时钟
        print(f"  [streaming_node] 产出 {chunk}（重置 idle 时钟）")
    return {"result": "流式输出完成"}


class ProgressSignalsDemo:
    """演示 refresh_on='auto'（默认）：stream_writer / state writes 等自动重置 idle 时钟。

    节点虽然总耗时 1.6s > idle_timeout 1.0s，但每次 stream_writer 调用都重置计时器，不会超时。
    对比示例 3（stalling_node 无进度信号 -> 超时），此处有进度信号 -> 不超时。
    """

    def run(self) -> None:
        builder = StateGraph(TimeoutState)
        builder.add_node(
            "streaming_node",
            streaming_node,
            # auto 模式（默认）：stream_writer、state writes 等自动重置 idle 计时器
            timeout=TimeoutPolicy(idle_timeout=1.0),
        )
        builder.add_edge(START, "streaming_node")
        builder.add_edge("streaming_node", END)
        graph = builder.compile()

        result = asyncio.run(graph.ainvoke({"result": ""}))
        print(f"  最终结果: {result['result']}")


# ---------------------------------------------------------------------------
# 示例 5：Manual heartbeats - 手动 heartbeat() 重置空闲计时器
# ---------------------------------------------------------------------------
async def long_running_node(state: TimeoutState, runtime: Runtime) -> dict:
    batches = ["batch-1", "batch-2", "batch-3"]
    for batch in batches:
        print(f"  [long_running_node] 处理 {batch}...")
        await asyncio.sleep(0.5)
        runtime.heartbeat()  # 重置空闲计时器
    return {"result": "所有批次处理完成"}


class HeartbeatDemo:
    """演示 refresh_on='heartbeat'：必须手动调用 runtime.heartbeat() 保活。

    适用于长时间运行且不会自然产出进度信号的工作。
    runtime.heartbeat() 在非 idle-timed 的尝试中是 no-op，可以无条件调用。
    """

    def run(self) -> None:
        builder = StateGraph(TimeoutState)
        builder.add_node(
            "long_running_node",
            long_running_node,
            timeout=TimeoutPolicy(idle_timeout=1.0, refresh_on="heartbeat"),
        )
        builder.add_edge(START, "long_running_node")
        builder.add_edge("long_running_node", END)
        graph = builder.compile()

        result = asyncio.run(graph.ainvoke({"result": ""}))
        print(f"  最终结果: {result['result']}")


# ---------------------------------------------------------------------------
# 示例 6：超时 + 重试组合
# ---------------------------------------------------------------------------
class TimeoutWithRetryDemo:
    """timeout 与 retry_policy 组合：超时后自动重试，每次超时时钟重置。"""

    def __init__(self):
        self._attempt = 0

    async def call_api(self, state: TimeoutState) -> dict:
        self._attempt += 1
        print(f"  [call_api] 第 {self._attempt} 次尝试")
        if self._attempt < 2:
            # 第一次：sleep 太久导致超时
            await asyncio.sleep(3)
        # 第二次：快速返回
        return {"result": "重试后成功"}

    def run(self) -> None:
        self._attempt = 0
        builder = StateGraph(TimeoutState)
        builder.add_node(
            "call_api",
            self.call_api,
            timeout=TimeoutPolicy(idle_timeout=1.0),
            retry_policy=RetryPolicy(max_attempts=3),
        )
        builder.add_edge(START, "call_api")
        builder.add_edge("call_api", END)
        graph = builder.compile()

        result = asyncio.run(graph.ainvoke({"result": ""}))
        print(f"  最终结果: {result['result']}")


if __name__ == "__main__":
    print("=== 1. 简单超时（数字秒 / timedelta） ===")
    SimpleTimeoutDemo().run()

    print("\n=== 2. run_timeout ===")
    RunTimeoutDemo().run()

    print("\n=== 3. idle_timeout（空闲超时） ===")
    IdleTimeoutDemo().run()

    print("\n=== 4. progress signals（stream_writer 自动重置 idle 时钟） ===")
    ProgressSignalsDemo().run()

    print("\n=== 5. manual heartbeats（heartbeat 模式） ===")
    HeartbeatDemo().run()

    print("\n=== 6. 超时 + 重试组合 ===")
    TimeoutWithRetryDemo().run()
