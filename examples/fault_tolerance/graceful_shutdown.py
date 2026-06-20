"""Graceful shutdown：协作式优雅关闭。

对应文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#graceful-shutdown

演示内容：
1. RunControl + request_drain 基本用法
2. 在节点内读取 drain 状态
3. SIGTERM 信号处理模式
"""

import os
import signal
import threading
import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl, Runtime
from typing_extensions import TypedDict


class ShutdownState(TypedDict, total=False):
    steps_done: int


# ---------------------------------------------------------------------------
# 示例 1：RunControl + request_drain 基本用法
# ---------------------------------------------------------------------------
class BasicDrainDemo:
    """演示 RunControl.request_drain()：图在完成当前 superstep 后停止。

    使用 checkpointer 保存进度，后续可从断点恢复。
    """

    @staticmethod
    def step_1(state: ShutdownState) -> dict:
        print("  [step_1] 执行完成")
        return {"steps_done": 1}

    @staticmethod
    def step_2(state: ShutdownState) -> dict:
        print("  [step_2] 执行完成")
        return {"steps_done": 2}

    @staticmethod
    def step_3(state: ShutdownState) -> dict:
        print("  [step_3] 执行完成")
        return {"steps_done": 3}

    def run(self) -> None:
        graph = (
            StateGraph(ShutdownState)
            .add_node("step_1", self.step_1)
            .add_node("step_2", self.step_2)
            .add_node("step_3", self.step_3)
            .add_edge(START, "step_1")
            .add_edge("step_1", "step_2")
            .add_edge("step_2", "step_3")
            .add_edge("step_3", END)
            .compile(checkpointer=InMemorySaver())
        )

        config = {"configurable": {"thread_id": "drain-demo"}}
        control = RunControl()

        # 模拟：在另一个线程 / 信号处理器中请求 drain
        # 这里直接在 invoke 前请求，图会在第一个 superstep 边界停下
        control.request_drain("manual-stop")

        try:
            graph.invoke({"steps_done": 0}, config, control=control)
        except GraphDrained as exc:
            print(f"  图被 drain！原因：{exc.reason}")

            # 查看当前进度
            state = graph.get_state(config)
            print(f"  已保存的进度：{state.values}")

            # 后续可以恢复执行
            print("  恢复执行...")
            result = graph.invoke(None, config)
            print(f"  恢复后最终状态：{result}")


# ---------------------------------------------------------------------------
# 示例 2：节点内读取 drain 状态
# ---------------------------------------------------------------------------
class DrainAwareNodeDemo:
    """节点内检查 runtime.drain_requested，据此调整行为。

    参考文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#read-drain-state-inside-a-node
    使用后台线程在 smart_node 执行期间请求 drain，节点内检测到后提前结束。
    """

    @staticmethod
    def smart_node(state: ShutdownState, runtime: Runtime) -> dict:
        """模拟耗时工作，期间周期性检查 drain 状态。"""
        print("  [smart_node] 开始耗时工作（模拟 3 步处理）...")
        for i in range(1, 4):
            time.sleep(0.3)
            if runtime.drain_requested:
                print(f"  [smart_node] 第 {i} 步检测到 drain（原因：{runtime.drain_reason}），提前结束")
                return {"steps_done": state.get("steps_done", 0) + i}
        print("  [smart_node] 耗时工作完成")
        return {"steps_done": state.get("steps_done", 0) + 3}

    def run(self) -> None:
        graph = (
            StateGraph(ShutdownState)
            .add_node("smart_node", self.smart_node)
            .add_edge(START, "smart_node")
            .add_edge("smart_node", END)
            .compile(checkpointer=InMemorySaver())
        )

        # 正常执行（不请求 drain）
        print("  --- 正常执行 ---")
        config_normal = {"configurable": {"thread_id": "drain-aware-normal"}}
        control = RunControl()
        result = graph.invoke({"steps_done": 0}, config_normal, control=control)
        print(f"  结果：{result}")

        # 后台线程延迟触发 drain：模拟外部信号在执行期间到达
        print("  --- 带 drain 请求执行（后台线程延迟触发）---")
        config_drain = {"configurable": {"thread_id": "drain-aware-drain"}}
        control2 = RunControl()
        timer = threading.Timer(0.5, lambda: control2.request_drain("test-drain"))
        timer.start()
        try:
            result = graph.invoke({"steps_done": 0}, config_drain, control=control2)
            print(f"  结果：{result}")
        except GraphDrained as exc:
            print(f"  图被 drain：{exc.reason}")
        finally:
            timer.cancel()


# ---------------------------------------------------------------------------
# 示例 3：SIGTERM 信号处理模式
# ---------------------------------------------------------------------------
class SigtermPatternDemo:
    """演示推荐的 SIGTERM 信号处理模式。

    参考文档：https://docs.langchain.com/oss/python/langgraph/fault-tolerance#sigterm-hook-pattern
    真实注册 SIGTERM 信号处理器，用 os.kill 向自身发送信号，展示 drain + checkpointer 恢复。
    """

    @staticmethod
    def step_1(state: ShutdownState) -> dict:
        print("  [step_1] 执行完成")
        return {"steps_done": 1}

    @staticmethod
    def step_2(state: ShutdownState) -> dict:
        print("  [step_2] 开始耗时操作...")
        time.sleep(2)
        print("  [step_2] 执行完成")
        return {"steps_done": 2}

    @staticmethod
    def step_3(state: ShutdownState) -> dict:
        print("  [step_3] 执行完成")
        return {"steps_done": 3}

    def run(self) -> None:
        graph = (
            StateGraph(ShutdownState)
            .add_node("step_1", self.step_1)
            .add_node("step_2", self.step_2)
            .add_node("step_3", self.step_3)
            .add_edge(START, "step_1")
            .add_edge("step_1", "step_2")
            .add_edge("step_2", "step_3")
            .add_edge("step_3", END)
            .compile(checkpointer=InMemorySaver())
        )

        config = {"configurable": {"thread_id": "sigterm-demo"}}
        control = RunControl()

        # 真实注册 SIGTERM 信号处理器
        signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))

        # 后台线程延迟发送真实 SIGTERM 信号给当前进程
        timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()

        print("  --- 真实 SIGTERM drain ---")
        try:
            graph.invoke({"steps_done": 0}, config, control=control)
        except GraphDrained as exc:
            print(f"  图被 drain：{exc.reason}")

            # 查看保存的进度
            state = graph.get_state(config)
            print(f"  已保存的进度：{state.values}")

            # 恢复执行
            print("  恢复执行...")
            result = graph.invoke(None, config)
            print(f"  恢复后最终状态：{result}")
        finally:
            timer.cancel()
            # 恢复默认信号处理
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


if __name__ == "__main__":
    print("=== 1. RunControl + request_drain 基本用法 ===")
    BasicDrainDemo().run()

    print("\n=== 2. 节点内读取 drain 状态 ===")
    DrainAwareNodeDemo().run()

    print("\n=== 3. SIGTERM 信号处理模式 ===")
    SigtermPatternDemo().run()
