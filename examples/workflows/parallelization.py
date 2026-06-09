"""并行化（Parallelization）示例：Graph API。

对应文档章节：Parallelization
https://docs.langchain.com/oss/python/langgraph/workflows-agents#parallelization

多个 LLM 并行执行子任务，然后聚合结果。适用于可拆分为独立子任务的场景。
"""

import time
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from models.model_factory import ModelFactory
from utils.graph_utils import GraphVisualizer


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class ParallelizationState(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# ── Graph API 版本 ─────────────────────────────────────────────────────────────

class ParallelizationGraph:
    """使用 StateGraph 实现并行化：同时生成笑话、故事、诗歌，然后聚合。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_dashscope_chat_model()
        self._start_time: float = 0
        self._workflow = self._build()

    # ── 节点函数 ────────────────────────────────────────────────

    def _call_llm_1(self, state: ParallelizationState) -> dict[str, Any]:
        """第一次 LLM 调用：生成笑话。"""
        t0 = time.perf_counter()
        print(f"  [生成笑话] 开始 @ {t0 - self._start_time:.3f}s")
        msg = self.llm.invoke(f"写一个关于{state['topic']}的笑话")
        print(f"  [生成笑话] 完成 @ {time.perf_counter() - self._start_time:.3f}s  耗时 {time.perf_counter() - t0:.3f}s")
        return {"joke": msg.content}

    def _call_llm_2(self, state: ParallelizationState) -> dict[str, Any]:
        """第二次 LLM 调用：生成故事。"""
        t0 = time.perf_counter()
        print(f"  [生成故事] 开始 @ {t0 - self._start_time:.3f}s")
        msg = self.llm.invoke(f"写一个关于{state['topic']}的故事")
        print(f"  [生成故事] 完成 @ {time.perf_counter() - self._start_time:.3f}s  耗时 {time.perf_counter() - t0:.3f}s")
        return {"story": msg.content}

    def _call_llm_3(self, state: ParallelizationState) -> dict[str, Any]:
        """第三次 LLM 调用：生成诗歌。"""
        t0 = time.perf_counter()
        print(f"  [生成诗歌] 开始 @ {t0 - self._start_time:.3f}s")
        msg = self.llm.invoke(f"写一首关于{state['topic']}的诗")
        print(f"  [生成诗歌] 完成 @ {time.perf_counter() - self._start_time:.3f}s  耗时 {time.perf_counter() - t0:.3f}s")
        return {"poem": msg.content}

    @staticmethod
    def _aggregator(state: ParallelizationState) -> dict[str, Any]:
        """聚合笑话、故事和诗歌。"""
        combined = f"以下是关于{state['topic']}的故事、笑话和诗歌！\n\n"
        combined += f"故事：\n{state['story']}\n\n"
        combined += f"笑话：\n{state['joke']}\n\n"
        combined += f"诗歌：\n{state['poem']}"
        return {"combined_output": combined}

    # ── 构建工作流 ──────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(ParallelizationState)

        workflow.add_node("生成笑话", self._call_llm_1)
        workflow.add_node("生成故事", self._call_llm_2)
        workflow.add_node("生成诗歌", self._call_llm_3)
        workflow.add_node("聚合", self._aggregator)

        # 并行执行三个 LLM 调用
        workflow.add_edge(START, "生成笑话")
        workflow.add_edge(START, "生成故事")
        workflow.add_edge(START, "生成诗歌")
        # 全部完成后聚合
        workflow.add_edge("生成笑话", "聚合")
        workflow.add_edge("生成故事", "聚合")
        workflow.add_edge("生成诗歌", "聚合")
        workflow.add_edge("聚合", END)

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "ParallelizationGraph")

    def run(self, topic: str = "猫") -> ParallelizationState:
        """运行并行化工作流。

        Args:
            topic: 主题。

        Returns:
            最终状态字典。
        """
        print(f"\n{'=' * 60}")
        print(f"开始并行执行，主题：{topic}")
        print(f"{'=' * 60}")
        self._start_time = time.perf_counter()
        state = self._workflow.invoke({"topic": topic})
        total = time.perf_counter() - self._start_time
        print(f"{'=' * 60}")
        print(f"全部完成！总耗时：{total:.3f}s")
        print(f"（若串行，总耗时 ≈ 三个子任务耗时之和；")
        print(f"  并行时，总耗时 ≈ 最慢的那一个子任务耗时）")
        print(f"{'=' * 60}\n")
        print(state["combined_output"])
        return state


if __name__ == "__main__":
    demo = ParallelizationGraph()
    demo.show_graph()
    demo.run("猫")
