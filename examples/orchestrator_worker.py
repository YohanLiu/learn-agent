"""编排者-工人（Orchestrator-Worker）示例：Graph API（Send）。

对应文档章节：Orchestrator-worker
https://docs.langchain.com/oss/python/langgraph/workflows-agents#orchestrator-worker

编排者将任务拆分为子任务，委派给工人并行执行，最后综合所有工人输出。
适用于子任务数量不固定、需要动态分配的场景。
"""

import time
from pathlib import Path
from typing import Annotated, Any, List
import operator

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models.model_factory import ModelFactory
from examples.graph_utils import GraphVisualizer


# ── 共享 Schema ───────────────────────────────────────────────────────────────

class Section(BaseModel):
    name: str = Field(description="报告章节的名称。")
    description: str = Field(
        description="本节要涵盖的主要主题和概念的简要概述。"
    )


class Sections(BaseModel):
    sections: List[Section] = Field(description="报告的各章节列表。")


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]  # 所有工人并行写入
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# ── Graph API 版本（使用 Send API 动态创建工人）─────────────────────────────────

class OrchestratorWorkerGraph:
    """使用 StateGraph + Send API 实现编排者-工人模式。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_yunwu_chat_model()
        self._planner = self.llm.with_structured_output(Sections)
        self._start_time: float = 0.0
        self._workflow = self._build()

    # ── 节点函数 ────────────────────────────────────────────────

    def _orchestrator(self, state: OrchestratorState) -> dict[str, Any]:
        """编排者：生成报告规划。"""
        report_sections = self._planner.invoke(
            [
                SystemMessage(content="为报告生成一份规划。"),
                HumanMessage(content=f"报告主题：{state['topic']}"),
            ]
        )
        return {"sections": report_sections.sections}

    def _llm_call(self, state: WorkerState) -> dict[str, Any]:
        """工人：撰写报告的某个章节。"""
        section_name = state['section'].name
        t_start = time.perf_counter() - self._start_time
        print(f"  [工人启动] 章节「{section_name}」 | 相对时间 {t_start:+.3f}s")

        section = self.llm.invoke(
            [
                SystemMessage(
                    content="根据提供的名称和描述撰写报告的一个章节。"
                            "不要写前言，直接使用 Markdown 格式。"
                ),
                HumanMessage(
                    content=f"章节名称：{state['section'].name}，"
                            f"描述：{state['section'].description}"
                ),
            ]
        )

        t_end = time.perf_counter() - self._start_time
        duration = t_end - t_start
        print(f"  [工人完成] 章节「{section_name}」 | 相对时间 {t_end:+.3f}s | 耗时 {duration:.3f}s")
        return {"completed_sections": [section.content]}

    @staticmethod
    def _synthesizer(state: OrchestratorState) -> dict[str, Any]:
        """综合所有章节生成最终报告。"""
        completed_report_sections = "\n\n---\n\n".join(state["completed_sections"])
        return {"final_report": completed_report_sections}

    @staticmethod
    def _assign_workers(state: OrchestratorState) -> list[Send]:
        """为计划中的每个章节分配一个工人（通过 Send API 并行派发）。"""
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    # ── 构建工作流 ──────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("orchestrator", self._orchestrator)
        workflow.add_node("llm_call", self._llm_call)
        workflow.add_node("synthesizer", self._synthesizer)

        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges("orchestrator", self._assign_workers, ["llm_call"])
        workflow.add_edge("llm_call", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "OrchestratorWorkerGraph")

    def run(self, topic: str = "撰写一份关于 LLM 缩放定律的报告") -> str:
        """运行编排者-工人工作流。

        Args:
            topic: 报告主题。

        Returns:
            最终报告文本。
        """
        self._start_time = time.perf_counter()
        print(f"[编排开始] 主题：{topic}")
        state = self._workflow.invoke({"topic": topic})
        total = time.perf_counter() - self._start_time

        print(f"\n[编排完成] 总耗时 {total:.3f}s")
        print(f"  若为串行，总耗时 ≈ 各工人耗时之和；")
        print(f"  若为并行，总耗时 ≈ 最慢工人的耗时。")
        print(f"  工人数量：{len(state['completed_sections'])}")

        try:
            from IPython.display import Markdown
            Markdown(state["final_report"])
        except ImportError:
            pass
        print(state["final_report"])
        return state["final_report"]


if __name__ == "__main__":
    demo = OrchestratorWorkerGraph()
    demo.show_graph()
    demo.run("撰写一份关于 LLM 缩放定律的报告")
