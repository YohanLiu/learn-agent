"""评估-优化（Evaluator-Optimizer）示例：Graph API。

对应文档章节：Evaluator-optimizer
https://docs.langchain.com/oss/python/langgraph/workflows-agents#evaluator-optimizer

一个 LLM 生成输出，另一个 LLM 评估输出。如果评估不合格则提供反馈并重新生成，
直到满足质量标准为止。适用于需要迭代才能达到标准的任务。
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from models.model_factory import ModelFactory
from examples.graph_utils import GraphVisualizer


# ── 评估 Schema ───────────────────────────────────────────────────────────────

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="判断这个笑话是否好笑。"
    )
    feedback: str = Field(
        description="如果笑话不好笑，提供改进建议。"
    )


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class EvaluatorOptimizerState(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


# ── Graph API 版本 ─────────────────────────────────────────────────────────────

class EvaluatorOptimizerGraph:
    """使用 StateGraph 实现评估-优化循环：生成笑话 → 评估 → 不合格则重新生成。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_yunwu_chat_model()
        self._evaluator = self.llm.with_structured_output(Feedback)
        self._workflow = self._build()

    # ── 节点函数 ────────────────────────────────────────────────

    def _llm_call_generator(self, state: EvaluatorOptimizerState) -> dict[str, Any]:
        """LLM 生成笑话（如有反馈则参考反馈）。"""
        if state.get("feedback"):
            msg = self.llm.invoke(
                f"写一个关于{state['topic']}的笑话，并参考以下反馈：{state['feedback']}"
            )
        else:
            msg = self.llm.invoke(f"写一个关于{state['topic']}的笑话")
        return {"joke": msg.content}

    def _llm_call_evaluator(self, state: EvaluatorOptimizerState) -> dict[str, Any]:
        """LLM 评估笑话。"""
        grade = self._evaluator.invoke(f"评估这个笑话：{state['joke']}")
        return {"funny_or_not": grade.grade, "feedback": grade.feedback}

    @staticmethod
    def _route_joke(state: EvaluatorOptimizerState) -> str:
        """根据评估结果决定是否继续迭代。"""
        if state["funny_or_not"] == "funny":
            return "Accepted"
        elif state["funny_or_not"] == "not funny":
            return "Rejected + Feedback"

    # ── 构建工作流 ──────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(EvaluatorOptimizerState)

        workflow.add_node("llm_call_generator", self._llm_call_generator)
        workflow.add_node("llm_call_evaluator", self._llm_call_evaluator)

        workflow.add_edge(START, "llm_call_generator")
        workflow.add_edge("llm_call_generator", "llm_call_evaluator")
        workflow.add_conditional_edges(
            "llm_call_evaluator",
            self._route_joke,
            {
                "Accepted": END,
                "Rejected + Feedback": "llm_call_generator",
            },
        )

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "EvaluatorOptimizerGraph")

    def run(self, topic: str = "猫") -> EvaluatorOptimizerState:
        """运行评估-优化工作流。

        Args:
            topic: 笑话主题。

        Returns:
            最终状态字典。
        """
        state = self._workflow.invoke({"topic": topic})
        print(state["joke"])
        return state


if __name__ == "__main__":
    demo = EvaluatorOptimizerGraph()
    demo.show_graph()
    demo.run("猫")
