"""路由（Routing）示例：Graph API。

对应文档章节：Routing
https://docs.langchain.com/oss/python/langgraph/workflows-agents#routing

根据输入内容将其路由到不同的专门处理流程，适用于需要不同处理策略的复杂任务。
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from models.model_factory import ModelFactory
from utils.graph_utils import GraphVisualizer


# ── 路由 Schema ───────────────────────────────────────────────────────────────

class Route(BaseModel):
    step: Literal["诗歌", "故事", "笑话"] = Field(
        None, description="路由过程中的下一步骤"
    )


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class RoutingState(TypedDict):
    input: str
    decision: str
    output: str


# ── Graph API 版本 ─────────────────────────────────────────────────────────────

class RoutingGraph:
    """使用 StateGraph 实现路由：根据输入决定生成故事、笑话或诗歌。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_yunwu_chat_model()
        self._router = self.llm.with_structured_output(Route)
        self._workflow = self._build()

    # ── 节点函数 ────────────────────────────────────────────────

    def _llm_call_1(self, state: RoutingState) -> dict[str, Any]:
        """写故事。"""
        result = self.llm.invoke(state["input"])
        return {"output": result.content}

    def _llm_call_2(self, state: RoutingState) -> dict[str, Any]:
        """写笑话。"""
        result = self.llm.invoke(state["input"])
        return {"output": result.content}

    def _llm_call_3(self, state: RoutingState) -> dict[str, Any]:
        """写诗歌。"""
        result = self.llm.invoke(state["input"])
        return {"output": result.content}

    def _llm_call_router(self, state: RoutingState) -> dict[str, Any]:
        """路由输入到适当的节点。"""
        decision = self._router.invoke(
            [
                SystemMessage(
                    content="根据用户的请求，将输入路由到故事、笑话或诗歌。"
                ),
                HumanMessage(content=state["input"]),
            ]
        )
        return {"decision": decision.step}

    @staticmethod
    def _route_decision(state: RoutingState) -> str:
        """根据决策路由到对应节点。"""
        if state["decision"] == "故事":
            return "写故事"
        elif state["decision"] == "笑话":
            return "写笑话"
        elif state["decision"] == "诗歌":
            return "写诗歌"

    # ── 构建工作流 ──────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(RoutingState)

        workflow.add_node("写故事", self._llm_call_1)
        workflow.add_node("写笑话", self._llm_call_2)
        workflow.add_node("写诗歌", self._llm_call_3)
        workflow.add_node("路由", self._llm_call_router)

        workflow.add_edge(START, "路由")
        workflow.add_conditional_edges(
            "路由",
            self._route_decision,
            {
                "写故事": "写故事",
                "写笑话": "写笑话",
                "写诗歌": "写诗歌",
            },
        )
        workflow.add_edge("写故事", END)
        workflow.add_edge("写笑话", END)
        workflow.add_edge("写诗歌", END)

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "RoutingGraph")

    def run(self, input_text: str = "给我写一个关于猫的笑话") -> RoutingState:
        """运行路由工作流。

        Args:
            input_text: 用户输入。

        Returns:
            最终状态字典。
        """
        state = self._workflow.invoke({"input": input_text})
        print(state["output"])
        return state


if __name__ == "__main__":
    demo = RoutingGraph()
    demo.show_graph()

    test_inputs = [
        "给我写一个关于猫的笑话",
        "写一个关于太空探险的短故事",
        "写一首关于秋天的诗",
    ]
    for text in test_inputs:
        print(f"\n{'=' * 40}")
        print(f"输入：{text}")
        print(f"{'=' * 40}")
        demo.run(text)
