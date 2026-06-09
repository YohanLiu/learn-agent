"""提示链（Prompt Chaining）示例：Graph API。

对应文档章节：Prompt chaining
https://docs.langchain.com/oss/python/langgraph/workflows-agents#prompt-chaining

每个 LLM 调用依次处理上一个调用的输出，适用于可拆分为可验证步骤的明确任务。
"""

from pathlib import Path
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from models.model_factory import ModelFactory
from utils.graph_utils import GraphVisualizer


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class PromptChainingState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# ── Graph API 版本 ─────────────────────────────────────────────────────────────

class PromptChainingGraph:
    """使用 StateGraph 实现提示链：生成笑话 → 检查 → 改进 → 润色。"""

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_dashscope_chat_model()
        self._workflow = self._build()

    # ── 节点函数 ────────────────────────────────────────────────

    def _generate_joke(self, state: PromptChainingState) -> dict[str, Any]:
        """第一次 LLM 调用：生成初始笑话。"""
        msg = self.llm.invoke(f"写一个关于{state['topic']}的短笑话")
        return {"joke": msg.content}

    @staticmethod
    def _check_punchline(state: PromptChainingState) -> str:
        """门控函数：检查笑话是否有包袱（punchline）。"""
        if "？" in state["joke"] or "！" in state["joke"] or "?" in state["joke"] or "!" in state["joke"]:
            return "Pass"
        return "Fail"

    def _improve_joke(self, state: PromptChainingState) -> dict[str, Any]:
        """第二次 LLM 调用：改进笑话。"""
        msg = self.llm.invoke(
            f"通过添加文字游戏让这个笑话更有趣：{state['joke']}"
        )
        return {"improved_joke": msg.content}

    def _polish_joke(self, state: PromptChainingState) -> dict[str, Any]:
        """第三次 LLM 调用：最终润色。"""
        msg = self.llm.invoke(
            f"给这个笑话加一个出人意料的反转：{state['improved_joke']}"
        )
        return {"final_joke": msg.content}

    # ── 构建工作流 ──────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(PromptChainingState)

        workflow.add_node("generate_joke", self._generate_joke)
        workflow.add_node("improve_joke", self._improve_joke)
        workflow.add_node("polish_joke", self._polish_joke)

        workflow.add_edge(START, "generate_joke")
        workflow.add_conditional_edges(
            "generate_joke",
            self._check_punchline,
            {"Fail": "improve_joke", "Pass": END},
        )
        workflow.add_edge("improve_joke", "polish_joke")
        workflow.add_edge("polish_joke", END)

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "PromptChainingGraph")

    def run(self, topic: str = "猫") -> PromptChainingState:
        """运行提示链工作流。

        Args:
            topic: 笑话主题。

        Returns:
            最终状态字典。
        """
        state = self._workflow.invoke({"topic": topic})
        print("初始笑话：")
        print(state["joke"])
        print("\n--- --- ---\n")
        if "improved_joke" in state:
            print("改进后的笑话：")
            print(state["improved_joke"])
            print("\n--- --- ---\n")
            print("最终笑话：")
            print(state["final_joke"])
        else:
            print("最终笑话：")
            print(state["joke"])
        return state


if __name__ == "__main__":
    demo = PromptChainingGraph()
    demo.show_graph()
    demo.run("猫")
