"""Agent 示例：ToolNode 预构建节点。

对应文档章节：Agents / ToolNode
https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents
https://docs.langchain.com/oss/python/langgraph/workflows-agents#toolnode

Graph API 版本已实现于 examples/langgraphAgent.py，本文件补充：
  ToolNode 预构建节点的用法。
"""

from pathlib import Path
from typing import Any

from langchain.tools import tool
from langchain.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

from models.model_factory import ModelFactory
from examples.graph_utils import GraphVisualizer


# ── 工具定义 ───────────────────────────────────────────────────────────────────

@tool
def multiply(a: int, b: int) -> int:
    """将 `a` 和 `b` 相乘。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """将 `a` 和 `b` 相加。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """将 `a` 除以 `b`。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a / b


_ARITHMETIC_TOOLS = [add, multiply, divide]
_TOOLS_BY_NAME = {t.name: t for t in _ARITHMETIC_TOOLS}


# ── ToolNode 预构建节点版本 ────────────────────────────────────────────────────

class AgentWithToolNode:
    """使用 LangGraph 预构建的 ToolNode 来自动执行工具调用。

    ToolNode 自动处理并行工具执行、错误处理和状态注入。
    """

    def __init__(self, model_factory: ModelFactory | None = None):
        self._factory = model_factory or ModelFactory()
        self.llm = self._factory.create_dashscope_chat_model()
        self._tools = _ARITHMETIC_TOOLS
        self._llm_with_tools = self.llm.bind_tools(self._tools)
        self._workflow = self._build()

    def _llm_call(self, state: MessagesState) -> dict[str, Any]:
        """LLM 节点。"""
        return {
            "messages": [
                self._llm_with_tools.invoke(
                    [
                        SystemMessage(
                            content="你是一个乐于助人的助手，"
                                    "负责对一组输入执行算术运算。"
                        )
                    ]
                    + state["messages"]
                )
            ]
        }

    @staticmethod
    def _should_continue(state: MessagesState):
        """根据是否有工具调用决定是否继续。"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    def _build(self):
        workflow = StateGraph(MessagesState)

        workflow.add_node("llm_call", self._llm_call)
        # 使用预构建的 ToolNode
        workflow.add_node("tool_node", ToolNode(self._tools))

        workflow.add_edge(START, "llm_call")
        workflow.add_conditional_edges(
            "llm_call", self._should_continue, ["tool_node", END]
        )
        workflow.add_edge("tool_node", "llm_call")

        return workflow.compile()

    def show_graph(self) -> Path | None:
        """显示或保存工作流图。"""
        return GraphVisualizer().show(self._workflow, "AgentWithToolNode", xray=True)

    def run(self, user_message: str = "把 3 和 4 相加。") -> None:
        """运行使用 ToolNode 的 Agent。

        Args:
            user_message: 用户输入消息。
        """
        messages = [HumanMessage(content=user_message)]
        result = self._workflow.invoke({"messages": messages})
        for m in result["messages"]:
            m.pretty_print()


if __name__ == "__main__":
    demo = AgentWithToolNode()
    demo.show_graph()
    demo.run("把 3 和 4 相加。")
