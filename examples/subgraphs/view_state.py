"""查看子图状态：通过 get_state(subgraphs=True) 检查子图内部状态。

对应文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

当启用持久化时，可以用 get_state(config, subgraphs=True) 查看子图内部状态。
注意：子图必须能被静态发现（作为节点添加或在节点内调用），
      通过工具函数间接调用的子图无法查看状态。
"""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# ── 定义子图 ──────────────────────────────────────────────────────────────────

class SubgraphState(TypedDict):
    """子图状态。"""
    foo: str


def subgraph_node_1(state: SubgraphState) -> dict[str, Any]:
    """子图节点 1：遇到中断点，等待用户输入。"""
    value = interrupt("请提供一个值：")
    return {"foo": state["foo"] + value}


# 构建子图（继承父图的 checkpointer）
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()


# ── 定义父图 ──────────────────────────────────────────────────────────────────

class ParentState(TypedDict):
    """父图状态。"""
    foo: str


class ViewStateDemo:
    """演示查看子图状态。

    关键 API：
    - graph.get_state(config, subgraphs=True)：获取包含子图状态的完整快照
    - .tasks[0].state：访问第一个子图任务的状态
    """

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> StateGraph:
        builder = StateGraph(ParentState)
        # 直接将子图作为节点添加（共享 foo key）
        builder.add_node("node_1", subgraph)
        builder.add_edge(START, "node_1")
        builder.add_edge("node_1", END)

        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def run(self) -> None:
        print("=== 查看子图状态 ===\n")

        config = {"configurable": {"thread_id": "1"}}

        # 运行到中断点
        print("步骤 1：运行子图，遇到中断点")
        try:
            result = self._graph.invoke({"foo": "初始值_"}, config)
        except Exception:
            # interrupt 会抛出异常或暂停
            pass

        # 查看子图状态
        print("\n步骤 2：查看子图状态")
        state_snapshot = self._graph.get_state(config, subgraphs=True)
        print(f"  父图状态: foo = {state_snapshot.values.get('foo', '(空)')}")
        print(f"  子图任务数: {len(state_snapshot.tasks)}")

        if state_snapshot.tasks:
            subgraph_state = state_snapshot.tasks[0].state
            print(f"  子图状态: foo = {subgraph_state.values.get('foo', '(空)')}")
            print(f"  子图下一个节点: {subgraph_state.next}")

        # 恢复子图执行
        print("\n步骤 3：恢复子图执行")
        result = self._graph.invoke(Command(resume="用户输入的数据"), config)
        print(f"  最终结果: foo = {result['foo']}")


if __name__ == "__main__":
    ViewStateDemo().run()
