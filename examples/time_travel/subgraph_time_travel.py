"""
带子图的时间旅行示例

本示例演示子图的时间旅行，展示继承检查点器（默认）和
拥有独立检查点器的子图之间的区别。

文档：https://docs.langchain.com/oss/python/langgraph/use-time-travel
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    value: list[str]


def step_a(state: State):
    """子图中带中断的第一个步骤。"""
    answer = interrupt("What is your name?")
    return {"value": [f"name:{answer}"]}


def step_b(state: State):
    """子图中带中断的第二个步骤。"""
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}


# ============================================================================
# 示例 1：继承检查点器的子图（默认）
# ============================================================================

def example_inherited_checkpointer():
    """
    默认情况下，子图继承父图的检查点器。
    父图将整个子图视为单个超级步骤。
    您无法在子图中的节点之间进行时间旅行。
    """
    print("=" * 80)
    print("示例 1：使用继承检查点器的子图（默认）")
    print("=" * 80)
    print()

    # 没有自己检查点器的子图（默认）
    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile()  # 没有检查点器 — 从父图继承
    )

    graph = (
        StateGraph(State)
        .add_node("subgraph_node", subgraph)
        .add_edge(START, "subgraph_node")
        .compile(checkpointer=InMemorySaver())
    )

    config = {"configurable": {"thread_id": "1"}}

    # 完成两个中断
    print("步骤 1：运行直到第一个中断（step_a）")
    result = graph.invoke({"value": []}, config)
    print(f"在 step_a 处暂停：{result}")
    print()

    print("步骤 2：恢复 step_a -> 遇到 step_b 中断")
    result = graph.invoke(Command(resume="Alice"), config)
    print(f"在 step_b 处暂停：{result}")
    print()

    print("步骤 3：完成执行")
    result = graph.invoke(Command(resume="30"), config)
    print(f"最终结果：{result}")
    print()

    # 从子图之前进行时间旅行
    print("步骤 4：从子图之前进行时间旅行")
    history = list(graph.get_state_history(config))
    before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]
    print(f"子图之前的检查点：{before_sub.config['configurable']['checkpoint_id']}")

    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    result = graph.invoke(None, fork_config)
    print(f"分叉在 step_a 处暂停：{result}")
    print("注意：整个子图从头开始重新执行")
    print("您无法在 step_a 和 step_b 之间进行时间旅行")
    print()

    # 完成分叉的执行
    result = graph.invoke(Command(resume="Bob"), fork_config)
    result = graph.invoke(Command(resume="25"), fork_config)
    print(f"分叉最终结果：{result}")
    print()
    print()


# ============================================================================
# 示例 2：拥有独立检查点器的子图
# ============================================================================

def example_subgraph_checkpointer():
    """
    在子图上设置 checkpointer=True 以拥有自己的检查点历史。
    这会在子图内的每个步骤创建检查点，允许您
    从子图内的特定点进行时间旅行。
    """
    print("=" * 80)
    print("示例 2：拥有独立检查点器的子图")
    print("=" * 80)
    print()

    # 拥有独立检查点器的子图
    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile(checkpointer=True)  # 独立的检查点历史
    )

    graph = (
        StateGraph(State)
        .add_node("subgraph_node", subgraph)
        .add_edge(START, "subgraph_node")
        .compile(checkpointer=InMemorySaver())
    )

    config = {"configurable": {"thread_id": "2"}}

    # 运行直到 step_a 中断
    print("步骤 1：运行直到 step_a 中断")
    result = graph.invoke({"value": []}, config)
    print(f"在 step_a 处暂停：{result}")
    print()

    # 恢复 step_a -> 遇到 step_b 中断
    print("步骤 2：恢复 step_a -> 遇到 step_b 中断")
    result = graph.invoke(Command(resume="Alice"), config)
    print(f"在 step_b 处暂停：{result}")
    print()

    # 获取子图的独立检查点（在 step_a 和 step_b 之间）
    print("步骤 3：获取子图的独立检查点")
    parent_state = graph.get_state(config, subgraphs=True)
    sub_config = parent_state.tasks[0].state.config
    print(f"子图检查点：{sub_config['configurable']['checkpoint_id']}")
    print()

    # 从子图检查点分叉
    print("步骤 4：从子图检查点分叉")
    fork_config = graph.update_state(sub_config, {"value": ["forked"]})
    print(f"分叉配置：{fork_config['configurable']['checkpoint_id']}")

    result = graph.invoke(None, fork_config)
    # step_b 重新执行，step_a 的结果被保留
    print(f"分叉在 step_b 处暂停：{result}")
    print("注意：step_a 的结果被保留，只有 step_b 重新执行")
    print()

    # 完成分叉的执行
    print("步骤 5：完成分叉的执行")
    final_result = graph.invoke(Command(resume="25"), fork_config)
    print(f"最终分叉结果：{final_result}")
    print(f"期望：['forked', 'name:Alice', 'age:25']")
    print()


def main():
    # 示例 1：继承的检查点器
    example_inherited_checkpointer()

    # 示例 2：拥有独立检查点器的子图
    example_subgraph_checkpointer()


if __name__ == "__main__":
    main()
