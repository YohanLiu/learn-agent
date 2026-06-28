"""
带多个中断的时间旅行示例

本示例演示在多步表单场景中从多个中断之间进行分叉。
您可以更改后续答案而无需重新询问之前的问题。

文档：https://docs.langchain.com/oss/python/langgraph/use-time-travel
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    value: list[str]


def ask_name(state: State):
    """第一个中断 - 询问姓名。"""
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}


def ask_age(state: State):
    """第二个中断 - 询问年龄。"""
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}


def final_step(state: State):
    """最终处理步骤。"""
    return {"value": ["completed"]}


def main():
    # 初始化带多个中断的图
    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("ask_name", ask_name)
        .add_node("ask_age", ask_age)
        .add_node("final_step", final_step)
        .add_edge(START, "ask_name")
        .add_edge("ask_name", "ask_age")
        .add_edge("ask_age", "final_step")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 步骤 1：运行直到第一个中断
    print("=" * 60)
    print("步骤 1：初始执行 - 遇到 ask_name 中断")
    print("=" * 60)
    result = graph.invoke({"value": []}, config)
    print(f"在 ask_name 处暂停：{result}")
    print()

    # 步骤 2：用姓名恢复 -> 遇到第二个中断
    print("=" * 60)
    print("步骤 2：用姓名 'Alice' 恢复 - 遇到 ask_age 中断")
    print("=" * 60)
    result = graph.invoke(Command(resume="Alice"), config)
    print(f"在 ask_age 处暂停：{result}")
    print()

    # 步骤 3：完成第二个中断
    print("=" * 60)
    print("步骤 3：用年龄 '30' 恢复 - 完成执行")
    print("=" * 60)
    result = graph.invoke(Command(resume="30"), config)
    print(f"最终结果：{result}")
    print()

    # 步骤 4：从两个中断之间分叉
    print("=" * 60)
    print("步骤 4：从 ask_name 和 ask_age 之间的检查点分叉")
    print("=" * 60)
    history = list(graph.get_state_history(config))
    # 找到 ask_name 之后但 ask_age 之前的检查点
    between = [s for s in history if s.next == ("ask_age",)][-1]
    print(f"分叉检查点：{between.config['configurable']['checkpoint_id']}")
    print(f"下一个要执行的节点：{between.next}")

    fork_config = graph.update_state(between.config, {"value": ["modified"]})
    print(f"更新后的分叉配置：{fork_config['configurable']['checkpoint_id']}")

    result = graph.invoke(None, fork_config)
    # ask_name 的结果被保留（"name:Alice"）
    # ask_age 在中断处暂停 — 等待新的答案
    print(f"分叉在 ask_age 处暂停：{result}")
    print(f"注意：ask_name 的结果被保留，只有 ask_age 重新执行")
    print()

    # 步骤 5：用不同的年龄恢复分叉的中断
    print("=" * 60)
    print("步骤 5：用年龄 '25' 恢复分叉的中断")
    print("=" * 60)
    final_result = graph.invoke(Command(resume="25"), fork_config)
    print(f"最终分叉结果：{final_result}")
    print(f"期望：['modified', 'name:Alice', 'age:25', 'completed']")
    print()

    # 步骤 6：演示从第一个中断重放
    print("=" * 60)
    print("步骤 6：从 ask_name 之前的检查点重放")
    print("=" * 60)
    history = list(graph.get_state_history(config))
    before_ask_name = [s for s in history if s.next == ("ask_name",)][-1]
    print(f"重放检查点：{before_ask_name.config['configurable']['checkpoint_id']}")

    replay_result = graph.invoke(None, before_ask_name.config)
    print(f"在 ask_name 处重放暂停：{replay_result}")
    print(f"注意：两个中断将再次被触发")
    print()


if __name__ == "__main__":
    main()
