"""
带中断的时间旅行示例

本示例演示在包含中断的人机交互工作流中进行时间旅行。
在中断期间，中断总是会被重新触发。

文档：https://docs.langchain.com/oss/python/langgraph/use-time-travel
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    value: list[str]


def ask_human(state: State):
    """使用中断询问人类输入。"""
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}


def final_step(state: State):
    """最终处理步骤。"""
    return {"value": ["Done"]}


def main():
    # 初始化带中断的图
    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("ask_human", ask_human)
        .add_node("final_step", final_step)
        .add_edge(START, "ask_human")
        .add_edge("ask_human", "final_step")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 步骤 1：首次运行 - 遇到中断
    print("=" * 60)
    print("步骤 1：初始执行 - 遇到中断")
    print("=" * 60)
    result = graph.invoke({"value": []}, config)
    print(f"中断后的状态：{result}")
    print()

    # 用答案恢复
    print("=" * 60)
    print("步骤 2：用答案 'Alice' 恢复")
    print("=" * 60)
    result = graph.invoke(Command(resume="Alice"), config)
    print(f"最终结果：{result}")
    print()

    # 步骤 3：从 ask_human 之前重放
    print("=" * 60)
    print("步骤 3：从 ask_human 之前的检查点重放")
    print("=" * 60)
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]
    print(f"检查点：{before_ask.config['configurable']['checkpoint_id']}")

    replay_result = graph.invoke(None, before_ask.config)
    # 在中断处暂停 — 等待新的 Command(resume=...)
    print(f"在中断处重放暂停：{replay_result}")
    print()

    # 步骤 4：从 ask_human 之前分叉
    print("=" * 60)
    print("步骤 4：从 ask_human 之前的检查点分叉")
    print("=" * 60)
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
    print(f"分叉检查点：{fork_config['configurable']['checkpoint_id']}")

    fork_result = graph.invoke(None, fork_config)
    # 在中断处暂停 — 等待新的 Command(resume=...)
    print(f"在中断处分叉暂停：{fork_result}")
    print()

    # 用不同的答案恢复分叉的中断
    print("=" * 60)
    print("步骤 5：用答案 'Bob' 恢复分叉的中断")
    print("=" * 60)
    final_fork_result = graph.invoke(Command(resume="Bob"), fork_config)
    print(f"最终分叉结果：{final_fork_result}")
    print(f"期望：['forked', 'Hello, Bob!', 'Done']")
    print()


if __name__ == "__main__":
    main()
