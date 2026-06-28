"""
时间旅行示例 - 重放和分叉

本示例演示如何在 LangGraph 中使用时间旅行：
- 重放（Replay）：从先前的检查点重新执行
- 分叉（Fork）：从先前的检查点分支，使用修改后的状态探索替代路径

文档：https://docs.langchain.com/oss/python/langgraph/use-time-travel
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils.uuid import uuid7


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: State):
    """生成笑话的主题。"""
    return {"topic": "socks in the dryer"}


def write_joke(state: State):
    """根据主题编写笑话。"""
    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


def main():
    # 初始化检查点器并编译图
    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("generate_topic", generate_topic)
        .add_node("write_joke", write_joke)
        .add_edge(START, "generate_topic")
        .add_edge("generate_topic", "write_joke")
        .compile(checkpointer=checkpointer)
    )

    # 步骤 1：运行图
    print("=" * 60)
    print("步骤 1：初始执行")
    print("=" * 60)
    config = {"configurable": {"thread_id": str(uuid7())}}
    result = graph.invoke({}, config)
    print(f"Result: {result}")
    print()

    # 步骤 2：查找要重放的检查点
    print("=" * 60)
    print("步骤 2：检查点历史")
    print("=" * 60)
    history = list(graph.get_state_history(config))
    # 历史记录按时间倒序排列
    for state in history:
        print(f"next={state.next}, checkpoint_id={state.config['configurable']['checkpoint_id']}")
    print()

    # 步骤 3：从特定检查点重放
    print("=" * 60)
    print("步骤 3：从 write_joke 之前的检查点重放")
    print("=" * 60)
    # 找到 write_joke 之前的检查点
    before_joke = next(s for s in history if s.next == ("write_joke",))
    print(f"Replaying from checkpoint: {before_joke.config['configurable']['checkpoint_id']}")
    replay_result = graph.invoke(None, before_joke.config)
    # write_joke 重新执行（再次运行），generate_topic 不执行
    print(f"Replay result: {replay_result}")
    print()

    # 步骤 4：从检查点分叉并修改状态
    print("=" * 60)
    print("步骤 4：从检查点分叉并修改主题")
    print("=" * 60)
    # 找到 write_joke 之前的检查点
    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))

    # 分叉：更新状态以更改主题
    fork_config = graph.update_state(
        before_joke.config,
        values={"topic": "chickens"},
    )
    print(f"Fork config checkpoint: {fork_config['configurable']['checkpoint_id']}")

    # 从分叉恢复 — write_joke 使用新主题重新执行
    fork_result = graph.invoke(None, fork_config)
    print(f"分叉结果：{fork_result}")
    print(f"关于鸡的笑话：{fork_result['joke']}")
    print(f"Fork result: {fork_result}")

    # 步骤 5：演示 as_node 参数
    print("=" * 60)
    print("步骤 5：使用显式 as_node 参数分叉")
    print("=" * 60)
    # 将此更新视为由 generate_topic 产生。
    # 执行从 write_joke（generate_topic 的后继节点）恢复。
    fork_config_explicit = graph.update_state(
        before_joke.config,
        values={"topic": "programmers"},
        as_node="generate_topic",
    )
    fork_result_explicit = graph.invoke(None, fork_config_explicit)
    print(f"使用 as_node 的分叉结果：{fork_result_explicit}")
    print(f"关于程序员的笑话：{fork_result_explicit['joke']}")
    print()


if __name__ == "__main__":
    main()
