"""流式输出快速入门（Streaming Quickstart）。

演示 graph.stream(stream_mode=..., version="v2") 这套「stream-mode」API：
  - v2 格式下，无论传单个模式（字符串）还是多模式（列表），
    每个 chunk 都是统一信封：{"type", "ns", "data"}
  - 用 get_stream_writer() 在节点内推送自定义数据（custom 通道）

对应文档：https://docs.langchain.com/oss/python/langgraph/streaming#basic-usage

运行：
    uv run python examples/streaming/quickstart.py
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ── 定义状态与节点 ────────────────────────────────────────────────────────────

class State(TypedDict):
    """演示用状态：累积消息。"""
    messages: Annotated[list, add_messages]


def refine_topic(state: State) -> dict[str, Any]:
    """第 1 步：把用户的模糊话题收敛成一个明确的笑话主题。"""
    last = state["messages"][-1].content
    topic = f"关于「{last}」的冷笑话"
    return {"messages": [AIMessage(content=topic)]}


def generate_joke(state: State) -> dict[str, Any]:
    """第 2 步：根据主题讲一个笑话。节点内通过 stream writer 推送进度。"""
    writer = get_stream_writer()
    writer({"status": "正在构思笑话..."})  # 进入 custom 通道
    topic = state["messages"][-1].content
    joke = f"问：{topic}？ 答：因为它能。"
    return {"messages": [AIMessage(content=joke)]}


workflow = StateGraph(State)
workflow.add_node("refine_topic", refine_topic)
workflow.add_node("generate_joke", generate_joke)
workflow.add_edge(START, "refine_topic")
workflow.add_edge("refine_topic", "generate_joke")
workflow.add_edge("generate_joke", END)
graph = workflow.compile()


# ── 演示 1：单模式 updates ───────────────────────────────────────────────────
# 即便只传一个模式（字符串），v2 下每个 chunk 仍是完整信封 {"type", "ns", "data"}。

print("=== 演示 1：单模式 updates（字符串参数，v2 仍是完整信封）===\n")

for chunk in graph.stream(
    {"messages": [HumanMessage(content="程序员")]},
    stream_mode="updates",
    version="v2",
):
    # chunk 是完整信封；data 形如 {节点名: 该节点返回的 state 更新}
    for node_name, state_update in chunk["data"].items():
        print(f"[节点 {node_name}] 更新键: {list(state_update.keys())}")

print("\n说明：v2 下即便传单个模式（字符串），chunk 也是完整信封，data 里才是节点更新。")


# ── 演示 2：多模式 [updates, custom] ─────────────────────────────────────────
# 传列表时，用 chunk["type"] 分支即可统一处理。

print("\n=== 演示 2：多模式 [updates, custom]（列表参数）===\n")

for chunk in graph.stream(
    {"messages": [HumanMessage(content="咖啡")]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    match chunk["type"]:
        case "updates":
            for node_name, state_update in chunk["data"].items():
                print(f"[updates] 节点 {node_name} → 键 {list(state_update.keys())}")
        case "custom":
            # data 是节点内 writer 推送的任意字典
            print(f"[custom] 进度: {chunk['data']}")

print("\n说明：v2 单模式与多模式的 chunk 结构一致，代码可统一用 chunk['type'] 分支。")


# ── 演示 3：四种模式一起消费 ─────────────────────────────────────────────────
# values（完整快照）、updates（增量）、custom（自定义）、messages（LLM token）

print("\n=== 演示 3：四种模式一起消费 [values, updates, custom, messages] ===\n")

counts: dict[str, int] = {}
for chunk in graph.stream(
    {"messages": [HumanMessage(content="猫")]},
    stream_mode=["values", "updates", "custom", "messages"],
    version="v2",
):
    mode = chunk["type"]
    counts[mode] = counts.get(mode, 0) + 1
    match mode:
        case "values":
            n = len(chunk["data"].get("messages", []))
            print(f"[values]  完整快照，消息数: {n}")
        case "updates":
            node = next(iter(chunk["data"]), "?")
            print(f"[updates] 节点 {node} 产出更新")
        case "custom":
            print(f"[custom]  {chunk['data']}")
        case "messages":
            # data = (message_chunk, metadata)
            msg_chunk, _meta = chunk["data"]
            text = msg_chunk.content if isinstance(msg_chunk.content, str) else ""
            if text:
                print(f"[messages] token: {text!r}")

print("\n--- 各模式收到的事件数 ---")
for mode, count in sorted(counts.items()):
    print(f"  {mode}: {count}")
