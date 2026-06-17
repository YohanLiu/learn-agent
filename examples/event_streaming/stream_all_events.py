"""原始协议事件流（Stream All Protocol Events）。

演示直接迭代 stream 对象获取所有原始协议事件：
  - 每个事件是 ProtocolEvent 信封，包含 seq、method、params
  - method 是 channel 名称（messages/values/updates/tools/lifecycle 等）
  - params.namespace 是事件来源的路径
  - 可按 method 过滤感兴趣的 channel

运行：
    uv run python examples/event_streaming/stream_all_events.py
"""

from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from models.model_factory import ModelFactory
from tools.math import ARITHMETIC_TOOLS
from tools.weather import get_weather


# ── 创建模型与 Agent ──────────────────────────────────────────────────────────

factory = ModelFactory()
model = factory.create_modelscope_chat_model()

agent = create_agent(
    model=model,
    tools=ARITHMETIC_TOOLS + [get_weather],
    system_prompt="你是一个全能助手，用中文回答。",
)


# ── 演示 1：遍历所有原始事件 ──────────────────────────────────────────────────

print("=== 演示 1：遍历所有原始协议事件 ===\n")

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "请计算 42 * 17"}]},
    version="v3",
)

event_count = 0
method_counts: dict[str, int] = {}

for event in stream:
    event_count += 1
    method = event["method"]
    namespace = event["params"]["namespace"]

    # 统计各 channel 的事件数量
    method_counts[method] = method_counts.get(method, 0) + 1

    # 只打印前 15 个事件的详情，避免输出过长
    if event_count <= 15:
        print(f"[事件 {event_count}] method={method}, namespace={namespace}")
        data = event["params"]["data"]
        # 简要展示 data
        data_str = str(data)
        if len(data_str) > 120:
            data_str = data_str[:120] + "..."
        print(f"  └─ data: {data_str}")

print(f"\n--- 共收到 {event_count} 个原始事件 ---")
print("--- 各 channel 事件数量 ---")
for method, count in sorted(method_counts.items()):
    print(f"  {method}: {count}")


# ── 演示 2：按 channel 过滤 — 只关注 messages channel ─────────────────────────

print("\n=== 演示 2：按 channel 过滤 — 只看 messages channel ===\n")

stream2 = agent.stream_events(
    {"messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}]},
    version="v3",
)

for event in stream2:
    if event["method"] != "messages":
        continue

    data = event["params"]["data"]
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        continue
    data = data[0] if isinstance(data, (list, tuple)) else data

    if not isinstance(data, dict):
        continue

    event_type = data.get("event", "")
    if event_type == "content-block-delta":
        block = data.get("delta") or {}
        block_type = block.get("type", "")
        if block_type == "text-delta":
            print(block.get("text", ""), end="", flush=True)
        elif block_type == "reasoning-delta":
            print(f"[思考]{block.get('reasoning', '')}", end="", flush=True)
    elif event_type == "message-start":
        pass  # 消息开始
    elif event_type == "message-finish":
        print()  # 消息结束，换行

print("\n\n--- messages channel 过滤完成 ---")


# ── 演示 3：观察 lifecycle 事件 ───────────────────────────────────────────────
# lifecycle 通道由 LifecycleTransformer 产生，仅在存在子图时才会触发。
# 事件类型包括：started / running / completed / failed / interrupted
# 因此需要构建一个包含子图的复合图来演示。

print("\n=== 演示 3：观察 lifecycle 事件（需要子图）===\n")


# 构建一个简单的子图
class SubState(TypedDict):
    """子图状态。"""
    messages: Annotated[list, add_messages]


sub_workflow = StateGraph(SubState)


def echo_node(state: dict) -> dict[str, Any]:
    """简单回显节点。"""
    msgs = state.get("messages", [])
    last = msgs[-1].content if msgs else ""
    return {"messages": [AIMessage(content=f"回显: {last}")]}


sub_workflow.add_node("echo", echo_node)
sub_workflow.add_edge(START, "echo")
sub_workflow.add_edge("echo", END)
echo_graph = sub_workflow.compile(name="echo_subgraph")


class MainState(TypedDict):
    """主图状态。"""
    messages: Annotated[list, add_messages]


main_workflow = StateGraph(MainState)
main_workflow.add_node("echo_subgraph", echo_graph)
main_workflow.add_edge(START, "echo_subgraph")
main_workflow.add_edge("echo_subgraph", END)
main_graph = main_workflow.compile()

stream3 = main_graph.stream_events(
    {"messages": [HumanMessage(content="你好")]},
    version="v3",
)

for event in stream3:
    if event["method"] != "lifecycle":
        continue
    data = event["params"]["data"]
    if isinstance(data, dict):
        event_type = data.get("event", "")
        graph_name = data.get("graph_name", "")
        ns = data.get("namespace", [])
        print(f"[lifecycle] event={event_type}, graph_name={graph_name}, namespace={ns}")

print("\n--- lifecycle channel 过滤完成 ---")
