"""多投影并发消费（Stream Multiple Projections）。

演示两种并发消费多个投影的方式：
  - 同步 interleave：stream.interleave("values", "messages", "subgraphs")
    按严格到达顺序交错消费，每个事件以 (name, item) 元组返回
  - 异步 asyncio.gather：astream_events + asyncio.gather
    并发消费多个投影，各投影独立迭代、互不阻塞

运行：
    uv run python examples/event_streaming/stream_multiple_projections.py
"""

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.math import ARITHMETIC_TOOLS
from tools.weather import get_weather
from utils.graph_utils import GraphVisualizer

# ── 创建模型与复合 Agent（带子图）──────────────────────────────────────────────

factory = ModelFactory()
model = factory.create_dashscope_chat_model()

# 子图 1：数学 + 天气 Agent（命名子图）
tool_agent = create_agent(
    model=model,
    tools=ARITHMETIC_TOOLS + [get_weather],
    system_prompt="你是一个全能助手，可以用中文回答问题、做数学计算、查询天气。",
    name="tool_expert",
)


# ── 定义状态类型 ──

class MainState(TypedDict):
    """主图状态。"""
    messages: Annotated[list, add_messages]
    route: str


class TextState(TypedDict):
    """文本子图状态。"""
    messages: Annotated[list, add_messages]


# 子图 2：文本处理节点（手动构建的简单子图）
text_workflow = StateGraph(TextState)


def summarize_node(state: dict) -> dict[str, Any]:
    """简单的文本总结节点。"""
    messages = state.get("messages", [])
    last_content = messages[-1].content if messages else ""
    summary = f"总结：用户提到了「{last_content[:30]}」"
    return {"messages": [AIMessage(content=summary)]}


text_workflow.add_node("summarizer", summarize_node)
text_workflow.add_edge(START, "summarizer")
text_workflow.add_edge("summarizer", END)
text_graph = text_workflow.compile(name="text_summarizer")


# ── 构建主图：路由到不同子图 ──

main_workflow = StateGraph(MainState)


def route_to_subgraph(state: dict) -> dict[str, Any]:
    """根据用户消息路由到对应子图。"""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""
    if any(kw in last_msg for kw in ["计算", "乘", "加", "减", "除", "天气"]):
        return {"route": "tool"}
    return {"route": "text"}


main_workflow.add_node("router", route_to_subgraph)
# 直接将编译好的子图作为节点添加，这样 stream_events 才能捕获子图事件
main_workflow.add_node("tool_expert", tool_agent)
main_workflow.add_node("text_summarizer", text_graph)
main_workflow.add_edge(START, "router")
main_workflow.add_conditional_edges(
    "router",
    lambda s: "tool_expert" if s.get("route") == "tool" else "text_summarizer",
)
main_workflow.add_edge("tool_expert", END)
main_workflow.add_edge("text_summarizer", END)

agent = main_workflow.compile()

GraphVisualizer().show(agent, 'stream_multiple_projections_graph', xray=True)


# ── 演示 1：同步 interleave 交错消费 values + messages + subgraphs ────────────

print("=== 演示 1：interleave 交错消费 values + messages + subgraphs ===\n")

stream = agent.stream_events(
    {"messages": [HumanMessage(content="请计算 15 * 23")], "route": ""},
    version="v3",
)

event_count = 0
for name, item in stream.interleave("values", "messages", "subgraphs"):
    event_count += 1
    if name == "values":
        # item 是状态快照（dict）
        print(f"[state] keys={list(item)}")
    elif name == "messages":
        # item 是 ChatModelStream，直接迭代可获取原始协议事件（按到达顺序）
        # 这比分别用 message.text / message.tool_calls 投影更能保证时序一致
        print(f"[llm] node={item.node}")
        for event in item:
            if event.get("event") != "content-block-delta":
                continue
            delta = event.get("delta") or {}
            dtype = delta.get("type", "")
            if dtype == "text-delta":
                text = delta.get("text", "")
                if text.strip():
                    display = text[:60] + "..." if len(text) > 60 else text
                    print(f"  [文本] {display}")
            elif dtype in ("block-delta", "legacy-block-delta"):
                fields = delta.get("fields") or {}
                if fields.get("type") == "tool_call_chunk":
                    tc_name = fields.get("name") or ""
                    tc_args = fields.get("args") or ""
                    if tc_name:
                        print(f"  [工具调用] {tc_name}({tc_args}")
                    elif tc_args:
                        print(f"  [工具调用] ...{tc_args}")
    elif name == "subgraphs":
        # item 是子图事件
        print(f"[子图] graph_name={item.graph_name}, path={item.path}")

print(f"\n--- 共收到 {event_count} 个交错事件 ---")


# ── 演示 2：asyncio.gather 异步并发消费 messages + subgraphs ─────────────────

async def async_demo():
    """异步并发消费：使用 astream_events + asyncio.gather 同时消费多个投影。"""
    print("\n=== 演示 2：asyncio.gather 异步并发消费 messages + subgraphs ===\n")

    stream = await agent.astream_events(
        {"messages": [HumanMessage(content="上海今天天气怎么样？")], "route": ""},
        version="v3",
    )

    async def consume_messages():
        """消费 LLM 消息投影。直接迭代原始协议事件，按到达顺序统一处理文本和工具调用。"""
        async for message in stream.messages:
            print(f"[llm] node={message.node}")
            # async for event in message 迭代原始 MessagesData 事件
            async for event in message:
                if event.get("event") != "content-block-delta":
                    continue
                delta = event.get("delta") or {}
                dtype = delta.get("type", "")
                if dtype == "text-delta":
                    text = delta.get("text", "")
                    if text.strip():
                        display = text[:60] + "..." if len(text) > 60 else text
                        print(f"  [文本] {display}")
                elif dtype in ("block-delta", "legacy-block-delta"):
                    fields = delta.get("fields") or {}
                    if fields.get("type") == "tool_call_chunk":
                        tc_name = fields.get("name") or ""
                        tc_args = fields.get("args") or ""
                        if tc_name:
                            print(f"  [工具调用] {tc_name}({tc_args}")
                        elif tc_args:
                            print(f"  [工具调用] ...{tc_args}")

    async def consume_subgraphs():
        """消费子图投影。"""
        async for subgraph in stream.subgraphs:
            print(f"[子图] graph_name={subgraph.graph_name}, path={subgraph.path}")

    # asyncio.gather 并发驱动两个消费者，互不阻塞
    await asyncio.gather(consume_messages(), consume_subgraphs())

    final = (await stream.output())["messages"][-1].content
    print(f"\n[最终回答] {final}")


asyncio.run(async_demo())
