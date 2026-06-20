"""子图流式观察（Stream Subgraphs）。

演示 stream.subgraphs 投影：
  - 观察嵌套图的执行过程，无需手动解析 namespace 字符串
  - subgraph.graph_name 是子图的名称
  - subgraph.path 是子图在父图中的路径
  - subgraph.messages 是子图内部的消息流

运行：
    uv run python examples/event_streaming/stream_subgraphs.py
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.math import ARITHMETIC_TOOLS
from utils.graph_utils import GraphVisualizer


# ── 构建带子图的复合 Agent ─────────────────────────────────────────────────────

factory = ModelFactory()
model = factory.create_dashscope_chat_model()

# 子图 1：数学计算 Agent（命名子图）
math_agent = create_agent(
    model=model,
    tools=ARITHMETIC_TOOLS,
    system_prompt="你是一个数学计算专家，只做数学计算，用中文回答。",
    name="math_expert",
)

# ── 定义状态类型（messages 使用 reducer 追加合并，避免被节点返回值覆盖）──

class MainState(TypedDict):
    """主图状态，messages 使用 add_messages reducer 实现追加合并。"""
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
    summary = f"总结：用户提到了「{last_content[:30]}...」"
    return {"messages": [AIMessage(content=summary)]}


text_workflow.add_node("summarizer", summarize_node)
text_workflow.add_edge(START, "summarizer")
text_workflow.add_edge("summarizer", END)
text_graph = text_workflow.compile(name="text_summarizer")


# 主图：路由到不同子图
main_workflow = StateGraph(MainState)


def route_to_subgraph(state: dict) -> dict[str, Any]:
    """根据用户消息路由到对应子图。"""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""
    # 简单关键词路由
    if any(kw in last_msg for kw in ["计算", "乘", "加", "减", "除"]):
        return {"route": "math"}
    return {"route": "text"}


# 将编译好的子图直接作为节点添加（而非包装在函数中调用 .invoke()），
# 这样 LangGraph 能将其识别为真正的子图，内部事件可传播到外层 stream_events。
main_workflow.add_node("router", route_to_subgraph)
main_workflow.add_node("math_expert", math_agent)
main_workflow.add_node("text_summarizer", text_graph)
main_workflow.add_edge(START, "router")
main_workflow.add_conditional_edges(
    "router",
    lambda s: "math_expert" if s.get("route") == "math" else "text_summarizer",
)
main_workflow.add_edge("math_expert", END)
main_workflow.add_edge("text_summarizer", END)

main_graph = main_workflow.compile()

GraphVisualizer().show(main_graph, 'stream_subgraphs_main_graph', xray=True)

# ── 演示 1：流式观察子图执行 ───────────────────────────────────────────────────

print("=== 演示 1：流式观察子图执行 ===\n")

stream = main_graph.stream_events(
    {"messages": [HumanMessage(content="请计算 42 * 17")], "route": ""},
    version="v3",
)

for subgraph in stream.subgraphs:
    print(f"[子图事件] graph_name={subgraph.graph_name}, path={subgraph.path}")
    for message in subgraph.messages:
        text = str(message.text)
        if text.strip():
            print(f"  └─ 消息: {text[:80]}")

print(f"\n[最终输出] {stream.output}")


# ── 演示 2：使用 interleave 同时观察消息与子图 ────────────────────────────
# 编译好的子图直接作为节点添加后，内部 LLM 流式事件能传播到顶层，
# interleave 可同时捕获 [消息] 和 [子图] 两类事件。

print("\n=== 演示 2：使用 interleave 同时观察消息与子图 ===\n")

stream2 = main_graph.stream_events(
    {"messages": [HumanMessage(content="计算 15 + 28")], "route": ""},
    version="v3",
)

for name, item in stream2.interleave("messages", "subgraphs"):
    if name == "messages":
        text = str(item.text)
        if text.strip():
            print(f"[消息] {text[:80]}")
    elif name == "subgraphs":
        print(f"[子图] graph_name={item.graph_name}, path={item.path}")

print(f"\nstream2[最终输出] {stream2.output}")
