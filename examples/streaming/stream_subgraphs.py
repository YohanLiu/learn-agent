"""子图流式输出（Stream Subgraph Outputs）。

演示 graph.stream(..., subgraphs=True, version="v2")：
  - 默认只看到父图节点的更新，子图内部的执行被「折叠」
  - 开启 subgraphs=True 后，子图内部节点的更新也会进入流
  - 用 chunk["ns"]（命名空间元组）区分一条更新来自根图还是某个子图

对应文档：https://docs.langchain.com/oss/python/langgraph/streaming#subgraph-outputs

运行：
    uv run python examples/streaming/stream_subgraphs.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class State(TypedDict):
    """父图与子图共享同一个状态结构。"""
    foo: str


# ── 定义子图 ──────────────────────────────────────────────────────────────────

def child_node(state: State) -> dict:
    """子图内部节点：在 foo 上追加后缀。"""
    return {"foo": state["foo"] + "_child"}


sub_builder = StateGraph(State)
sub_builder.add_node("child_node", child_node)
sub_builder.add_edge(START, "child_node")
sub_builder.add_edge("child_node", END)
subgraph = sub_builder.compile()


# ── 定义父图 ──────────────────────────────────────────────────────────────────

def parent_first(state: State) -> dict:
    """父图前置节点。"""
    return {"foo": state["foo"] + "_p1"}


def call_subgraph(state: State) -> dict:
    """父图节点：直接调用（invoke）子图。"""
    return subgraph.invoke({"foo": state["foo"]})


builder = StateGraph(State)
builder.add_node("parent_first", parent_first)
builder.add_node("call_subgraph", call_subgraph)
builder.add_edge(START, "parent_first")
builder.add_edge("parent_first", "call_subgraph")
builder.add_edge("call_subgraph", END)
graph = builder.compile()


# ── 演示 1：不开启 subgraphs —— 子图被折叠 ─────────────────────────────────

print("=== 演示 1：subgraphs=False（默认，子图被折叠）===\n")

for chunk in graph.stream(
    {"foo": "start"},
    stream_mode="updates",
    version="v2",
    subgraphs=False,
):
    # v2 下 chunk 是完整信封；data 里是 {节点名: 更新}，且只能看到父图节点
    for node, update in chunk["data"].items():
        print(f"  [父图] {node} → foo={update['foo']}")

print("\n说明：默认看不到子图内部执行，call_subgraph 像个黑盒。\n")


# ── 演示 2：开启 subgraphs=True —— 子图内部更新也进流 ────────────────────────

print("=== 演示 2：subgraphs=True（子图内部更新也进入流）===\n")

for chunk in graph.stream(
    {"foo": "start"},
    stream_mode=["updates"],  # 用列表，拿到完整信封含 ns
    version="v2",
    subgraphs=True,
):
    # chunk 是完整信封 {"type", "ns", "data"}
    ns = chunk["ns"]
    if len(ns) == 0:
        scope = "根图"
    else:
        # ns 元组的元素形如 "call_subgraph:<task_id>"
        scope = f"子图({ns[0].split(':')[0]})"
    for node, update in chunk["data"].items():
        print(f"  [{scope}] {node} → foo={update['foo']}")

print("\n说明：ns 为空元组 → 根图更新；ns 非空 → 子图内部更新。")


# ── 演示 3：只过滤子图内部的事件 ─────────────────────────────────────────────

print("=== 演示 3：只看子图内部事件（ns 非空）===\n")

subgraph_events = 0
for chunk in graph.stream(
    {"foo": "start"},
    stream_mode=["updates", "custom"],
    version="v2",
    subgraphs=True,
):
    if len(chunk["ns"]) > 0:  # 只关心子图事件
        subgraph_events += 1
        node = next(iter(chunk["data"]), "?")
        print(f"  [子图事件] 节点 {node}")

print(f"\n子图内部事件数: {subgraph_events}")
print("说明：用 chunk['ns'] 是否为空，可精准分离根图与子图事件。")
