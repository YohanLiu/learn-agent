"""消息流式输出（Stream LLM Tokens: messages 模式）。

演示 graph.stream(stream_mode="messages", version="v2")：
  - 逐 token 获取 LLM 输出，配合 metadata 做精细化过滤
  - 按 tags 过滤（给模型实例打标签）
  - 用 nostream 标签让内部模型 token 不进流
  - 按 langgraph_node 过滤指定节点

注意：本示例需要调用真实 LLM，会消耗 API 额度。

对应文档：https://docs.langchain.com/oss/python/langgraph/streaming#llm-tokens

运行：
    uv run python examples/streaming/stream_messages.py
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from models.model_factory import ModelFactory

# ── 准备模型 ──────────────────────────────────────────────────────────────────
# 文档里硬编码 gpt/claude，这里统一换成项目已配置的 ModelFactory。
_factory = ModelFactory()
base_model = _factory.create_dashscope_chat_model()

# 通过 .with_config(tags=[...]) 给模型实例打标签——后续可按 tag 过滤 token。
joke_model = base_model.with_config(tags=["joke"])   # 讲笑话的模型
poem_model = base_model.with_config(tags=["poem"])   # 写诗的模型


# ── 定义状态与节点 ────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


def tell_joke(state: State) -> dict[str, Any]:
    """用 joke 模型讲一个笑话。"""
    messages = state["messages"] + [HumanMessage(content="讲一个关于程序员的笑话")]
    resp: AIMessage = joke_model.invoke(messages)
    return {"messages": [resp]}


def write_poem(state: State) -> dict[str, Any]:
    """用 poem 模型写一首诗。"""
    messages = state["messages"] + [HumanMessage(content="写两句关于秋天的诗")]
    resp: AIMessage = poem_model.invoke(messages)
    return {"messages": [resp]}


workflow = StateGraph(State)
workflow.add_node("tell_joke", tell_joke)
workflow.add_node("write_poem", write_poem)
workflow.add_edge(START, "tell_joke")
workflow.add_edge("tell_joke", "write_poem")
workflow.add_edge("write_poem", END)
graph = workflow.compile()

INPUT = {"messages": [HumanMessage(content="你好")]}


# ── 演示 1：基础逐 token 流 ──────────────────────────────────────────────────

print("=== 演示 1：基础逐 token 流（messages 模式）===\n")

print("[tell_joke] ", end="", flush=True)
for chunk in graph.stream(INPUT, stream_mode="messages", version="v2"):
    # v2 下 chunk 是完整信封；data = (message_chunk, metadata)
    msg_chunk, metadata = chunk["data"]
    text = msg_chunk.content if isinstance(msg_chunk.content, str) else ""
    if text:
        print(text, end="", flush=True)
print("\n\n说明：messages 模式可拿到 LLM 的逐 token 输出（此处是完整消息块）。\n")


# ── 演示 2：按 tags 过滤 token ───────────────────────────────────────────────

print("=== 演示 2：按 tags 过滤（只接收 poem 标签的 token）===\n")

print("[仅 poem 模型] ", end="", flush=True)
for chunk in graph.stream(INPUT, stream_mode="messages", version="v2"):
    msg_chunk, metadata = chunk["data"]
    tags = metadata.get("tags", [])
    if "poem" not in tags:
        continue  # 跳过 joke 模型的 token
    text = msg_chunk.content if isinstance(msg_chunk.content, str) else ""
    if text:
        print(text, end="", flush=True)
print("\n\n说明：给模型实例打 tag 后，消费端可按 tag 只看感兴趣的模型 token。\n")


# ── 演示 3：按 langgraph_node 过滤 ───────────────────────────────────────────

print("=== 演示 3：按 langgraph_node 过滤（只接收 write_poem 节点）===\n")

print("[仅 write_poem 节点] ", end="", flush=True)
for chunk in graph.stream(INPUT, stream_mode="messages", version="v2"):
    msg_chunk, metadata = chunk["data"]
    if metadata.get("langgraph_node") != "write_poem":
        continue  # 只看 write_poem 节点产出的 token
    text = msg_chunk.content if isinstance(msg_chunk.content, str) else ""
    if text:
        print(text, end="", flush=True)
print("\n\n说明：metadata['langgraph_node'] 标记了 token 来自哪个图节点。\n")


# ── 演示 4：nostream 标签 —— 关闭某模型的 token 流 ───────────────────────────
# 在调用侧用 with_config(tags=["nostream"])，再消费时跳过 nostream，
# 即可让「内部辅助模型」的 token 不出现在流里。

print("=== 演示 4：nostream 标签 ===\n")

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph

stream_model = _factory.create_yunwu_chat_model(model="claude-haiku-4-5-20251001:floor")
internal_model = _factory.create_yunwu_chat_model(model="claude-haiku-4-5-20251001:floor").with_config(
    {"tags": ["nostream"]}
)


class State2(TypedDict):
    topic: str
    answer: str
    notes: str


def answer(state: State2) -> dict[str, Any]:
    r = stream_model.invoke(
        [{"role": "user", "content": f"简要回复关于 {state['topic']} 的内容"}]
    )
    return {"answer": r.content}


def internal_notes(state: State2) -> dict[str, Any]:
    # Tokens from this model are omitted from stream_mode="messages" because of nostream
    r = internal_model.invoke(
        [{"role": "user", "content": f"关于 {state['topic']} 的私人笔记"}]
    )
    return {"notes": r.content}


graph = (
    StateGraph(State2)
    .add_node("write_answer", answer)
    .add_node("internal_notes", internal_notes)
    .add_edge(START, "write_answer")
    .add_edge("write_answer", "internal_notes")
    .compile()
)

initial_state: State2 = {"topic": "AI", "answer": "", "notes": ""}
stream = graph.stream_events(initial_state,version="v3")

# 逐 token 打印消息输出
# 私人笔记的token消息这里没打印出来
print("[流式输出] ", end="", flush=True)
for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)
print()

# 获取最终状态
final_state = stream.output
# 私人笔记的回复在最终消息里出现
print(f"[最终消息topic] {final_state['topic']}")
print(f"[最终消息answer] {final_state['answer']}")
print(f"[最终消息notes] {final_state['notes']}")

print("\n\n说明：用 nostream 标签可让某些模型的 token 不进流，常用于内部辅助调用。")