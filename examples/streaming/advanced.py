"""流式进阶（Streaming Advanced：禁用流式 / 迁移 v2 / 异步）。

合并演示文档 Streaming 章节的进阶主题（version="v2"）：
  - Disable streaming：用 disable_streaming 关闭某模型的 token 流
  - Migrate to v2：invoke(..., version="v2") 返回 GraphOutput，含 .value / .interrupts
  - Async：异步 astream，以及节点签名里显式接收 writer / config

注意：演示 1 的「禁用流式」消费段需要真实 LLM 调用，会消耗 API 额度；
      其余演示为纯本地逻辑，可直接运行。

对应文档：
  - https://docs.langchain.com/oss/python/langgraph/streaming#disable-streaming
  - https://docs.langchain.com/oss/python/langgraph/streaming#migrate-to-v2
  - https://docs.langchain.com/oss/python/langgraph/streaming#async

运行：
    uv run python examples/streaming/advanced.py
"""

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from models.model_factory import ModelFactory


# ── 1. Disable streaming：禁用某模型的 token 流 ──────────────────────────────
# 文档里用 ChatOpenAI(streaming=False)；本项目统一用 ModelFactory，对应开关是
# disable_streaming（当前 langchain 版本的正确字段）。开启后该模型的 token 不进流。

_factory = ModelFactory()
streaming_model = _factory.create_dashscope_chat_model()               # 默认可流式
nostream_model = _factory.create_dashscope_chat_model(                 # 关闭流式
    disable_streaming=True,
)


class MsgState(TypedDict):
    messages: Annotated[list, add_messages]


def call_nostream_model(state: MsgState, config: RunnableConfig) -> dict[str, Any]:
    """调用「关闭流式」的模型——其 token 不会出现在 messages 流里。"""
    resp: AIMessage = nostream_model.invoke(state["messages"])
    return {"messages": [resp]}


msg_builder = StateGraph(MsgState)
msg_builder.add_node("call", call_nostream_model)
msg_builder.add_edge(START, "call")
msg_builder.add_edge("call", END)
msg_graph = msg_builder.compile()


# ── 2. Migrate to v2：GraphOutput ─────────────────────────────────────────────

class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


def step(state: SimpleState) -> dict[str, Any]:
    return {"result": "已完成", "messages": [AIMessage(content="ok")]}


simple_builder = StateGraph(SimpleState)
simple_builder.add_node("step", step)
simple_builder.add_edge(START, "step")
simple_builder.add_edge("step", END)
simple_graph = simple_builder.compile()


# ── 3. Async：异步流 + 节点显式接收 writer / config ─────────────────────────

class AsyncState(TypedDict):
    messages: Annotated[list, add_messages]


def emit_with_writer(state: AsyncState, writer: StreamWriter) -> dict[str, Any]:
    """节点签名里显式声明 writer 参数（等价于 get_stream_writer()）。"""
    for i in range(3):
        writer({"step": i, "msg": f"阶段 {i}"})  # 显式 writer 推送 custom 数据
    return {"messages": [AIMessage(content="异步节点完成")]}


async_builder = StateGraph(AsyncState)
async_builder.add_node("emit", emit_with_writer)
async_builder.add_edge(START, "emit")
async_builder.add_edge("emit", END)
async_graph = async_builder.compile()


# ── 演示 1：disable_streaming（关闭流式）──────────────────────────────────────

print("=== 演示 1：disable_streaming（关闭某模型的 token 流）===\n")
print("提示：本演示需调用真实 LLM。若未配置密钥可跳过本段。\n")

try:
    token_count = 0
    for chunk in msg_graph.stream(
        {"messages": [HumanMessage(content="用一个词回答：你好")]},
        stream_mode="messages",
        version="v2",
    ):
        # v2 下 chunk 是完整信封；data = (message_chunk, metadata)
        msg_chunk, _metadata = chunk["data"]
        token_count += 1
    print(f"messages 流收到的事件数: {token_count}")
    print("说明：开启 disable_streaming 后，整段回答以单个事件返回，无逐 token 流。\n")
except Exception as exc:  # 缺密钥或网络问题时优雅降级
    print(f"  （跳过实际调用：{type(exc).__name__}: {exc}）\n"
          "  说明：开启 disable_streaming=True 后，该模型 token 不会逐块进入流。\n")


# ── 演示 2：Migrate to v2 —— GraphOutput ──────────────────────────────────────

print("=== 演示 2：invoke(..., version='v2') 返回 GraphOutput ===\n")

# 关键：传 version="v2"，返回值不再是裸 dict，而是 GraphOutput 对象
output = simple_graph.invoke(
    {"messages": [HumanMessage(content="hi")]},
    version="v2",
)
print(f"  返回类型: {type(output).__name__}")  # GraphOutput
print(f"  .value   : {output.value}")          # 最终状态
print(f"  .interrupts: {output.interrupts}")   # 中断信息（空元组表示无中断）
print(
    "\n说明：迁移到 v2 后，invoke/stream 返回 GraphOutput，"
    "通过 .value 取最终状态、.interrupts 取中断（人机交互场景常用）。\n"
)


# ── 演示 3：Async —— 异步流 + 显式 writer ─────────────────────────────────────

print("=== 演示 3：异步流 astream（+ 节点显式 writer 参数）===\n")


async def run_async() -> None:
    """异步消费 astream。"""
    async for chunk in async_graph.astream(
        {"messages": [HumanMessage(content="start")]},
        stream_mode=["custom", "updates"],
        version="v2",
    ):
        match chunk["type"]:
            case "custom":
                d = chunk["data"]
                print(f"  [custom] 阶段 {d['step']}: {d['msg']}")
            case "updates":
                print(f"  [updates] 节点完成")


asyncio.run(run_async())
print(
    "\n说明：①异步用 astream/ainvoke；②节点签名可显式声明 writer: StreamWriter 参数"
    "（等价于 get_stream_writer()），也适用于 3.11 以下版本的 config 透传场景。"
)
