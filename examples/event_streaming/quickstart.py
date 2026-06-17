"""事件流式输出（Event Streaming）快速入门。

演示 LangGraph 推荐的进程内流式模型：
  graph.stream_events(input, version="v3") 返回 GraphRunStream，
  可同时消费 stream.messages（逐 token）和 stream.output（最终状态）。

运行：
    uv run python examples/event_streaming/quickstart.py
"""

from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.math import ARITHMETIC_TOOLS


# ── 创建模型与 Agent ──────────────────────────────────────────────────────────

factory = ModelFactory()
model = factory.create_dashscope_chat_model()

agent = create_agent(
    model=model,
    tools=ARITHMETIC_TOOLS,
    system_prompt="你是一个乐于助人的数学助手，请用中文回答。",
)


# ── 使用 stream_events 流式消费 ────────────────────────────────────────────────

print("=== 快速入门：stream_events ===\n")

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "请计算 42 * 17 是多少？"}]},
    version="v3",
)

# 逐 token 打印消息输出
print("[流式输出] ", end="", flush=True)
for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)
print()

# 获取最终状态
final_state = stream.output
print(f"\n\n[最终状态] 消息数量: {len(final_state['messages'])}")
print(f"[最后一条消息] {final_state['messages'][-1].content}")
