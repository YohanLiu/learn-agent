"""消息流式输出（Stream Messages）。

演示 stream.messages 投影：
  - 逐消息获取文本与 usage_metadata
  - message.text 可逐 token 迭代，也可 str() 获取完整文本
  - message.tool_calls 获取工具调用参数块

运行：
    uv run python examples/event_streaming/stream_messages.py
"""

from langchain.agents import create_agent
from models.model_factory import ModelFactory
from tools.math import ARITHMETIC_TOOLS
from tools.weather import get_weather


# ── 创建模型与 Agent ──────────────────────────────────────────────────────────

factory = ModelFactory()
model = factory.create_dashscope_chat_model()

agent = create_agent(
    model=model,
    tools=ARITHMETIC_TOOLS + [get_weather],
    system_prompt="你是一个全能助手，可以用中文回答问题、做数学计算、查询天气。",
)


# ── 演示 1：逐消息获取文本与 usage ──────────────────────────────────────────────

print("=== 演示 1：逐消息获取文本与 usage ===\n")

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "请计算 123 + 456，并告诉我北京今天天气如何？"}]},
    version="v3",
)

message_count = 0
for message in stream.messages:
    message_count += 1
    text = str(message.text)
    usage = message.output.usage_metadata

    if text.strip():
        print(f"[消息 {message_count}] {text}")
    if usage:
        print(f"  └─ usage: input_tokens={usage.get('input_tokens', 0)}, "
              f"output_tokens={usage.get('output_tokens', 0)}")

print(f"\n共收到 {message_count} 条消息流事件")


# ── 演示 2：逐 token 打印（打字机效果）─────────────────────────────────────────

print("\n=== 演示 2：逐 token 打印（打字机效果）===\n")

stream2 = agent.stream_events(
    {"messages": [{"role": "user", "content": "用三句话解释什么是 LangGraph？"}]},
    version="v3",
)

print("[打字机效果] ", end="", flush=True)
for message in stream2.messages:
    for token in message.text:
        print(token, end="", flush=True)
print()


# ── 演示 3：工具调用块 ─────────────────────────────────────────────────────────

print("\n=== 演示 3：工具调用块（tool_calls）===\n")

stream3 = agent.stream_events(
    {"messages": [{"role": "user", "content": "帮我计算 99 * 88"}]},
    version="v3",
)

seen_tool_ids: set[str] = set()
tool_info: dict[int, dict] = {}  # index -> {name, args}

for message in stream3.messages:
    if message.tool_calls:
        for tc in message.tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", "")
            tc_index = tc.get("index", 0)

            # 首次出现的工具调用记录名称
            if tc_id and tc_id not in seen_tool_ids:
                seen_tool_ids.add(tc_id)
                tool_info[tc_index] = {"name": tc_name, "args": ""}

            # args 是渐进式快照，每次都覆盖（取最新的即为完整参数）
            if tc_args and tc_index in tool_info:
                tool_info[tc_index]["args"] = tc_args

# 打印每个工具调用的名称和完整参数
for idx, info in tool_info.items():
    print(f"[工具调用] name={info['name']}, args={info['args']}")

# 等待最终输出
final = stream3.output
print(f"\n[最终回答] {final['messages'][-1].content}")
