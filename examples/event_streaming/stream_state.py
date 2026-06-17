"""状态快照流式输出（Stream State / Values）。

演示 stream.values 投影：
  - 每一步执行后获取完整的状态快照
  - stream.output 等待最终输出

运行：
    uv run python examples/event_streaming/stream_state.py
"""

from operator import add
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """多步流水线状态。"""
    input_text: str
    step_results: Annotated[list[str], add]
    final_output: str


# ── 构建多步流水线图 ──────────────────────────────────────────────────────────

def step1_validate(state: PipelineState) -> dict[str, Any]:
    """第 1 步：验证输入。"""
    text = state["input_text"]
    result = f"验证通过：输入长度 {len(text)} 字符"
    print(f"  方法内部 print [step1_validate] {result}")
    return {"step_results": [result]}


def step2_transform(state: PipelineState) -> dict[str, Any]:
    """第 2 步：转换文本。"""
    text = state["input_text"]
    transformed = text.upper()
    result = f"转换完成：{transformed}"
    print(f"  方法内部 print [step2_transform] {result}")
    return {"step_results": [result]}


def step3_summarize(state: PipelineState) -> dict[str, Any]:
    """第 3 步：生成摘要。"""
    steps_done = len(state["step_results"])
    summary = f"流水线已完成 {steps_done} 个步骤，输入为：{state['input_text']}"
    print(f"  方法内部 print [step3_summarize] {summary}")
    return {"step_results": [summary], "final_output": summary}


workflow = StateGraph(PipelineState)
workflow.add_node("step1_validate", step1_validate)
workflow.add_node("step2_transform", step2_transform)
workflow.add_node("step3_summarize", step3_summarize)

workflow.add_edge(START, "step1_validate")
workflow.add_edge("step1_validate", "step2_transform")
workflow.add_edge("step2_transform", "step3_summarize")
workflow.add_edge("step3_summarize", END)

graph = workflow.compile()


# ── 演示 1：逐步观察状态快照 ───────────────────────────────────────────────────

print("=== 演示 1：逐步观察状态快照（stream.values）===\n")

stream = graph.stream_events(
    {
        "input_text": "Hello LangGraph Event Streaming",
        "step_results": [],
        "final_output": "",
    },
    version="v3",
)

step = 0
for snapshot in stream.values:
    step += 1
    print(f"\n[快照 {step}]")
    for key, value in snapshot.items():
        print(f"  {key}: {value}")

print(f"\n--- 共收到 {step} 个状态快照 ---")


# ── 演示 2：获取最终输出 ───────────────────────────────────────────────────────

print("\n=== 演示 2：获取最终输出（stream.output）===\n")

stream2 = graph.stream_events(
    {
        "input_text": "测试 stream.output 功能",
        "step_results": [],
        "final_output": "",
    },
    version="v3",
)

# 先消费所有中间事件
for _ in stream2.values:
    pass

# 再获取最终输出
final = stream2.output
print(f"[最终输出] final_output: {final['final_output']}")
print(f"[最终输出] step_results: {final['step_results']}")
