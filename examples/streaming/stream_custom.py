"""自定义数据流（Stream Custom Data）。

演示 graph.stream(stream_mode="custom", version="v2")：
  - 节点内用 get_stream_writer() 推送任意字典到 custom 通道
  - 进度、心跳、阶段标记等非状态信息走 custom，不污染图状态
  - Advanced：配合「任意 LLM」——非 LangChain 的生成器也能把分块推入流
  - Tool Calling：在 @tool 装饰的工具函数中推送进度，配合真实 LLM 调用

对应文档：
  - https://docs.langchain.com/oss/python/langgraph/streaming#custom-data
  - https://docs.langchain.com/oss/python/langgraph/streaming#use-with-any-llm

运行：
    uv run python examples/streaming/stream_custom.py
"""

import time
from typing import Annotated, Any, Iterator, TypedDict

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from models.model_factory import ModelFactory


# ── 定义状态与节点 ────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


def slow_work(state: State) -> dict[str, Any]:
    """模拟一个耗时节点：分阶段处理，并通过 writer 推送进度。"""
    writer = get_stream_writer()

    steps = ["下载数据", "解析格式", "聚合结果", "生成报告"]
    for i, step in enumerate(steps, start=1):
        time.sleep(0.1)  # 模拟耗时
        # custom 通道可推送任意可序列化字典
        writer({"phase": step, "progress": f"{i}/{len(steps)}", "percent": i / len(steps) * 100})

    return {"result": "报告已生成", "messages": [AIMessage(content="处理完成")]}


workflow = StateGraph(State)
workflow.add_node("slow_work", slow_work)
workflow.add_edge(START, "slow_work")
workflow.add_edge("slow_work", END)
graph = workflow.compile()

INPUT = {"messages": [HumanMessage(content="开始处理")]}


# ── 演示 1：custom 单模式 —— 只看进度 ────────────────────────────────────────

print("=== 演示 1：custom 模式（只接收自定义进度）===\n")

for chunk in graph.stream(INPUT, stream_mode="custom", version="v2"):
    # v2 下 chunk 是完整信封；data 才是节点内 writer 推送的字典
    data = chunk["data"]
    bar = "█" * int(data["percent"] // 10)
    print(f"  [{data['progress']}] {data['percent']:5.1f}% {bar} {data['phase']}")

print("\n说明：custom 通道专门承载进度/心跳等非状态信息，不进 State。\n")


# ── 演示 2：[updates, custom] 一起 —— 进度 + 结果 ───────────────────────────

print("=== 演示 2：[updates, custom]（进度 + 最终状态更新）===\n")

for chunk in graph.stream(INPUT, stream_mode=["updates", "custom"], version="v2"):
    match chunk["type"]:
        case "custom":
            print(f"  [custom]  阶段 {chunk['data']['progress']}: {chunk['data']['phase']}")
        case "updates":
            for node, update in chunk["data"].items():
                print(f"  [updates] 节点 {node} 完成，result={update.get('result')}")


# ── 演示 3：配合真实 LLM 分块推流 ───────────────────────────────────────────
# 文档强调：即使是自定义生成器（非 LangChain 模型），也能用 writer 把分块推入流，
# 消费端用 messages 模式或 custom 模式都能收到。这里用真实模型演示。

print("=== 演示 3：配合真实 LLM 流式输出 ===\n")


def generate_with_real_llm(state: State) -> dict[str, Any]:
    """节点内调用真实 LLM，把每个 token 经 writer 推入 custom 通道。"""
    writer = get_stream_writer()
    prompt = state["messages"][-1].content
    
    # 创建真实模型实例
    model_factory = ModelFactory()
    llm = model_factory.create_yunwu_chat_model(
        model="gpt-5.4-nano:floor",
        temperature=0.7,
        streaming=True,  # 启用流式输出
    )
    
    pieces: list[str] = []
    # 流式调用 LLM
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if hasattr(chunk, 'content') and chunk.content:
            token = chunk.content
            pieces.append(token)
            # 关键：将每个 token 推入 custom 通道
            writer({"token": token})
    
    return {"result": "".join(pieces), "messages": [AIMessage(content="".join(pieces))]}


workflow2 = StateGraph(State)
workflow2.add_node("generate", generate_with_real_llm)
workflow2.add_edge(START, "generate")
workflow2.add_edge("generate", END)
graph2 = workflow2.compile()

print("LLM 流式输出: ", end="", flush=True)
final_result = None
for chunk in graph2.stream(INPUT, stream_mode=["custom", "updates"], version="v2"):
    match chunk["type"]:
        case "custom":
            print(chunk["data"]["token"], end="", flush=True)
        case "updates":
            final_result = chunk["data"]["generate"]["result"]

print(f"\n\n最终结果: {final_result}")
print(f"\n\n最终结果长度: {len(final_result) if final_result else 0} 字符")

print(
    "\n说明：真实 LLM 的流式输出可通过 writer 逐 token 推入 custom 通道，"
    "前端可实时展示打字机效果。"
)


# ── 演示 4：Tool Calling + custom 流 —— 真实 LLM 调用带进度的工具 ──────────
# 这是最实用的场景：LLM 自主决定调用工具，工具内部通过 writer 推送进度

print("\n=== 演示 4：Tool Calling + custom 流（真实 LLM 调用工具）===\n")


@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库，返回相关信息。"""
    writer = get_stream_writer()
    
    # 模拟知识库搜索的各个阶段
    writer({"step": "初始化", "message": "🔍 正在连接知识库...", "progress": 0})
    time.sleep(0.2)
    
    writer({"step": "检索", "message": f"📖 搜索关键词: '{query}'", "progress": 25})
    time.sleep(0.2)
    
    writer({"step": "筛选", "message": "✅ 找到 15 条相关文档，正在筛选...", "progress": 50})
    time.sleep(0.2)
    
    writer({"step": "排序", "message": "📊 按相关性排序中...", "progress": 75})
    time.sleep(0.2)
    
    writer({"step": "完成", "message": "✨ 检索完成，返回 Top 3 结果", "progress": 100})
    time.sleep(0.1)
    
    # 返回模拟的知识库内容
    return f"""关于"{query}"的相关知识：
1. 这是第一条相关知识，包含重要的背景信息。
2. 这是第二条相关知识，提供了详细的解释。
3. 这是第三条相关知识，给出了实际应用案例。"""


@tool
def calculate_complex_metric(data_points: int) -> str:
    """计算复杂的业务指标，需要较长时间处理。"""
    writer = get_stream_writer()
    
    writer({"step": "数据加载", "message": f"📥 正在加载 {data_points} 个数据点...", "progress": 0})
    time.sleep(0.2)
    
    for i in range(1, 4):
        writer({
            "step": "计算中",
            "message": f"⚙️  处理批次 {i}/3 ({i*100} 个数据点)",
            "progress": i * 25
        })
        time.sleep(0.2)
    
    writer({"step": "聚合", "message": "📈 正在聚合计算结果...", "progress": 85})
    time.sleep(0.1)
    
    writer({"step": "完成", "message": "✅ 指标计算完成", "progress": 100})
    time.sleep(0.1)
    
    result = data_points * 3.14159
    return f"计算结果：基于 {data_points} 个数据点，复杂指标值为 {result:.2f}"


# 创建支持工具调用的 Agent
print("正在初始化 Agent（使用 YUNWU GPT 模型）...")
model_factory = ModelFactory()
llm = model_factory.create_yunwu_chat_model(
    model="gpt-5.4-nano:floor",
    temperature=0.7,
)

tools = [search_knowledge_base, calculate_complex_metric]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个智能助手，可以搜索知识库或计算复杂指标。请用中文回答。",
)

# 测试问题：让 LLM 自主决定调用哪个工具
test_questions = [
    "帮我搜索一下 Python 异步编程的最佳实践",
    "请计算一个复杂指标，我有 500 个数据点需要分析",
]

for idx, question in enumerate(test_questions, start=1):
    print(f"\n{'='*60}")
    print(f"问题 {idx}: {question}")
    print('='*60)
    
    input_messages = {"messages": [HumanMessage(content=question)]}
    
    # 同时接收 custom（工具进度）和 messages（LLM 回复）
    print("\n📡 实时进度:")
    print("-" * 60)
    
    final_answer = None
    for chunk in agent.stream(input_messages, stream_mode=["custom", "messages"], version="v2"):
        match chunk["type"]:
            case "custom":
                # 显示工具推送的进度信息
                data = chunk["data"]
                progress_bar = "█" * int(data.get("progress", 0) // 10)
                print(f"  [{data.get('progress', 0):3d}%] {progress_bar:<10} {data.get('message', '')}")
            
            case "messages":
                # 显示 LLM 的消息
                for msg in chunk["data"]:
                    if hasattr(msg, 'content') and msg.content:
                        if isinstance(msg, AIMessage):
                            final_answer = msg.content
    
    print("-" * 60)
    if final_answer:
        print(f"\n💬 LLM 回答:\n{final_answer}")
    print()

print("\n" + "="*60)
print("说明：")
print("  1. LLM 根据用户问题自主决定调用哪个工具")
print("  2. 工具执行过程中通过 writer() 推送详细进度")
print("  3. 前端可以实时展示进度条和阶段性信息")
print("  4. custom 通道与 messages 通道互不干扰，各司其职")
print("="*60)
