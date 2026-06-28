"""工具内的中断（Interrupts in Tools）。

演示内容（完整示例）：
  - 在工具函数内部调用 interrupt()，使工具本身在执行前暂停等待审批
  - Agent 节点调用 LLM 决策工具调用，工具节点执行工具
  - 工具暂停时，interrupt 负载包含邮件详情，供调用方审核/编辑/取消
  - 恢复时可传入修改后的参数，工具使用恢复值执行实际操作
  - 将审批逻辑内聚在工具本身，便于跨图复用

注意：
  - 本示例需要有效的 LLM API 密钥（YUNWU_API_KEY）才能完整运行
  - 使用 YUNWU（GPT）模型，确保 tool calling 能力可靠
  - 若无需真实 LLM 调用，可注释 agent_node 部分，手动构造 tool_call 消息测试

运行：
    uv run python examples/interrupts/interrupts_in_tools.py
"""

import operator
from typing import Annotated, Literal

from langchain.tools import tool
from langchain.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from models.model_factory import ModelFactory


# ── 定义状态 ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Agent 状态，包含消息历史（自动追加）。"""
    messages: Annotated[list[AnyMessage], operator.add]


# ── 定义带中断的工具 ──────────────────────────────────────────────────────────

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    在发送前调用 interrupt() 暂停，等待人工审批。
    恢复时可传入修改后的参数，或直接取消操作。
    """
    # 暂停：将邮件详情暴露给调用方审核
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    # 根据恢复值决定是否执行
    if response.get("action") == "approve":
        # 恢复值可覆盖原始参数（编辑后再发送）
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # 实际发送邮件（此处仅打印）
        print(f"  [send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"

    return "Email cancelled by user"


# ── 初始化 LLM 与工具映射 ─────────────────────────────────────────────────────

factory = ModelFactory()
# 绑定工具，让 LLM 知道可以调用 send_email
model = factory.create_yunwu_chat_model(model="gpt-4o-mini:floor").bind_tools([send_email])
tools_by_name = {"send_email": send_email}


# ── 定义节点 ──────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM 决定是否调用工具。"""
    print("  [agent_node] 调用 LLM 决策...")
    result = model.invoke(state["messages"])
    return {"messages": [result]}


def tool_node(state: AgentState) -> dict:
    """工具节点：执行 LLM 决策的工具调用。

    工具内的 interrupt() 会在此处暂停执行，等待审批后恢复。
    """
    print("  [tool_node] 执行工具调用...")
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        observation = tool_fn.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """根据最后一条消息判断是否需要执行工具。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# ── 构建图 ────────────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "agent")  # 工具执行后回到 agent 继续对话

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "email-workflow"}}

    # ── 第 1 步：发送请求，LLM 决策调用 send_email，工具在 interrupt 处暂停 ──
    print("=== 第 1 步：请求发送邮件 ===\n")

    initial = graph.stream_events(
        {
            "messages": [
                {"role": "user", "content": "Send an email to alice@example.com about the meeting"}
            ]
        },
        config=config,
        version="v3",
    )
    _ = initial.output  # 驱动流执行完毕
    print(f"\n[中断] {initial.interrupts}")
    # 输出：(Interrupt(value={'action': 'send_email', 'to': 'alice@example.com', ...}),)

    if initial.interrupted:
        # 获取中断信息中的邮件详情
        interrupt_info = initial.interrupts[0].value
        print(f"\n[邮件详情]")
        print(f"  to:      {interrupt_info.get('to')}")
        print(f"  subject: {interrupt_info.get('subject')}")
        print(f"  body:    {interrupt_info.get('body')}")

        # ── 第 2 步：审批并恢复（可修改主题等参数）──
        print("\n=== 第 2 步：批准发送（可编辑参数）===\n")

        resumed = graph.stream_events(
            Command(resume={
                "action": "approve",
                "subject": "Updated: Meeting at 3pm",  # 覆盖原始主题
            }),
            config=config,
            version="v3",
        )
        print(f"\n[最终消息] {resumed.output['messages'][-1]}")
