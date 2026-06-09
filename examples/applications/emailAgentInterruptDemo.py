"""interrupt 人机协作最小演示（LangGraph Thinking in LangGraph 文档 Testing 部分）。"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class EmailState(TypedDict):
    email_content: str
    response_text: str | None


def human_review_node(state: EmailState):
    interrupt(
        {
            "approved": False,
            "edited_response": state.get("response_text") or "",
        }
    )
    return {"response_text": "placeholder"}


app = (
    StateGraph(EmailState)
    .add_node("human_review", human_review_node)
    .add_edge(START, "human_review")
    .add_edge("human_review", END)
    .compile(checkpointer=InMemorySaver())
)


if __name__ == "__main__":
    initial_state = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "response_text": "Draft response",
    }

    config = {"configurable": {"thread_id": "customer_123"}}
    stream = app.stream_events(initial_state, config, version="v3")
    _ = stream.output
    print(f"human review interrupt:{stream.interrupts}")

    approved = input("批准这封邮件回复吗？(y/n): ").strip().lower() == "y"
    if approved:
        edited_response = ""
    else:
        edited_response = input("请输入修改后的回复内容: ")

    human_response = Command(
        resume={
            "approved": approved,
            "edited_response": edited_response,
        }
    )

    resumed = app.stream_events(human_response, config, version="v3")
    _ = resumed.output
    print("Email sent successfully!")
