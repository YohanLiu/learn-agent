from typing import TypedDict, Literal

# Define the structure for email classification
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_results: list[str] | None  # List of raw document chunks
    customer_history: dict | None  # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    messages: list[str] | None


from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain.messages import HumanMessage

from models.model_factory import ModelFactory

factory = ModelFactory()
llm = factory.create_yunwu_chat_model()



def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    # In production, this would connect to your email service
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response directly as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # Store classification as a single dict in state
    return Command(
        update={"classification": classification},
        goto=goto
    )

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""

    # Build search query from classification
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        # Implement your search logic here
        # Store raw search results, not formatted text
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols"
        ]
    except SearchAPIError as e:
        # For recoverable search errors, store error and continue
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(
        update={"search_results": search_results},  # Store raw results or error
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""

    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # Would be created via API

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    classification = state.get('classification', {})

    # Format context from raw state data on-demand
    context_sections = []

    if state.get('search_results'):
        # Format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get('customer_history'):
        # Format customer data for the prompt
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    # Build the prompt with formatted context
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """

    response = llm.invoke(draft_prompt)

    # Determine if human review needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # Route to appropriate next node
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # Store only the raw response
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})

    # interrupt() must come first - any code before it will re-run on resume
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response',''))},
            goto="send_reply"
        )
    else:
        # Rejection means human will handle directly
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    # Integrate with email service
    print(f"Sending reply: {state['draft_response'][:100]}...")
    return {}


from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

GRAPH_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "graphs"


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def show_agent_graph(compiled_agent, state_class: type) -> Path | None:
    """In IPython: inline display; otherwise save PNG named after state_class."""
    png = compiled_agent.get_graph(xray=True).draw_mermaid_png()
    if _in_ipython():
        from IPython.display import Image, display

        display(Image(png))
        return None

    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_OUTPUT_DIR / f"{state_class.__name__}.png"
    path.write_bytes(png)
    print(f"Graph saved to {path}")
    return path

# Create the graph
workflow = StateGraph(EmailAgentState)

# Add nodes with appropriate error handling
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# Add retry policy for nodes that might have transient failures
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# Add only the essential edges
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    show_agent_graph(app, EmailAgentState)

    def make_initial_state(email_content: str, email_id: str) -> EmailAgentState:
        return {
            "email_content": email_content,
            "sender_email": "customer@example.com",
            "email_id": email_id,
            "classification": None,
            "search_results": None,
            "customer_history": None,
            "draft_response": None,
            "messages": None,
        }

    def run_email_agent(initial_state: EmailAgentState, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}

        stream = app.stream_events(initial_state, config, version="v3")
        result = stream.output

        if stream.interrupts:
            print(f"\n[{thread_id}] human review interrupt:")
            for item in stream.interrupts:
                print(item)

            draft = (result or {}).get("draft_response") or ""
            if draft:
                print(f"\n草稿回复:\n{draft}\n")

            approved = input("批准这封邮件回复吗？(y/n): ").strip().lower() == "y"
            if approved:
                edited = input("如需修改请输入新回复（直接回车使用草稿）: ").strip()
                edited_response = edited or draft
            else:
                edited_response = ""

            human_response = Command(
                resume={
                    "approved": approved,
                    "edited_response": edited_response,
                }
            )

            resumed = app.stream_events(human_response, config, version="v3")
            result = resumed.output

            if resumed.interrupts:
                print(f"[{thread_id}] still interrupted:", resumed.interrupts)

        return result or {}

    # Demo 1：普通问题 → classify → search → draft → send（不中断）
    print("=" * 60)
    print("Demo 1: simple question (no interrupt)")
    result1 = run_email_agent(
        make_initial_state("How do I reset my password?", "email_001"),
        "email_agent_demo_simple",
    )
    print("Classification:", result1.get("classification"))
    print("Draft response:", (result1.get("draft_response") or "")[:200], "...")

    # Demo 2：账单/紧急 → 会进 human_review 并 interrupt，等待终端输入后 resume
    print("\n" + "=" * 60)
    print("Demo 2: billing / urgent (interrupt + interactive resume)")
    result2 = run_email_agent(
        make_initial_state(
            "I was charged twice for my subscription! This is urgent!",
            "email_002",
        ),
        "email_agent_demo_billing",
    )
    print("Classification:", result2.get("classification"))
    print("Final draft:", (result2.get("draft_response") or "")[:200], "...")
    print("Email sent successfully!")