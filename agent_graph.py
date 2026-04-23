"""
AutoStream Conversational AI Agent
Built with LangGraph for state management.

Supports two LLM backends — auto-selected based on available API key:
  1. Groq  (GROQ_API_KEY)      → llama-3.3-70b-versatile  [default, ultra-fast]
  2. Anthropic (ANTHROPIC_API_KEY) → claude-haiku-4-5       [fallback]

Architecture:
  User Input → Intent Classifier → [Greeting | RAG Retrieval | Lead Capture]
             → Memory Update → Generate Response → (loop)
"""

import os
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent_state import AgentState
from agent_rag_pipeline import RAGPipeline
from tools_lead_capture import mock_lead_capture

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  LLM initialisation — Groq preferred, Anthropic as fallback
# ─────────────────────────────────────────────────────────────────────────────

def _init_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if groq_key:
        try:
            from langchain_groq import ChatGroq
            print("[LLM] Using Groq - llama-3.3-70b-versatile (ultra-fast)")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                max_tokens=1024,
                temperature=0.3,
            )
        except ImportError:
            print("[LLM] langchain-groq not installed. Falling back to Anthropic.")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        print("[LLM] Using Anthropic — claude-haiku-4-5")
        return ChatAnthropic(
            model="claude-haiku-4-5",
            api_key=anthropic_key,
            max_tokens=1024,
            temperature=0.3,
        )

    raise EnvironmentError(
        "\n[ERROR] No LLM API key found!\n"
        "Please set either GROQ_API_KEY or ANTHROPIC_API_KEY in your .env file.\n"
        "  Groq (free):     https://console.groq.com/\n"
        "  Anthropic:       https://console.anthropic.com/\n"
    )


llm = _init_llm()

rag = RAGPipeline()

# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

Your personality:
- Warm, helpful, and enthusiastic about video creation
- Concise — keep replies under 4 sentences unless explaining a plan
- Never pushy, but guide high-intent users toward signing up naturally

Your capabilities:
- Answer questions about AutoStream pricing and features using the provided context
- Detect when a user is ready to sign up and collect their details
- Always be honest — if something isn't in your knowledge base, say so

Strict rules:
- NEVER make up pricing or features not in the provided context
- NEVER ask for lead details unless the user has shown clear sign-up intent
- NEVER call the lead capture tool unless you have collected name, email, AND platform
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Node: Intent Classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Classifies the latest user message into one of:
    - greeting
    - inquiry  (product / pricing question)
    - high_intent (ready to sign up / try the product)
    - lead_collection (currently filling out lead form slots)
    - unknown
    """

    # If we're mid-slot-collection, stay in lead_collection mode
    if state.get("awaiting_slot") and not state.get("lead_captured"):
        return {**state, "intent": "lead_collection"}

    messages = state["messages"]
    last_human_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    classifier_prompt = f"""Classify the intent of this message from a potential customer talking to AutoStream (a video editing SaaS).

Message: "{last_human_msg}"

Respond with EXACTLY one word — no explanation:
- "greeting"      → hello, hi, hey, how are you, etc.
- "inquiry"       → asking about features, pricing, plans, refunds, support, or how the product works
- "high_intent"   → wants to sign up, try it, start a plan, or explicitly wants to buy/subscribe
- "unknown"       → anything else

Reply with only the single word."""

    response = llm.invoke([HumanMessage(content=classifier_prompt)])
    raw = response.content.strip().lower().replace('"', "").replace("'", "")

    # Map to valid intents
    intent_map = {
        "greeting": "greeting",
        "inquiry": "inquiry",
        "high_intent": "high_intent",
        "unknown": "unknown",
    }
    intent = intent_map.get(raw, "unknown")

    return {**state, "intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
#  Node: Greeting Handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_greeting(state: AgentState) -> AgentState:
    """Generates a warm greeting response. No RAG needed."""

    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "Hi"
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *messages[-6:],  # Last 6 messages for context
        HumanMessage(
            content=f"{last_human}\n\n"
            "[Instruction: Respond with a friendly greeting and briefly mention you can help with AutoStream pricing and features.]"
        ),
    ]

    response = llm.invoke(prompt_messages)
    new_messages = list(messages) + [AIMessage(content=response.content)]

    return {
        **state,
        "messages": new_messages,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node: RAG Retrieval + Response
# ─────────────────────────────────────────────────────────────────────────────

def handle_inquiry(state: AgentState) -> AgentState:
    """Retrieves relevant KB context and generates a grounded answer."""

    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # RAG retrieval
    context = rag.get_context(last_human, top_k=3)

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *messages[-6:],
        HumanMessage(
            content=f"{last_human}\n\n"
            f"[KNOWLEDGE BASE CONTEXT — use this to answer accurately]:\n{context}\n\n"
            "[Instruction: Answer the question using only the context above. "
            "If context doesn't cover it, say you don't have that information. "
            "Be concise and friendly.]"
        ),
    ]

    response = llm.invoke(prompt_messages)
    new_messages = list(messages) + [AIMessage(content=response.content)]

    return {
        **state,
        "messages": new_messages,
        "last_rag_context": context,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node: Lead Capture — start collection
# ─────────────────────────────────────────────────────────────────────────────

def start_lead_capture(state: AgentState) -> AgentState:
    """
    Triggered when high_intent is detected.
    Acknowledges intent and starts collecting the first missing slot (name).
    """

    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # Build a context-aware opener
    context = rag.get_context(last_human, top_k=2)

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *messages[-6:],
        HumanMessage(
            content=f"{last_human}\n\n"
            f"[CONTEXT]: {context}\n\n"
            "[Instruction: The user wants to sign up or try AutoStream. "
            "Acknowledge their interest warmly, then ask for their full name to get started. "
            "Do NOT ask for email or platform yet — only name.]"
        ),
    ]

    response = llm.invoke(prompt_messages)
    new_messages = list(messages) + [AIMessage(content=response.content)]

    return {
        **state,
        "messages": new_messages,
        "intent": "lead_collection",
        "awaiting_slot": "name",
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node: Lead Collection — slot filling
# ─────────────────────────────────────────────────────────────────────────────

def collect_lead_slot(state: AgentState) -> AgentState:
    """
    Fills lead slots one at a time: name → email → platform → fire tool.
    The guard ensures the tool is NEVER called prematurely.
    """

    messages = state["messages"]
    awaiting = state.get("awaiting_slot")
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    ).strip()

    updated_state = dict(state)

    # ── Save the answer to the current awaiting slot ──────────────────────
    if awaiting == "name":
        updated_state["lead_name"] = last_human
        updated_state["awaiting_slot"] = "email"

        prompt = (
            f"The user's name is {last_human}. "
            "Now politely ask for their email address to complete the sign-up."
        )

    elif awaiting == "email":
        updated_state["lead_email"] = last_human
        updated_state["awaiting_slot"] = "platform"

        prompt = (
            f"The user's email is {last_human}. "
            "Now ask which content platform they primarily create for "
            "(e.g. YouTube, Instagram, TikTok, LinkedIn, etc.)."
        )

    elif awaiting == "platform":
        updated_state["lead_platform"] = last_human
        updated_state["awaiting_slot"] = None  # All slots filled

        # ── All 3 slots collected — FIRE THE TOOL ─────────────────────────
        tool_result = mock_lead_capture(
            name=updated_state["lead_name"],
            email=updated_state["lead_email"],
            platform=updated_state["lead_platform"],
        )

        updated_state["lead_captured"] = True

        if tool_result["status"] == "success":
            prompt = (
                f"Lead captured successfully! Lead ID: {tool_result['lead_id']}. "
                f"The user's name is {updated_state['lead_name']}, "
                f"email {updated_state['lead_email']}, "
                f"platform {updated_state['lead_platform']}. "
                "Write a warm confirmation message, mention their lead ID, "
                "tell them the team will be in touch within 24 hours, "
                "and wish them happy creating!"
            )
        else:
            prompt = (
                f"There was an error capturing the lead: {tool_result['message']}. "
                "Apologize and ask the user to provide the information again."
            )
            updated_state["lead_captured"] = False
            updated_state["awaiting_slot"] = "name"
            updated_state["lead_name"] = None
            updated_state["lead_email"] = None
            updated_state["lead_platform"] = None

    else:
        # Fallback — should not normally be reached
        prompt = "Ask the user for their name to get started."
        updated_state["awaiting_slot"] = "name"

    # ── Generate response ─────────────────────────────────────────────────
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    new_messages = list(messages) + [AIMessage(content=response.content)]
    updated_state["messages"] = new_messages
    updated_state["turn_count"] = state.get("turn_count", 0) + 1

    return updated_state


# ─────────────────────────────────────────────────────────────────────────────
#  Node: Unknown intent fallback
# ─────────────────────────────────────────────────────────────────────────────

def handle_unknown(state: AgentState) -> AgentState:
    """Handles off-topic or unclear messages gracefully."""

    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *messages[-4:],
        HumanMessage(
            content=f"{last_human}\n\n"
            "[Instruction: The user's message is unclear or off-topic. "
            "Politely redirect them to ask about AutoStream's plans, pricing, or features. "
            "Keep it short and friendly.]"
        ),
    ])

    new_messages = list(messages) + [AIMessage(content=response.content)]
    return {
        **state,
        "messages": new_messages,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Routing logic
# ─────────────────────────────────────────────────────────────────────────────

def route_intent(
    state: AgentState,
) -> Literal["handle_greeting", "handle_inquiry", "start_lead_capture", "collect_lead_slot", "handle_unknown"]:
    """Conditional edge — routes to the correct handler based on classified intent."""

    intent = state.get("intent", "unknown")
    awaiting = state.get("awaiting_slot")
    lead_captured = state.get("lead_captured", False)

    # If we're in mid-collection and lead not yet captured, continue filling slots
    if awaiting and not lead_captured:
        return "collect_lead_slot"

    routing = {
        "greeting": "handle_greeting",
        "inquiry": "handle_inquiry",
        "high_intent": "start_lead_capture",
        "unknown": "handle_unknown",
    }
    return routing.get(intent, "handle_unknown")


# ─────────────────────────────────────────────────────────────────────────────
#  Build the LangGraph StateGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Constructs and compiles the LangGraph agent."""

    graph = StateGraph(AgentState)

    # ── Register all nodes ─────────────────────────────────────────────────
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_inquiry", handle_inquiry)
    graph.add_node("start_lead_capture", start_lead_capture)
    graph.add_node("collect_lead_slot", collect_lead_slot)
    graph.add_node("handle_unknown", handle_unknown)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("classify_intent")

    # ── Conditional routing from intent classifier ─────────────────────────
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "start_lead_capture": "start_lead_capture",
            "collect_lead_slot": "collect_lead_slot",
            "handle_unknown": "handle_unknown",
        },
    )

    # ── All handler nodes end the graph turn (loop handled in main.py) ─────
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("start_lead_capture", END)
    graph.add_edge("collect_lead_slot", END)
    graph.add_edge("handle_unknown", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
#  Initial state factory
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_state() -> AgentState:
    """Creates a fresh agent state for a new conversation."""
    return AgentState(
        messages=[],
        intent="unknown",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        turn_count=0,
        last_rag_context=None,
        awaiting_slot=None,
    )
