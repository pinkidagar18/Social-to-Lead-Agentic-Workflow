"""
Agent state schema for the AutoStream LangGraph agent.
All nodes read from and write to this typed state object.
"""

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Central state object persisted across all conversation turns.
    LangGraph passes this between every node in the graph.
    """

    # ── Conversation history ──────────────────────────────────────────────
    messages: List[BaseMessage]
    """Full conversation history including system, human, and AI messages."""

    # ── Intent classification ─────────────────────────────────────────────
    intent: Literal["greeting", "inquiry", "high_intent", "lead_collection", "unknown"]
    """Classified intent of the latest user message."""

    # ── Lead capture slots ────────────────────────────────────────────────
    lead_name: Optional[str]
    """Collected full name of the prospect."""

    lead_email: Optional[str]
    """Collected email address of the prospect."""

    lead_platform: Optional[str]
    """Collected creator platform (YouTube, Instagram, TikTok, etc.)."""

    lead_captured: bool
    """Guard flag — True once mock_lead_capture() has been called. Prevents double-firing."""

    # ── Metadata ──────────────────────────────────────────────────────────
    turn_count: int
    """Number of completed conversation turns."""

    last_rag_context: Optional[str]
    """Last retrieved RAG context, stored for debugging / logging."""

    awaiting_slot: Optional[Literal["name", "email", "platform"]]
    """
    Tracks which lead slot the agent is currently asking for.
    When set, the next user message is treated as the answer to this slot.
    """
