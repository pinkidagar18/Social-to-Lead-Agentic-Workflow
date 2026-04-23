"""
AutoStream Agent — CLI Entry Point
Run this file to start an interactive conversation with the agent.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


def check_env():
    """Validate required environment variables."""
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not groq_key and not anthropic_key:
        print("\n[ERROR] Neither GROQ_API_KEY nor ANTHROPIC_API_KEY is set.")
        print("Please create a .env file with at least one of these keys.\n")
        sys.exit(1)


def print_banner():
    print("\n" + "=" * 60)
    print("   AutoStream AI Agent — Social-to-Lead Workflow")
    print("   Powered by LangGraph + Claude 3 Haiku + RAG")
    print("=" * 60)
    print("   Type 'quit' or 'exit' to end the conversation")
    print("   Type 'status' to see current lead collection state")
    print("=" * 60 + "\n")


def print_status(state: dict):
    """Print current conversation state for debugging."""
    print("\n── State Snapshot ─────────────────────────────")
    print(f"  Intent       : {state.get('intent', 'N/A')}")
    print(f"  Turn count   : {state.get('turn_count', 0)}")
    print(f"  Lead name    : {state.get('lead_name', 'Not collected')}")
    print(f"  Lead email   : {state.get('lead_email', 'Not collected')}")
    print(f"  Lead platform: {state.get('lead_platform', 'Not collected')}")
    print(f"  Lead captured: {state.get('lead_captured', False)}")
    print(f"  Awaiting slot: {state.get('awaiting_slot', 'None')}")
    print("───────────────────────────────────────────────\n")


def run():
    check_env()

    # Lazy import after env check
    from agent_graph import build_graph, create_initial_state

    print("\n[INFO] Initializing RAG pipeline and loading knowledge base...")
    agent = build_graph()
    state = create_initial_state()
    print("[INFO] Agent ready.\n")

    print_banner()

    # Opening message from agent
    opening = (
        "Hi there! I'm Alex from AutoStream. "
        "I can help you learn about our AI video editing plans and features. "
        "What would you like to know?"
    )
    print(f"AutoStream Agent: {opening}\n")
    state["messages"].append(AIMessage(content=opening))

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Have a great day!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("\nAutoStream Agent: Thanks for chatting! Have a great day!")
            break

        if user_input.lower() == "status":
            print_status(state)
            continue

        # Append user message to state
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # Run the graph for one turn
        try:
            state = agent.invoke(state)
        except Exception as e:
            print(f"\n[ERROR] Agent error: {e}")
            print("Please try again.\n")
            # Remove the failed human message to avoid corrupted state
            state["messages"] = list(state["messages"])[:-1]
            continue

        # Print the latest AI response
        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            "I'm sorry, something went wrong.",
        )
        print(f"\nAutoStream Agent: {last_ai}\n")

        # If lead was just captured, show a celebratory note
        if state.get("lead_captured") and state.get("awaiting_slot") is None:
            if state.get("turn_count", 0) > 0:
                # Only print once — check if last message was the capture confirmation
                pass  # Already handled in collect_lead_slot node


if __name__ == "__main__":
    run()
