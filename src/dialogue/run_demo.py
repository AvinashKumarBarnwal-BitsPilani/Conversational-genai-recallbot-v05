"""Run RecallBot with the real JSON tools and local hybrid retrieval.

From the project root:
    python -m src.dialogue.run_demo
    python -m src.dialogue.run_demo --reset-db

Qwen-based NLG is enabled by default; deterministic responses are retained as
the fallback when prompt preparation or Ollama generation fails.
"""
from __future__ import annotations

import argparse

from ..config import CHAT_MODEL, CONTEXT_WINDOW_TOKENS
from .dialogue_manager import DialogueManager
from .state_tracker import DialogueStateTracker
from .tools import RecallBotToolGateway
from .qwen_nlg import QwenNLG


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RecallBot with real local tools")
    parser.add_argument("--user-id", default="CUST-MAYA-1042")
    parser.add_argument("--session", default="A")
    parser.add_argument("--chunk-size", type=int, choices=(128, 512), default=128)
    parser.add_argument("--memory-strategy", choices=("full", "trim", "summary"), default="full")
    parser.add_argument(
        "--reset-db", action="store_true",
        help="Reset runtime pickup slots from db/seed_pickup_slots.json before starting",
    )
    parser.add_argument(
        "--disable-nlg", action="store_true",
        help="Use deterministic response templates instead of local Qwen NLG",
    )
    args = parser.parse_args()

    tools = RecallBotToolGateway(chunk_size=args.chunk_size)
    if args.reset_db:
        tools.reset_runtime_database()
        print("Runtime pickup database reset from the immutable seed fixture.")

    tracker = DialogueStateTracker(session_id=args.session, user_id=args.user_id)
    nlg = None if args.disable_nlg else QwenNLG()
    manager = DialogueManager(
        tracker=tracker,
        tools=tools,
        user_id=args.user_id,
        config=args.memory_strategy,
        memory_strategy=args.memory_strategy,
        nlg=nlg,
    )

    print("\nRecallBot local tool demo")
    print(f"Customer: {args.user_id} | Session: {args.session}")
    print(f"Configured chat model: {CHAT_MODEL}")
    print(f"NLG: {'disabled (deterministic templates)' if args.disable_nlg else 'Qwen via Ollama'}")
    print("Decoding: task turns temperature=0.0, open turns temperature=0.3, top_p=0.9")
    print(f"Variant V05 context budget: {CONTEXT_WINDOW_TOKENS} tokens")
    print("Type 'exit' or 'quit' to stop.\n")

    turn = 0
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0
        if text.casefold() in {"exit", "quit"}:
            print("RecallBot: Goodbye!")
            return 0
        if not text:
            continue
        turn += 1
        result = manager.handle_turn(text, turn=turn)
        print(f"RecallBot: {result['reply']}")
        if result.get("nlg_status", "disabled") != "qwen" and not args.disable_nlg:
            print(f"[NLG fallback: {result.get('nlg_status')}]")
        if result.get("tool_calls"):
            for call in result["tool_calls"]:
                print("Tool:", call)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
