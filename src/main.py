from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

from .config import CHAT_MODEL, CONTEXT_WINDOW_TOKENS, DATA_DIR, KB_DIR, TRIM_WINDOW_K_TURNS

ROOT = Path(__file__).resolve().parents[1]


def inspect_project() -> int:
    """Validate the Day 1 data files without invoking a model or fabricating logs."""
    required = [
        DATA_DIR / "test_conversation.json",
        DATA_DIR / "kb_questions.json",
        DATA_DIR / "probes.json",
        DATA_DIR / "surprise_probes.json",
        ROOT / "db" / "seed_orders.json",
        ROOT / "skills" / "system_prompt_v1.txt",
        ROOT / "skills" / "return_pickup_workflow_v1.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    conversation = json.loads((DATA_DIR / "test_conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA_DIR / "kb_questions.json").read_text(encoding="utf-8"))
    probes = json.loads((DATA_DIR / "probes.json").read_text(encoding="utf-8"))
    kb_files = sorted(KB_DIR.glob("*.md"))
    total_words = sum(len(re.findall(r"\b[\w'-]+\b", p.read_text(encoding="utf-8"))) for p in kb_files)
    turns = conversation["turns"]
    counts = {
        "kb_documents": len(kb_files),
        "kb_words": total_words,
        "user_turns": len(turns),
        "session_a_turns": sum(t["session"] == "A" for t in turns),
        "session_b_turns": sum(t["session"] == "B" for t in turns),
        "kb_questions": len(questions),
        "probes": len(probes),
        "twist_probes": sum(p.get("twist") == "E" for p in probes),
    }
    print("RecallBot Day 1 project check")
    print(f"Variant: V05 | Twist: Small context | Context: {CONTEXT_WINDOW_TOKENS} tokens | trim k={TRIM_WINDOW_K_TURNS}")
    print(f"Local chat model configured: {CHAT_MODEL}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    errors = []
    if counts["kb_documents"] < 15: errors.append("Need at least 15 KB documents")
    if not 4000 <= counts["kb_words"] <= 8000: errors.append("KB word count must be 4,000–8,000")
    if counts["user_turns"] < 40 or counts["session_a_turns"] != 25 or counts["session_b_turns"] < 15: errors.append("Conversation must have at least 40 turns split A=1–25, B=26 onward")
    if counts["kb_questions"] < 25: errors.append("Need at least 25 KB questions")
    if counts["probes"] < 14 or counts["twist_probes"] < 2: errors.append("Need 14+ probes including 2+ twist probes")
    if errors:
        print("\nCHECK FAILED:")
        for err in errors: print(f"  - {err}")
        return 1
    print("\nCHECK PASSED: Day 1 data minimums satisfied.")
    return 0


def test_ollama() -> int:
    """Send one real local test request and log actual Ollama usage if successful."""
    try:
        import ollama
    except ImportError:
        print("Ollama Python package missing. Install with: pip install ollama")
        return 2
    from .logger import log_call, timer
    start = timer()
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: RecallBot local model ready."}],
            stream=False,
            options={"temperature": 0},
        )
        prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
        completion_tokens = int(getattr(response, "eval_count", 0) or 0)
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        log_call(config="full", user="LOCAL-TEST", session="A", turn=0,
                 call_type="chat", model=CHAT_MODEL, input_tokens=prompt_tokens,
                 output_tokens=completion_tokens, retrieved_tokens=0, start_time=start,
                 extra={"purpose": "local_model_smoke_test"})
        print(f"Model response: {content}")
        print(f"Actual Ollama token usage: input={prompt_tokens}, output={completion_tokens}")
        print("A real chat call was appended to logs/calls.jsonl.")
        return 0
    except Exception as exc:
        # The chat request was attempted. Record its measured latency and zero returned
        # tokens so failed calls are visible in the audit trail too.
        log_call(config="full", user="LOCAL-TEST", session="A", turn=0,
                 call_type="chat", model=CHAT_MODEL, input_tokens=0,
                 output_tokens=0, retrieved_tokens=0, start_time=start,
                 extra={"purpose": "local_model_smoke_test", "status": "failed", "error_type": type(exc).__name__})
        print(f"Could not reach local Ollama/model: {exc}")
        print(f"The failed attempt was recorded in logs/calls.jsonl.")
        print(f"Check: ollama list  |  ollama run {CHAT_MODEL}")
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="RecallBot V05 project utilities")
    parser.add_argument("--check", action="store_true", help="validate Day 1 data and project structure")
    parser.add_argument("--test-ollama", action="store_true", help="run and log one real local Qwen chat call")
    args = parser.parse_args()
    if args.test_ollama:
        return test_ollama()
    return inspect_project()

if __name__ == "__main__":
    raise SystemExit(main())
