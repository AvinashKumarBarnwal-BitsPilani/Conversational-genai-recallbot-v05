"""Evaluate rule-based NLU against the labelled Task 1 conversation.

Run from the project root:
    python -m src.dialogue.evaluate_nlu

The script writes measured per-utterance predictions and a compact accuracy
summary. It does not modify the gold labels or the scripted conversation.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, RESULTS_DIR
from .nlu import analyze_utterance


def evaluate() -> dict[str, Any]:
    """Run NLU on all labelled utterances and write CSV/JSON evidence."""
    cases_path = DATA_DIR / "nlu_test_cases.json"
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload["cases"]

    rows: list[dict[str, Any]] = []
    per_intent: Counter[str] = Counter()
    correct = 0

    for case in cases:
        result = analyze_utterance(case["utterance"])
        expected = case["expected_intent"]
        predicted = result["intent"]
        is_correct = predicted == expected
        correct += int(is_correct)
        per_intent[expected] += 1

        rows.append({
            "turn": case["turn"],
            "session": case["session"],
            "utterance": case["utterance"],
            "expected_intent": expected,
            "predicted_intent": predicted,
            "correct": int(is_correct),
            "rule_id": result["rule_id"],
            "entities": json.dumps(result["entities"], ensure_ascii=False),
        })

    total = len(cases)
    accuracy = correct / total if total else 0.0
    summary = {
        "evaluation": "rule_based_nlu_on_labelled_scripted_utterances",
        "source": "data/test_conversation.json turns 1-40, labelled in data/nlu_test_cases.json",
        "approach": "deterministic rules and regular expressions",
        "model": None,
        "total_utterances": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "expected_intent_counts": dict(sorted(per_intent.items())),
        "note": "This measures accuracy on the labelled scripted utterances; it is not a claim of generalisation to unseen customer traffic.",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "nlu_accuracy.csv"
    json_path = RESULTS_DIR / "nlu_accuracy.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"NLU approach: {summary['approach']}")
    print(f"Labelled utterances: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    return summary


if __name__ == "__main__":
    evaluate()
