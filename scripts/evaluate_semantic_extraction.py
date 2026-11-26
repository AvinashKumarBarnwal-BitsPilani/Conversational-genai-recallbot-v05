"""Semantic-memory extraction precision/recall from a logged run (Task 3(c)).

Predictions come only from the run's own artifacts:
  * memory/runs/<exp>/semantic_facts.json - every surviving record (active and
    superseded) with key, value, source turn and extraction method;
  * logs/calls.jsonl extract rows - which turns were attempted and which rule fired;
  * forget_tombstones.json - a fact deleted at turn 24 is no longer stored, so its
    value is verified by matching the planted value against the SHA-256 tombstone
    fingerprint (the value itself is never kept on disk).

Match rule: same key, same source turn, and the normalised content words of one
value contained in the other ("home pickup" == "pickup from home";
"side door" within "side door, when carrier permits").

    python scripts/evaluate_semantic_extraction.py --experiment t3_hm_01
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, MEMORY_DIR, RESULTS_DIR  # noqa: E402
from src.memory.forgetting import contains_fingerprint  # noqa: E402

_STOP = {"a", "an", "the", "when", "where", "from", "my", "at", "is", "to", "of", "for", "and", "if"}


def words(value: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", value.casefold()) if w not in _STOP}


def value_match(a: str, b: str) -> bool:
    wa, wb = words(a), words(b)
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="t3_hm_01")
    args = parser.parse_args()
    run_dir = MEMORY_DIR / "runs" / args.experiment
    run_id = f"{args.experiment}_A"
    conversation = json.loads((DATA_DIR / "test_conversation.json").read_text(encoding="utf-8"))
    planted = [f for f in conversation["planted_user_facts"] if f["source_turn"] <= 25]
    records = json.loads((run_dir / "semantic_facts.json").read_text(encoding="utf-8"))["records"]
    tombstones = json.loads((run_dir / "forget_tombstones.json").read_text(encoding="utf-8"))["tombstones"]
    log = [json.loads(line) for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
           if f'"{run_id}"' in line]
    extract_rows = [r for r in log if r["run_id"] == run_id and r["call_type"] == "extract"]

    predictions = [{"key": r["key"], "value": r["value"], "source_turn": r["source_turn"],
                    "method": r["extraction_method"], "status": r["status"], "evidence": "semantic_facts.json"}
                   for r in records]
    stored_keys_turns = {(p["key"], p["source_turn"]) for p in predictions}
    # Facts extracted, stored, then deleted by a forget request: recover from the log + tombstone.
    for row in extract_rows:
        for fact in row.get("extra", {}).get("facts", []):
            if (fact["key"], row["turn"]) in stored_keys_turns:
                continue
            stone = next((s for s in tombstones if s["key"] == fact["key"]), None)
            gold = next((p for p in planted if p["key"] == fact["key"] and p["source_turn"] == row["turn"]), None)
            verified = None
            if stone and gold:
                for candidate in sorted({gold["value"], *re.split(r",\s*", gold["value"])}):
                    if any(contains_fingerprint(candidate, [fp]) and len(words(candidate)) == fp["words"]
                           for fp in stone["fingerprints"]):
                        verified = candidate
            predictions.append({"key": fact["key"], "value": verified or "[deleted; value unverified]",
                                "source_turn": row["turn"], "method": "rule", "status": "deleted_by_forget",
                                "evidence": "extract log row + tombstone fingerprint"})

    def evaluate(preds):
        tp = [p for p in preds if any(g["key"] == p["key"] and g["source_turn"] == p["source_turn"]
                                      and value_match(g["value"], p["value"]) for g in planted)]
        found = [g for g in planted if any(g["key"] == p["key"] and g["source_turn"] == p["source_turn"]
                                           and value_match(g["value"], p["value"]) for p in preds)]
        return {"predicted": len(preds), "true_positive": len(tp), "planted": len(planted),
                "precision": round(len(tp) / len(preds), 4) if preds else 0.0,
                "recall": round(len(found) / len(planted), 4),
                "missed": [f"turn {g['source_turn']} {g['key']}" for g in planted if g not in found]}

    rules_only = [p for p in predictions if p["method"] == "rule"]
    attempted_turns = sorted({r["turn"] for r in extract_rows})
    filler = [t for t in conversation["low_importance_turns"] if t <= 25]
    result = {
        "experiment": args.experiment, "run_id": run_id,
        "match_rule": "same key and source turn; content words of one value contained in the other",
        "rules_plus_llm_fallback": evaluate(predictions),
        "rules_only": evaluate(rules_only),
        "llm_fallback_turns": sorted({r["turn"] for r in extract_rows if r["model"] != "rule-based"}),
        "extraction_attempted_turns": attempted_turns,
        "filler_turns": filler,
        "filler_turns_with_records": sorted({p["source_turn"] for p in predictions} & set(filler)),
        "filler_turns_with_extraction_attempts": sorted(set(attempted_turns) & set(filler)),
        "predictions": predictions,
    }
    out = RESULTS_DIR / "task3_semantic_memory_metrics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
    for p in predictions:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
