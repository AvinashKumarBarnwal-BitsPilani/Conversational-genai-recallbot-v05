"""Extract and explicitly adjudicate the six official Session A probe rows.

The correctness decisions below are human-reviewed outcomes for P08/P10;
they are never inferred from substring matches or generated automatically.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "data" / "probes.json"
RESULTS = ROOT / "results"
RUNS = {
    "full": "task2c_full_snapshot1",
    "trim": "task2c_trim_snapshot1",
    "summary": "task2c_summary_snapshot1",
}
ADJUDICATION = {
    "P08": {
        "correctness": "correct",
        "judging_rationale": (
            "The response directly defines ERR-RMA-04 as a missing or invalid return reason. "
            "It cites the retrieved active KB-007 chunk. The additional error-code guidance is "
            "supported by that retrieved chunk; no material factual error was found. Qwen itself "
            "fell back because prompt preparation exceeded budget, but the recorded deterministic "
            "answer satisfies the probe."
        ),
    },
    "P10": {
        "correctness": "correct",
        "judging_rationale": (
            "The tracker retained the active 28 November 2026 pickup date and merged the after-6-pm "
            "time. check_pickup_slots returned two valid windows and the response presented both, "
            "asking the customer to choose. The response does not select or book a slot."
        ),
    },
}


def main() -> int:
    probes = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    official = {item["probe_id"]: item for item in probes if item.get("session") == "A"}
    if set(official) != {"P08", "P10"}:
        raise ValueError(f"Expected official Session A probes P08 and P10; got {sorted(official)}")

    rows = []
    for strategy, run_id in RUNS.items():
        path = RESULTS / f"task2c_snapshot_{strategy}_session_A_{run_id}.json"
        transcript = json.loads(path.read_text(encoding="utf-8"))
        if transcript.get("status") != "complete" or transcript.get("successfully_processed_user_turns") != 25:
            raise ValueError(f"Incomplete transcript: {path}")
        turns = {int(row["turn"]): row for row in transcript["turns"]}
        for probe_id, turn in (("P08", 18), ("P10", 15)):
            probe = official[probe_id]
            result = turns[turn]
            evidence = {
                "nlg_status": result["nlg_status"],
                "nlg_error": result.get("nlg_error"),
                "sources": result.get("sources", []),
            }
            if probe_id == "P08":
                evidence["retrieved_chunks"] = [
                    {"doc_id": item.get("doc_id"), "chunk_id": item.get("chunk_id"),
                     "status": item.get("status"), "content": item.get("content")}
                    for call in result.get("tool_results", [])
                    for item in call.get("result", [])
                ]
            else:
                before = turns[14]["dialogue_state"]
                after = result["dialogue_state"]
                evidence.update({
                    "tracker_before": {
                        "order_id": before["slots"].get("order_id"),
                        "sku": before["slots"].get("sku"),
                        "pickup_date": before["slots"].get("pickup_date"),
                        "pickup_time_window": before["slots"].get("pickup_time_window"),
                    },
                    "nlu_intent": result["intent"],
                    "tracker_after": {
                        "order_id": after["slots"].get("order_id"),
                        "sku": after["slots"].get("sku"),
                        "pickup_date": after["slots"].get("pickup_date"),
                        "pickup_time_window": after["slots"].get("pickup_time_window"),
                    },
                    "tool_results": result.get("tool_results", []),
                })
            rows.append({
                "probe_id": probe_id,
                "category": probe["category"],
                "strategy": strategy,
                "run_id": run_id,
                "turn": turn,
                "expected_behavior": probe["expected_behavior"],
                "actual_response": result["assistant_response"],
                "evidence": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
                "correctness": ADJUDICATION[probe_id]["correctness"],
                "judging_rationale": ADJUDICATION[probe_id]["judging_rationale"],
            })

    output = RESULTS / "task2c_probe_results.csv"
    fields = [
        "probe_id", "category", "strategy", "run_id", "turn", "expected_behavior",
        "actual_response", "evidence", "correctness", "judging_rationale",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} human-adjudicated official Session A rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
