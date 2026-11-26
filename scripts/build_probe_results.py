"""Write results/probe_results.csv rows for one Session A/B experiment.

Answers are copied from the run transcripts (never typed by hand). Correctness
comes from a judgements file that the student has reviewed; the script refuses
to write judged_by=manual rows until ``reviewed_by_student`` is true. Existing
rows for other run IDs are preserved.

    python scripts/build_probe_results.py --judgements results/task3_probe_judgements.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["run_id", "config", "probe_id", "category", "expected", "answer", "correct", "judged_by"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", required=True)
    args = parser.parse_args()
    judgements = json.loads((ROOT / args.judgements).read_text(encoding="utf-8"))
    if not judgements.get("reviewed_by_student"):
        print("Refusing to write: set reviewed_by_student to true after reviewing every mark.")
        return 2
    experiment, config = judgements["experiment"], judgements["config"]
    probes = {p["probe_id"]: p for p in json.loads((ROOT / "data" / "probes.json").read_text(encoding="utf-8"))}
    sessions = ROOT / "results" / "task3_sessions"
    a_turns = {r["turn"]: r for r in json.loads((sessions / f"{experiment}_session_A.json")
                                                 .read_text(encoding="utf-8"))["turns"]}
    b_probes = {r["probe_id"]: r for r in json.loads((sessions / f"{experiment}_session_B.json")
                                                      .read_text(encoding="utf-8"))["probe_turns"]}
    new_rows = []
    for item in judgements["judgements"]:
        probe = probes[item["probe_id"]]
        if probe["session"] == "A":
            answer, run_id = a_turns[probe["turn"]]["assistant_response"], f"{experiment}_A"
        else:
            answer, run_id = b_probes[item["probe_id"]]["assistant_response"], f"{experiment}_B"
        new_rows.append({"run_id": run_id, "config": config, "probe_id": item["probe_id"],
                         "category": probe["category"], "expected": probe["expected"], "answer": answer,
                         "correct": int(item["correct"]), "judged_by": "manual"})
    path = ROOT / "results" / "probe_results.csv"
    keep = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            keep = [r for r in csv.DictReader(handle) if not r["run_id"].startswith(f"{experiment}_")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(keep + new_rows)
    by_category: dict[str, list[int]] = {}
    for row in new_rows:
        by_category.setdefault(row["category"], []).append(row["correct"])
    print(f"Wrote {len(new_rows)} rows for {experiment} ({sum(r['correct'] for r in new_rows)}/{len(new_rows)} correct)")
    for category, marks in sorted(by_category.items()):
        print(f"  {category}: {sum(marks)}/{len(marks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
