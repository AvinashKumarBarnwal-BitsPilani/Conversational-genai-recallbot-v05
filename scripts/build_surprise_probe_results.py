"""Write the surprise-probe rows (S1-S5, category SP) into results/probe_results.csv.

Answers are copied from the Session C transcripts in results/surprise_probes/
(never typed by hand). Marks come from results/surprise_probe_judgements.json,
which must have reviewed_by_student set to true. Rows of other runs are kept.

    python scripts/build_surprise_probe_results.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["run_id", "config", "probe_id", "category", "expected", "answer", "correct", "judged_by"]


def probe_answers(experiment: str) -> dict[str, tuple[str, str]]:
    """probe_id -> (run_id, answer). S2 is judged on the reply to its last booking turn."""
    answers: dict[str, tuple[str, str]] = {}
    for part in (1, 2):
        path = ROOT / "results" / "surprise_probes" / f"{experiment}_session_C{part}.json"
        transcript = json.loads(path.read_text(encoding="utf-8"))
        for row in transcript["turns"]:
            if row["probe_id"] == "S2" and part == 1:
                continue  # the planted instruction itself; its effect is judged after the restart
            answers[row["probe_id"]] = (transcript["run_id"], row["assistant_response"])
    return answers


def main() -> int:
    judgements = json.loads((ROOT / "results" / "surprise_probe_judgements.json").read_text(encoding="utf-8"))
    if not judgements.get("reviewed_by_student"):
        print("Refusing to write: set reviewed_by_student to true after reviewing every mark.")
        return 2
    probes = {p["id"]: p for p in json.loads((ROOT / "data" / "surprise_probes.json").read_text(encoding="utf-8"))}
    new_rows = []
    for run in judgements["runs"]:
        answers = probe_answers(run["experiment"])
        for item in run["judgements"]:
            run_id, answer = answers[item["probe_id"]]
            new_rows.append({"run_id": run_id, "config": judgements["config"], "probe_id": item["probe_id"],
                             "category": "SP", "expected": probes[item["probe_id"]]["pass_if"], "answer": answer,
                             "correct": int(item["correct"]), "judged_by": "manual"})
    path = ROOT / "results" / "probe_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        keep = [r for r in csv.DictReader(handle) if r["category"] != "SP"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(keep + new_rows)
    for run in judgements["runs"]:
        marks = [r["correct"] for r in new_rows if r["run_id"].startswith(run["experiment"])]
        print(f"{run['experiment']}: {sum(marks)}/{len(marks)} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
