"""Task 4(a) rubric table and LLM-judge agreement.

Reads the student-reviewed manual ratings (results/task4_rubric_ratings.json)
and the Qwen judge ratings (results/task4_judge_ratings.json), then writes
  * results/task4_rubric_ratings.csv - one row per rated reply (reply text
    copied from the transcript), judged_by=manual;
  * results/task4_rubric_summary.json - mean score per dimension per config,
    and judge-vs-manual agreement (exact, within one point, Cohen's kappa
    and mean absolute difference) per dimension.

    python scripts/build_rubric_results.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("coherence", "relevance", "groundedness", "safety")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def main() -> int:
    manual = json.loads((ROOT / "results" / "task4_rubric_ratings.json").read_text(encoding="utf-8"))
    if not manual.get("reviewed_by_student"):
        print("Refusing to write: set reviewed_by_student to true after reviewing every rating.")
        return 2
    judge_path = ROOT / "results" / "task4_judge_ratings.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {"ratings": []}
    judged = {(r["config"], r["turn"]): r["scores"] for r in judge["ratings"] if r["valid"]}

    rows, summary = [], {"per_config": {}, "agreement": {}}
    for item in manual["ratings"]:
        experiment = f"{manual['prefix']}_{item['config']}"
        transcript = json.loads((ROOT / "results" / "task3_sessions" / f"{experiment}_session_B.json")
                                .read_text(encoding="utf-8"))
        reply = next(r for r in transcript["turns"] if r["turn"] == item["turn"])
        rows.append({"run_id": f"{experiment}_B", "config": item["config"], "turn": item["turn"],
                     "user_message": reply["user_message"], "reply": reply["assistant_response"],
                     **{d: item[d] for d in DIMENSIONS}, "note": item.get("note", ""), "judged_by": "manual"})
    with (ROOT / "results" / "task4_rubric_ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        summary["per_config"][config] = {d: round(statistics.mean(r[d] for r in mine), 2) for d in DIMENSIONS}
        summary["per_config"][config]["overall"] = round(
            statistics.mean(r[d] for r in mine for d in DIMENSIONS), 2)
        summary["per_config"][config]["replies"] = len(mine)

    paired = [(r, judged[(r["config"], r["turn"])]) for r in rows if (r["config"], r["turn"]) in judged]
    for dimension in DIMENSIONS:
        human = [int(r[dimension]) for r, _ in paired]
        model = [int(s[dimension]) for _, s in paired]
        if human:
            summary["agreement"][dimension] = {
                "pairs": len(human),
                "exact": round(sum(h == m for h, m in zip(human, model)) / len(human), 3),
                "within_1": round(sum(abs(h - m) <= 1 for h, m in zip(human, model)) / len(human), 3),
                "cohen_kappa": cohen_kappa(human, model),
                "mean_abs_diff": round(statistics.mean(abs(h - m) for h, m in zip(human, model)), 3),
                "judge_mean": round(statistics.mean(model), 2), "manual_mean": round(statistics.mean(human), 2),
            }
    (ROOT / "results" / "task4_rubric_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
