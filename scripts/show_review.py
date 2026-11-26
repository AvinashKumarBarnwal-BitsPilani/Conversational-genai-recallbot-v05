"""Print proposed marks next to the actual answers, for the student's review.

    python scripts/show_review.py probes results/task4_probe_judgements_hybrid_mem.json
    python scripts/show_review.py rubric results/task4_rubric_ratings.json [--config hybrid]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "results" / "task3_sessions"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("probes", "rubric"))
    parser.add_argument("path")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    data = json.loads((ROOT / args.path).read_text(encoding="utf-8"))
    if args.kind == "probes":
        probes = {p["probe_id"]: p for p in json.loads((ROOT / "data" / "probes.json").read_text(encoding="utf-8"))}
        a = {r["turn"]: r for r in json.loads((SESSIONS / f"{data['experiment']}_session_A.json").read_text(encoding="utf-8"))["turns"]}
        b = {r["probe_id"]: r for r in json.loads((SESSIONS / f"{data['experiment']}_session_B.json").read_text(encoding="utf-8"))["probe_turns"]}
        for j in data["judgements"]:
            p = probes[j["probe_id"]]
            answer = (a[p["turn"]] if p["session"] == "A" else b[j["probe_id"]])["assistant_response"]
            print(f"\n{j['probe_id']} [{p['category']}]  PROPOSED {j['correct']}  -- {j['rationale']}")
            print(f"  Q:        {p['question']}\n  EXPECTED: {p['expected']}\n  GOT:      {' '.join(answer.split())}")
    else:
        transcripts = {}
        for r in data["ratings"]:
            if args.config and r["config"] != args.config:
                continue
            key = f"{data['prefix']}_{r['config']}"
            if key not in transcripts:
                transcripts[key] = {t["turn"]: t for t in json.loads(
                    (SESSIONS / f"{key}_session_B.json").read_text(encoding="utf-8"))["turns"]}
            turn = transcripts[key][r["turn"]]
            print(f"\n{r['config']} turn {r['turn']}  C{r['coherence']} R{r['relevance']} G{r['groundedness']} "
                  f"S{r['safety']}  -- {r['note']}")
            print(f"  USER: {turn['user_message']}\n  BOT:  {' '.join(turn['assistant_response'].split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
