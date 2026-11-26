"""Task 4(a) ablation table, computed only from logs/calls.jsonl and results/probe_results.csv.

Definitions (stated so evaluators can recompute them):
  * accuracy per category: probe rows of the config's runs (Session A probes from
    run <prefix>_<config>_A, Session B probes from <prefix>_<config>_B);
  * mean input tokens / turn: mean `input_tokens` of customer-facing `chat` calls on
    the 40 scripted turns (turns 1-40; probe questions excluded so every config is
    measured on identical turns);
  * median latency: median `latency_ms` of those chat calls; also the median
    end-to-end turn latency (sum of every logged call on that turn);
  * total tokens: input + output tokens of every logged call of both runs (all
    call types, probe questions included), with a per-call-type breakdown.

    python scripts/ablation_metrics.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("no_rag", "hybrid", "hybrid_sem", "hybrid_mem")
CATEGORIES = ("R", "U", "F", "E", "K", "N", "S", "TWIST_SMALL_CONTEXT")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="t4")
    args = parser.parse_args()
    log = [json.loads(line) for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    with (ROOT / "results" / "probe_results.csv").open(encoding="utf-8", newline="") as handle:
        probes = list(csv.DictReader(handle))

    rows, breakdown = [], {}
    for config in CONFIGS:
        runs = {f"{args.prefix}_{config}_A", f"{args.prefix}_{config}_B"}
        records = [r for r in log if r["run_id"] in runs]
        if not records:
            continue
        chat = [r for r in records if r["call_type"] == "chat" and 1 <= r["turn"] <= 40]
        per_turn = defaultdict(int)
        for r in records:
            if 1 <= r["turn"] <= 40:
                per_turn[(r["run_id"], r["turn"])] += r["latency_ms"]
        marks = [p for p in probes if p["run_id"] in runs]
        row = {"config": config, "runs": " + ".join(sorted(runs))}
        for category in CATEGORIES:
            hits = [int(p["correct"]) for p in marks if p["category"] == category]
            row[f"acc_{category}"] = f"{sum(hits)}/{len(hits)}" if hits else "n/a"
        row["acc_overall"] = (f"{sum(int(p['correct']) for p in marks)}/{len(marks)}" if marks else "n/a")
        row["mean_input_tokens_per_turn"] = round(statistics.mean(r["input_tokens"] for r in chat), 1)
        row["median_chat_latency_ms"] = statistics.median(r["latency_ms"] for r in chat)
        row["median_turn_latency_ms"] = statistics.median(per_turn.values())
        row["total_tokens"] = sum(r["input_tokens"] + r["output_tokens"] for r in records)
        row["chat_fallbacks"] = sum(r.get("extra", {}).get("status") not in (None, "success")
                                    for r in records if r["call_type"] == "chat")
        row["mean_retrieved_tokens_per_turn"] = round(statistics.mean(r["retrieved_tokens"] for r in chat), 1)
        rows.append(row)
        by_type = defaultdict(lambda: [0, 0])
        for r in records:
            by_type[r["call_type"]][0] += 1
            by_type[r["call_type"]][1] += r["input_tokens"] + r["output_tokens"]
        breakdown[config] = {k: {"calls": v[0], "tokens": v[1]} for k, v in sorted(by_type.items())}

    out_dir = ROOT / "results"
    with (out_dir / "task4_ablation_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "task4_token_breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n", encoding="utf-8")

    headers = ["Config", *CATEGORIES[:-1], "Twist", "Overall", "Mean input tok/turn", "Median chat latency (ms)",
               "Median turn latency (ms)", "Total tokens"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["config"], *(r[f"acc_{c}"] for c in CATEGORIES), r["acc_overall"],
            f"{r['mean_input_tokens_per_turn']:,}", f"{r['median_chat_latency_ms']:,}",
            f"{r['median_turn_latency_ms']:,}", f"{r['total_tokens']:,}"]) + " |")
    (out_dir / "task4_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nCalls and tokens by call type:")
    for config, types in breakdown.items():
        print(f"  {config}: " + ", ".join(f"{k} {v['calls']}/{v['tokens']:,}" for k, v in types.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
