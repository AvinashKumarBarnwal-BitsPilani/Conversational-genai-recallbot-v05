"""Compare episodic retrieval weightings on the E probes + episodic queries (Task 3(d)).

Uses the memory saved by a run's Session A and the Session B logical clock.
Each query is embedded once with BGE (logged under its own run ID); every
weighting then ranks the same candidates with the same raw signals.

    python scripts/compare_episodic_weightings.py --experiment t3_hm_01
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import logger as logger_module  # noqa: E402
from src.config import MEMORY_DIR, RESULTS_DIR  # noqa: E402
from src.memory.clock import SessionClock  # noqa: E402
from src.memory.episodic_memory import WEIGHTINGS, score_episodes  # noqa: E402
from src.memory.memory_manager import EPISODE_MIN_RELEVANCE_QUERY, MemoryManager  # noqa: E402

USER = "CUST-MAYA-1042"
QUERY_TURN = 28  # Session B turn of the first episodic question


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    logger_module.set_run_id(args.run_id or f"{args.experiment}_weighting", reject_existing=True)
    memory = MemoryManager(MEMORY_DIR / "runs" / args.experiment, config="hybrid_mem", user=USER, autosave=False)
    memory.set_call_context(user=USER, session="B", turn=QUERY_TURN)
    now = SessionClock.for_session("B").iso_at(QUERY_TURN)
    queries = json.loads((ROOT / "data" / "episodic_queries.json").read_text(encoding="utf-8"))["queries"]
    vectors = memory._embed([q["query"] for q in queries], is_query=True, purpose="episodic_weighting_comparison")
    episodes = memory.episodic.episodes(USER)

    rows, summary = [], {}
    for key, weights in WEIGHTINGS.items():
        hits1, rr, probe_hits1 = [], [], []
        for query, vector in zip(queries, vectors):
            relevance = memory.episodic.index.similarities(vector)
            ranked = score_episodes(episodes, relevance, now=now, weights=weights,
                                    min_relevance=EPISODE_MIN_RELEVANCE_QUERY)
            gold = {e.episode_id for e in episodes
                    if e.type == query["gold_type"] and e.source_turn == query["gold_source_turn"]}
            ids = [s.episode.episode_id for s in ranked]
            rank = next((i for i, eid in enumerate(ids, start=1) if eid in gold), None)
            hits1.append(int(rank == 1))
            rr.append(1.0 / rank if rank else 0.0)
            if query["source"].startswith("probe"):
                probe_hits1.append(int(rank == 1))
            top = ranked[0] if ranked else None
            rows.append({
                "weighting": weights.name, "w_recency": round(weights.recency, 4),
                "w_importance": round(weights.importance, 4), "w_relevance": round(weights.relevance, 4),
                "query_id": query["query_id"], "source": query["source"], "gold": "|".join(sorted(gold)),
                "gold_rank": rank or "", "hit_at_1": int(rank == 1),
                "top1": top.episode.episode_id if top else "", "top1_type": top.episode.type if top else "",
                "top1_score": round(top.score, 4) if top else "",
                "top3": "|".join(ids[:3]),
            })
        summary[key] = {
            "name": weights.name, "weights": [weights.recency, weights.importance, weights.relevance],
            "hit_at_1": round(sum(hits1) / len(hits1), 4), "mrr": round(sum(rr) / len(rr), 4),
            "e_probe_hit_at_1": round(sum(probe_hits1) / len(probe_hits1), 4), "queries": len(hits1),
        }
    out_csv = RESULTS_DIR / "task3_episodic_weight_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_json = RESULTS_DIR / "task3_episodic_weight_comparison.json"
    out_json.write_text(json.dumps({
        "experiment": args.experiment, "memory": f"memory/runs/{args.experiment} after Session A",
        "now": now, "recency": "0.995 ** hours (logical clock)", "min_relevance": EPISODE_MIN_RELEVANCE_QUERY,
        "candidates": len(episodes), "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        print(row["weighting"][:2], row["query_id"], "gold", row["gold"], "rank", row["gold_rank"],
              "top1", row["top1"], row["top1_type"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
