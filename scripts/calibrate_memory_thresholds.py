"""Calibrate the memory relevance thresholds on the PILOT memory (Task 3).

Positive queries need a specific memory; negative queries are turns where no
long-term memory should enter the prompt. The script embeds every query once
with BGE (logged under its own run ID), records the raw cosine similarity to
each fact and episode, and writes the evidence the thresholds were chosen from.

    python scripts/calibrate_memory_thresholds.py --experiment t3_hm_pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import logger as logger_module  # noqa: E402
from src.config import MEMORY_DIR, RESULTS_DIR  # noqa: E402
from src.memory.memory_manager import MemoryManager  # noqa: E402

USER = "CUST-MAYA-1042"
# (query, source, relevant fact key or None, relevant episode type or None)
FACT_QUERIES = [
    ("What delivery window do I currently prefer?", "turn 27", "preferred_delivery_window"),
    ("Which delivery window did I most recently ask you to save?", "P01", "preferred_delivery_window"),
    ("Which city do I usually use for delivery?", "P02", "preferred_city"),
    ("Which channel should routine updates use now?", "P03", "notification_channel"),
    ("Can you still recall my latest delivery preference even though it is older than the last three turns?",
     "P16", "preferred_delivery_window"),
    ("What is my pet's name?", "P09 (never stated)", None),
    ("I had a frustrating late delivery on order ORD-88216. Please remember that issue.", "turn 10", None),
    ("I need to return the wireless earbuds from order ORD-74021.", "turn 12", None),
    ("The left earbud does not charge even after I tried the charging case.", "turn 13", None),
    ("Saturday, 28 November 2026 would work.", "turn 14", None),
    ("Before we continue, what is the electronics return window under the current policy?", "turn 16", None),
    ("What does code ERR-RMA-04 mean?", "turn 18", None),
    ("What happens if the driver misses my return pickup?", "turn 21", None),
    ("Can you check the latest status of order ORD-74021?", "turn 32", None),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="t3_hm_pilot")
    parser.add_argument("--run-id", default="t3_calibration")
    args = parser.parse_args()
    logger_module.set_run_id(args.run_id, reject_existing=True)
    memory = MemoryManager(MEMORY_DIR / "runs" / args.experiment, config="hybrid_mem", user=USER, autosave=False)
    memory.set_call_context(user=USER, session="B", turn=0)
    episodic_queries = json.loads((ROOT / "data" / "episodic_queries.json").read_text(encoding="utf-8"))["queries"]
    negatives = [(q, s) for q, s, key in FACT_QUERIES if key is None]
    texts = [q for q, _, _ in FACT_QUERIES] + [q["query"] for q in episodic_queries]
    vectors = memory._embed(texts, is_query=True, purpose="threshold_calibration")
    facts = {f.memory_id: f for f in memory.semantic.active_facts(USER)}
    episodes = {e.episode_id: e for e in memory.episodic.episodes(USER)}

    rows = []
    for (query, source, key), vector in zip(FACT_QUERIES, vectors):
        fact_scores = memory.semantic.index.similarities(vector, allowed_ids=set(facts))
        episode_scores = memory.episodic.index.similarities(vector)
        best_fact = max(fact_scores, key=fact_scores.get)
        relevant = [mid for mid, f in facts.items() if f.key == key]
        rows.append({
            "query": query, "source": source, "kind": "fact", "relevant_key": key,
            "relevant_score": max((fact_scores[m] for m in relevant), default=None),
            "best_fact": f"{best_fact} {facts[best_fact].key}", "best_fact_score": round(fact_scores[best_fact], 4),
            "best_irrelevant_fact_score": round(max(s for m, s in fact_scores.items() if m not in relevant), 4),
            "best_episode_score": round(max(episode_scores.values()), 4),
        })
    for query, vector in zip(episodic_queries, vectors[len(FACT_QUERIES):]):
        scores = memory.episodic.index.similarities(vector)
        gold = [eid for eid, e in episodes.items()
                if e.type == query["gold_type"] and e.source_turn == query["gold_source_turn"]]
        rows.append({
            "query": query["query"], "source": query["query_id"], "kind": "episode",
            "gold": gold, "gold_score": round(max((scores[g] for g in gold), default=float("nan")), 4),
            "best_episode": max(scores, key=scores.get), "best_episode_score": round(max(scores.values()), 4),
        })
    neg_fact = max(r["best_fact_score"] for r in rows if r["kind"] == "fact" and r["relevant_key"] is None)
    pos_fact = min(r["relevant_score"] for r in rows if r["kind"] == "fact" and r["relevant_key"])
    neg_episode = max(r["best_episode_score"] for r in rows if r["kind"] == "fact" and r["relevant_key"] is None)
    pos_episode = min(r["gold_score"] for r in rows if r["kind"] == "episode")
    summary = {
        "memory_source": f"memory/runs/{args.experiment} (pilot Session A)",
        "embedding_model": "BAAI/bge-small-en-v1.5 (CLS pooling, query instruction on queries)",
        "fact_scores": {"max_negative": neg_fact, "min_positive_relevant": round(pos_fact, 4)},
        "episode_scores": {"max_negative_turn": neg_episode, "min_gold": pos_episode},
        "negatives": [s for _, s in negatives],
        "rows": rows,
    }
    out = RESULTS_DIR / "task3_threshold_calibration.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
