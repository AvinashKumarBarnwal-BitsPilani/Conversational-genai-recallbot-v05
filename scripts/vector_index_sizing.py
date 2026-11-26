"""Task 5(b) part 1: one-year vector count and index memory by the CS3 formula.

Per-vector memory (CS3 Lecture 3, slides 48-49, following hnswlib):
  flat float32 : d x 4 bytes
  HNSW         : d x 4 (vector) + 1 (layer byte) + 2M x 4 (layer-0 links, 4 bytes each)
                 + a small upper-layer cost (ignored, as on the slide)
  practical budget: 1.2-1.5 x raw vector memory (growth, deletions, allocator overhead)

Vectors written per conversation are measured from the final hybrid_mem run
(the indexes saved after Session A and Session B). Scale: 25,000 daily users,
one conversation per user per day, 365 days.

    python scripts/vector_index_sizing.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAILY_USERS = 25_000
DAYS = 365
DIM = 384               # BAAI/bge-small-en-v1.5
M = 16
EXPERIMENT = "t4_hybrid_mem"


def per_vector_bytes(m: int | None) -> int:
    return DIM * 4 if m is None else DIM * 4 + 1 + 2 * m * 4


def gb(n_bytes: float) -> float:
    return n_bytes / 1e9


def main() -> int:
    run = ROOT / "memory" / "runs" / EXPERIMENT
    semantic = len(json.loads((run / "semantic_index.json").read_text(encoding="utf-8"))["ids"])
    episodic = len(json.loads((run / "episodic_index.json").read_text(encoding="utf-8"))["ids"])
    kb_chunks = len(json.loads((ROOT / "index" / "dense_128" / "chunks.json").read_text(encoding="utf-8")))

    scenarios = {
        # Facts supersede per key, so a returning user keeps ~`semantic` active fact vectors in total;
        # every daily conversation adds new episode vectors (events + reflection).
        "realistic (returning users; facts supersede, episodes accumulate)": {
            "semantic": semantic * DAILY_USERS,
            "episodic": episodic * DAILY_USERS * DAYS,
        },
        # Upper bound: every daily conversation behaves like the scripted one and adds new facts too.
        "upper bound (every conversation adds facts and episodes)": {
            "semantic": semantic * DAILY_USERS * DAYS,
            "episodic": episodic * DAILY_USERS * DAYS,
        },
    }
    rows = []
    for name, counts in scenarios.items():
        total = kb_chunks + counts["semantic"] + counts["episodic"]
        flat = total * per_vector_bytes(None)
        hnsw = total * per_vector_bytes(M)
        rows.append({
            "scenario": name, "kb_chunks": kb_chunks, "semantic_vectors": counts["semantic"],
            "episodic_vectors": counts["episodic"], "total_vectors": total,
            "flat_gb": round(gb(flat), 1), "hnsw_m16_gb": round(gb(hnsw), 1),
            "hnsw_graph_overhead_pct": round(100 * (hnsw - flat) / flat, 1),
            "practical_budget_gb": [round(gb(flat) * 1.2, 1), round(gb(flat) * 1.5, 1)],
        })
    report = {
        "measured_per_conversation": {"experiment": EXPERIMENT, "semantic_vectors": semantic,
                                      "episodic_vectors": episodic, "kb_chunks_128": kb_chunks},
        "assumptions": {"daily_users": DAILY_USERS, "days": DAYS, "dimension": DIM, "M": M,
                        "formula_source": "CS3 Lecture 3 slides 48-49 (hnswlib per-vector estimate)"},
        "bytes_per_vector": {"flat": per_vector_bytes(None), "hnsw_m16": per_vector_bytes(M),
                             **{f"hnsw_m{m}": per_vector_bytes(m) for m in (8, 32, 64)}},
        "scenarios": rows,
    }
    out = ROOT / "results" / "task5_vector_sizing.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
