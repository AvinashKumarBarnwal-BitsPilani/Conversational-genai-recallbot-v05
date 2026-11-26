"""Task 5(b) part 2: flat vs HNSW on >= 100,000 vectors, measured on this machine.

Corpus: the real 384-d KB chunk vectors (index/dense_128) and the final hybrid_mem
memory vectors, padded with random unit vectors to N (CS3 / assignment: "pad your
index with random vectors if needed"). Cosine = inner product on unit vectors.

Query sets:
  * kb_questions - the 32 retrieval questions, embedded with the same BGE model
    (logged embedding call). This is the recall the assignment asks for.
  * random       - 200 random unit queries. Random padding sits far from real
    questions, which makes the kb_questions case easy; random queries over random
    data are the hard case for HNSW, so both are reported.

Exact search (faiss.IndexFlatIP) is the ground truth. HNSW (faiss.IndexHNSWFlat,
inner product, ef_construction = 200) is swept over M and ef_search. Search
latency is single-threaded, one query at a time (an online chatbot request).

Padding modes:
  * random    - uniform random unit vectors (literal reading of the assignment).
  * clustered - noisy copies of the real vectors (cosine ~0.6-0.9 to their source), so each
                real question has thousands of plausible near neighbours: the realistic case
                for a large production index of similar embeddings.

    python scripts/benchmark_ann.py                      # N = 100,000, random padding
    python scripts/benchmark_ann.py --padding clustered
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402

from src import logger as logger_module  # noqa: E402

M_GRID = (8, 16, 32, 64)
EF_GRID = (16, 50, 100, 200, 500)
EF_CONSTRUCTION = 200
K = 5
SEED = 7


def unit(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype("float32")
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def latencies_ms(index, queries: np.ndarray, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        for q in queries:
            start = time.perf_counter()
            index.search(q[None, :], K)
            times.append((time.perf_counter() - start) * 1000)
    return times


def recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--run-id", default="t5_ann_benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--padding", choices=("random", "clustered"), default="random")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    real = [np.load(ROOT / "index" / "dense_128" / "vectors.npy")]
    for name in ("semantic", "episodic"):
        path = ROOT / "memory" / "runs" / "t4_hybrid_mem" / f"{name}_vectors.npy"
        if path.exists():
            real.append(np.load(path))
    real_vectors = unit(np.vstack(real))
    dim = real_vectors.shape[1]
    n_pad = args.n - len(real_vectors)
    if args.padding == "random":
        padding = unit(rng.standard_normal((n_pad, dim)))
    else:
        # Noise of norm s on a unit vector gives cosine ~ 1/sqrt(1 + s^2); s in [0.5, 1.3] -> ~0.6-0.9.
        sources = real_vectors[rng.integers(0, len(real_vectors), n_pad)]
        scale = rng.uniform(0.5, 1.3, (n_pad, 1)) / np.sqrt(dim)
        padding = unit(sources + rng.standard_normal((n_pad, dim)) * scale)
    corpus = np.vstack([real_vectors, padding]).astype("float32")

    # Query embeddings: the same BGE model and query instruction as retrieval (one logged embed call).
    logger_module.set_run_id(args.run_id, reject_existing=True)
    from src.retrieval.embeddings import SentenceTransformerEmbedder
    questions = json.loads((ROOT / "data" / "kb_questions.json").read_text(encoding="utf-8"))
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    kb_queries = unit(embedder.encode([q["question"] for q in questions], purpose="ann_benchmark_queries",
                                      is_query=True, user="BENCHMARK", session="A", turn=0, config="hybrid"))
    query_sets = {"kb_questions": kb_queries}
    if args.padding == "random":
        query_sets["random"] = unit(rng.standard_normal((200, dim)))

    faiss.omp_set_num_threads(1)  # single-threaded search latency (one online request)
    flat = faiss.IndexFlatIP(dim)
    flat.add(corpus)
    truth = {name: flat.search(q, K)[1] for name, q in query_sets.items()}
    flat_lat = latencies_ms(flat, query_sets["kb_questions"], args.repeats)
    flat_bytes = len(faiss.serialize_index(flat))
    rows = [{"index": "flat", "M": "", "ef_search": "", "build_s": 0.0,
             "index_mb": round(flat_bytes / 1e6, 1), "bytes_per_vector": round(flat_bytes / args.n, 1),
             "p50_ms": round(statistics.median(flat_lat), 3),
             "p95_ms": round(float(np.percentile(flat_lat, 95)), 3),
             "recall1_kb": 1.0, "recall5_kb": 1.0,
             "recall1_random": 1.0 if "random" in query_sets else "", "recall5_random": 1.0 if "random" in query_sets else ""}]
    print(f"flat: {rows[0]['p50_ms']} ms p50, {rows[0]['index_mb']} MB")

    for m in M_GRID:
        faiss.omp_set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # build may use several threads
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        start = time.perf_counter()
        index.add(corpus)
        build_s = time.perf_counter() - start
        size = len(faiss.serialize_index(index))
        faiss.omp_set_num_threads(1)
        for ef in EF_GRID:
            index.hnsw.efSearch = ef
            found = {name: index.search(q, K)[1] for name, q in query_sets.items()}
            lat = latencies_ms(index, query_sets["kb_questions"], args.repeats)
            row = {"index": "hnsw", "M": m, "ef_search": ef, "build_s": round(build_s, 1),
                   "index_mb": round(size / 1e6, 1), "bytes_per_vector": round(size / args.n, 1),
                   "p50_ms": round(statistics.median(lat), 3), "p95_ms": round(float(np.percentile(lat, 95)), 3),
                   "recall1_kb": round(recall(found["kb_questions"], truth["kb_questions"], 1), 3),
                   "recall5_kb": round(recall(found["kb_questions"], truth["kb_questions"], K), 3),
                   "recall1_random": round(recall(found["random"], truth["random"], 1), 3) if "random" in found else "",
                   "recall5_random": round(recall(found["random"], truth["random"], K), 3) if "random" in found else ""}
            rows.append(row)
            print(f"M={m:<2} ef={ef:<3} build {build_s:5.1f}s  p50 {row['p50_ms']:.3f} ms  "
                  f"recall@1 kb {row['recall1_kb']:.3f} recall@5 kb {row['recall5_kb']:.3f} random {row['recall5_random']}  {row['index_mb']} MB")

    suffix = "" if args.padding == "random" else f"_{args.padding}"
    out_csv = ROOT / "results" / f"task5_ann_benchmark{suffix}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"padding": args.padding, "n_vectors": args.n, "real_vectors": int(len(real_vectors)), "random_padding": int(len(padding)),
            "dimension": dim, "k": K, "ef_construction": EF_CONSTRUCTION, "seed": SEED,
            "query_sets": {k: int(len(v)) for k, v in query_sets.items()}, "repeats": args.repeats,
            "search_threads": 1, "faiss_version": faiss.__version__,
            "machine_note": "measured on the student's Windows laptop (CPU only)"}
    (ROOT / "results" / f"task5_ann_benchmark{suffix}_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
