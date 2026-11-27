"""Task 3(b) hit-rate chart, drawn only from results/hit_rate.csv.

    python scripts/make_hit_rate_figure.py

Writes images/retrieval-hit-rate-comparison.png (answer-span hit rate @1/@3/@5 for
dense, BM25 and hybrid at 128- and 512-token chunks, all 32 kb_questions).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("dense", "Dense"), ("bm25", "BM25"), ("hybrid", "Hybrid (RRF)"))
SIZES = (("128", "128-token chunks", "#2a6fbb"), ("512", "512-token chunks", "#e0822b"))


def main() -> int:
    hits: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    with (ROOT / "results" / "hit_rate.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hits[(row["method"], row["chunk_size"], row["k"])].append(int(row["hit"]))
    groups = [(m, label, k) for k in ("1", "3", "5") for m, label in METHODS]
    n_questions = len(hits[("dense", "128", "1")])

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    width = 0.38
    for offset, (size, label, colour) in zip((-width / 2, width / 2), SIZES):
        values = [100 * sum(hits[(m, size, k)]) / len(hits[(m, size, k)]) for m, _, k in groups]
        xs = [i + offset for i in range(len(groups))]
        ax.bar(xs, values, width, label=label, color=colour)
        for x, v in zip(xs, values):
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(groups)), [f"{label} @{k}" for _, label, k in groups], rotation=25, ha="right", fontsize=8.5)
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="#bbbbbb", lw=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Answer-span hit rate (%)")
    ax.set_title(f"Retrieval hit rate on {n_questions} kb_questions (from results/hit_rate.csv)", fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    out = ROOT / "images" / "retrieval-hit-rate-comparison.png"
    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
