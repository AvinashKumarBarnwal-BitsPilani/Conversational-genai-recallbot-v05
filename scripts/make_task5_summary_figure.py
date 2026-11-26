"""Draw the Task 5 summary figure from results/task5_*.json/csv (no typed-in numbers).

    python scripts/make_task5_summary_figure.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
M_COLORS = {8: "#a6c6ee", 16: "#72a5e3", 32: "#2a78d6", 64: "#1b4f8f"}  # one hue, light -> dark with M
RECOMMENDED = (32, 200)


def style(axis):
    axis.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(RULE)
    axis.tick_params(colors=INK2, labelsize=8)


def main() -> int:
    econ = json.loads((RES / "task5_token_economics.json").read_text(encoding="utf-8"))
    sizing = json.loads((RES / "task5_vector_sizing.json").read_text(encoding="utf-8"))
    with (RES / "task5_ann_benchmark_clustered.csv").open(encoding="utf-8", newline="") as handle:
        bench = list(csv.DictReader(handle))
    with (RES / "task5_ann_benchmark.csv").open(encoding="utf-8", newline="") as handle:
        bench_random = list(csv.DictReader(handle))

    fig = plt.figure(figsize=(16, 10.4), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    a = econ["assumptions"]
    ax.text(0.02, 0.975, "Task 5 · What RecallBot would cost, and how big its vector index gets",
            fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.942, f"{a['daily_users']:,} daily users × 1 conversation × {a['turns_per_conversation']} turns × "
            f"{a['days_per_month']} days = {a['turns_per_month'] / 1e6:.0f}M turns/month · prices: CS3 slide 17, "
            "GPT-4o row · caching rules: CS3 slide 18 · memory formula: CS3 slides 48–49",
            fontsize=10, color=INK2, va="top")

    # ------------------------------------------------ 5(a) monthly cost
    ax.text(0.02, 0.895, "5(a) Monthly cost, GPT-4o ($2.50 in / $10.00 out / $1.25 cached per 1M)", fontsize=12,
            fontweight="bold", color=INK)
    chart = fig.add_axes([0.06, 0.53, 0.4, 0.32])
    style(chart)
    order = ["trim", "no_rag (reference)", "hybrid_mem"]
    gpt = {s["config"]: s for s in econ["scenarios"] if s["price_row"].startswith("GPT-4o")}
    xs = range(len(order))
    no_cache = [gpt[c]["cost_no_cache_usd"] / 1000 for c in order]
    cached = [gpt[c]["cost_with_cache_usd"] / 1000 for c in order]
    chart.bar([x - 0.18 for x in xs], no_cache, width=0.34, color=RULE, label="no caching")
    chart.bar([x + 0.18 for x in xs], cached, width=0.34, color=BLUE, label="prefix caching")
    for x, v1, v2, c in zip(xs, no_cache, cached, order):
        chart.text(x - 0.18, v1 + 1.5, f"${v1:,.1f}k", ha="center", fontsize=8, color=INK)
        chart.text(x + 0.18, v2 + 1.5, f"${v2:,.1f}k", ha="center", fontsize=8, color=INK)
    labels = []
    for c in order:
        cfg = econ["configs"][c]
        tag = "cacheable" if cfg["prefix_cacheable"] else "too short to cache"
        labels.append(f"{c.replace(' (reference)', ' (ref.)')}\nprefix {cfg['fixed_prefix_tokens']:,} tok\n{tag}")
    chart.set_xticks(list(xs), labels, fontsize=8)
    chart.set_yticks([])
    chart.set_ylim(0, max(no_cache) * 1.2)
    chart.legend(frameon=False, fontsize=8.5, loc="upper right")
    pad = next(p for p in econ["padding_what_if"] if p["price_row"].startswith("GPT-4o"))
    ax.text(0.02, 0.455,
            f"• trim (v1 prompt, 1,100-token prefix) clears the 1,024 minimum: caching saves "
            f"{gpt['trim']['saving_pct']:.0f}%.\n"
            f"• hybrid_mem's compact prefix ({econ['configs']['hybrid_mem']['fixed_prefix_tokens']} tokens) cannot be "
            "cached, yet it is the cheaper design.\n"
            f"• Padding it to 1,024 tokens would add ${pad['monthly_change_usd'] / 1000:,.1f}k/month on GPT-4o and use "
            f"{pad['input_budget_used_by_padding_pct']:.0f}% of the 1,856-token input budget: don't pad.\n"
            "• no_rag uses the same v2 prompt: KB + long-term memory add "
            f"${(gpt['hybrid_mem']['cost_no_cache_usd'] - gpt['no_rag (reference)']['cost_no_cache_usd']) / 1000:,.1f}k/month.",
            fontsize=8.8, color=INK2, va="top", linespacing=1.55)

    # ------------------------------------------------ 5(b) sizing
    ax.text(0.53, 0.895, "5(b) Vectors after one year (CS3 per-vector formula, d = 384)", fontsize=12,
            fontweight="bold", color=INK)
    bpv = sizing["bytes_per_vector"]
    ax.text(0.53, 0.86, f"flat = d×4 = {bpv['flat']:,} B/vector · HNSW M=16 = d×4 + 1 + 2M×4 = {bpv['hnsw_m16']:,} "
            f"B/vector (+{100 * (bpv['hnsw_m16'] - bpv['flat']) / bpv['flat']:.1f}%)", fontsize=8.8, color=INK2)
    headers = ["Scenario", "Vectors", "Flat", "HNSW M=16", "Budget ×1.2–1.5"]
    col = [0.53, 0.72, 0.79, 0.86, 0.94]
    for j, h in enumerate(headers):
        ax.text(col[j], 0.82, h, fontsize=8.8, fontweight="bold", color=INK, ha="left" if j == 0 else "center")
    for i, row in enumerate(sizing["scenarios"]):
        y = 0.785 - i * 0.045
        ax.plot([0.53, 0.98], [y - 0.017, y - 0.017], color=RULE, linewidth=0.8)
        name = "Realistic: facts supersede,\nepisodes accumulate" if i == 0 else "Upper bound: every chat\nadds new facts too"
        ax.text(col[0], y, name, fontsize=8.2, color=INK2, va="center", linespacing=1.2)
        ax.text(col[1], y, f"{row['total_vectors'] / 1e6:,.1f} M", fontsize=9, color=INK, ha="center", va="center")
        ax.text(col[2], y, f"{row['flat_gb']:,.0f} GB", fontsize=9, color=INK, ha="center", va="center")
        ax.text(col[3], y, f"{row['hnsw_m16_gb']:,.0f} GB", fontsize=9, color=INK, ha="center", va="center")
        lo, hi = row["practical_budget_gb"]
        ax.text(col[4], y, f"{lo:,.0f}–{hi:,.0f} GB", fontsize=9, color=INK, ha="center", va="center")
    m = sizing["measured_per_conversation"]
    ax.text(0.53, 0.675, f"Measured per conversation (final hybrid_mem run): {m['semantic_vectors']} semantic + "
            f"{m['episodic_vectors']} episodic vectors; KB index {m['kb_chunks_128']} chunks. Episodes are "
            f"{100 * sizing['scenarios'][0]['episodic_vectors'] / sizing['scenarios'][0]['total_vectors']:.1f}% of "
            "the realistic total.", fontsize=8.6, color=INK2, wrap=True)

    # ------------------------------------------------ 5(b) benchmark
    ax.text(0.53, 0.615, "Measured: HNSW vs flat on 100,000 vectors (realistic clustered padding)", fontsize=12,
            fontweight="bold", color=INK)
    curve = fig.add_axes([0.565, 0.19, 0.41, 0.38])
    style(curve)
    flat_row = next(r for r in bench if r["index"] == "flat")
    for mval, color in M_COLORS.items():
        rows = [r for r in bench if r["index"] == "hnsw" and int(r["M"]) == mval]
        lat = [float(r["p50_ms"]) for r in rows]
        rec = [float(r["recall5_kb"]) for r in rows]
        curve.plot(lat, rec, color=color, linewidth=2, marker="o", markersize=4.5, label=f"M = {mval}")
        curve.text(lat[-1] * 1.06, rec[-1], f"M={mval}", fontsize=7.8, color=INK, va="center")
    rec_row = next(r for r in bench if r["index"] == "hnsw" and (int(r["M"]), int(r["ef_search"])) == RECOMMENDED)
    curve.scatter([float(rec_row["p50_ms"])], [float(rec_row["recall5_kb"])], s=130, facecolor="none",
                  edgecolor=ORANGE, linewidth=2, zorder=5)
    curve.annotate(f"recommended M=32, ef=200\n{float(rec_row['p50_ms']):.2f} ms, recall@5 {float(rec_row['recall5_kb']):.3f}",
                   (float(rec_row["p50_ms"]), float(rec_row["recall5_kb"])), xytext=(0.06, 0.985),
                   fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1})
    curve.axvline(float(flat_row["p50_ms"]), color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    curve.text(float(flat_row["p50_ms"]) * 0.95, 0.62, f"flat (exact)\n{float(flat_row['p50_ms']):.1f} ms",
               fontsize=8, color=INK2, ha="right")
    curve.axhline(0.95, color=RULE, linewidth=1)
    curve.set_xscale("log")
    curve.set_xlabel("p50 query latency, ms (log scale; single thread)", fontsize=8.6, color=INK2)
    curve.set_ylabel("recall@5 vs flat (32 KB questions)", fontsize=8.6, color=INK2)
    curve.set_ylim(0.3, 1.02)
    curve.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(0.86, 0.0))
    easy = [float(r["recall5_kb"]) for r in bench_random if r["index"] == "hnsw" and int(r["ef_search"]) >= 50]
    worst = max(float(r["recall5_random"]) for r in bench_random if r["index"] == "hnsw")
    ax.text(0.53, 0.115,
            "Each line runs ef_search 16 → 500 (left to right). With purely random padding the KB questions are\n"
            f"trivial (recall@5 ≥ {min(easy):.2f} for ef ≥ 50) while random queries on random data never pass "
            f"{worst:.2f}.\nNeither extreme is realistic, so the recommendation uses clustered padding.",
            fontsize=8.4, color=INK2, va="top", linespacing=1.45)

    # ------------------------------------------------ recommendation tiles
    tiles = [
        ("Cheapest design", "hybrid_mem", f"${gpt['hybrid_mem']['cost_no_cache_usd'] / 1000:,.1f}k/month on GPT-4o\n"
         f"vs ${gpt['trim']['cost_with_cache_usd'] / 1000:,.1f}k for cached trim"),
        (f"Index for {sizing['scenarios'][0]['total_vectors'] / 1e6:,.0f}M vectors", "HNSW",
         f"flat would take ~{float(flat_row['p50_ms']) * sizing['scenarios'][0]['total_vectors'] / 100_000 / 1000:,.1f} s "
         f"per query\nat this scale ({float(flat_row['p50_ms']):.1f} ms per 100k, linear)"),
        ("Parameters", "M=32, ef=200", f"recall@5 {float(rec_row['recall5_kb']):.3f} at "
         f"{float(rec_row['p50_ms']):.2f} ms;\n{float(rec_row['index_mb']):.0f} MB per 100k vectors"),
    ]
    tw = 0.152
    for i, (label, value, caption) in enumerate(tiles):
        x = 0.02 + i * (tw + 0.008)
        ax.add_patch(FancyBboxPatch((x, 0.035), tw, 0.19, boxstyle="round,pad=0,rounding_size=0.006",
                                    linewidth=1, edgecolor=RULE, facecolor="white"))
        ax.text(x + 0.01, 0.21, label, fontsize=8.6, color=INK2, va="top")
        ax.text(x + 0.01, 0.175, value, fontsize=15, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.01, 0.05, caption, fontsize=8, color=INK2, va="bottom", linespacing=1.35)

    out = ROOT / "images" / "task5-summary.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
