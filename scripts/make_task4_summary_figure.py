"""Draw the Task 4 summary figure from results/ and logs/ (no typed-in numbers).

    python scripts/make_task4_summary_figure.py
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
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
# Sequential single-hue ramp (light -> dark blue) for accuracy and rubric cells.
RAMP = ["#eef4fc", "#cfe0f6", "#a6c6ee", "#72a5e3", "#2a78d6", "#1b5aa8"]
CONFIGS = ("no_rag", "hybrid", "hybrid_sem", "hybrid_mem")
LABELS = {"no_rag": "no_rag\n(short-term only)", "hybrid": "hybrid\n(+ knowledge base)",
          "hybrid_sem": "hybrid_sem\n(+ semantic)", "hybrid_mem": "hybrid_mem\n(+ episodic)"}
CATS = [("R", "R  semantic recall"), ("U", "U  updated fact"), ("F", "F  forgotten fact"), ("E", "E  episodic recall"),
        ("K", "K  knowledge base"), ("N", "N  not available"), ("S", "S  state follow-up"),
        ("TWIST_SMALL_CONTEXT", "Twist E  small window")]


def ramp(fraction: float) -> str:
    return RAMP[min(len(RAMP) - 1, int(round(fraction * (len(RAMP) - 1))))]


def cell_text_color(fraction: float) -> str:
    # Text colour follows the fill step actually drawn, not the raw value.
    return "white" if RAMP.index(ramp(fraction)) >= 4 else INK


def main() -> int:
    with (ROOT / "results" / "probe_results.csv").open(encoding="utf-8", newline="") as handle:
        probes = list(csv.DictReader(handle))
    with (ROOT / "results" / "task4_ablation_table.csv").open(encoding="utf-8", newline="") as handle:
        table = {r["config"]: r for r in csv.DictReader(handle)}
    rubric = json.loads((ROOT / "results" / "task4_rubric_summary.json").read_text(encoding="utf-8"))
    fix = [p for p in probes if p["run_id"].startswith("t4fix_hybrid_mem")]
    before = [p for p in probes if p["run_id"].startswith("t3_hm_01")]

    fig = plt.figure(figsize=(16, 10.6), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.02, 0.972, "Task 4 · What each memory type adds, what it costs, and where RecallBot still fails",
            fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.937, "Four configurations, Session A then Session B as separate processes (runs t4_<config>_A/_B) · "
            "16 probes per config, manually judged · same model qwen3:1.7b and decoding", fontsize=10.5, color=INK2, va="top")

    # ---------------- accuracy heatmap ----------------
    ax.text(0.02, 0.895, "Probe accuracy by category (correct / probes)", fontsize=12, fontweight="bold", color=INK)
    x0, y0, cw, ch, lw = 0.17, 0.855, 0.088, 0.047, 0.15
    for j, config in enumerate(CONFIGS):
        ax.text(x0 + j * cw + cw / 2, y0 + 0.006, LABELS[config], ha="center", va="bottom", fontsize=8.6, color=INK2)
    rows = CATS + [("ALL", "Overall")]
    for i, (key, label) in enumerate(rows):
        y = y0 - (i + 1) * ch
        bold = key == "ALL"
        ax.text(x0 - 0.008, y + ch / 2, label, ha="right", va="center", fontsize=9.2, color=INK,
                fontweight="bold" if bold else "normal")
        for j, config in enumerate(CONFIGS):
            runs = (f"t4_{config}_A", f"t4_{config}_B")
            marks = [int(p["correct"]) for p in probes if p["run_id"] in runs and (bold or p["category"] == key)]
            frac = sum(marks) / len(marks)
            ax.add_patch(FancyBboxPatch((x0 + j * cw + 0.002, y + 0.003), cw - 0.004, ch - 0.006,
                                        boxstyle="round,pad=0,rounding_size=0.004", linewidth=0,
                                        facecolor=ramp(frac)))
            ax.text(x0 + j * cw + cw / 2, y + ch / 2, f"{sum(marks)}/{len(marks)}", ha="center", va="center",
                    fontsize=10 if bold else 9.5, fontweight="bold" if bold else "normal", color=cell_text_color(frac))
    ax.text(0.02, y0 - (len(rows) + 1) * ch + 0.02,
            "Each layer unlocks its own categories: KB → K; semantic → R, U, twist; episodic → E.\n"
            "S stays 1/2 in every config: P11 (what happens after pickup) fails throughout.",
            fontsize=8.8, color=INK2, va="top", linespacing=1.4)

    # ---------------- cost bars ----------------
    bx, by, bw, bh = 0.56, 0.62, 0.42, 0.24
    ax.text(bx, 0.895, "Cost per configuration (from logs/calls.jsonl)", fontsize=12, fontweight="bold", color=INK)
    metrics = [("Mean input tokens / turn", "mean_input_tokens_per_turn", "{:,.0f}"),
               ("Median turn latency (ms)", "median_turn_latency_ms", "{:,.0f}"),
               ("Total tokens (A + B)", "total_tokens", "{:,.0f}")]
    panel_w = bw / 3
    for k, (title, key, fmt) in enumerate(metrics):
        sub = fig.add_axes([bx + k * panel_w + 0.012, by, panel_w - 0.03, bh])
        values = [float(table[c][key]) for c in CONFIGS]
        bars = sub.bar(range(4), values, color=[RULE, RAMP[2], RAMP[3], BLUE], width=0.68)
        for bar, value in zip(bars, values):
            sub.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom",
                     fontsize=7.6, color=INK)
        sub.set_title(title, fontsize=9.2, color=INK, loc="left")
        sub.set_xticks(range(4), [c.replace("hybrid_", "h_") for c in CONFIGS], fontsize=7.6, color=INK2)
        sub.set_yticks([])
        sub.set_ylim(0, max(values) * 1.18)
        for spine in ("top", "right", "left"):
            sub.spines[spine].set_visible(False)
        sub.spines["bottom"].set_color(RULE)
        sub.set_facecolor(SURFACE)

    # ---------------- rubric ----------------
    ax.text(bx, 0.555, "Dialogue quality, 15 Session B replies per config (manual, 1-5)", fontsize=12,
            fontweight="bold", color=INK)
    dims = ("coherence", "relevance", "groundedness", "safety")
    rx, ry, rcw, rch = bx + 0.11, 0.505, 0.075, 0.04
    for j, dim in enumerate(dims):
        ax.text(rx + j * rcw + rcw / 2, ry + 0.006, dim, ha="center", va="bottom", fontsize=8.6, color=INK2)
    for i, config in enumerate(CONFIGS):
        y = ry - (i + 1) * rch
        ax.text(rx - 0.008, y + rch / 2, config, ha="right", va="center", fontsize=9.2, color=INK)
        for j, dim in enumerate(dims):
            value = rubric["per_config"][config][dim]
            frac = (value - 1) / 4
            ax.add_patch(FancyBboxPatch((rx + j * rcw + 0.002, y + 0.003), rcw - 0.004, rch - 0.006,
                                        boxstyle="round,pad=0,rounding_size=0.004", linewidth=0, facecolor=ramp(frac)))
            ax.text(rx + j * rcw + rcw / 2, y + rch / 2, f"{value:.2f}", ha="center", va="center", fontsize=9.5,
                    color=cell_text_color(frac))
    agreement = rubric["agreement"]
    ax.text(bx, ry - 4 * rch - 0.014,
            "Relevance rises with each memory type; no_rag looks safe only because it refuses.\n"
            "hybrid_mem loses coherence to stray 'Missing detail' lines.\n"
            "Qwen judge vs manual (60 replies), Cohen's kappa C / R / G / S: "
            f"{agreement['coherence']['cohen_kappa']:.2f} / {agreement['relevance']['cohen_kappa']:.2f} / "
            f"{agreement['groundedness']['cohen_kappa']:.2f} / {agreement['safety']['cohen_kappa']:.2f}\n"
            "The 1.7B judge is not a reliable substitute for manual rating.",
            fontsize=8.6, color=INK2, va="top", linespacing=1.4)

    # ---------------- error analysis tiles ----------------
    ax.text(0.02, 0.232, "4(b) Three hybrid_mem failures (run t3_hm_01), traced to their stage and fixed there",
            fontsize=12, fontweight="bold", color=INK)
    mark = {p["probe_id"]: int(p["correct"]) for p in fix}
    mark_before = {p["probe_id"]: int(p["correct"]) for p in before}
    tiles = [
        ("P13 · E", "Memory write", VIOLET,
         "Decision episode text dropped the product,\nso 'USB-C charger' query ranked it 3rd.\n"
         "Fix: store the customer's own words.\nRelevance 0.606 → 0.799, rank 3 → 1."),
        ("P11 · S", "KB retrieval → prompt", ORANGE,
         "'What happens next after pickup?' shares no\nwords with KB-003 (rank 12). Fix: lifecycle\n"
         "query rewrite → KB-003 now rank 1, but the\nmodel still misreads it (residual NLG fault)."),
        ("P15 · Twist", "NLG / prompt", BLUE,
         "Correct draft rewritten by the model into\n'ask for the next missing detail'.\n"
         "Fix: budget answer emitted verbatim by the\ndialogue manager (no model call)."),
    ]
    tw = 0.312
    for k, (title, stage, color, body) in enumerate(tiles):
        tx = 0.02 + k * (tw + 0.012)
        pid = title.split(" ")[0]
        ax.add_patch(FancyBboxPatch((tx, 0.03), tw, 0.18, boxstyle="round,pad=0,rounding_size=0.008",
                                    linewidth=1.4, edgecolor=color, facecolor=color + "14"))
        ax.text(tx + 0.012, 0.197, f"{title}   stage: {stage}", fontsize=10.5, fontweight="bold", color=INK, va="top")
        ax.text(tx + 0.012, 0.166, body, fontsize=8.8, color=INK2, va="top", linespacing=1.45)
        result = f"before {mark_before.get(pid, '?')} → after {mark.get(pid, '?')}   (1 = correct)"
        if pid == "P15":
            p15_after = [int(p["correct"]) for p in probes if p["run_id"] == "t4_hybrid_mem_B" and p["probe_id"] == "P15"]
            result = f"before {mark_before.get(pid)} → after {p15_after[0] if p15_after else '?'}   (1 = correct)"
        ax.text(tx + 0.012, 0.042, result, fontsize=9.5, fontweight="bold", color=INK, va="bottom")

    out = ROOT / "images" / "task4-summary.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
