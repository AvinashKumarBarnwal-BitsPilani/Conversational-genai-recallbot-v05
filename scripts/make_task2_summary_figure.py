"""Draw the Task 2 summary figure from the Task 2 result files (no typed-in numbers).

Token curves come from results/task2b_token_per_turn.csv, which
scripts/analyze_task2b_logs.py derives from logs/calls.jsonl.

    python scripts/make_task2_summary_figure.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CONTEXT_WINDOW_TOKENS, TRIM_WINDOW_K_TURNS  # noqa: E402

SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
SERIES = {"full": (BLUE, "full history"), "trim": (ORANGE, f"trim, last k = {TRIM_WINDOW_K_TURNS}"),
          "summary": (AQUA, "rolling summary + last 2")}


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", linewidth=1.5,
                                edgecolor=color, facecolor=color + "1a"))
    ax.text(x + w / 2, y + h - 0.014, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.04, body, ha="center", va="top", fontsize=8.1, color=INK2, linespacing=1.35)


def arrow(ax, a, b, color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=color,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    res = ROOT / "results"
    with (res / "task2b_token_per_turn.csv").open(encoding="utf-8-sig", newline="") as handle:
        per_turn = list(csv.DictReader(handle))
    with (res / "task2b_token_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        summary = {r["strategy"]: r for r in csv.DictReader(handle)}
    with (res / "task2c_probe_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        probes = list(csv.DictReader(handle))
    nlu = json.loads((res / "nlu_accuracy.json").read_text(encoding="utf-8"))
    facts = json.loads((ROOT / "data" / "test_conversation.json").read_text(encoding="utf-8"))["planted_user_facts"]

    fig = plt.figure(figsize=(16, 10.8), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.02, 0.975, "Task 2 · The dialogue pipeline and three ways to keep short-term history",
            fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.942, "Rule-based NLU → state tracker → dialogue manager → tools → Qwen NLG, with procedural memory "
            "loaded from skills/ · Session A replayed once per history strategy", fontsize=10.5, color=INK2, va="top")

    # ---------------------------------------------------------------- pipeline
    ax.text(0.02, 0.9, "How a turn is processed (function names from src/dialogue/)", fontsize=12,
            fontweight="bold", color=INK)
    y, h = 0.755, 0.11
    box(ax, 0.02, y, 0.115, h, "Customer turn", '"And for Saturday,\nafter 6 pm?"', BLUE)
    box(ax, 0.155, y, 0.14, h, "NLU", "analyze_utterance()\nrules + regex: intent,\norder / SKU / date / time", BLUE)
    box(ax, 0.315, y, 0.15, h, "State tracker", "DialogueStateTracker\nslots, missing slots,\npending confirmation", BLUE)
    box(ax, 0.485, y, 0.15, h, "Dialogue manager", "DialogueManager\n.handle_turn(): route,\nask, call tools, confirm", BLUE)
    box(ax, 0.655, y, 0.15, h, "Qwen NLG", "QwenNLG.generate_reply()\nqwen3:1.7b, temp 0 task\n/ 0.3 open turns", BLUE)
    box(ax, 0.825, y, 0.155, h, "Reply", "grounded wording; the\ndeterministic draft is the\nfallback on any failure", BLUE)
    for a, b in ((0.135, 0.155), (0.295, 0.315), (0.465, 0.485), (0.635, 0.655), (0.805, 0.825)):
        arrow(ax, (a, y + h / 2), (b, y + h / 2))
    ty, th = 0.6, 0.11
    box(ax, 0.315, ty, 0.32, th, "Tools (RecallBotToolGateway)",
        "search_kb (hybrid KB) · get_order (verified owner)\ncheck_pickup_slots · book_slot: writes ONLY after an\n"
        "explicit 'yes' to the exact pending summary", ORANGE)
    box(ax, 0.655, ty, 0.15, th, "History strategy", "ConversationHistory\nfull / trim\nRollingSummaryMemory", AQUA)
    box(ax, 0.825, ty, 0.155, th, "Procedural memory", "skills/system_prompt_v1.txt\nskills/return_pickup_\nworkflow_v1.json", VIOLET)
    box(ax, 0.02, ty, 0.275, th, "Slot filling (return pickup)",
        "order → item → reason → date → window →\naddress → summary → explicit confirmation\n"
        "→ book_slot → real RMA reference", YELLOW)
    arrow(ax, (0.56, y), (0.52, ty + th))
    arrow(ax, (0.73, ty + th), (0.73, y))
    arrow(ax, (0.9, ty + th), (0.76, y))
    arrow(ax, (0.39, y), (0.2, ty + th))

    # ---------------------------------------------------------------- token curves
    ax.text(0.02, 0.555, "Input tokens per turn, Session A, from logs/calls.jsonl", fontsize=12,
            fontweight="bold", color=INK)
    chart = fig.add_axes([0.055, 0.2, 0.5, 0.31])
    chart.set_facecolor(SURFACE)
    for strategy, (color, label) in SERIES.items():
        rows = sorted((r for r in per_turn if r["strategy"] == strategy), key=lambda r: int(r["turn"]))
        turns = [int(r["turn"]) for r in rows]
        tokens = [int(r["chat_input_tokens_logged_attempted"]) for r in rows]
        chart.plot(turns, tokens, color=color, linewidth=2, label=label, zorder=3)
        sent = [(t, v) for t, v, r in zip(turns, tokens, rows) if r["chat_status"] == "success"]
        unsent = [(t, v) for t, v, r in zip(turns, tokens, rows) if r["chat_status"] != "success"]
        chart.scatter(*zip(*sent), s=22, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        if unsent:
            chart.scatter(*zip(*unsent), s=34, facecolor=SURFACE, edgecolor=color, linewidth=1.6, zorder=5)
        chart.text(turns[-1] + 0.4, tokens[-1], label.split(",")[0].split(" +")[0], color=INK, fontsize=8.5,
                   va="center")
    chart.axhline(CONTEXT_WINDOW_TOKENS, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    chart.text(1, CONTEXT_WINDOW_TOKENS + 60, f"context window {CONTEXT_WINDOW_TOKENS:,} tokens", fontsize=8.2,
               color=INK2)
    chart.set_xlim(0.5, 28)
    chart.set_xlabel("Session A turn", fontsize=9, color=INK2)
    chart.set_ylabel("input tokens (attempted)", fontsize=9, color=INK2)
    chart.tick_params(colors=INK2, labelsize=8)
    chart.grid(axis="y", color=RULE, linewidth=0.6)
    for spine in ("top", "right"):
        chart.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        chart.spines[spine].set_color(RULE)
    chart.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.text(0.055, 0.112, "Filled marker = prompt sent to Qwen; hollow = over the safe input budget, not sent "
            "(deterministic fallback reply).\nFull history grows every turn and stops being sendable from turn 14; "
            "trim stays flattest.", fontsize=8.6, color=INK2, va="top", linespacing=1.4)

    # ---------------------------------------------------------------- totals + outcomes
    rx = 0.6
    ax.text(rx, 0.555, "Totals for the 25 Session A turns (all logged calls)", fontsize=12, fontweight="bold",
            color=INK)
    headers = ["", "full", "trim", "summary"]
    rows = [
        ("Total tokens", [f"{int(summary[s]['logged_total_tokens_input_plus_output']):,}" for s in SERIES]),
        ("of which summary calls", ["0", "0", f"{int(summary['summary']['summary_generation_input_tokens_logged']) + int(summary['summary']['summary_generation_output_tokens_logged']):,}"]),
        ("Fallback turns (not sent)", [summary[s]["prompt_preparation_failed_chat_calls_not_sent"] for s in SERIES]),
        ("Median chat latency (ms)", [f"{float(summary[s]['median_latency_ms_chat_calls']):,.0f}" for s in SERIES]),
        ("Official probes (P08, P10)",
         [f"{sum(p['correctness'].startswith('correct') for p in probes if p['strategy'] == s)}/"
          f"{sum(p['strategy'] == s for p in probes)}" for s in SERIES]),
    ]
    col_x = [rx, rx + 0.2, rx + 0.27, rx + 0.34]
    for j, head in enumerate(headers):
        ax.text(col_x[j], 0.515, head, fontsize=9.2, fontweight="bold", color=INK, ha="left" if j == 0 else "center")
    for i, (label, values) in enumerate(rows):
        yy = 0.48 - i * 0.034
        ax.plot([rx, 0.98], [yy - 0.013, yy - 0.013], color=RULE, linewidth=0.8)
        ax.text(col_x[0], yy, label, fontsize=9, color=INK2)
        for j, value in enumerate(values, start=1):
            ax.text(col_x[j], yy, value, fontsize=9.4, color=INK, ha="center")

    first_fact = min(f["source_turn"] for f in facts)
    tiles = [
        ("NLU accuracy", f"{nlu['correct']}/{nlu['total_utterances']}", "labelled utterances,\nrule-based (no LLM)"),
        ("Trim forgets first at", f"turn {first_fact + TRIM_WINDOW_K_TURNS + 1}",
         f"turn-{first_fact} name leaves the\nlast-{TRIM_WINDOW_K_TURNS} window"),
        ("Database write", "only after 'yes'", "book_slot needs the exact\npending summary confirmed"),
    ]
    tw = 0.122
    for i, (label, value, caption) in enumerate(tiles):
        x = rx + i * (tw + 0.007)
        ax.add_patch(FancyBboxPatch((x, 0.04), tw, 0.21, boxstyle="round,pad=0,rounding_size=0.006",
                                    linewidth=1, edgecolor=RULE, facecolor="white"))
        ax.text(x + 0.01, 0.235, label, fontsize=8.6, color=INK2, va="top")
        ax.text(x + 0.01, 0.195, value, fontsize=13.5, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.01, 0.055, caption, fontsize=8, color=INK2, va="bottom", linespacing=1.35)

    # Table rules are drawn in data coordinates: pin the canvas so nothing rescales.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out = ROOT / "images" / "task2-summary.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
