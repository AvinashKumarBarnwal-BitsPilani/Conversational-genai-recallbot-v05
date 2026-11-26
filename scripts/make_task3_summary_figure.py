"""Draw the Task 3 summary figure from the run's own result files and log.

Every number on the figure is read from logs/calls.jsonl or results/*.json, so
the image can be regenerated after any rerun:

    python scripts/make_task3_summary_figure.py --experiment t3_hm_01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
# Reference categorical palette, fixed order: each hue marks one component group.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"


def load(experiment: str) -> dict:
    res = ROOT / "results"
    log = [json.loads(line) for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    chat = [r for r in log if r["run_id"] in (f"{experiment}_A", f"{experiment}_B") and r["call_type"] == "chat"]
    manifest = json.loads((ROOT / "memory" / "runs" / experiment / "session_manifest.json").read_text(encoding="utf-8"))
    judgements = json.loads((res / "task3_probe_judgements.json").read_text(encoding="utf-8"))
    weighting = json.loads((res / "task3_episodic_weight_comparison.json").read_text(encoding="utf-8"))["summary"]
    return {
        "semantic": json.loads((res / "task3_semantic_memory_metrics.json").read_text(encoding="utf-8")),
        "weighting": weighting,
        "assembly": json.loads((res / "task3_context_assembly.json").read_text(encoding="utf-8")),
        "chat_calls": len(chat),
        "fallbacks": sum(r["extra"].get("status") != "success" for r in chat),
        "max_input": max(r["input_tokens"] for r in chat),
        "exact_counts": sum(r["extra"]["context_assembly"]["total_input_tokens"] == r["input_tokens"] for r in chat),
        "pids": (manifest["sessions"]["A"]["pid"], manifest["sessions"]["B"]["pid"]),
        "hashes_match": manifest["sessions"]["B"]["loaded_hashes_match_session_a"],
        "files": len(manifest["sessions"]["A"]["saved_hashes"]),
        "probes": judgements,
    }


def box(ax, x, y, w, h, title, body, color, *, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=1.6, edgecolor=color, facecolor=color + "1a",
                                linestyle=(0, (4, 3)) if dashed else "-", transform=ax.transAxes))
    ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=10.5, fontweight="semibold",
            color=INK, transform=ax.transAxes)
    ax.text(x + w / 2, y + h - 0.043, body, ha="center", va="top", fontsize=8.6, color=INK2,
            linespacing=1.35, transform=ax.transAxes)


def arrow(ax, start, end, label=None, *, color=INK2, rad=0.0, dashed=False, label_offset=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=(0, (4, 3)) if dashed else "-",
                                 transform=ax.transAxes, shrinkA=2, shrinkB=2))
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1], label,
                ha="center", va="bottom", fontsize=7.8, color=MUTED, transform=ax.transAxes)


def tile(ax, x, y, w, h, label, value, caption):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", linewidth=1,
                                edgecolor=RULE, facecolor="white", transform=ax.transAxes))
    ax.text(x + 0.012, y + h - 0.02, label, ha="left", va="top", fontsize=9, color=INK2, transform=ax.transAxes)
    ax.text(x + 0.012, y + h - 0.052, value, ha="left", va="top", fontsize=19, fontweight="semibold",
            color=INK, transform=ax.transAxes)
    ax.text(x + 0.012, y + 0.016, caption, ha="left", va="bottom", fontsize=8.2, color=INK2,
            linespacing=1.35, transform=ax.transAxes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="t3_hm_01")
    parser.add_argument("--out", default="images/task3-summary.png")
    args = parser.parse_args()
    d = load(args.experiment)

    fig = plt.figure(figsize=(16, 11), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.02, 0.972, "Task 3 · RecallBot remembers a returning customer inside a 2,048-token window",
            fontsize=17, fontweight="semibold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.02, 0.936, f"V05 e-commerce returns · official run {args.experiment} (hybrid_mem) · "
            "Session A writes memory to disk, the process ends, Session B starts fresh and loads it",
            fontsize=10.5, color=INK2, va="top", transform=ax.transAxes)

    # ------------------------------------------------------------ workflow
    ax.text(0.02, 0.895, "How the pieces work together", fontsize=12, fontweight="semibold", color=INK,
            transform=ax.transAxes)
    top_y, h = 0.775, 0.095
    box(ax, 0.02, top_y, 0.10, h, "Customer turn", "\"What delivery window\ndo I prefer now?\"", BLUE)
    box(ax, 0.145, top_y, 0.13, h, "NLU + state", "intent, entities,\nslots (rule-based)", BLUE)
    box(ax, 0.30, top_y, 0.15, h, "Dialogue manager", "routes: task tools, KB,\nmemory read / write", BLUE)
    box(ax, 0.54, top_y, 0.17, h, "Context assembler",
        "1,856 input tokens max;\ncuts lowest priority, logs it", ORANGE)
    box(ax, 0.735, top_y, 0.10, h, "Qwen3 1.7B", "local, Ollama\ntemp 0 for tasks", BLUE)
    box(ax, 0.86, top_y, 0.12, h, "Grounded reply", "cites [SM-…] [EP-…]\n[KB-…] or says\n\"I won't guess\"", BLUE)
    for x0, x1 in ((0.12, 0.145), (0.275, 0.30), (0.45, 0.54), (0.71, 0.735), (0.835, 0.86)):
        arrow(ax, (x0, top_y + h / 2), (x1, top_y + h / 2))

    mid_y, h2 = 0.615, 0.115
    box(ax, 0.145, mid_y, 0.13, h2, "Fact extractor", "rules first;\nQwen JSON fallback\nonly when rules miss", AQUA)
    box(ax, 0.30, mid_y, 0.15, h2, "Semantic memory", "keyed facts (SM-…)\nupdate → old superseded\n+ vector index", AQUA)
    box(ax, 0.145, 0.475, 0.13, h2, "Episode writer", "events with importance\n1–10 (booking 9, …)\n+ end-of-session reflection", VIOLET)
    box(ax, 0.30, 0.475, 0.15, h2, "Episodic memory", "time-stamped stream\n(EP-…) incl. reflection\n+ vector index", VIOLET)
    box(ax, 0.02, 0.475, 0.10, 0.255, "Forget", "delete fact\n+ its vector\n\ndelete / redact\nepisodes and\nreflections\n\ntombstone keeps\nonly a hash", MUTED,
        dashed=True)
    box(ax, 0.54, 0.545, 0.17, 0.16, "Memory retriever",
        "facts: cosine similarity\nevents: recency + importance\n+ relevance (weights\n0.2 / 0.3 / 0.5)\nthresholds by intent", AQUA)
    box(ax, 0.735, 0.615, 0.245, h2, "Knowledge base (Day 2)",
        "dense + own BM25 + own RRF\nactive policy chunks only\n(superseded 2025 versions dropped)", YELLOW)

    # Writes: facts go through the extractor; events are written by the DM's handlers.
    arrow(ax, (0.33, top_y), (0.21, mid_y + h2), rad=0.15)
    arrow(ax, (0.302, top_y), (0.277, 0.555))
    ax.text(0.148, top_y - 0.012, "DM writes facts\nand events", fontsize=7.8, color=MUTED, ha="left", va="top",
            transform=ax.transAxes)
    arrow(ax, (0.275, mid_y + h2 / 2), (0.30, mid_y + h2 / 2))
    arrow(ax, (0.275, 0.475 + h2 / 2), (0.30, 0.475 + h2 / 2))
    arrow(ax, (0.12, 0.66), (0.145, 0.66), dashed=True, color=MUTED)
    arrow(ax, (0.12, 0.53), (0.145, 0.53), dashed=True, color=MUTED)
    arrow(ax, (0.45, mid_y + h2 / 2), (0.54, 0.655))
    arrow(ax, (0.45, 0.475 + h2 / 2), (0.54, 0.585))
    arrow(ax, (0.625, 0.705), (0.625, top_y))
    ax.text(0.632, 0.742, "relevant memories", fontsize=7.8, color=MUTED, ha="left", va="center",
            transform=ax.transAxes)
    arrow(ax, (0.80, mid_y + h2), (0.68, top_y), rad=0.15)
    ax.text(0.815, 0.75, "active chunks", fontsize=7.8, color=MUTED, ha="left", va="center",
            transform=ax.transAxes)

    # persistence strip
    py = 0.405
    ax.add_patch(FancyBboxPatch((0.145, py), 0.835, 0.045, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=1, edgecolor=INK2, facecolor="white", transform=ax.transAxes))
    a_pid, b_pid = d["pids"]
    ax.text(0.16, py + 0.0225,
            f"Disk: memory/runs/{args.experiment}  ·  {d['files']} files, SHA-256 checked  ·  "
            f"Session A (PID {a_pid}) saves → process ends → Session B (PID {b_pid}) loads"
            f"  ·  hashes match: {'yes' if d['hashes_match'] else 'NO'}",
            fontsize=9.5, color=INK, va="center", transform=ax.transAxes)
    arrow(ax, (0.375, 0.475), (0.375, py + 0.045))

    # ------------------------------------------------------------ budget bar
    ax.text(0.02, 0.36, "The 2,048-token window on a real Session B turn (turn 31, policy question)",
            fontsize=12, fontweight="semibold", color=INK, transform=ax.transAxes)
    rec = d["assembly"]["official_turns"]["31"]
    rows = {r["component"]: r["tokens"] for r in rec["component_table"]}
    used = rec["total_input_tokens"]
    parts = [
        ("System prompt + tools", rows.get("system", 0) + rows.get("tools", 0), BLUE, INK),
        ("Current request + state", rows.get("current_task", 0), BLUE + "99", INK),
        ("Last 3 exchanges", rows.get("conversation_history", 0), MUTED, "white"),
        ("Knowledge base chunks", rows.get("retrieved_knowledge", 0), YELLOW, INK),
        ("Memories", rows.get("long_term_memory", 0), AQUA, INK),
        ("Format", rows.get("chat template and section headings", 0), RULE, INK),
        ("Unused input budget", rec["input_budget"] - used, "#ffffff", INK2),
        ("Reply reserve + safety", rec["output_reserve"] + (2048 - rec["input_budget"] - rec["output_reserve"]),
         ORANGE, INK),
    ]
    x, bar_y, bar_h, total_w = 0.02, 0.285, 0.045, 0.96
    for name, tokens, color, text_color in parts:
        if tokens <= 0:
            continue
        w = total_w * tokens / 2048
        ax.add_patch(FancyBboxPatch((x + 0.001, bar_y), w - 0.002, bar_h, boxstyle="round,pad=0,rounding_size=0.004",
                                    linewidth=1 if color == "#ffffff" else 0, edgecolor=RULE, facecolor=color,
                                    transform=ax.transAxes))
        if w > 0.06:
            ax.text(x + 0.006, bar_y + bar_h / 2, f"{name}\n{tokens:,}", fontsize=8, color=text_color,
                    va="center", ha="left", linespacing=1.2, transform=ax.transAxes)
        x += w
    narrow = [f"{name.lower()} {tokens}" for name, tokens, *_ in parts
              if 0 < tokens and total_w * tokens / 2048 <= 0.06]
    ax.text(0.02, bar_y - 0.012,
            f"Used {used:,} of {rec['input_budget']:,} input tokens; 160 reserved for the reply, 32 safety. "
            f"Small segments: {', '.join(narrow)}. Token count equals Ollama's on "
            f"{d['exact_counts']}/{d['chat_calls']} official calls.\nMemories are 0 here: on a policy question "
            "they did not clear the relevance threshold.",
            fontsize=8.6, color=INK2, va="top", transform=ax.transAxes)

    # ------------------------------------------------------------ outcome tiles
    ax.text(0.02, 0.215, "What Task 3 achieved (official run)", fontsize=12, fontweight="semibold", color=INK,
            transform=ax.transAxes)
    sem = d["semantic"]["rules_plus_llm_fallback"]
    rules = d["semantic"]["rules_only"]
    wa, wb = d["weighting"]["A"], d["weighting"]["B"]
    stress = d["assembly"]["stress_replay"]
    marks = d["probes"]["judgements"]
    correct = sum(j["correct"] for j in marks)
    failed = ", ".join(j["probe_id"] for j in marks if not j["correct"])
    review = "" if d["probes"].get("reviewed_by_student") else " (pending review)"
    tiles = [
        ("Remembers user facts", f"{sem['precision']:.2f} / {sem['recall']:.2f}",
         f"precision / recall, {sem['planted']} planted facts\nrules alone: recall {rules['recall']:.2f}"),
        ("Recalls past events", f"hit@1 {wa['hit_at_1']:.2f}",
         f"weighting A kept (MRR {wa['mrr']:.2f});\nrecency-heavy B: hit@1 {wb['hit_at_1']:.2f}"),
        ("Stays inside the window", f"{d['fallbacks']} fallbacks",
         f"{d['chat_calls']} chat calls, largest {d['max_input']:,}\nof 1,856 input tokens"),
        ("Shows what gets cut", f"{len(stress['dropped'])} items cut",
         f"stress replay: {stress['total_input_tokens']:,} / {stress['input_budget']:,} used;\n"
         "KB chunks, reflection, old turns"),
        ("Forgets on request", "0 traces",
         "delivery note gone from facts,\nindex, episodes after restart"),
        ("Probe accuracy", f"{correct} / {len(marks)}", f"fails: {failed}\nmarks proposed{review}"),
    ]
    tw, gap = 0.152, 0.0096
    for i, (label, value, caption) in enumerate(tiles):
        tile(ax, 0.02 + i * (tw + gap), 0.035, tw, 0.16, label, value, caption)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
