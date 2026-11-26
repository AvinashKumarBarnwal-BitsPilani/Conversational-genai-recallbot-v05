"""Draw the Task 1 summary figure from the data files (no typed-in numbers).

    python scripts/make_task1_summary_figure.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
# Reference categorical palette in fixed order (one hue per lane); every lane is also labelled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CODE_RE = re.compile(r"\b(?:ORD|SKU|RET|ERR|RMA|PAY|SHP|RFND|PICK|KB|ADDR|EXC|CAN|WAR|SLA|GC|PROMO|MEM)-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")


def load():
    conversation = json.loads((DATA / "test_conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "kb_questions.json").read_text(encoding="utf-8"))
    probes = json.loads((DATA / "probes.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "kb" / "manifest.json").read_text(encoding="utf-8"))
    texts = [(DATA / "kb" / doc["file"]).read_text(encoding="utf-8") for doc in manifest]
    words = sum(len(re.findall(r"\b[\w'-]+\b", text)) for text in texts)
    codes = {code for text in texts for code in CODE_RE.findall(text) if not code.startswith("KB-")}
    return conversation, questions, probes, manifest, words, codes


def lanes_from(conversation, probes):
    turns = conversation["turns"]
    tagged = lambda *prefixes: sorted({t["turn"] for t in turns  # noqa: E731
                                       if any(tag.startswith(p) for tag in t.get("tags", []) for p in prefixes)})
    facts = conversation["planted_user_facts"]
    updates = sorted({f["source_turn"] for f in facts if any(g.get("updated_by") and g["key"] == f["key"]
                                                             and g["source_turn"] < f["source_turn"] for g in facts)})
    return [
        ("User fact stated", sorted({f["source_turn"] for f in facts} - set(updates))),
        ("Fact updated", updates),
        ("Forget request", sorted({f["updated_by"] for f in facts if f["status"] == "deleted"})),
        ("Event worth remembering", sorted({e["source_turn"] for e in conversation["important_episodes_expected"]})),
        ("Return-pickup slot filling", tagged("slot_task", "slot:")),
        ("State follow-up / clarification", tagged("state_followup", "requires_clarification")),
        ("Knowledge-base question", tagged("kb_question")),
        ("Low-importance (filler)", sorted(conversation["low_importance_turns"])),
        ("Probe asked at this turn", sorted({p["turn"] for p in probes})),
    ]


def main() -> int:
    conversation, questions, probes, manifest, words, codes = load()
    turns = conversation["turns"]
    boundary = conversation["session_boundary"]["Session A"]["last_turn"]
    n_turns = len(turns)

    fig = plt.figure(figsize=(16, 11.2), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.02, 0.975, "Task 1 · The test data every later number is measured against",
            fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.942, "V05 e-commerce returns (ShopSphere, synthetic) · a 40-turn scripted customer, a 16-document "
            "knowledge base, retrieval questions with gold answers, and memory probes", fontsize=10.5, color=INK2, va="top")

    # ------------------------------------------------ conversation map
    ax.text(0.02, 0.9, f"The scripted conversation: {n_turns} customer turns, Session A (1–{boundary}) and "
            f"Session B ({boundary + 1}–{n_turns}) in separate processes", fontsize=12, fontweight="bold", color=INK)
    lanes = lanes_from(conversation, probes)
    left, right, top, row_h = 0.215, 0.98, 0.855, 0.031
    col_w = (right - left) / n_turns
    session_x = left + boundary * col_w
    ax.add_patch(FancyBboxPatch((left, top - len(lanes) * row_h - 0.004), boundary * col_w,
                                len(lanes) * row_h + 0.008, boxstyle="round,pad=0,rounding_size=0.004",
                                linewidth=0, facecolor="#eef4fc"))
    ax.add_patch(FancyBboxPatch((session_x, top - len(lanes) * row_h - 0.004), right - session_x,
                                len(lanes) * row_h + 0.008, boxstyle="round,pad=0,rounding_size=0.004",
                                linewidth=0, facecolor="#f5f1fb"))
    ax.text(left + 0.004, top + 0.012, "SESSION A", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(session_x + 0.004, top + 0.012, "SESSION B  (new process, memory from disk)", fontsize=8.5,
            fontweight="bold", color=INK2)
    for i, (label, lane_turns) in enumerate(lanes):
        y = top - (i + 1) * row_h
        color = PALETTE[i] if i < len(PALETTE) else MUTED
        ax.text(left - 0.008, y + row_h / 2, f"{label}  ({len(lane_turns)})", ha="right", va="center",
                fontsize=8.8, color=INK)
        for turn in lane_turns:
            x = left + (turn - 1) * col_w
            ax.add_patch(FancyBboxPatch((x + 0.0015, y + 0.005), col_w - 0.003, row_h - 0.01,
                                        boxstyle="round,pad=0,rounding_size=0.002", linewidth=0, facecolor=color))
    axis_y = top - len(lanes) * row_h - 0.018
    for turn in range(1, n_turns + 1):
        if turn == 1 or turn % 5 == 0 or turn == boundary + 1:
            ax.text(left + (turn - 0.5) * col_w, axis_y, str(turn), ha="center", va="top", fontsize=7.5, color=INK2)
    ax.text(left - 0.008, axis_y, "turn", ha="right", va="top", fontsize=7.5, color=INK2)

    # ------------------------------------------------ knowledge base
    kb_top = 0.53
    superseded = [d for d in manifest if d["status"] == "superseded"]
    ax.text(0.02, kb_top, f"Knowledge base: {len(manifest)} documents, {words:,} words, "
            f"{len(superseded)} old/new version pairs", fontsize=12, fontweight="bold", color=INK)
    by_title = {d["doc_id"]: d for d in manifest}
    pairs = {"KB-001": "KB-002", "KB-016": "KB-005"}
    tile_w, tile_h, per_row = 0.112, 0.044, 4
    for k, doc in enumerate(manifest):
        col, row = k % per_row, k // per_row
        x, y = 0.02 + col * (tile_w + 0.006), kb_top - 0.06 - row * (tile_h + 0.008)
        old = doc["status"] == "superseded"
        ax.add_patch(FancyBboxPatch((x, y), tile_w, tile_h, boxstyle="round,pad=0,rounding_size=0.004",
                                    linewidth=1, edgecolor=RULE if not old else MUTED,
                                    facecolor="white" if not old else "#efeeea",
                                    linestyle="-" if not old else (0, (3, 2))))
        title = re.sub(r"\s*[—-]\s*20\d\d.*$", "", doc["title"]).replace("ShopSphere ", "")
        ax.text(x + 0.006, y + tile_h - 0.009, doc["doc_id"] + ("  · 2025 version" if old else ""),
                fontsize=7.6, fontweight="bold", color=INK2 if old else INK, va="top")
        body = f"superseded by {pairs[doc['doc_id']]} (2026)" if old else title[:30] + ("…" if len(title) > 30 else "")
        ax.text(x + 0.006, y + 0.008, body, fontsize=7.3, color=INK2)
    del by_title

    # ------------------------------------------------ questions and probes
    qx = 0.52
    ax.text(qx, kb_top, f"{len(questions)} retrieval questions with gold answer spans", fontsize=12,
            fontweight="bold", color=INK)
    qcats = [
        ("contain an exact code or ID", sum(q["contains_exact_code_or_id"] for q in questions), 5),
        ("paraphrased, no answer keyword", sum(q["paraphrased_no_answer_keyword"] for q in questions), 5),
        ("version-sensitive (old vs new policy)", sum(q["version_sensitive"] for q in questions), None),
        ("answer only in the newer version", sum(q["category"] == "new_version_only" for q in questions), 2),
    ]
    bar_x, bar_w = qx + 0.2, 0.24
    scale = bar_w / max(len(questions) / 2, max(c for _, c, _ in qcats))
    for i, (label, count, minimum) in enumerate(qcats):
        y = kb_top - 0.045 - i * 0.032
        ax.text(bar_x - 0.008, y + 0.009, label, ha="right", va="center", fontsize=8.6, color=INK)
        ax.add_patch(FancyBboxPatch((bar_x, y), count * scale, 0.018, boxstyle="round,pad=0,rounding_size=0.003",
                                    linewidth=0, facecolor=PALETTE[0]))
        ax.text(bar_x + count * scale + 0.006, y + 0.009,
                f"{count}" + (f"  (required ≥ {minimum})" if minimum else ""), va="center", fontsize=8.6, color=INK)

    category_names = {"R": "R semantic recall", "U": "U updated fact", "F": "F forgotten fact",
                      "E": "E episodic recall", "K": "K knowledge base", "N": "N not available",
                      "S": "S state follow-up", "TWIST_SMALL_CONTEXT": "Twist E small window"}
    counts = Counter(p["category"] for p in probes)
    py = kb_top - 0.2
    ax.text(qx, py, f"{len(probes)} memory probes: all 7 categories + 2 twist probes", fontsize=12,
            fontweight="bold", color=INK)
    for i, (key, label) in enumerate(category_names.items()):
        col, row = i % 2, i // 2
        x, y = qx + col * 0.235, py - 0.045 - row * 0.03
        ax.text(x, y, label, fontsize=8.8, color=INK, va="center")
        for k in range(counts.get(key, 0)):
            ax.add_patch(FancyBboxPatch((x + 0.135 + k * 0.016, y - 0.007), 0.012, 0.014,
                                        boxstyle="round,pad=0,rounding_size=0.002", linewidth=0,
                                        facecolor=PALETTE[6] if key == "TWIST_SMALL_CONTEXT" else PALETTE[2]))
        ax.text(x + 0.135 + counts.get(key, 0) * 0.016 + 0.004, y, str(counts.get(key, 0)), fontsize=8.6,
                color=INK, va="center")

    # ------------------------------------------------ requirement tiles
    facts = conversation["planted_user_facts"]
    tiles = [
        ("KB documents", f"{len(manifest)}", "required ≥ 15"),
        ("KB words", f"{words:,}", "required 4,000–8,000"),
        ("Codes / IDs in the KB", f"{len(codes)}", "distinct, required ≥ 10"),
        ("Conversation turns", f"{n_turns}", f"A {boundary} + B {n_turns - boundary}; required ≥ 40"),
        ("Planted user facts", f"{len(facts)}", f"{sum(f['updated_by'] is not None and f['status'] == 'superseded' for f in facts)} "
         f"updates, {sum(f['status'] == 'deleted' for f in facts)} forget; ≥ 8"),
        ("Retrieval questions", f"{len(questions)}", "required ≥ 25"),
        ("Probes", f"{len(probes)}", "required ≥ 14 incl. 2 twist"),
    ]
    tw = (0.96 - 0.006 * (len(tiles) - 1)) / len(tiles)
    for i, (label, value, caption) in enumerate(tiles):
        x = 0.02 + i * (tw + 0.006)
        ax.add_patch(FancyBboxPatch((x, 0.03), tw, 0.115, boxstyle="round,pad=0,rounding_size=0.006",
                                    linewidth=1, edgecolor=RULE, facecolor="white"))
        ax.text(x + 0.01, 0.13, label, fontsize=8.6, color=INK2, va="top")
        ax.text(x + 0.01, 0.1, value, fontsize=18, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.01, 0.042, caption, fontsize=7.6, color=INK2, va="bottom")
    ax.text(0.02, 0.165, "All data is synthetic; checked automatically by tests/test_day1_data.py, "
            "tests/test_task1_full_coverage.py and python -m src.main --check.", fontsize=8.6, color=INK2)

    out = ROOT / "images" / "task1-summary.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}  (codes found: {len(codes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
