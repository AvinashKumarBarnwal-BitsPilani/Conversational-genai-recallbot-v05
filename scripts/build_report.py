"""Build report.pdf from docs/report/report_template.html.

    python scripts/build_report.py --bits-id <BITS_ID>

* Hit-rate rows are computed from results/hit_rate.csv.
* Appendix B prompts are read from the files and constants the code uses.
* The B.3 layout example is produced by the real build_turn_items() and
  ContextAssembler.build_messages() with stored items from memory/ and index/.
* The PDF is printed with headless Microsoft Edge (Chromium) and written to report.pdf.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context.context_assembler import FALLBACK_SENTENCE, TOOLS_TEXT, ContextAssembler, build_turn_items  # noqa: E402
from src.dialogue.qwen_nlg import _SUMMARY_INSTRUCTION  # noqa: E402
from src.memory.episodic_memory import IMPORTANCE_BY_TYPE, SMALL_TALK_IMPORTANCE, WRITE_THRESHOLD  # noqa: E402
from src.memory.fact_extractor import _LLM_SYSTEM_PROMPT  # noqa: E402
from src.memory.reflection import REFLECTION_SYSTEM_PROMPT  # noqa: E402

REPORT_DIR = ROOT / "docs" / "report"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
VIDEO_LINK = (ROOT / "video_link.txt").read_text(encoding="utf-8").strip()


def judge_rubric() -> str:
    spec = importlib.util.spec_from_file_location("llm_judge", ROOT / "scripts" / "llm_judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RUBRIC


def hit_rows() -> str:
    """Every cell computed from results/hit_rate.csv and the question flags in data/kb_questions.json."""
    questions = json.loads((ROOT / "data" / "kb_questions.json").read_text(encoding="utf-8"))
    groups = {
        "all": {q["question_id"] for q in questions},
        "code": {q["question_id"] for q in questions if q["contains_exact_code_or_id"]},
        "para": {q["question_id"] for q in questions if q["paraphrased_no_answer_keyword"]},
    }
    hits: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for row in csv.DictReader((ROOT / "results" / "hit_rate.csv").open(encoding="utf-8")):
        hits[(row["method"], row["chunk_size"], row["k"])][row["question_id"]] = int(row["hit"])
    rows = []
    for method, label in (("dense", "Dense"), ("bm25", "BM25"), ("hybrid", "Hybrid (RRF)")):
        cells = []
        for group in ("all", "code", "para"):
            for size in ("128", "512"):
                for k in ("1", "3", "5"):
                    values = [hits[(method, size, k)][q] for q in groups[group]]
                    cells.append(f'<td class="n">{sum(values) / len(values):.2f}</td>')
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    assert (len(groups["all"]), len(groups["code"]), len(groups["para"])) == (32, 10, 5)
    return "\n".join(rows)


def rag_layout() -> str:
    facts = {r["memory_id"]: r for r in json.loads((ROOT / "memory" / "semantic_facts.json").read_text(encoding="utf-8"))["records"]}
    episodes = {e["episode_id"]: e for e in json.loads((ROOT / "memory" / "episodes.json").read_text(encoding="utf-8"))["episodes"]}
    chunks = {c["chunk_id"]: c for c in json.loads((ROOT / "index" / "dense_128" / "chunks.json").read_text(encoding="utf-8"))}
    fact = SimpleNamespace(**facts["SM-0008"])
    episode = SimpleNamespace(episode=SimpleNamespace(**episodes["EP-0013"]))
    hit = chunks["KB-002-C128-0002"]
    items, _ = build_turn_items(
        intent="knowledge_base_query",
        system_prompt="<skills/system_prompt_v2.txt, B.1>",
        system_prompt_version="system_prompt_v2",
        workflow_step=None,
        user_message="For a 2026 delivery, how many days do I have to return electronic accessories?",
        state={"turn": 31, "current_intent": "knowledge_base_query", "active_task": None},
        draft="<deterministic draft written by the dialogue manager>",
        facts=[(fact, 0.0)], episodes=[episode], kb_hits=[hit],
        history=[{"role": "user", "content": "<previous user turn>"},
                 {"role": "assistant", "content": "<previous reply>"}],
        truncate=lambda text, n: " ".join(text.split()[:45]) + " ...",
    )
    messages = ContextAssembler.build_messages(items)
    out = []
    for message in messages:
        content = message["content"].replace(TOOLS_TEXT, "<TOOLS_TEXT, B.2>")
        out.append(f"--- {message['role']} ---\n{content}")
    note = ("Format produced by build_turn_items() + build_messages() with real stored items "
            "(SM-0008, EP-0013, KB-002-C128-0002); chunk text shortened here, up to 180 tokens at run time. "
            "A WORKFLOW-STEP line is added to the system message while a task is active; the assembler "
            "drops optional items (memories, chunks, older history) whole when the 1,856-token budget would be exceeded.\n\n")
    return note + "\n\n".join(out)


def importance_text() -> str:
    lines = [f"{kind:<18} {score}" for kind, score in sorted(IMPORTANCE_BY_TYPE.items(), key=lambda kv: -kv[1])]
    lines.append(f"{'small talk / other':<18} {SMALL_TALK_IMPORTANCE}")
    lines.append(f"\nAn episode is written only if importance >= {WRITE_THRESHOLD}. Each importance decision is "
                 "logged as call_type=importance (model=rule-based).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits-id", default="&lt;BITS_ID&gt;")
    parser.add_argument("--out", default=str(ROOT / "report.pdf"))
    args = parser.parse_args()

    template = (REPORT_DIR / "report_template.html").read_text(encoding="utf-8")
    svg = (REPORT_DIR / "architecture.svg").read_text(encoding="utf-8")
    esc = html.escape
    values = {
        "BITS_ID": args.bits_id,
        "VIDEO_LINK": f'<a href="{esc(VIDEO_LINK)}">{esc(VIDEO_LINK)}</a>',
        "ARCH_SVG": svg,
        "HIT_ROWS": hit_rows(),
        "FALLBACK": esc(FALLBACK_SENTENCE),
        "SYSTEM_V2": esc((ROOT / "skills" / "system_prompt_v2.txt").read_text(encoding="utf-8").strip()),
        "SYSTEM_V1": esc((ROOT / "skills" / "system_prompt_v1.txt").read_text(encoding="utf-8").strip()),
        "TOOLS": esc(TOOLS_TEXT),
        "RAG_LAYOUT": esc(rag_layout()),
        "EXTRACTION": esc(_LLM_SYSTEM_PROMPT),
        "IMPORTANCE": esc(importance_text()),
        "REFLECTION": esc(REFLECTION_SYSTEM_PROMPT),
        "SUMMARY": esc(_SUMMARY_INSTRUCTION),
        "JUDGE": esc(judge_rubric()),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise SystemExit("Unfilled placeholder in the report template")
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(template, encoding="utf-8")

    out = Path(args.out).resolve()
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", f"--print-to-pdf={out}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=120)
    from pypdf import PdfReader
    reader = PdfReader(str(out))
    appendix_page = next((i + 1 for i, page in enumerate(reader.pages) if "Appendix A" in (page.extract_text() or "")), None)
    print(f"{out}: {len(reader.pages)} pages; main body = pages 1-{(appendix_page or 1) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
