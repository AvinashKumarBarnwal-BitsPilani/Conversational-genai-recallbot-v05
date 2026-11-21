import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def test_kb_document_and_word_count():
    docs = list((ROOT / "data" / "kb").glob("*.md"))
    words = sum(len(re.findall(r"\b[\w\'-]+\b", p.read_text(encoding="utf-8"))) for p in docs)
    assert len(docs) >= 15
    assert 4000 <= words <= 8000

def test_kb_gold_spans_exist_in_source():
    questions = read_json("data/kb_questions.json")
    manifest = read_json("data/kb/manifest.json")
    files = {item["doc_id"]: item["file"] for item in manifest}
    assert len(questions) >= 25
    for q in questions:
        source = (ROOT / "data" / "kb" / files[q["document_id"]]).read_text(encoding="utf-8")
        assert q["gold_answer_span"] in source, q["question_id"]

def test_conversation_structure_and_facts():
    data = read_json("data/test_conversation.json")
    turns = data["turns"]
    assert len(turns) >= 40
    assert [t["turn"] for t in turns[:25]] == list(range(1, 26))
    assert all(t["session"] == "A" for t in turns[:25])
    assert all(t["session"] == "B" for t in turns[25:])
    assert len(data["planted_user_facts"]) >= 8

def test_probe_coverage_including_small_context_twist():
    probes = read_json("data/probes.json")
    categories = {p["category"] for p in probes}
    assert {"R", "U", "F", "E", "K", "N", "S"}.issubset(categories)
    assert sum(p.get("twist") == "E" for p in probes) >= 2
