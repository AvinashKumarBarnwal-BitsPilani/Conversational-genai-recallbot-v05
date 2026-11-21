"""Full Task 1 rubric coverage checks for the scripted V05 test assets."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _all_tags(turns):
    return [tag for turn in turns for tag in turn.get("tags", [])]


def test_kb_has_required_volume_codes_and_two_version_pairs():
    kb_dir = ROOT / "data" / "kb"
    documents = sorted(kb_dir.glob("*.md"))
    words = sum(
        len(re.findall(r"\b[\w'-]+\b", path.read_text(encoding="utf-8")))
        for path in documents
    )
    manifest = _read_json("data/kb/manifest.json")
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    codes = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b", all_text))

    assert len(documents) >= 15
    assert 4000 <= words <= 8000
    assert len(codes) >= 10

    # Confirm two policy families have both a historical and current version.
    file_names = {item["file"] for item in manifest}
    assert {"returns_policy_2025_v1.md", "returns_policy_2026_v2.md"}.issubset(file_names)
    assert {"shipping_delivery_policy_2025_v1.md", "shipping_delivery_policy.md"}.issubset(file_names)


def test_kb_questions_have_codes_paraphrases_and_new_version_cases():
    questions = _read_json("data/kb_questions.json")
    assert len(questions) >= 25
    assert sum(bool(item.get("contains_exact_code_or_id")) for item in questions) >= 5
    assert sum(bool(item.get("paraphrased_no_answer_keyword")) for item in questions) >= 5
    assert sum(bool(item.get("version_sensitive")) for item in questions) >= 2
    assert sum(item.get("category") == "new_version_only" for item in questions) >= 2


def test_conversation_contains_all_required_task1_elements():
    conversation = _read_json("data/test_conversation.json")
    turns = conversation["turns"]
    facts = conversation["planted_user_facts"]
    tags = _all_tags(turns)

    assert len(turns) >= 40
    assert [turn["turn"] for turn in turns[:25]] == list(range(1, 26))
    assert all(turn["session"] == "A" for turn in turns[:25])
    assert all(turn["session"] == "B" for turn in turns[25:])
    assert len(facts) >= 8
    assert sum(tag.startswith("fact_update:") for tag in tags) >= 2
    assert any(tag.startswith("forget:") for tag in tags)
    assert len(conversation["important_episodes_expected"]) >= 3
    assert len(conversation["low_importance_turns"]) >= 5
    assert "slot_task:confirmed_write" in tags
    assert sum(tag == "state_followup" for tag in tags) >= 2
    assert "requires_clarification" in tags
    assert sum("kb_question" in turn.get("tags", []) for turn in turns) >= 5
    assert sum("kb_question" in turn.get("tags", []) and turn["session"] == "A" for turn in turns) >= 1
    assert sum("kb_question" in turn.get("tags", []) and turn["session"] == "B" for turn in turns) >= 1


def test_probes_cover_seven_categories_and_two_small_context_twists():
    probes = _read_json("data/probes.json")
    required = {"R", "U", "F", "E", "K", "N", "S"}
    categories = {item["category"] for item in probes}
    twist_probes = [item for item in probes if item.get("twist") == "E"]

    assert len(probes) >= 14
    assert required.issubset(categories)
    assert len(twist_probes) >= 2
