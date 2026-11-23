"""Unit tests for the deterministic RecallBot NLU (Task 2a)."""
from __future__ import annotations

import json
from pathlib import Path

from src.dialogue.nlu import analyze_utterance

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "nlu_test_cases.json"


def load_cases():
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return payload["cases"]


def test_all_40_scripted_utterances_have_expected_primary_intent():
    """The labelled NLU set uses the exact 40 utterances from Task 1."""
    cases = load_cases()
    assert len(cases) == 40
    mismatches = []
    for case in cases:
        predicted = analyze_utterance(case["utterance"])["intent"]
        if predicted != case["expected_intent"]:
            mismatches.append(
                (case["turn"], case["expected_intent"], predicted, case["utterance"])
            )
    assert not mismatches, f"NLU intent mismatches: {mismatches}"


def test_expected_entities_are_extracted_for_labelled_examples():
    """Check representative order, code, date, time and profile entities."""
    for case in load_cases():
        expected = case.get("expected_entities", {})
        if not expected:
            continue
        actual = analyze_utterance(case["utterance"])["entities"]
        for key, value in expected.items():
            assert actual.get(key) == value, (
                f"Turn {case['turn']} entity {key!r}: "
                f"expected {value!r}, got {actual.get(key)!r}"
            )


def test_short_context_followup_is_classified_and_time_is_extracted():
    result = analyze_utterance("And for Saturday, please make it after 6 pm.")
    assert result["intent"] == "state_followup"
    assert result["entities"]["time_window"]["start"] == "18:00"


def test_ambiguous_other_item_requires_clarification():
    result = analyze_utterance("Can you book a pickup for the other item as well?")
    assert result["intent"] == "clarification_required"


def test_question_about_forgotten_fact_is_not_a_forget_command():
    result = analyze_utterance("What was the delivery note I asked you to forget?")
    assert result["intent"] == "forgotten_memory_query"


def test_empty_input_is_safe():
    result = analyze_utterance("   ")
    assert result["intent"] == "unknown"
    assert result["entities"] == {}


def test_exact_24_hour_slot_label_is_classified_and_extracted():
    """The assistant's displayed HH:MM–HH:MM option can be selected verbatim."""
    result = analyze_utterance("18:00–20:00")
    assert result["intent"] == "provide_pickup_window"
    assert result["entities"]["time_window"] == {
        "start": "18:00",
        "end": "20:00",
        "raw": "18:00–20:00",
    }
