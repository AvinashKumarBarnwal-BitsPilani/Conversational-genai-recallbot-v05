from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.dialogue.nlu import analyze_utterance
from src.memory.fact_extractor import FactExtractor
from src.memory.llm_calls import LoggedLLM
from src.memory.semantic_memory import SemanticMemoryStore
from src.memory.vector_store import MemoryVectorIndex
from memory_fakes import FakeEmbedder, FakeOllamaClient, RecordingLogger

ROOT = Path(__file__).resolve().parents[1]
CONVERSATION = json.loads((ROOT / "data" / "test_conversation.json").read_text(encoding="utf-8"))
USER = "CUST-MAYA-1042"
EMBED = FakeEmbedder()


def _vec(text: str) -> np.ndarray:
    return EMBED.encode([text], purpose="test")[0]


def _upsert(store, key, value, turn, user=USER):
    return store.upsert(user=user, key=key, value=value, source_turn=turn, session="A",
                        date="2026-11-21T10:00:00+05:30", vector=_vec(f"{key}: {value}"))


# ----------------------------------------------------------------- store
def test_update_supersedes_previous_value_and_removes_its_vector():
    store = SemanticMemoryStore()
    action1, first, _ = _upsert(store, "preferred_delivery_window", "18:00-21:00", 7)
    action2, second, superseded = _upsert(store, "preferred_delivery_window", "19:00-21:00", 19)

    assert (action1, action2) == ("created", "updated")
    assert superseded.memory_id == first.memory_id
    assert first.status == "superseded" and first.superseded_by == second.memory_id
    assert second.supersedes == first.memory_id
    assert [f.value for f in store.active_facts(USER)] == ["19:00-21:00"]
    assert first.memory_id not in store.index and second.memory_id in store.index
    hits = store.search(USER, _vec("preferred delivery window 18:00-21:00"), top_k=5)
    assert [fact.value for fact, _ in hits] == ["19:00-21:00"]


def test_repeating_the_same_value_does_not_duplicate_the_record():
    store = SemanticMemoryStore()
    _upsert(store, "preferred_city", "Bengaluru", 5)
    action, record, superseded = _upsert(store, "preferred_city", "  bengaluru ", 30)
    assert action == "unchanged" and superseded is None and record.source_turn == 5
    assert len(store.all_records(USER)) == 1


def test_forget_key_deletes_active_and_superseded_records_and_vectors():
    store = SemanticMemoryStore()
    _upsert(store, "delivery_note", "side door", 6)
    _upsert(store, "delivery_note", "porch", 9)
    _upsert(store, "preferred_city", "Bengaluru", 5)
    removed = store.forget_key(USER, "delivery_note")
    assert {r.value for r in removed} == {"side door", "porch"}
    assert [r.key for r in store.all_records(USER)] == ["preferred_city"]
    assert store.index.ids == [store.get_active(USER, "preferred_city").memory_id]


def test_search_is_isolated_per_user():
    store = SemanticMemoryStore()
    _upsert(store, "preferred_city", "Bengaluru", 5, user="U1")
    _upsert(store, "preferred_city", "Chennai", 5, user="U2")
    hits = store.search("U2", _vec("which city"), top_k=5)
    assert [fact.value for fact, _ in hits] == ["Chennai"]


def test_store_and_index_round_trip_and_reject_stale_vectors(tmp_path):
    store = SemanticMemoryStore()
    _upsert(store, "preferred_name", "Maya", 2)
    _upsert(store, "notification_channel", "email", 4)
    _upsert(store, "notification_channel", "SMS", 20)
    store.index.save(tmp_path, "semantic")
    restored = SemanticMemoryStore.from_dict(store.to_dict(), MemoryVectorIndex.load(tmp_path, "semantic"))
    assert [f.to_dict() for f in restored.all_records()] == [f.to_dict() for f in store.all_records()]
    assert restored.index.ids == store.index.ids

    stale_index = MemoryVectorIndex.load(tmp_path, "semantic")
    stale_index.add("SM-0002", _vec("email"))  # the superseded record
    with pytest.raises(ValueError, match="non-active"):
        SemanticMemoryStore.from_dict(store.to_dict(), stale_index)


# ------------------------------------------------------------- extraction
def _session_a_rule_facts():
    extractor = FactExtractor(enable_llm_fallback=False, logger=None, token_counter=lambda t: len(t.split()))
    found = {}
    for turn in CONVERSATION["turns"]:
        if turn["session"] != "A":
            continue
        result = extractor.extract(turn["text"], analyze_utterance(turn["text"]), context={"turn": turn["turn"]})
        found[turn["turn"]] = (result, [(f.key, f.value) for f in result.facts])
    return found


def test_rule_pass_on_session_a_finds_eight_planted_facts_and_misses_turn_8():
    found = _session_a_rule_facts()
    extracted = {turn: facts for turn, (_, facts) in found.items() if facts}
    assert extracted == {
        2: [("preferred_name", "Maya")],
        4: [("notification_channel", "email")],
        5: [("preferred_city", "Bengaluru")],
        6: [("delivery_note", "side door")],
        7: [("preferred_delivery_window", "18:00-21:00")],
        9: [("packaging_preference", "recyclable")],
        19: [("preferred_delivery_window", "19:00-21:00")],
        20: [("notification_channel", "SMS")],
    }
    # Turn 8 is fact-like but no rule covers "a pickup from my home": this is
    # exactly the case the Qwen fallback exists for.
    assert found[8][0].attempted and found[8][0].status == "none"


def test_filler_and_task_turns_produce_no_records_or_attempts():
    found = _session_a_rule_facts()
    for turn in CONVERSATION["low_importance_turns"]:
        if turn <= 25:
            assert found[turn][1] == [] and found[turn][0].status == "skipped"
    # Pickup "after 6 pm" (turn 15) is a slot value, not a delivery preference.
    assert found[15][1] == []


def test_llm_fallback_runs_only_when_rules_miss_and_is_logged_as_extract():
    logger = RecordingLogger()
    client = FakeOllamaClient(json.dumps({"facts": [{"key": "return_method_preference", "value": "pickup from my home"}]}))
    extractor = FactExtractor(llm=LoggedLLM(client=client, logger=logger), logger=logger,
                              token_counter=lambda t: len(t.split()))
    text = "For returns, I prefer a pickup from my home instead of dropping the parcel at a counter."
    result = extractor.extract(text, analyze_utterance(text), context={"turn": 8, "session": "A",
                                                                       "user": USER, "config": "hybrid_mem"})
    assert result.fallback_used and result.status == "llm"
    assert [(f.key, f.value, f.method) for f in result.facts] == [
        ("return_method_preference", "pickup from my home", "llm")
    ]
    extract_rows = logger.of_type("extract")
    assert [row["model"] for row in extract_rows] == ["rule-based", "qwen3:1.7b"]
    assert extract_rows[1]["input_tokens"] == 40 and extract_rows[1]["output_tokens"] == 12
    assert client.requests[0]["format"] == "json"

    # A rule hit never pays for an LLM call.
    covered = "My name is Maya Rao. Please call me Maya."
    extractor.extract(covered, analyze_utterance(covered), context={"turn": 2})
    assert len(client.requests) == 1


@pytest.mark.parametrize("payload, reason", [
    ({"facts": [{"key": "preferred_city", "value": "Mumbai"}]}, "value_not_grounded_in_message"),
    ({"facts": [{"key": "Bad Key!", "value": "home"}]}, "invalid_key"),
    ({"facts": [{"key": "card_number", "value": "home"}]}, "secret"),
])
def test_llm_fallback_output_is_validated_before_storage(payload, reason):
    client = FakeOllamaClient(json.dumps(payload))
    extractor = FactExtractor(llm=LoggedLLM(client=client, logger=None), logger=None,
                              token_counter=lambda t: len(t.split()))
    text = "For returns, I prefer a pickup from my home."
    result = extractor.extract(text, analyze_utterance(text), context={"turn": 8})
    assert result.facts == [] and result.status == "llm_rejected"
    assert result.rejected[0]["reason"] == reason


def test_invalid_json_from_llm_is_rejected():
    extractor = FactExtractor(llm=LoggedLLM(client=FakeOllamaClient("not json"), logger=None), logger=None,
                              token_counter=lambda t: 1)
    text = "For returns, I prefer a pickup from my home."
    result = extractor.extract(text, analyze_utterance(text), context={"turn": 8})
    assert result.facts == [] and result.rejected == [{"reason": "invalid_json"}]
