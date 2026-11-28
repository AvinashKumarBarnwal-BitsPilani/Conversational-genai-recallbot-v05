"""Surprise probes S1-S5 (Task 4(c)): the fixes made after the first Session C run."""
from __future__ import annotations

import json
from pathlib import Path

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.nlu import analyze_utterance
from src.dialogue.state_tracker import DialogueStateTracker
from src.memory.clock import SessionClock
from test_dialogue_memory_integration import USER, run_session_a
from test_tools import make_gateway

ROOT = Path(__file__).resolve().parents[1]
PROBES = {p["id"]: p for p in json.loads((ROOT / "data" / "surprise_probes.json").read_text(encoding="utf-8"))}


def test_probe_file_has_five_adapted_probes_for_session_c():
    assert sorted(PROBES) == ["S1", "S2", "S3", "S4", "S5"]
    assert all(p["session"] == "C" and p["user"] == "U1" and "<" not in p["text"] for p in PROBES.values())


def test_nlu_routes_the_probe_wordings():
    assert analyze_utterance(PROBES["S1"]["text"])["intent"] == "list_memories"
    assert analyze_utterance(PROBES["S4"]["text"])["intent"] == "knowledge_base_query"
    assert analyze_utterance(PROBES["S5"]["text"])["intent"] == "episodic_memory_query"
    assert analyze_utterance("You said my refund would be instant, right?")["intent"] == "episodic_memory_query"


def test_bare_slot_ids_are_not_treated_as_kb_codes():
    for text in ("ORD-74021", "SKU-PHC-11", "RMA-20261126-AAA3511C"):
        assert analyze_utterance(text)["intent"] == "unknown"
    assert analyze_utterance("ERR-RMA-04")["intent"] == "knowledge_base_query"


def test_session_c_has_a_logical_start_time():
    assert SessionClock.for_session("C").iso_at(60) == "2026-11-28T10:00:00+05:30"


def test_order_without_address_uses_customer_default(tmp_path):
    gateway, _, _ = make_gateway(tmp_path)
    order = gateway.get_order("ORD-88216", "CUST-MAYA-1042")
    assert order["address_source"] == "customer_default"
    assert order["address_summary"].startswith("Customer saved Bengaluru")
    assert "address_source" not in gateway.get_order("ORD-74021", "CUST-MAYA-1042")


def _session_c(tmp_path):
    manager_a, memory, _, logger, _ = run_session_a(tmp_path)
    manager = DialogueManager(tracker=DialogueStateTracker(session_id="C", user_id=USER), tools=manager_a.tools,
                              user_id=USER, config="hybrid_mem", logger=logger, memory=memory,
                              token_counter=lambda t: len(t.split()))
    return manager, tmp_path / "runtime_pickup_slots.json"


def test_s1_lists_only_current_facts_and_events(tmp_path):
    manager, _ = _session_c(tmp_path)
    result = manager.handle_turn(PROBES["S1"]["text"], turn=60)
    reply = result["reply"]
    assert result["nlg_status"] == "verbatim"
    assert "SMS" in reply and "19:00-21:00" in reply and "Maya" in reply
    for gone in ("email", "18:00-21:00", "side door"):
        assert gone not in reply.casefold()
    assert "ORD-88216" in reply and "exchange rather than refund" in reply


def test_s2_planted_instruction_does_not_skip_confirmation(tmp_path):
    manager, db_path = _session_c(tmp_path)
    before = db_path.read_bytes()
    manager.handle_turn(PROBES["S2"]["plant_text"], turn=61)
    results = [manager.handle_turn(text, turn=62 + i) for i, text in enumerate(PROBES["S2"]["after_restart_turns"])]
    last = results[-1]
    assert "confirm" in last["reply"].casefold()
    assert last["state"]["slots"]["explicit_confirmation"] is False
    assert not any(c.get("tool") == "book_slot" for r in results for c in r["tool_calls"])
    assert db_path.read_bytes() == before


def test_s5_claim_about_past_reply_is_not_confirmed(tmp_path):
    manager, _ = _session_c(tmp_path)
    result = manager.handle_turn(PROBES["S5"]["text"], turn=68)
    assert result["reply"].startswith("I have no record of telling you that")
    assert result["nlg_status"] == "verbatim"
