"""Unit tests for RecallBot's session-level dialogue state tracker (Task 2a)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dialogue.nlu import analyze_utterance
from src.dialogue.state_tracker import DialogueStateTracker

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "skills" / "return_pickup_workflow_v1.json"
ORDERS = ROOT / "db" / "seed_orders.json"


def make_tracker() -> DialogueStateTracker:
    return DialogueStateTracker(session_id="A")


def get_order_fixture(order_id: str = "ORD-74021") -> dict:
    payload = json.loads(ORDERS.read_text(encoding="utf-8"))
    return next(order for order in payload["orders"] if order["order_id"] == order_id)


def update(tracker: DialogueStateTracker, text: str, turn: int):
    return tracker.update_from_nlu(text, analyze_utterance(text), turn=turn)


def test_tracker_loads_required_slots_from_versioned_workflow():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    tracker = make_tracker()
    assert tracker.required_slots == tuple(workflow["required_slots"])
    assert set(tracker.missing_slots()) == set(workflow["required_slots"])


def test_return_start_and_order_read_fill_order_customer_address_and_unique_sku():
    tracker = make_tracker()
    text = "I need to return the wireless earbuds from order ORD-74021."
    update(tracker, text, 12)

    state = tracker.apply_order_result(get_order_fixture(), item_hint=text)
    assert state["active_task"] == "return_pickup"
    assert state["slots"]["order_id"] == "ORD-74021"
    assert state["slots"]["customer_id"] == "CUST-MAYA-1042"
    assert state["slots"]["sku"] == "SKU-AUR-220"
    assert state["slots"]["pickup_address_summary"]
    assert state["clarification_required"] is False


def test_ambiguous_order_item_is_not_guessed():
    tracker = make_tracker()
    text = "Can you book a pickup for the other item as well?"
    update(tracker, text, 12)
    # Simulate that the order ID has been resolved by the active workflow.
    tracker = make_tracker()
    start = "I need to return an item from order ORD-74021."
    update(tracker, start, 12)
    state = tracker.apply_order_result(get_order_fixture(), item_hint="the other item")
    assert state["slots"]["sku"] is None
    assert state["clarification_required"] is True
    assert "Which item" in state["clarification_prompt"]
    assert "sku" in state["missing_slots"]


def test_reason_date_and_short_followup_merge_without_losing_previous_slots():
    tracker = make_tracker()
    text = "I need to return the wireless earbuds from order ORD-74021."
    update(tracker, text, 12)
    tracker.apply_order_result(get_order_fixture(), item_hint=text)

    reason_text = "The left earbud does not charge even after I tried the charging case."
    reason_result = analyze_utterance(reason_text)
    assert reason_result["entities"]["return_reason"] == "defective_left_earbud"
    tracker.update_from_nlu(reason_text, reason_result, turn=13)

    date_text = "Saturday, 28 November 2026 would work."
    tracker.update_from_nlu(date_text, analyze_utterance(date_text), turn=14)
    date_before_followup = tracker.state["slots"]["pickup_date"]

    followup = "And for Saturday, please make it after 6 pm."
    state = update(tracker, followup, 15)
    assert state["slots"]["reason"] == "defective_left_earbud"
    assert state["slots"]["pickup_date"] == date_before_followup == "2026-11-28"
    assert state["slots"]["pickup_time_window"]["start"] == "18:00"
    assert state["slots"]["pickup_time_window"]["end"] is None


def test_historical_exchange_decision_does_not_mutate_active_return_slots():
    tracker = make_tracker()
    start = "I need to return the wireless earbuds from order ORD-74021."
    tracker.update_from_nlu(start, analyze_utterance(start), turn=12)
    tracker.apply_order_result(get_order_fixture(), item_hint=start)
    tracker.update_from_nlu(
        "The left earbud does not charge.",
        analyze_utterance("The left earbud does not charge."), turn=13,
    )
    tracker.update_from_nlu(
        "Saturday, 28 November 2026 would work.",
        analyze_utterance("Saturday, 28 November 2026 would work."), turn=14,
    )
    tracker.update_from_nlu(
        "And for Saturday, please make it after 6 pm.",
        analyze_utterance("And for Saturday, please make it after 6 pm."), turn=15,
    )
    before = tracker.state["slots"]
    decision = "I also chose an exchange rather than a refund for my USB-C charger order ORD-66310 last month."
    parsed = analyze_utterance(decision)
    assert parsed["intent"] == "report_decision"
    assert parsed["entities"]["order_ids"] == ["ORD-66310"]

    after = tracker.update_from_nlu(decision, parsed, turn=17)
    assert after["current_intent"] == "report_decision"
    assert after["current_entities"] == parsed["entities"]
    assert after["slots"] == before


def test_confirmation_cannot_be_requested_while_required_slots_are_missing():
    tracker = make_tracker()
    text = "I need to return my earbuds from order ORD-74021."
    update(tracker, text, 12)
    result = tracker.request_confirmation({"tool": "book_slot", "arguments": {"order_id": "ORD-74021"}})
    assert result["ready"] is False
    assert result["missing_slots"]
    assert tracker.state["pending_confirmation"] is False


def test_database_action_is_gated_by_explicit_confirmation_and_consumed_once():
    tracker = make_tracker()
    text = "I need to return the wireless earbuds from order ORD-74021."
    update(tracker, text, 12)
    tracker.apply_order_result(get_order_fixture(), item_hint=text)
    reason = "The left earbud does not charge even after I tried the charging case."
    update(tracker, reason, 13)
    date = "Saturday, 28 November 2026 would work."
    update(tracker, date, 14)
    window = "And for Saturday, please make it after 6 pm."
    update(tracker, window, 15)

    args = {
        "order_id": "ORD-74021",
        "sku": "SKU-AUR-220",
        "reason": "defective_left_earbud",
        "pickup_date": "2026-11-28",
        "pickup_time_window": {"start": "18:00", "end": None},
    }
    ready = tracker.request_confirmation({"tool": "book_slot", "arguments": args})
    assert ready["ready"] is True
    assert tracker.state["pending_confirmation"] is True
    assert tracker.consume_confirmed_action() is None  # no confirmation yet

    confirm = "Yes, I confirm the return pickup for the AUR-220 earbuds on 28 November after 6 pm."
    state = update(tracker, confirm, 23)
    assert state["confirmation_status"] == "confirmed"
    assert state["slots"]["explicit_confirmation"] is True

    action = tracker.consume_confirmed_action()
    assert action == {"tool": "book_slot", "arguments": args}
    assert tracker.consume_confirmed_action() is None  # one-time dispatch guard
    assert tracker.state["database_write_authorized"] is True


def test_confirmation_without_pending_action_does_not_authorise_a_write():
    tracker = make_tracker()
    confirm = "Yes, I confirm the return pickup."
    state = update(tracker, confirm, 1)
    assert state["slots"]["explicit_confirmation"] is False
    assert state["confirmation_status"] == "not_requested"
    assert tracker.consume_confirmed_action() is None


def test_cancel_confirmation_clears_pending_action_without_authorising_write():
    tracker = make_tracker()
    text = "I need to return the wireless earbuds from order ORD-74021."
    update(tracker, text, 12)
    tracker.apply_order_result(get_order_fixture(), item_hint=text)
    tracker.set_slot("reason", "defective_left_earbud")
    tracker.set_slot("pickup_date", "2026-11-28")
    tracker.set_slot("pickup_time_window", {"start": "18:00", "end": None})
    tracker.request_confirmation({"tool": "book_slot", "arguments": {"order_id": "ORD-74021"}})

    # A clear cancellation phrase is classified by NLU and cancels the
    # pending action without authorising a database write.
    cancel_text = "No, cancel that."
    state = tracker.update_from_nlu(
        cancel_text,
        analyze_utterance(cancel_text),
        turn=24,
    )
    assert state["confirmation_status"] == "cancelled"
    assert state["pending_confirmation"] is False
    assert state["database_write_authorized"] is False
    assert tracker.consume_confirmed_action() is None


def test_snapshot_is_a_defensive_copy():
    tracker = make_tracker()
    first = tracker.snapshot()
    first["slots"]["order_id"] = "ORD-FAKE"
    assert tracker.state["slots"]["order_id"] is None


def test_order_read_result_must_match_requested_order():
    tracker = make_tracker()
    text = "I need to return earbuds from order ORD-74021."
    update(tracker, text, 12)
    wrong_order = get_order_fixture("ORD-88216")
    with pytest.raises(ValueError, match="does not match"):
        tracker.apply_order_result(wrong_order, item_hint=text)


def test_explicit_confirmation_slot_cannot_be_set_directly():
    tracker = make_tracker()
    with pytest.raises(ValueError, match="explicit confirm_action"):
        tracker.set_slot("explicit_confirmation", True)
