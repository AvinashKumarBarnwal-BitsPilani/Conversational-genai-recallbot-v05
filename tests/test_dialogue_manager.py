from __future__ import annotations

import json
from pathlib import Path

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker

ROOT = Path(__file__).resolve().parents[1]
ORDERS = json.loads((ROOT / "db" / "seed_orders.json").read_text(encoding="utf-8"))


class FakeTools:
    """In-memory tool doubles; tests never modify the real seed database."""

    def __init__(self):
        self.calls = []

    def get_order(self, order_id, customer_id=None):
        self.calls.append(("get_order", order_id))
        order = next((x for x in ORDERS["orders"] if x["order_id"] == order_id), None)
        if order is None:
            raise LookupError("order_not_found")
        if customer_id and order["customer_id"] != customer_id:
            raise PermissionError("order_not_owned")
        return order

    def search_kb(self, query, top_k=3):
        self.calls.append(("search_kb", query))
        return [{
            "doc_id": "KB-002", "chunk_id": "KB-002-C128-0001",
            "version": "2026", "content": "The current return policy allows eligible returns within the stated return window."
        }][:top_k]

    def check_pickup_slots(self, pickup_date, requested_window, address_summary):
        self.calls.append(("check_pickup_slots", pickup_date))
        # A single deterministic slot is returned only when the requested
        # start time is compatible with the slot. This is a test double.
        if requested_window.get("start") == "18:00" and requested_window.get("end") in (None, "20:00"):
            return [{
                "slot_id": "PICK-TEST-001", "window_start": "18:00", "window_end": "20:00",
                "date": pickup_date, "status": "available"
            }]
        return []

    def book_slot(self, arguments):
        self.calls.append(("book_slot", dict(arguments)))
        return {"status": "success", "booking_id": "PICKUP-TEST-001", "booking_status": "confirmed"}


def make_manager():
    tools = FakeTools()
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(
        tracker=tracker,
        tools=tools,
        user_id="CUST-MAYA-1042",
        logger=None,
        token_counter=lambda value: len(value.split()),
    )
    return manager, tools, tracker


def test_greeting_returns_concise_response_without_tool_call():
    manager, tools, _ = make_manager()
    result = manager.handle_turn("Hi there!", turn=1)
    assert result["intent"] == "greeting"
    assert "Hello" in result["reply"]
    assert tools.calls == []


def test_start_return_reads_order_and_prompts_for_first_missing_slot():
    manager, tools, tracker = make_manager()
    result = manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    assert tools.calls[0] == ("get_order", "ORD-74021")
    assert tracker.state["slots"]["order_id"] == "ORD-74021"
    assert tracker.state["slots"]["sku"] == "SKU-AUR-220"
    assert "why" in result["reply"].lower() or "reason" in result["reply"].lower()


def test_policy_question_calls_kb_and_returns_source_citation():
    manager, tools, _ = make_manager()
    result = manager.handle_turn("What is the return policy?", turn=1)
    assert tools.calls[0][0] == "search_kb"
    assert "KB-002" in result["reply"]
    assert result["sources"] == ["KB-002 / KB-002-C128-0001 (version 2026)"]


def test_ambiguous_item_does_not_get_guessed_and_manager_asks_clarification():
    manager, tools, tracker = make_manager()
    result = manager.handle_turn("I need to return an item from order ORD-74021.", turn=12)
    assert tracker.state["clarification_required"] is True
    assert "Which item" in result["reply"]


def test_state_followup_preserves_date_and_asks_for_remaining_reason_or_slot():
    manager, tools, tracker = make_manager()
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    result = manager.handle_turn("And for Saturday, please make it after 6 pm.", turn=15)
    assert tracker.state["slots"]["pickup_date"] == "2026-11-28"
    assert tracker.state["slots"]["pickup_time_window"]["start"] == "18:00"
    assert "address" not in result["reply"].lower() or tracker.state["slots"]["pickup_address_summary"]


def test_session_a_historical_decision_does_not_route_or_overwrite_active_return():
    manager, tools, tracker = make_manager()
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    manager.handle_turn("And for Saturday, please make it after 6 pm.", turn=15)
    before = tracker.state["slots"]
    order_reads_before = [call for call in tools.calls if call[0] == "get_order"]
    slot_checks_before = [call for call in tools.calls if call[0] == "check_pickup_slots"]

    manager.handle_turn(
        "I also chose an exchange rather than a refund for my USB-C charger order ORD-66310 last month.",
        turn=17,
    )

    assert tracker.state["slots"] == before
    assert tracker.state["current_intent"] == "report_decision"
    assert tracker.state["current_entities"]["order_ids"] == ["ORD-66310"]
    assert [call for call in tools.calls if call[0] == "get_order"] == order_reads_before
    assert [call for call in tools.calls if call[0] == "check_pickup_slots"] == slot_checks_before


def test_confirm_without_pending_action_never_calls_booking_tool():
    manager, tools, tracker = make_manager()
    result = manager.handle_turn("Yes, I confirm the pickup.", turn=1)
    assert tracker.state["database_write_authorized"] is False
    assert not any(name == "book_slot" for name, *_ in tools.calls)
    assert "haven't made" in result["reply"].lower() or "no pending" in result["reply"].lower()


def test_summary_request_with_ambiguous_slot_asks_for_choice_without_pending_action():
    class MultipleSlotTools(FakeTools):
        def check_pickup_slots(self, pickup_date, requested_window, address_summary):
            self.calls.append(("check_pickup_slots", pickup_date))
            return [
                {"slot_id": "SLOT-A", "window_start": "18:00", "window_end": "20:00"},
                {"slot_id": "SLOT-B", "window_start": "20:00", "window_end": "22:00"},
            ]

    tools = MultipleSlotTools()
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(tracker=tracker, tools=tools, user_id="CUST-MAYA-1042", logger=None)
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    manager.handle_turn("And for Saturday, please make it after 6 pm.", turn=15)

    response = manager.handle_turn("Please repeat the complete return pickup summary so I can confirm it.", turn=22)
    assert "choose one" in response["reply"].lower()
    assert "SLOT-A" in response["reply"] and "SLOT-B" in response["reply"]
    assert tracker.state["pending_confirmation"] is False
    assert tracker.state["pending_action"] is None
    assert not any(name == "book_slot" for name, *_ in tools.calls)

    confirmation = manager.handle_turn("Yes, I confirm the return pickup.", turn=23)
    assert "no pending" in confirmation["reply"].lower()
    assert not any(name == "book_slot" for name, *_ in tools.calls)


def test_confirmation_details_must_match_the_pending_booking():
    manager, tools, tracker = make_manager()
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    manager.handle_turn("And for Saturday, make it after 6 pm.", turn=15)
    assert tracker.state["pending_confirmation"] is True

    mismatch = manager.handle_turn(
        "Yes, I confirm the return pickup for order ORD-66310 and SKU-CHG-45.", turn=16
    )
    assert "does not match" in mismatch["reply"].lower()
    assert tracker.state["pending_confirmation"] is True
    assert tracker.state["slots"]["order_id"] == "ORD-74021"
    assert tracker.state["slots"]["sku"] == "SKU-AUR-220"
    assert not any(name == "book_slot" for name, *_ in tools.calls)

    successful = manager.handle_turn("Yes, I confirm the pickup booking.", turn=17)
    assert successful["completed"] is True
    assert any(name == "book_slot" for name, *_ in tools.calls)


def test_booking_tool_runs_only_after_tracker_confirmation_gate():
    manager, tools, tracker = make_manager()
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    result = manager.handle_turn("And for Saturday, make it after 6 pm.", turn=15)
    assert tracker.state["pending_confirmation"] is True
    assert not any(name == "book_slot" for name, *_ in tools.calls)
    assert "confirm" in result["reply"].lower()

    result = manager.handle_turn("Yes, I confirm the pickup booking.", turn=23)
    assert any(name == "book_slot" for name, *_ in tools.calls)
    assert tracker.state["task_status"] == "completed"
    assert result["completed"] is True
    assert "PICKUP-TEST-001" in result["reply"]


def test_cancel_pending_booking_does_not_call_write_tool():
    manager, tools, tracker = make_manager()
    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    manager.handle_turn("And for Saturday, make it after 6 pm.", turn=15)
    result = manager.handle_turn("No, cancel that.", turn=22)
    assert tracker.state["confirmation_status"] == "cancelled"
    assert not any(name == "book_slot" for name, *_ in tools.calls)
    assert "cancelled" in result["reply"].lower()


def test_tool_failure_does_not_claim_order_lookup_success():
    class FailingTools(FakeTools):
        def get_order(self, order_id, customer_id=None):
            raise TimeoutError("backend timeout")

    tools = FailingTools()
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(tracker=tracker, tools=tools, user_id="CUST-MAYA-1042", logger=None)
    result = manager.handle_turn("I need to return earbuds from order ORD-74021.", turn=12)
    assert "couldn't retrieve" in result["reply"].lower()
    assert tracker.state["task_status"] != "completed"


def test_order_id_provided_in_followup_turn_triggers_order_lookup():
    manager, tools, tracker = make_manager()
    first = manager.handle_turn("I want to return an item.", turn=12)
    assert "order ID" in first["reply"]
    assert tracker.state["active_task"] == "return_pickup"

    second = manager.handle_turn("ORD-74021", turn=13)
    assert ("get_order", "ORD-74021") in tools.calls
    assert tracker.state["slots"]["order_id"] == "ORD-74021"
    assert tracker.state["clarification_required"] is True
    assert "Which item" in second["reply"]


def test_plain_language_item_selection_resolves_prior_clarification():
    manager, tools, tracker = make_manager()
    first = manager.handle_turn("I need to return an item from order ORD-74021.", turn=12)
    assert tracker.state["clarification_required"] is True
    assert "Which item" in first["reply"]

    second = manager.handle_turn("The protective phone case.", turn=13)
    assert tracker.state["slots"]["sku"] == "SKU-PHC-11"
    assert tracker.state["clarification_required"] is False
    assert "why" in second["reply"].lower()
    assert sum(1 for call in tools.calls if call[0] == "get_order") == 2


def test_session_b_other_item_request_clarifies_without_any_tool_call():
    manager, tools, tracker = make_manager()
    text = "Can you book a pickup for the other item as well?"
    result = manager.handle_turn(text, turn=37)

    assert result["intent"] == "clarification_required"
    assert tracker.state["active_task"] is None
    assert tracker.state["slots"]["order_id"] is None
    assert tracker.state["slots"]["sku"] is None
    assert result["reply"] == "Which order and item do you mean? Please provide the order ID or select an item."
    assert tools.calls == []
    assert result["tool_calls"] == []


def test_session_b_turn_33_uses_preceding_context_and_kb_for_general_pickup_followup():
    kb3_lifecycle = (
        "A refund has several distinct stages: return requested, return authorised, "
        "item collected, item received, inspection completed, refund initiated and "
        "payment provider posting. A customer may see the return marked accepted "
        "before the credit is visible in the bank account."
    )
    kb14_receipt = (
        "A return inspection update is normally provided within three business days "
        "after receipt at the returns centre."
    )

    class SessionBTools(FakeTools):
        def search_kb(self, query, top_k=3):
            self.calls.append(("search_kb", query))
            if "after the pickup" in query.casefold():
                return [
                    {
                        "doc_id": "KB-003", "chunk_id": "KB-003-C128-0001",
                        "version": "1.3", "status": "active", "content": kb3_lifecycle,
                    },
                    {
                        "doc_id": "KB-014", "chunk_id": "KB-014-C128-0001",
                        "version": "1.0", "status": "active", "content": kb14_receipt,
                    },
                ][:top_k]
            return super().search_kb(query, top_k=top_k)

    tools = SessionBTools()
    tracker = DialogueStateTracker(session_id="B", user_id="CUST-MAYA-1042")

    def grounded_test_response(query, evidence):
        if query != "And what happens next after the pickup is completed?":
            return "The current policy specifies an eligible return window."
        assert {item["doc_id"] for item in evidence} == {"KB-003", "KB-014"}
        assert "item received" in evidence[0]["content"]
        assert "refund initiated" in evidence[0]["content"]
        assert "returns centre" in evidence[1]["content"]
        return (
            "After collection, the return is received at the returns centre and inspected. "
            "If accepted, the refund is then initiated; this does not confirm that this "
            "pickup or a refund has occurred."
        )

    manager = DialogueManager(
        tracker=tracker,
        tools=tools,
        user_id="CUST-MAYA-1042",
        logger=None,
        token_counter=lambda value: len(value.split()),
        response_generator=grounded_test_response,
    )

    # Replay the exact preceding Session B user turns. No Session A tracker,
    # return slots, booking event or persistent memory is injected.
    turn31 = manager.handle_turn(
        "For a 2026 delivery, how many days do I have to return electronic accessories?",
        turn=31,
    )
    turn32 = manager.handle_turn(
        "Can you check the latest status of order ORD-74021?",
        turn=32,
    )
    assert turn31["intent"] == "knowledge_base_query"
    assert turn32["intent"] == "order_status_query"
    assert "ORD-74021" in turn32["reply"]
    assert tracker.state["active_task"] is None
    assert tracker.state["slots"]["order_id"] is None

    turn33 = manager.handle_turn(
        "And what happens next after the pickup is completed?",
        turn=33,
    )

    assert turn33["intent"] == "state_followup"
    assert turn33["nlu"]["rule_id"] == "R14_STATE_FOLLOWUP"
    assert "returns centre" in turn33["reply"].lower()
    assert "inspection" in turn33["reply"].lower() or "inspected" in turn33["reply"].lower()
    assert "if accepted" in turn33["reply"].lower()
    assert "refund is then initiated" in turn33["reply"].lower()
    assert "pickup is complete" not in turn33["reply"].lower()
    assert "refund has been initiated" not in turn33["reply"].lower()
    assert "RFND-" not in turn33["reply"]
    assert turn33["sources"] == [
        "KB-003 / KB-003-C128-0001 (version 1.3)",
        "KB-014 / KB-014-C128-0001 (version 1.0)",
    ]
    assert turn33["tool_calls"] == [{
        "tool": "search_kb",
        "status": "success",
        "candidate_count": 2,
        "result_count": 2,
        "version_filter": "none",
    }]
    assert tools.calls[-1] == ("search_kb", "And what happens next after the pickup is completed?")
    assert [message["content"] for message in manager.conversation_history.get_history("full")[-6:-2]] == [
        "For a 2026 delivery, how many days do I have to return electronic accessories?",
        turn31["reply"],
        "Can you check the latest status of order ORD-74021?",
        turn32["reply"],
    ]
