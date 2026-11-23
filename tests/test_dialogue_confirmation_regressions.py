from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.nlu import analyze_utterance
from src.dialogue.state_tracker import DialogueStateTracker


class FakeTools:
    def __init__(self):
        self.book_calls = []
        self.slot_checks = 0

    def set_call_context(self, **kwargs):
        pass

    def get_order(self, order_id, customer_id=None):
        return {
            "order_id": order_id,
            "customer_id": "CUST-MAYA-1042",
            "address_summary": "Bengaluru test address",
            "items": [{"sku": "SKU-AUR-220", "name": "Wireless earbuds"}],
        }

    def search_kb(self, query, top_k=3):
        return []

    def check_pickup_slots(self, pickup_date, requested_window, address_summary):
        self.slot_checks += 1
        return [
            {"slot_id": "PICK-E1", "window_start": "18:00", "window_end": "20:00"},
            {"slot_id": "PICK-E2", "window_start": "20:00", "window_end": "22:00"},
        ]

    def book_slot(self, arguments):
        self.book_calls.append(dict(arguments))
        return {
            "status": "success",
            "booking_status": "confirmed",
            "booking_id": "PICKUP-TEST-001",
            "rma_id": "RMA-TEST-001",
        }


def _tracker_with_pending_booking():
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    start_text = "I need to return wireless earbuds from order ORD-74021."
    tracker.update_from_nlu(start_text, analyze_utterance(start_text), turn=1)

    tracker.set_slot("customer_id", "CUST-MAYA-1042")
    tracker.set_slot("order_id", "ORD-74021")
    tracker.set_slot("sku", "SKU-AUR-220")
    tracker.set_slot("reason", "defective_left_earbud")
    tracker.set_slot("pickup_date", "2026-11-28")
    tracker.set_slot("pickup_time_window", {"start": "18:00", "end": "20:00"})
    tracker.set_slot("pickup_address_summary", "Bengaluru test address")
    tracker.request_confirmation({
        "tool": "book_slot",
        "arguments": {
            "customer_id": "CUST-MAYA-1042",
            "order_id": "ORD-74021",
            "sku": "SKU-AUR-220",
            "reason": "defective_left_earbud",
            "pickup_date": "2026-11-28",
            "pickup_time_window": {"start": "18:00", "end": "20:00"},
            "pickup_address_summary": "Bengaluru test address",
            "slot_id": "PICK-E1",
        },
    })
    return tracker


def test_quoted_yes_consumes_existing_pending_action_once():
    tracker = _tracker_with_pending_booking()
    tools = FakeTools()
    manager = DialogueManager(
        tracker=tracker, tools=tools, user_id="CUST-MAYA-1042",
        nlg=None, logger=None, token_counter=lambda text: len(text.split()),
    )

    first = manager.handle_turn('"Yes, I confirm the pickup booking."', turn=2)
    assert first["completed"] is True
    assert len(tools.book_calls) == 1

    second = manager.handle_turn('"Yes, I confirm the pickup booking."', turn=3)
    assert len(tools.book_calls) == 1
    assert second["completed"] is False


def test_unknown_message_while_confirmation_pending_does_not_recreate_action():
    tracker = _tracker_with_pending_booking()
    tools = FakeTools()
    manager = DialogueManager(
        tracker=tracker, tools=tools, user_id="CUST-MAYA-1042",
        nlg=None, logger=None, token_counter=lambda text: len(text.split()),
    )

    result = manager.handle_turn("Could you repeat the available pickup details?", turn=2)

    assert result["needs_user_input"] is True
    assert "awaiting your confirmation" in result["reply"]
    assert tools.slot_checks == 0
    assert len(tools.book_calls) == 0


def test_exact_slot_selection_filters_broader_gateway_results():
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    start_text = "I need to return wireless earbuds from order ORD-74021."
    tracker.update_from_nlu(start_text, analyze_utterance(start_text), turn=1)
    tracker.set_slot("customer_id", "CUST-MAYA-1042")
    tracker.set_slot("order_id", "ORD-74021")
    tracker.set_slot("sku", "SKU-AUR-220")
    tracker.set_slot("reason", "defective_left_earbud")
    tracker.set_slot("pickup_date", "2026-11-28")
    tracker.set_slot("pickup_time_window", {"start": "18:00", "end": None})
    tracker.set_slot("pickup_address_summary", "Bengaluru test address")

    tools = FakeTools()
    manager = DialogueManager(
        tracker=tracker, tools=tools, user_id="CUST-MAYA-1042",
        nlg=None, logger=None, token_counter=lambda text: len(text.split()),
    )

    result = manager.handle_turn("18:00–20:00", turn=2)

    pending = tracker.snapshot()["pending_action"]
    assert result["needs_user_input"] is True
    assert pending["status"] == "awaiting_confirmation"
    assert pending["arguments"]["slot_id"] == "PICK-E1"
