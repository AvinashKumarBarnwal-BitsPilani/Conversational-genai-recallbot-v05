from __future__ import annotations

import json
from pathlib import Path

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker
from src.dialogue.tools import RecallBotToolGateway

ROOT = Path(__file__).resolve().parents[1]


def test_dialogue_manager_reads_real_order_database(tmp_path):
    tool_events = []
    tools = RecallBotToolGateway(
        orders_path=ROOT / "db" / "seed_orders.json",
        seed_slots_path=ROOT / "db" / "seed_pickup_slots.json",
        runtime_slots_path=tmp_path / "runtime.json",
        logger=lambda **record: tool_events.append(record),
        token_counter=lambda text: len(text.split()),
    )
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(tracker=tracker, tools=tools, user_id="CUST-MAYA-1042", logger=None)

    result = manager.handle_turn("Check the status of order ORD-74021.", turn=1)
    assert "ORD-74021" in result["reply"]
    assert "ORD-DEL" in result["reply"]
    assert tool_events[-1]["extra"]["tool_name"] == "get_order"
    assert tool_events[-1]["turn"] == 1


def test_real_database_write_runs_only_after_dialogue_manager_confirmation(tmp_path):
    tool_events = []
    runtime_path = tmp_path / "runtime_pickup_slots.json"
    tools = RecallBotToolGateway(
        orders_path=ROOT / "db" / "seed_orders.json",
        seed_slots_path=ROOT / "db" / "seed_pickup_slots.json",
        runtime_slots_path=runtime_path,
        logger=lambda **record: tool_events.append(record),
        token_counter=lambda text: len(text.split()),
    )
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(tracker=tracker, tools=tools, user_id="CUST-MAYA-1042", logger=None)

    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    before_confirmation = manager.handle_turn("After 6 pm.", turn=15)

    assert tracker.state["pending_confirmation"] is False
    assert not runtime_path.exists() or not json.loads(runtime_path.read_text(encoding="utf-8")).get("bookings")
    assert "choose" in before_confirmation["reply"].lower() or "available" in before_confirmation["reply"].lower()

    confirmation_prompt = manager.handle_turn("6 pm to 8 pm", turn=16)
    assert tracker.state["pending_confirmation"] is True
    assert not runtime_path.exists() or not json.loads(runtime_path.read_text(encoding="utf-8")).get("bookings")
    assert "confirm" in confirmation_prompt["reply"].lower()

    booked = manager.handle_turn("Yes, I confirm the pickup booking.", turn=17)
    assert booked["completed"] is True
    assert "RMA-" in booked["reply"]
    persisted = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert len(persisted["bookings"]) == 1
    assert persisted["bookings"][0]["confirmation_received"] is True
    assert persisted["bookings"][0]["order_id"] == "ORD-74021"
    assert any(event["extra"].get("tool_name") == "book_slot" and event["extra"].get("status") == "success" for event in tool_events)
    assert any(event["extra"].get("tool_name") == "book_slot" and event["turn"] == 17 for event in tool_events)


def test_user_can_select_displayed_24_hour_slot_label_verbatim(tmp_path):
    """A displayed 18:00–20:00 slot can be selected without retyping am/pm."""
    tool_events = []
    runtime_path = tmp_path / "runtime_pickup_slots.json"
    tools = RecallBotToolGateway(
        orders_path=ROOT / "db" / "seed_orders.json",
        seed_slots_path=ROOT / "db" / "seed_pickup_slots.json",
        runtime_slots_path=runtime_path,
        logger=lambda **record: tool_events.append(record),
        token_counter=lambda text: len(text.split()),
    )
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(tracker=tracker, tools=tools, user_id="CUST-MAYA-1042", logger=None)

    manager.handle_turn("I need to return the wireless earbuds from order ORD-74021.", turn=12)
    manager.handle_turn("The left earbud does not charge.", turn=13)
    manager.handle_turn("Saturday, 28 November 2026 would work.", turn=14)
    offered = manager.handle_turn("After 6 pm.", turn=15)
    assert "18:00–20:00" in offered["reply"]
    assert "20:00–22:00" in offered["reply"]
    assert tracker.state["pending_confirmation"] is False

    selected = manager.handle_turn("18:00–20:00", turn=16)
    assert tracker.state["pending_confirmation"] is True
    assert "confirm" in selected["reply"].lower()
    assert tracker.state["slots"]["pickup_time_window"] == {"start": "18:00", "end": "20:00"}
    pending_payload = tracker.state["pending_action"]["arguments"]
    assert pending_payload == {
        "customer_id": "CUST-MAYA-1042",
        "order_id": "ORD-74021",
        "sku": "SKU-AUR-220",
        "reason": "defective_left_earbud",
        "pickup_date": "2026-11-28",
        "pickup_time_window": {"start": "18:00", "end": "20:00"},
        "pickup_address_summary": "Customer saved Bengaluru delivery address (masked summary)",
        "slot_id": "PICK-20261128-E1",
    }
    assert "ORD-74021" in selected["reply"]
    assert "SKU-AUR-220" in selected["reply"]
    assert "2026-11-28" in selected["reply"]
    assert "PICK-20261128-E1" in selected["reply"]
    assert not runtime_path.exists() or not json.loads(runtime_path.read_text(encoding="utf-8")).get("bookings")

    booked = manager.handle_turn("Yes, I confirm the pickup booking.", turn=17)
    assert booked["completed"] is True
    persisted = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert len(persisted["bookings"]) == 1
    assert persisted["bookings"][0]["slot_id"] == "PICK-20261128-E1"
