from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.dialogue.tools import (
    ConfirmationRequired,
    OrderOwnershipError,
    RecallBotToolGateway,
    SlotUnavailable,
)
from src.retrieval.dense import DenseIndex
from src.retrieval.models import KnowledgeChunk

ROOT = Path(__file__).resolve().parents[1]


def make_gateway(tmp_path, **kwargs):
    runtime_path = tmp_path / "runtime_pickup_slots.json"
    events = []
    gateway = RecallBotToolGateway(
        orders_path=ROOT / "db" / "seed_orders.json",
        seed_slots_path=ROOT / "db" / "seed_pickup_slots.json",
        runtime_slots_path=runtime_path,
        logger=lambda **record: events.append(record),
        token_counter=lambda text: len(text.split()),
        **kwargs,
    )
    gateway.set_call_context(user="CUST-MAYA-1042", session="A", turn=1, config="full")
    return gateway, events, runtime_path


def valid_booking_arguments():
    return {
        "customer_id": "CUST-MAYA-1042",
        "order_id": "ORD-74021",
        "sku": "SKU-AUR-220",
        "reason": "defective_left_earbud",
        "pickup_date": "2026-11-28",
        "pickup_time_window": {"start": "18:00", "end": "20:00"},
        "pickup_address_summary": "Customer saved Bengaluru delivery address (masked summary)",
        "slot_id": "PICK-20261128-E1",
        "explicit_confirmation": True,
    }


def test_tool_logs_zero_tokens_and_retains_latency_for_success_and_failure(
    tmp_path, monkeypatch
):
    from src import logger as logger_module

    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(logger_module, "RUN_ID", "tool-zero-token-test")
    gateway = RecallBotToolGateway(logger=logger_module.log_call)
    gateway.set_call_context(
        user="CUST-MAYA-1042", session="A", turn=7, config="trim"
    )

    gateway._run_logged_tool(
        tool_name="test_success",
        model="test-tool",
        request={"value": "ok"},
        operation=lambda: {"status": "ok"},
    )

    def fail():
        raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"):
        gateway._run_logged_tool(
            tool_name="test_failure",
            model="test-tool",
            request={"value": "bad"},
            operation=fail,
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["extra"]["status"] for row in rows] == ["success", "failed"]
    assert [row["extra"]["tool_name"] for row in rows] == [
        "test_success", "test_failure"
    ]
    for row in rows:
        assert row["input_tokens"] == 0
        assert row["output_tokens"] == 0
        assert row["retrieved_tokens"] == 0
        assert row["latency_ms"] >= 0
        assert row["run_id"] == "tool-zero-token-test"
        assert row["user"] == "CUST-MAYA-1042"
        assert row["session"] == "A"
        assert row["turn"] == 7
        assert row["extra"]["request"] in ({"value": "ok"}, {"value": "bad"})


def test_get_order_reads_seed_json_and_enforces_customer_ownership(tmp_path):
    gateway, events, _ = make_gateway(tmp_path)
    order = gateway.get_order("ORD-74021", "CUST-MAYA-1042")
    assert order["order_id"] == "ORD-74021"
    assert order["items"][0]["sku"] == "SKU-AUR-220"
    assert events[-1]["call_type"] == "tool"
    assert events[-1]["extra"]["tool_name"] == "get_order"
    assert events[-1]["extra"]["status"] == "success"

    with pytest.raises(OrderOwnershipError):
        gateway.get_order("ORD-74021", "CUST-OTHER-9999")
    assert events[-1]["extra"]["status"] == "failed"


def test_pickup_slot_lookup_filters_by_date_time_and_address(tmp_path):
    gateway, _, runtime_path = make_gateway(tmp_path)
    after_six = gateway.check_pickup_slots(
        "2026-11-28", {"start": "18:00", "end": None},
        "Customer saved Bengaluru delivery address",
    )
    assert [slot["slot_id"] for slot in after_six] == [
        "PICK-20261128-E1", "PICK-20261128-E2"
    ]

    exact_window = gateway.check_pickup_slots(
        "2026-11-28", {"start": "18:00", "end": "20:00"},
        "Customer saved Bengaluru delivery address",
    )
    assert [slot["slot_id"] for slot in exact_window] == ["PICK-20261128-E1"]

    wrong_city = gateway.check_pickup_slots(
        "2026-11-28", {"start": "18:00", "end": "20:00"},
        "Customer saved Chennai delivery address",
    )
    assert wrong_city == []
    assert runtime_path.exists()


def test_booking_rejects_missing_explicit_confirmation_without_write(tmp_path):
    gateway, events, runtime_path = make_gateway(tmp_path)
    arguments = valid_booking_arguments()
    arguments.pop("explicit_confirmation")

    with pytest.raises(ConfirmationRequired):
        gateway.book_slot(arguments)

    assert not runtime_path.exists()
    assert events[-1]["extra"]["tool_name"] == "book_slot"
    assert events[-1]["extra"]["status"] == "failed"
    assert events[-1]["extra"]["error_type"] == "ConfirmationRequired"


def test_confirmed_booking_writes_runtime_json_and_prevents_double_booking(tmp_path):
    gateway, events, runtime_path = make_gateway(tmp_path)
    result = gateway.book_slot(valid_booking_arguments())

    assert result["status"] == "success"
    assert result["booking_id"].startswith("PICKUP-")
    assert result["rma_id"].startswith("RMA-")

    persisted = json.loads(runtime_path.read_text(encoding="utf-8"))
    booked_slot = next(row for row in persisted["slots"] if row["slot_id"] == "PICK-20261128-E1")
    assert booked_slot["status"] == "booked"
    assert len(persisted["bookings"]) == 1
    assert persisted["bookings"][0]["confirmation_received"] is True

    with pytest.raises(SlotUnavailable):
        gateway.book_slot(valid_booking_arguments())
    assert len(json.loads(runtime_path.read_text(encoding="utf-8"))["bookings"]) == 1
    assert events[-1]["extra"]["error_type"] == "SlotUnavailable"


def test_book_slot_rejects_sku_not_belonging_to_order(tmp_path):
    gateway, _, runtime_path = make_gateway(tmp_path)
    arguments = valid_booking_arguments()
    arguments["sku"] = "SKU-NOT-IN-ORDER"
    with pytest.raises(ValueError, match="not part of the requested order"):
        gateway.book_slot(arguments)
    assert not runtime_path.exists()


def test_search_kb_uses_dense_bm25_and_rrf_and_returns_source_metadata(tmp_path):
    chunks = [
        KnowledgeChunk(
            chunk_id="KB-TEST-C1", doc_id="KB-TEST", title="Returns Policy",
            section="Return window", version="2026", effective_from="2026-01-01",
            effective_to="", status="active", source_file="test.md", chunk_size=128,
            overlap_tokens=19, token_count=20,
            content="Eligible products may be returned within 30 days with proof of purchase.",
        ),
        KnowledgeChunk(
            chunk_id="KB-TEST-C2", doc_id="KB-TEST", title="Returns Policy",
            section="Refund method", version="2026", effective_from="2026-01-01",
            effective_to="", status="active", source_file="test.md", chunk_size=128,
            overlap_tokens=19, token_count=20,
            content="Refunds are returned to the original payment method after inspection.",
        ),
    ]
    index = DenseIndex(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    gateway, events, _ = make_gateway(
        tmp_path,
        index_loader=lambda _path: index,
        embedder_factory=lambda _name: FakeEmbedder(),
        candidate_k=2,
    )
    results = gateway.search_kb("What is the return period?", top_k=2)

    assert results
    assert results[0]["method"] == "hybrid_rrf"
    assert results[0]["doc_id"] == "KB-TEST"
    assert results[0]["version"] == "2026"
    assert "content" in results[0]
    assert events[-1]["extra"]["tool_name"] == "search_kb"
    assert events[-1]["retrieved_tokens"] == 0
