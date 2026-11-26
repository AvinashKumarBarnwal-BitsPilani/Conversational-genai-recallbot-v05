"""Offline end-to-end Session A: real DM + real JSON tools + MemoryManager fakes."""
from __future__ import annotations

import json
from pathlib import Path

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker
from src.dialogue.tools import RecallBotToolGateway
from src.memory.fact_extractor import FactExtractor
from src.memory.llm_calls import LoggedLLM
from src.memory.memory_manager import MemoryManager
from memory_fakes import FakeEmbedder, FakeOllamaClient, RecordingLogger

ROOT = Path(__file__).resolve().parents[1]
CONVERSATION = json.loads((ROOT / "data" / "test_conversation.json").read_text(encoding="utf-8"))
USER = CONVERSATION["user_id"]


class OfflineKBTools(RecallBotToolGateway):
    """Real order/slot/booking tools; KB search stubbed to avoid model loading."""

    def search_kb(self, query, top_k=3):
        return [{"doc_id": "KB-002", "chunk_id": "KB-002-C128-0001", "version": "2026", "status": "active",
                 "content": "Electronics may be returned within 14 calendar days of delivery."}]


def run_session_a(tmp_path, config="hybrid_mem"):
    logger = RecordingLogger()
    tools = OfflineKBTools(
        orders_path=ROOT / "db" / "seed_orders.json",
        seed_slots_path=ROOT / "db" / "seed_pickup_slots.json",
        runtime_slots_path=tmp_path / "runtime_pickup_slots.json",
        logger=logger, token_counter=lambda t: len(t.split()),
    )
    fallback = FakeOllamaClient(json.dumps({"facts": [{"key": "return_method_preference", "value": "home pickup"}]}))
    memory = MemoryManager(
        tmp_path / "memory", config=config, user=USER, embedder=FakeEmbedder(),
        extractor=FactExtractor(llm=LoggedLLM(client=fallback, logger=logger), logger=logger,
                                token_counter=lambda t: len(t.split())),
        logger=logger, token_counter=lambda t: len(t.split()),
    )
    tracker = DialogueStateTracker(session_id="A", user_id=USER)
    manager = DialogueManager(tracker=tracker, tools=tools, user_id=USER, config=config, logger=logger,
                              memory=memory, token_counter=lambda t: len(t.split()))
    results = {}
    for turn in CONVERSATION["turns"]:
        if turn["session"] == "A":
            results[turn["turn"]] = manager.handle_turn(turn["text"], turn=turn["turn"])
    return manager, memory, results, logger, fallback


def test_session_a_builds_semantic_and_episodic_memory_from_the_real_pipeline(tmp_path):
    manager, memory, results, logger, fallback = run_session_a(tmp_path)

    active = {f.key: f.value for f in memory.semantic.active_facts(USER)}
    assert active == {
        "preferred_name": "Maya", "notification_channel": "SMS", "preferred_city": "Bengaluru",
        "preferred_delivery_window": "19:00-21:00", "return_method_preference": "home pickup",
        "packaging_preference": "recyclable",
    }
    assert len(fallback.requests) == 1  # only turn 8 needed the LLM fallback
    assert memory.semantic.get_active(USER, "return_method_preference").extraction_method == "llm"

    by_type = {}
    for episode in memory.episodic.episodes(USER):
        by_type.setdefault(episode.type, []).append(episode)
    assert [ep.source_turn for ep in by_type["complaint"]] == [10]
    assert "ORD-88216" in by_type["complaint"][0].description
    assert "no formal complaint case" in by_type["complaint"][0].description
    assert [ep.source_turn for ep in by_type["decision"]] == [17]
    assert "exchange rather than refund" in by_type["decision"][0].description
    assert [ep.source_turn for ep in by_type["task_event"]] == [12]
    booking = by_type["booking"][0]
    assert booking.source_turn == 23 and booking.entities["slot_id"] == "PICK-20261128-E1"
    persisted = json.loads((tmp_path / "runtime_pickup_slots.json").read_text(encoding="utf-8"))
    assert booking.entities["rma_id"] == persisted["bookings"][0]["rma_id"]
    assert persisted["bookings"][0]["rma_id"] in results[23]["reply"]
    assert [ep.source_turn for ep in by_type["forget_event"]] == [24]

    # Filler turns create neither facts nor episodes.
    written_turns = {f.source_turn for f in memory.semantic.all_records(USER)} | {
        ep.source_turn for ep in memory.episodic.episodes(USER)}
    assert not written_turns & {1, 3, 11, 25}


def test_session_a_forget_leaves_no_trace_of_the_delivery_note_on_disk(tmp_path):
    _, memory, results, _, _ = run_session_a(tmp_path)
    assert "deleted your saved delivery note" in results[24]["reply"]
    assert memory.semantic.get_active(USER, "delivery_note") is None
    for path in (tmp_path / "memory").iterdir():
        assert "side door" not in path.read_bytes().decode("utf-8", "ignore").casefold(), path.name


def test_memory_calls_are_logged_with_the_task4_config_name(tmp_path):
    _, _, _, logger, _ = run_session_a(tmp_path)
    configs = {record["config"] for record in logger.records}
    assert configs == {"hybrid_mem"}
    assert {"extract", "importance", "nlu", "tool"} <= {record["call_type"] for record in logger.records}
    complaint_importance = [r for r in logger.of_type("importance") if r["extra"]["episode_type"] == "complaint"]
    assert complaint_importance[0]["turn"] == 10


def test_hybrid_config_keeps_dialogue_working_without_long_term_memory(tmp_path):
    _, memory, results, logger, _ = run_session_a(tmp_path, config="hybrid")
    assert memory.semantic.all_records(USER) == [] and memory.episodic.episodes(USER) == []
    assert "won't be saved" in results[2]["reply"]
    assert results[23]["completed"] is True  # the booking task itself is unaffected
    assert not logger.of_type("extract") and not logger.of_type("importance")
