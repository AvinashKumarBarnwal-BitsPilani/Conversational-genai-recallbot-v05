from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.dialogue.nlu import analyze_utterance
from src.memory.fact_extractor import FactExtractor
from src.memory.forgetting import fingerprint
from src.memory.llm_calls import LoggedLLM
from src.memory.memory_manager import MemoryManager
from memory_fakes import FakeEmbedder, FakeOllamaClient, RecordingLogger

ROOT = Path(__file__).resolve().parents[1]
USER = "CUST-MAYA-1042"
FACT_TURNS = [
    (2, "My name is Maya Rao. Please call me Maya."),
    (4, "Email is the best way to send me routine order updates."),
    (6, "If I am not home, leave small parcels at the side door when the carrier permits it."),
    (7, "My preferred delivery time is between 6 pm and 9 pm."),
    (19, "My preferred delivery window has changed to 7 pm to 9 pm."),
    (20, "Please use SMS instead of email for future routine updates."),
]
FORGET_TEXT = "Please forget my saved delivery note about leaving parcels at the side door."


def make_manager(memory_dir, *, config="hybrid_mem", logger=None, reflection_client=None, extractor_client=None):
    logger = logger if logger is not None else RecordingLogger()
    extractor = FactExtractor(
        llm=LoggedLLM(client=extractor_client or FakeOllamaClient(), logger=logger),
        logger=logger, token_counter=lambda t: len(t.split()),
    )
    return MemoryManager(
        memory_dir, config=config, user=USER, embedder=FakeEmbedder(), extractor=extractor,
        reflection_llm=LoggedLLM(client=reflection_client or FakeOllamaClient(), logger=logger),
        logger=logger, token_counter=lambda t: len(t.split()),
    )


def run_turn(manager, turn, text, session="A"):
    manager.set_call_context(user=USER, session=session, turn=turn)
    return manager.observe_user_turn(text, analyze_utterance(text))


def seed_session_a(manager):
    for turn, text in FACT_TURNS:
        run_turn(manager, turn, text)
    manager.set_call_context(user=USER, session="A", turn=10)
    manager.record_episode("complaint", "Customer reported a late delivery for order ORD-88216; no formal case created.",
                           entities={"order_id": "ORD-88216"})
    manager.set_call_context(user=USER, session="A", turn=23)
    manager.record_episode("booking", "Return pickup booked for SKU-AUR-220 (ORD-74021) on 2026-11-28 18:00-20:00, "
                           "RMA RMA-TEST-1; parcel is collected from the side door.",
                           entities={"order_id": "ORD-74021", "rma_id": "RMA-TEST-1"})


def all_memory_text(memory_dir: Path) -> str:
    return "\n".join(path.read_bytes().decode("utf-8", "ignore") for path in memory_dir.iterdir() if path.is_file())


def test_session_a_facts_supersede_and_write_linked_preference_episodes(tmp_path):
    manager = make_manager(tmp_path)
    seed_session_a(manager)
    active = {f.key: f.value for f in manager.semantic.active_facts(USER)}
    assert active == {
        "preferred_name": "Maya", "delivery_note": "side door",
        "preferred_delivery_window": "19:00-21:00", "notification_channel": "SMS",
    }
    superseded = {f.key: f.value for f in manager.semantic.all_records(USER) if f.status == "superseded"}
    assert superseded == {"preferred_delivery_window": "18:00-21:00", "notification_channel": "email"}
    updates = manager.episodic.episodes(USER, ["preference_update"])
    assert {ep.source_turn for ep in updates} == {19, 20}
    assert all(len(ep.linked_memory_ids) == 2 for ep in updates)
    window = manager.semantic.get_active(USER, "preferred_delivery_window")
    assert window.date == "2026-11-21T10:36:00+05:30" and window.source_turn == 19


def test_config_gating_controls_which_memory_is_written(tmp_path):
    for config, expect_facts, expect_episodes in (("hybrid", 0, 0), ("hybrid_sem", 4, 0), ("hybrid_mem", 4, 6)):
        logger = RecordingLogger()
        manager = make_manager(tmp_path / config, config=config, logger=logger)
        for turn, text in FACT_TURNS:
            run_turn(manager, turn, text)
        assert len(manager.semantic.active_facts(USER)) == expect_facts
        assert len(manager.episodic.episodes(USER)) == expect_episodes
        if config == "hybrid":
            assert logger.records == []  # a disabled memory type costs nothing
        assert len(logger.of_type("importance")) == expect_episodes


def test_retrieval_returns_current_fact_and_relevant_episode(tmp_path):
    manager = make_manager(tmp_path)
    seed_session_a(manager)
    manager.set_call_context(user=USER, session="B", turn=27)
    facts = manager.retrieve_facts("What delivery window do I currently prefer?", top_k=1)
    assert facts[0][0].value == "19:00-21:00"
    episodes = manager.retrieve_episodes("What problem with a late delivery did I register for ORD-88216?", top_k=1)
    assert episodes[0].episode.type == "complaint"


def test_forget_cascade_removes_fact_index_linked_episodes_and_redacts_others(tmp_path):
    manager = make_manager(tmp_path)
    seed_session_a(manager)
    note = manager.semantic.get_active(USER, "delivery_note")
    linked_episode = next(ep for ep in manager.episodic.episodes(USER) if note.memory_id in ep.linked_memory_ids)

    manager.set_call_context(user=USER, session="A", turn=24)
    report = manager.forget(FORGET_TEXT, analyze_utterance(FORGET_TEXT))

    assert report["keys"] == ["delivery_note"]
    assert report["facts_removed"] == [note.memory_id]
    assert report["episodes_deleted"] == [linked_episode.episode_id]
    booking = manager.episodic.episodes(USER, ["booking"])[0]
    assert report["episodes_redacted"] == [booking.episode_id]
    assert "side door" not in booking.description and "[forgotten delivery note]" in booking.description
    assert note.memory_id not in manager.semantic.index
    assert manager.episodic.episodes(USER, ["forget_event"])[0].description == (
        "Customer asked RecallBot to forget their saved delivery note."
    )
    text = all_memory_text(tmp_path).casefold()
    assert "side door" not in text
    tombstone = json.loads((tmp_path / "forget_tombstones.json").read_text(encoding="utf-8"))["tombstones"][0]
    assert tombstone["fingerprints"] == [fingerprint("side door")]


def test_forgotten_fact_stays_gone_after_restart_in_a_new_process(tmp_path):
    manager = make_manager(tmp_path)
    seed_session_a(manager)
    manager.set_call_context(user=USER, session="A", turn=24)
    manager.forget(FORGET_TEXT, analyze_utterance(FORGET_TEXT))
    saved_hashes = manager.save()

    script = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from src.memory.memory_manager import MemoryManager\n"
        "m = MemoryManager(Path(sys.argv[1]), config='hybrid_mem', user=sys.argv[2], embedder=object(), logger=None)\n"
        "print(json.dumps({'pid': os.getpid(), 'hashes': m.file_hashes(),\n"
        "  'facts': {f.key: f.value for f in m.semantic.active_facts(sys.argv[2])},\n"
        "  'all_keys': [f.key for f in m.semantic.all_records(sys.argv[2])],\n"
        "  'index_ids': m.semantic.index.ids,\n"
        "  'episodes': [e.description for e in m.episodic.episodes(sys.argv[2])],\n"
        "  'leaks': m.contains_forgotten(' '.join(e.description for e in m.episodic.episodes(sys.argv[2])) + ' parcels at the side door')}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path), USER], cwd=ROOT,
                               capture_output=True, text=True, check=True)
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded["pid"] != os.getpid()
    assert loaded["hashes"] == saved_hashes
    assert "delivery_note" not in loaded["all_keys"]
    assert loaded["facts"]["preferred_delivery_window"] == "19:00-21:00"
    assert all("side door" not in text.casefold() for text in loaded["episodes"])
    # The tombstone fingerprint still recognises the value if it reappears in new text.
    assert loaded["leaks"] is True


def test_reflection_is_logged_stored_linked_and_cannot_reintroduce_a_forgotten_value(tmp_path):
    logger = RecordingLogger()
    reflection = ("Maya has an open return pickup RMA-TEST-1 for ORD-74021 on 2026-11-28 and prefers SMS; "
                  "she also likes parcels left at the side door.")
    manager = make_manager(tmp_path, logger=logger, reflection_client=FakeOllamaClient(reflection))
    seed_session_a(manager)
    manager.set_call_context(user=USER, session="A", turn=24)
    manager.forget(FORGET_TEXT, analyze_utterance(FORGET_TEXT))

    stored = manager.write_reflection(session="A", last_turn=25)

    assert stored.type == "reflection" and stored.importance == 6
    assert stored.timestamp == "2026-11-21T10:50:00+05:30"
    assert "side door" not in stored.description and "RMA-TEST-1" in stored.description
    linked_types = {manager.episodic.get(eid).type for eid in stored.linked_memory_ids}
    assert linked_types == {"booking", "complaint", "preference_update"}
    assert len(logger.of_type("reflect")) == 1
    sent = json.loads(manager.reflection_llm.client.requests[0]["messages"][1]["content"])
    assert "side door" not in json.dumps(sent).casefold()  # forgotten before reflection: never sent


def test_save_load_save_is_byte_stable(tmp_path):
    manager = make_manager(tmp_path)
    seed_session_a(manager)
    first = manager.save()
    reloaded = MemoryManager(tmp_path, config="hybrid_mem", user=USER, embedder=FakeEmbedder(), logger=None)
    assert reloaded.save() == first
