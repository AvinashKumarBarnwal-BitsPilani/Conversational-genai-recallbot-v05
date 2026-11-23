from __future__ import annotations

import json

from src.dialogue.conversation_history import ConversationHistory
from src.dialogue.rolling_summary import RollingSummaryMemory
from src.dialogue.state_tracker import DialogueStateTracker
from src.dialogue import probe_evaluation
from src.config import CONTEXT_WINDOW_TOKENS
from src.dialogue.qwen_nlg import DECODING_PRESETS


def _history(count: int = 3) -> ConversationHistory:
    history = ConversationHistory()
    for index in range(1, count + 1):
        history.record_exchange(
            {"role": "user", "content": f"user turn {index}"},
            {"role": "assistant", "content": f"assistant turn {index}"},
        )
    return history


def test_history_tracker_and_redaction_snapshots_round_trip(tmp_path):
    history = ConversationHistory()
    for index in range(1, 5):
        content = "Please use the side door" if index == 1 else f"user turn {index}"
        history.record_exchange(
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"assistant turn {index}"},
        )
    memory = RollingSummaryMemory(history)
    memory.add_redaction("side door")
    summary_inputs = []

    def summarize(previous, aged):
        summary_inputs.extend(aged)
        return "Actual generated summary of the first two exchanges"

    assert memory.update_summary(summarize) is True
    assert len(summary_inputs) == 2
    assert "side door" not in json.dumps(summary_inputs).casefold()

    history_copy = ConversationHistory.from_snapshot(history.to_snapshot())
    memory_copy = RollingSummaryMemory(history_copy)
    memory_copy.restore_snapshot(memory.to_snapshot())
    assert memory_copy.summary == memory.summary
    assert memory_copy.last_summarized_exchange_count == 2
    assert "side door" not in json.dumps(memory_copy.get_context()).casefold()
    assert history_copy.get_history("full") == history.get_history("full")

    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    tracker_copy = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    tracker_copy.restore_snapshot(tracker.snapshot())
    assert tracker_copy.snapshot() == tracker.snapshot()


def test_probe_evaluator_restores_a_fresh_copy_for_each_probe(tmp_path, monkeypatch):
    history = _history()
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    runtime_state = json.loads(
        (probe_evaluation.DATA_DIR.parent / "db" / "seed_pickup_slots.json").read_text(encoding="utf-8")
    )
    snapshot = {
        "snapshot_version": 1,
        "status": "complete",
        "strategy": "full",
        "run_id": "source-run",
        "session": "A",
        "user_id": "CUST-MAYA-1042",
        "tracker": tracker.snapshot(),
        "conversation_history": history.to_snapshot(),
        "rolling_summary": None,
        "versions": {"prompt": "system_prompt_v1", "workflow": "1.0"},
        "model": "qwen3:1.7b",
        "decoding": DECODING_PRESETS,
        "context_window_tokens": CONTEXT_WINDOW_TOKENS,
        "database_reset_fingerprint": "seed-fingerprint",
        "final_database_state": runtime_state,
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    class FakeNLG:
        def __init__(self, *, model):
            self.model = model
            self.context_window = CONTEXT_WINDOW_TOKENS
            self.prompt_version = "system_prompt_v1"
            self.workflow_version = "1.0"

    seen_history_lengths = []
    seen_summaries = []

    def fake_handle_turn(self, text, *, turn):
        seen_history_lengths.append(self.conversation_history.completed_exchange_count)
        seen_summaries.append(
            self.rolling_summary_memory.summary
            if self.rolling_summary_memory is not None else ""
        )
        self.conversation_history.record_exchange(
            {"role": "user", "content": text},
            {"role": "assistant", "content": "raw mocked response"},
        )
        return {
            "reply": "raw mocked response",
            "nlg_status": "qwen",
            "nlg_error": None,
            "state": self.tracker.snapshot(),
            "tool_calls": [],
            "sources": [],
        }

    monkeypatch.setattr(probe_evaluation, "QwenNLG", FakeNLG)
    monkeypatch.setattr(probe_evaluation.DialogueManager, "handle_turn", fake_handle_turn)
    monkeypatch.setattr(probe_evaluation.logger_module, "set_run_id", lambda *args, **kwargs: None)

    one = probe_evaluation.evaluate_probe(
        strategy="full", probe_id="P08", snapshot_path=snapshot_path,
        run_id="isolated_p08", output_dir=tmp_path / "out",
    )
    two = probe_evaluation.evaluate_probe(
        strategy="full", probe_id="P08", snapshot_path=snapshot_path,
        run_id="isolated_p08_again", output_dir=tmp_path / "out",
    )

    summary_history = _history()
    summary_memory = RollingSummaryMemory(summary_history)
    summary_memory.add_redaction("user turn 1")
    summary_memory.restore_snapshot({
        "summary": "Actual saved Summary strategy text",
        "last_summarized_exchange_count": 1,
        "redactions": ["user turn 1"],
    })
    summary_snapshot = dict(snapshot)
    summary_snapshot.update({
        "strategy": "summary",
        "conversation_history": summary_history.to_snapshot(),
        "rolling_summary": summary_memory.to_snapshot(),
    })
    summary_snapshot_path = tmp_path / "summary_snapshot.json"
    summary_snapshot_path.write_text(json.dumps(summary_snapshot), encoding="utf-8")
    probe_evaluation.evaluate_probe(
        strategy="summary", probe_id="P08", snapshot_path=summary_snapshot_path,
        run_id="isolated_summary_p08", output_dir=tmp_path / "out",
    )

    assert seen_history_lengths == [3, 3, 3]
    assert seen_summaries == ["", "", "Actual saved Summary strategy text"]
    assert one["assistant_response"] == two["assistant_response"] == "raw mocked response"
    assert one["correctness"] is None
    assert one["probe_id"] == "P08" and one["source_turn"] == 18
    assert one["expected"] == two["expected"]
    assert json.loads((tmp_path / "out" / "isolated_p08.json").read_text(encoding="utf-8"))["correctness"] is None


def test_probe_loader_rejects_session_b_as_official_session_a_probe():
    try:
        probe_evaluation.load_probe("P01")
    except ValueError as exc:
        assert "only evaluates the official Session A" in str(exc)
    else:
        raise AssertionError("Session B probe must not be scored in the Session A evaluation")


def test_supplemental_memory_diagnostics_use_isolated_history_without_tools(tmp_path, monkeypatch):
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    captured_calls = []

    class FakeNLG:
        def __init__(self, *, model):
            self.model = model
            self.context_window = CONTEXT_WINDOW_TOKENS
            self.prompt_version = "system_prompt_v1"
            self.workflow_version = "1.0"

        def set_call_context(self, **kwargs):
            self.call_context = kwargs

        def _system_message(self):
            return "same system prompt"

        def generate_reply(self, draft, **kwargs):
            captured_calls.append({"draft": draft, **kwargs, "call_context": self.call_context})
            return "mock raw response"

    monkeypatch.setattr(probe_evaluation, "QwenNLG", FakeNLG)
    monkeypatch.setattr(probe_evaluation.logger_module, "set_run_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        probe_evaluation,
        "_read_run_records",
        lambda run_id: [{
            "call_type": "chat", "input_tokens": 27, "output_tokens": 4,
            "retrieved_tokens": 0, "latency_ms": 510,
            "extra": {"status": "success", "purpose": "qwen_nlg_response"},
        }],
    )
    monkeypatch.setattr(
        probe_evaluation,
        "RecallBotToolGateway",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("direct diagnostic must not call tools")),
    )

    snapshot_paths = {}
    for strategy in ("full", "trim", "summary"):
        history = _history(4)
        summary_state = None
        if strategy == "summary":
            memory = RollingSummaryMemory(history)
            memory.restore_snapshot({
                "summary": "Actual generated summary text",
                "last_summarized_exchange_count": 2,
                "redactions": [],
            })
            summary_state = memory.to_snapshot()
        snapshot = {
            "snapshot_version": 1,
            "status": "complete",
            "strategy": strategy,
            "run_id": f"snapshot-{strategy}",
            "session": "A",
            "user_id": "CUST-MAYA-1042",
            "tracker": tracker.snapshot(),
            "conversation_history": history.to_snapshot(),
            "rolling_summary": summary_state,
            "versions": {"prompt": "system_prompt_v1", "workflow": "1.0"},
            "model": "qwen3:1.7b",
            "decoding": DECODING_PRESETS,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "database_reset_fingerprint": "seed-fingerprint",
            "final_database_state": {},
        }
        path = tmp_path / f"{strategy}.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        snapshot_paths[strategy] = path

    results = probe_evaluation.run_supplemental_memory_diagnostics(
        snapshot_paths,
        output_path=tmp_path / "supplemental.json",
        output_dir=tmp_path / "runs",
    )

    assert len(results) == 6
    assert {item["probe_id"] for item in results} == {"P01", "P03"}
    assert all(item["correctness"] is None for item in results)
    assert all(item["official_session_a_accuracy"] is False for item in results)
    assert [len(call["conversation_history"]) for call in captured_calls[:3]] == [8, 6, 4]
    assert captured_calls[2]["conversation_summary"] == "Actual generated summary text"
    assert len({json.dumps(call["state"], sort_keys=True) for call in captured_calls}) == 1
    assert all(call["evidence"] is None and call["sources"] == [] for call in captured_calls)
    assert all(item["token_usage"][0]["input_tokens"] == 27 for item in results)
