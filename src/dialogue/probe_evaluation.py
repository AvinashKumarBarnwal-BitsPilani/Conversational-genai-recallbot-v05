"""Isolated Task 2(c) probe execution from captured post-Session-A snapshots.

This module prepares individual real-Qwen evaluations; it does not assign
correctness labels. Each invocation gets a fresh tracker, memory objects and
runtime database copy so probes cannot contaminate each other.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import uuid
from typing import Any, Mapping

from .. import logger as logger_module
from ..config import DATA_DIR, RESULTS_DIR
from .conversation_history import ConversationHistory
from .dialogue_manager import DialogueManager
from .qwen_nlg import DECODING_PRESETS, QwenNLG
from .rolling_summary import RollingSummaryMemory
from .state_tracker import DialogueStateTracker
from .tools import RecallBotToolGateway
from .replay_session import _write_json_atomic
from .replay_session import _read_run_records


def load_probe(probe_id: str) -> dict[str, Any]:
    payload = json.loads((DATA_DIR / "probes.json").read_text(encoding="utf-8"))
    matches = [item for item in payload if item.get("probe_id") == probe_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one probe with ID {probe_id!r}")
    probe = dict(matches[0])
    if probe.get("session") != "A":
        raise ValueError(
            f"{probe_id} belongs to Session {probe.get('session')}; this evaluator "
            "only evaluates the official Session A short-term-memory probes"
        )
    return probe


def evaluate_probe(
    *,
    strategy: str,
    probe_id: str,
    snapshot_path: Path,
    run_id: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Restore one strategy snapshot and execute one unscored real-Qwen probe."""
    if strategy not in {"full", "trim", "summary"}:
        raise ValueError("strategy must be full, trim or summary")
    probe = load_probe(probe_id)
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if snapshot.get("strategy") != strategy:
        raise ValueError("snapshot strategy does not match requested strategy")
    if snapshot.get("status") != "complete":
        raise ValueError("probe evaluation requires a complete post-Session-A snapshot")
    if snapshot.get("session") != "A":
        raise ValueError("probe snapshots must be captured after Session A")
    if strategy == "summary" and not isinstance(snapshot.get("rolling_summary"), Mapping):
        raise ValueError("summary snapshot is missing its actual rolling-summary state")

    run_id = run_id or f"task2c_{strategy}_{probe_id}_{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores and hyphens")
    output_dir = Path(output_dir or (RESULTS_DIR / "task2c_probe_evaluations"))
    output_path = output_dir / f"{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite probe result: {output_path}")
    logger_module.set_run_id(run_id, reject_existing=True)

    user_id = str(snapshot["user_id"])
    session = str(snapshot["session"])
    tracker = DialogueStateTracker(session_id=session, user_id=user_id)
    tracker.restore_snapshot(snapshot["tracker"])
    history = ConversationHistory.from_snapshot(snapshot["conversation_history"])

    # Use an isolated copy of the post-replay database. This avoids changing
    # either the official runtime DB or another probe's database state.
    run_dir = output_dir / f"{run_id}_state"
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime_db_path = run_dir / "runtime_pickup_slots.json"
    final_db = snapshot.get("final_database_state")
    if not isinstance(final_db, Mapping):
        raise ValueError("snapshot is missing final database state")
    runtime_db_path.write_text(json.dumps(final_db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tools = RecallBotToolGateway(runtime_slots_path=runtime_db_path)
    nlg = QwenNLG(model=str(snapshot["model"]))
    if snapshot.get("decoding") != DECODING_PRESETS:
        raise ValueError("snapshot decoding configuration differs from the active Qwen configuration")
    if int(snapshot.get("context_window_tokens", 0)) != nlg.context_window:
        raise ValueError("snapshot context window differs from the active Qwen configuration")
    if nlg.prompt_version != snapshot["versions"]["prompt"] or nlg.workflow_version != snapshot["versions"]["workflow"]:
        raise ValueError("snapshot prompt/workflow versions differ from the active files")
    manager = DialogueManager(
        tracker=tracker,
        tools=tools,
        user_id=user_id,
        config=strategy,
        memory_strategy=strategy,
        nlg=nlg,
    )
    manager.conversation_history = history
    if strategy == "summary":
        manager.rolling_summary_memory = RollingSummaryMemory(history)
        manager.rolling_summary_memory.restore_snapshot(snapshot["rolling_summary"])

    source_turn = int(probe["turn"])
    execution_turn = max(source_turn, int(tracker.state.get("turn", 0)) + 1)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    failure_type = None
    try:
        result = manager.handle_turn(str(probe["question"]), turn=execution_turn)
    except Exception as exc:
        # Preserve an unhandled probe failure as an artifact too; do not
        # manufacture a customer-facing answer or correctness judgment.
        result = None
        failure_type = type(exc).__name__
    record = {
        "run_id": run_id,
        "experiment_id": "task2c",
        "strategy": strategy,
        "probe_id": probe["probe_id"],
        "category": probe["category"],
        "question": probe["question"],
        "expected": probe.get("expected"),
        "expected_behavior": probe.get("expected_behavior"),
        "source_turn": source_turn,
        "source_turns": deepcopy(probe.get("source_turns", [])),
        "execution_turn": execution_turn,
        "session": session,
        "user_id": user_id,
        "snapshot_run_id": snapshot["run_id"],
        "snapshot_path": str(Path(snapshot_path)),
        "snapshot_summary_cursor": (
            snapshot["rolling_summary"].get("last_summarized_exchange_count")
            if strategy == "summary" else None
        ),
        "database_reset_fingerprint": snapshot["database_reset_fingerprint"],
        "prompt_version": snapshot["versions"]["prompt"],
        "workflow_version": snapshot["versions"]["workflow"],
        "model": snapshot["model"],
        "decoding": deepcopy(snapshot["decoding"]),
        "context_window_tokens": snapshot["context_window_tokens"],
        "started_at": started_at,
        "status": "processed" if result is not None else "failed_unhandled",
        "assistant_response": result["reply"] if result is not None else None,
        "nlg_status": result["nlg_status"] if result is not None else "unhandled_failure",
        "nlg_error": result.get("nlg_error") if result is not None else failure_type,
        "state_after": result["state"] if result is not None else tracker.snapshot(),
        "tool_calls": result["tool_calls"] if result is not None else [],
        "sources": result.get("sources", []) if result is not None else [],
        "correctness": None,
    }
    _write_json_atomic(output_path, record)
    record["result_path"] = str(output_path)
    return record


def _load_diagnostic_probe(probe_id: str) -> dict[str, Any]:
    if probe_id not in {"P01", "P03"}:
        raise ValueError("supplemental memory diagnostics are limited to existing probes P01 and P03")
    payload = json.loads((DATA_DIR / "probes.json").read_text(encoding="utf-8"))
    matches = [item for item in payload if item.get("probe_id") == probe_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one probe with ID {probe_id!r}")
    return dict(matches[0])


def evaluate_memory_diagnostic(
    *,
    strategy: str,
    probe_id: str,
    snapshot_path: Path,
    run_id: str,
    output_dir: Path | None = None,
    shared_tracker_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Qwen directly with one memory strategy, bypassing DM routing/RAG."""
    if strategy not in {"full", "trim", "summary"}:
        raise ValueError("strategy must be full, trim or summary")
    probe = _load_diagnostic_probe(probe_id)
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if snapshot.get("status") != "complete" or snapshot.get("strategy") != strategy:
        raise ValueError("a complete snapshot for the requested strategy is required")
    if snapshot.get("session") != "A":
        raise ValueError("supplemental diagnostics require a post-Session-A snapshot")
    if snapshot.get("decoding") != DECODING_PRESETS:
        raise ValueError("snapshot decoding configuration differs from the active Qwen configuration")

    tracker_snapshot = deepcopy(dict(shared_tracker_snapshot or snapshot["tracker"]))
    if tracker_snapshot != snapshot["tracker"]:
        raise ValueError("strategy snapshot tracker differs from the shared Session A tracker")
    tracker = DialogueStateTracker(session_id=str(snapshot["session"]), user_id=str(snapshot["user_id"]))
    tracker.restore_snapshot(tracker_snapshot)
    history = ConversationHistory.from_snapshot(snapshot["conversation_history"])
    summary_text = ""
    if strategy == "full":
        selected_messages = history.get_history("full")
    elif strategy == "trim":
        selected_messages = history.get_history("trim")
    else:
        rolling = snapshot.get("rolling_summary")
        if not isinstance(rolling, Mapping):
            raise ValueError("summary diagnostic requires the captured rolling-summary state")
        summary_memory = RollingSummaryMemory(history)
        summary_memory.restore_snapshot(rolling)
        context = summary_memory.get_context()
        summary_text = context["summary"]
        selected_messages = context["messages"]

    nlg = QwenNLG(model=str(snapshot["model"]))
    if int(snapshot.get("context_window_tokens", 0)) != nlg.context_window:
        raise ValueError("snapshot context window differs from the active Qwen configuration")
    if nlg.prompt_version != snapshot["versions"]["prompt"] or nlg.workflow_version != snapshot["versions"]["workflow"]:
        raise ValueError("snapshot prompt/workflow versions differ from the active files")

    run_id = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores and hyphens")
    output_dir = Path(output_dir or (RESULTS_DIR / "task2c_supplemental_runs"))
    output_path = output_dir / f"{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite supplemental result: {output_path}")
    logger_module.set_run_id(run_id, reject_existing=True)

    session = str(snapshot["session"])
    user_id = str(snapshot["user_id"])
    execution_turn = int(tracker.snapshot().get("turn", 0)) + 1
    draft = (
        "Answer the question using only the supplied conversation context. "
        "If the context does not contain the requested fact, say that you cannot determine it. "
        "Do not use external or persistent memory."
    )
    nlg.set_call_context(
        user=user_id,
        session=session,
        turn=execution_turn,
        config=strategy,
        metadata={
            "experiment_id": "task2c_supplemental_memory",
            "probe_id": probe_id,
            "probe_dataset_session": probe.get("session"),
            "source_turn": probe.get("turn"),
            "source_turns": probe.get("source_turns", []),
            "evaluation_purpose": "supplemental_memory_diagnostic",
        },
    )
    error_type = None
    status = "success"
    response = None
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        response = nlg.generate_reply(
            draft,
            user_message=str(probe["question"]),
            intent="supplemental_memory_diagnostic",
            state=tracker_snapshot,
            evidence=None,
            sources=[],
            turn_type="open",
            conversation_history=selected_messages,
            conversation_summary=summary_text,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        status = (
            "context_budget_failure"
            if isinstance(exc, ValueError) and "above the safe input budget" in str(exc)
            else "generation_failure"
        )

    run_records = _read_run_records(run_id)
    token_usage = [
        {
            "call_type": record.get("call_type"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "retrieved_tokens": record.get("retrieved_tokens"),
            "latency_ms": record.get("latency_ms"),
            "extra": record.get("extra", {}),
        }
        for record in run_records
    ]
    dataset = json.loads((DATA_DIR / "test_conversation.json").read_text(encoding="utf-8"))
    turn_by_user_text = {
        item["text"]: int(item["turn"])
        for item in dataset.get("turns", [])
        if item.get("session") == "A" and item.get("role") == "user"
    }
    source_turns_in_context = [
        turn_by_user_text[message["content"]]
        for message in selected_messages
        if message.get("role") == "user" and message.get("content") in turn_by_user_text
    ]
    serialized_context = json.dumps(
        {"summary": summary_text, "messages": selected_messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    record = {
        "run_id": run_id,
        "experiment_id": "task2c_supplemental_memory",
        "evaluation_scope": "supplemental_post_session_a_short_term_memory_diagnostic",
        "official_session_a_accuracy": False,
        "probe_id": probe_id,
        "category": probe["category"],
        "probe_dataset_session": probe["session"],
        "strategy": strategy,
        "question": probe["question"],
        "expected": probe["expected"],
        "expected_behavior": probe.get("expected_behavior"),
        "source_turn": probe.get("turn"),
        "fact_source_turns": deepcopy(probe.get("source_turns", [])),
        "source_turns_in_selected_raw_context": source_turns_in_context,
        "execution_session": session,
        "execution_turn": execution_turn,
        "snapshot_run_id": snapshot["run_id"],
        "snapshot_path": str(Path(snapshot_path)),
        "tracker_snapshot_sha256": hashlib.sha256(
            json.dumps(tracker_snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "model": snapshot["model"],
        "prompt_version": snapshot["versions"]["prompt"],
        "workflow_version": snapshot["versions"]["workflow"],
        "context_window_tokens": snapshot["context_window_tokens"],
        "turn_type": "open",
        "summary_cursor": (
            snapshot["rolling_summary"].get("last_summarized_exchange_count")
            if strategy == "summary" else None
        ),
        "actual_summary": summary_text,
        "selected_history_messages": deepcopy(selected_messages),
        "selected_context_sha256": hashlib.sha256(serialized_context.encode("utf-8")).hexdigest(),
        "system_message": nlg._system_message(),
        "draft_response": draft,
        "status": status,
        "raw_response": response,
        "error_type": error_type,
        "token_usage": token_usage,
        "correctness": None,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json_atomic(output_path, record)
    record["result_path"] = str(output_path)
    return record


def run_supplemental_memory_diagnostics(
    snapshot_paths: Mapping[str, Path],
    *,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run P01/P03 independently across strategies; preserve raw, unjudged output."""
    expected_strategies = {"full", "trim", "summary"}
    if set(snapshot_paths) != expected_strategies:
        raise ValueError("provide exactly one post-Session-A snapshot for full, trim and summary")
    snapshots = {
        strategy: json.loads(Path(path).read_text(encoding="utf-8"))
        for strategy, path in snapshot_paths.items()
    }
    shared_tracker = snapshots["full"]["tracker"]
    for strategy, snapshot in snapshots.items():
        if snapshot.get("strategy") != strategy or snapshot.get("status") != "complete":
            raise ValueError(f"invalid completed snapshot for strategy {strategy}")
        if snapshot.get("tracker") != shared_tracker:
            raise ValueError("the three strategy runs do not have identical tracker snapshots")

    records: list[dict[str, Any]] = []
    for probe_id in ("P01", "P03"):
        for strategy in ("full", "trim", "summary"):
            run_id = f"task2c_supp_{strategy}_{probe_id}_{uuid.uuid4().hex[:12]}"
            records.append(evaluate_memory_diagnostic(
                strategy=strategy,
                probe_id=probe_id,
                snapshot_path=Path(snapshot_paths[strategy]),
                run_id=run_id,
                output_dir=output_dir,
                shared_tracker_snapshot=shared_tracker,
            ))
    aggregate_path = Path(output_path or (RESULTS_DIR / "task2c_supplemental_memory_diagnostics.json"))
    if aggregate_path.exists():
        raise FileExistsError(f"Refusing to overwrite supplemental diagnostics: {aggregate_path}")
    _write_json_atomic(aggregate_path, {
        "evaluation_scope": "supplemental_post_session_a_short_term_memory_diagnostic",
        "official_session_a_accuracy": False,
        "strategies": ["full", "trim", "summary"],
        "probe_ids": ["P01", "P03"],
        "correctness_policy": "unjudged; review raw responses semantically",
        "results": records,
    })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated Task 2(c) probe")
    parser.add_argument("--strategy", choices=("full", "trim", "summary"), required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = evaluate_probe(
        strategy=args.strategy,
        probe_id=args.probe_id,
        snapshot_path=args.snapshot,
        run_id=args.run_id,
    )
    print(f"Saved unscored raw probe response: {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
