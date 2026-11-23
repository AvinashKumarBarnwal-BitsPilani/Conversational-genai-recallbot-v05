"""Reproducibly replay one dataset session with real RecallBot tools and Qwen."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .. import logger as logger_module
from ..config import CHAT_MODEL, CONTEXT_WINDOW_TOKENS, DATA_DIR, RESULTS_DIR
from .dialogue_manager import DialogueManager
from .qwen_nlg import QwenNLG
from .qwen_nlg import DECODING_PRESETS
from .state_tracker import DialogueStateTracker
from .tools import RecallBotToolGateway


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_session_turns(session: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_path = DATA_DIR / "test_conversation.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    boundary = dataset.get("session_boundary", {}).get(f"Session {session}")
    if not isinstance(boundary, Mapping):
        raise ValueError(f"Dataset does not define Session {session}")
    first_turn = int(boundary["first_turn"])
    last_turn = int(boundary["last_turn"])
    turns = [
        dict(turn) for turn in dataset.get("turns", [])
        if turn.get("session") == session
        and turn.get("role") == "user"
        and first_turn <= int(turn.get("turn", -1)) <= last_turn
    ]
    turns.sort(key=lambda item: int(item["turn"]))
    expected = list(range(first_turn, last_turn + 1))
    actual = [int(turn["turn"]) for turn in turns]
    if actual != expected:
        raise ValueError(
            f"Session {session} must contain turns {expected}; found {actual}"
        )
    return dataset, turns


def _read_run_records(run_id: str) -> list[dict[str, Any]]:
    path = logger_module.LOG_PATH
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_id") == run_id:
                records.append(record)
    return records


def _write_manifest_entry(entry: dict[str, Any], *, purpose: str = "task2b") -> None:
    manifest_name = (
        "task2b_session_A_run_manifest.json"
        if purpose == "task2b" else "task2c_session_A_run_manifest.json"
    )
    manifest_path = RESULTS_DIR / manifest_name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": "data/test_conversation.json",
            "session": "A",
            "model": CHAT_MODEL,
            "runs": [],
        }
    runs = manifest.setdefault("runs", [])
    if any(run.get("run_id") == entry["run_id"] for run in runs):
        raise ValueError(f"Manifest already contains run ID {entry['run_id']!r}")
    runs.append(entry)
    _write_json_atomic(manifest_path, manifest)


def _booking_write_validation(
    write_events: list[dict[str, Any]],
    tool_records: list[dict[str, Any]],
    confirmation_turns: list[int],
) -> tuple[bool, bool | None, bool | None]:
    """Report booking safeguards without treating no attempt as a pass."""
    if not write_events:
        return False, None, None
    first_confirmation_turn = min(confirmation_turns, default=None)
    booking_records = [
        record for record in tool_records
        if record.get("extra", {}).get("tool_name") == "book_slot"
    ]
    guard_ok = all(
        event.get("request", {}).get("explicit_confirmation") is True
        for event in write_events
    )
    after_confirmation = all(
        first_confirmation_turn is not None
        and int(record.get("turn", 0)) >= first_confirmation_turn
        for record in booking_records
    )
    return True, guard_ok, after_confirmation


def replay_session(
    *, session: str, strategy: str, run_id: str, purpose: str = "task2b"
) -> dict[str, Any]:
    if session != "A":
        raise ValueError("Stage 4 replay is scoped to Session A")
    if strategy not in {"full", "trim", "summary"}:
        raise ValueError("strategy must be full, trim or summary")
    if purpose not in {"task2b", "task2c_snapshot"}:
        raise ValueError("purpose must be task2b or task2c_snapshot")
    dataset, source_turns = _load_session_turns(session)
    # Include the unique run ID so interrupted or superseded attempts remain
    # intact beside the valid evaluation transcript.
    prefix = "task2b" if purpose == "task2b" else "task2c_snapshot"
    transcript_path = RESULTS_DIR / f"{prefix}_{strategy}_session_{session}_{run_id}.json"
    if transcript_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing replay transcript: {transcript_path}"
        )
    logger_module.set_run_id(run_id, reject_existing=True)
    manifest_path = RESULTS_DIR / (
        "task2b_session_A_run_manifest.json"
        if purpose == "task2b" else "task2c_session_A_run_manifest.json"
    )
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(run.get("run_id") == run_id for run in existing_manifest.get("runs", [])):
            raise ValueError(f"Manifest already contains run ID {run_id!r}")

    user_id = str(dataset["user_id"])
    tools = RecallBotToolGateway()
    tools.reset_runtime_database()
    reset_state = json.loads(tools.runtime_slots_path.read_text(encoding="utf-8"))
    reset_fingerprint = hashlib.sha256(
        json.dumps(reset_state, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    tracker = DialogueStateTracker(session_id=session, user_id=user_id)
    nlg = QwenNLG()
    manager = DialogueManager(
        tracker=tracker,
        tools=tools,
        user_id=user_id,
        config=strategy,
        memory_strategy=strategy,
        nlg=nlg,
    )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    transcript: dict[str, Any] = {
        "run_id": run_id,
        "purpose": purpose,
        "strategy": strategy,
        "session": session,
        "user_id": user_id,
        "dataset": "data/test_conversation.json",
        "model": nlg.model,
        "prompt_version": nlg.prompt_version,
        "workflow_version": nlg.workflow_version,
        "database_reset_fingerprint": reset_fingerprint,
        "expected_user_turns": len(source_turns),
        "status": "running",
        "turns": [
            {
                "turn": int(source_turn["turn"]),
                "status": "not_started",
                "user_message": source_turn["text"],
            }
            for source_turn in source_turns
        ],
    }
    _write_json_atomic(transcript_path, transcript)

    processed = 0
    failed = 0
    fallback_turns = 0
    confirmation_turns: list[int] = []
    for index, source_turn in enumerate(source_turns):
        turn = int(source_turn["turn"])
        row = transcript["turns"][index]
        first_tool_event = len(tools.call_history)
        print(f"[RecallBot replay {run_id}] Turn {turn}/25 started")
        try:
            result = manager.handle_turn(str(source_turn["text"]), turn=turn)
            tool_results = deepcopy(tools.call_history[first_tool_event:])
            row.update({
                "status": "processed",
                "assistant_response": result["reply"],
                "intent": result["intent"],
                "nlg_status": result["nlg_status"],
                "nlg_error": result.get("nlg_error"),
                "dialogue_state": result["state"],
                "tool_calls": result["tool_calls"],
                "tool_results": tool_results,
                "sources": result.get("sources", []),
                "completed": result.get("completed", False),
            })
            if result["intent"] == "confirm_action":
                confirmation_turns.append(turn)
            if str(result["nlg_status"]).startswith("fallback:"):
                fallback_turns += 1
            processed += 1
            print(
                f"[RecallBot replay {run_id}] Turn {turn}/25 processed; "
                f"intent={result['intent']}; nlg={result['nlg_status']}; tools={len(tool_results)}"
            )
        except Exception as exc:
            failed += 1
            row.update({
                "status": "failed_unhandled",
                "error_type": type(exc).__name__,
                "error": " ".join(str(exc).split())[:300],
                "tool_results": deepcopy(tools.call_history[first_tool_event:]),
            })
            for remaining in transcript["turns"][index + 1:]:
                remaining["status"] = "not_run_after_unhandled_error"
            print(
                f"[RecallBot replay {run_id}] Turn {turn}/25 failed; "
                f"stopping because turn state may be incomplete ({type(exc).__name__})"
            )
            break
        finally:
            _write_json_atomic(transcript_path, transcript)

    records = _read_run_records(run_id)
    chat_records = [record for record in records if record.get("call_type") == "chat"]
    summary_records = [record for record in records if record.get("call_type") == "summary"]
    tool_records = [record for record in records if record.get("call_type") == "tool"]
    write_events = [event for event in tools.call_history if event.get("tool_name") == "book_slot"]
    booking_write_attempted, booking_guard_ok, booking_after_confirmation = (
        _booking_write_validation(write_events, tool_records, confirmation_turns)
    )
    runtime_database_after = json.loads(tools.runtime_slots_path.read_text(encoding="utf-8"))
    transcript.update({
        "status": "complete" if processed == len(source_turns) and failed == 0 else "incomplete",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "successfully_processed_user_turns": processed,
        "failed_turns": failed,
        "deterministic_fallback_turns": fallback_turns,
        "customer_facing_qwen_calls": len(chat_records),
        "successful_customer_facing_qwen_calls": sum(
            record.get("extra", {}).get("status") == "success" for record in chat_records
        ),
        "summary_generation_calls": len(summary_records),
        "successful_summary_generation_calls": sum(
            record.get("extra", {}).get("status") == "success" for record in summary_records
        ),
        "tool_calls": len(tool_records),
        "booking_write_events": write_events,
        "booking_write_attempted": booking_write_attempted,
        "confirmation_guard_verified": booking_guard_ok,
        "booking_after_explicit_confirmation": booking_after_confirmation,
        "call_records_for_run": len(records),
        "summary_cursor": (
            manager.rolling_summary_memory.last_summarized_exchange_count
            if manager.rolling_summary_memory is not None else None
        ),
        "runtime_database_after": runtime_database_after,
    })
    _write_json_atomic(transcript_path, transcript)
    snapshot_path: Path | None = None
    if transcript["status"] == "complete":
        snapshot_dir = RESULTS_DIR / "task2c_snapshots"
        snapshot_path = snapshot_dir / f"{strategy}_{run_id}.json"
        if snapshot_path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation snapshot: {snapshot_path}")
        summary_snapshot = (
            manager.rolling_summary_memory.to_snapshot()
            if manager.rolling_summary_memory is not None else None
        )
        snapshot_payload = {
            "snapshot_version": 1,
            "status": transcript["status"],
            "strategy": strategy,
            "run_id": run_id,
            "session": session,
            "user_id": user_id,
            "tracker": tracker.snapshot(),
            "conversation_history": manager.conversation_history.to_snapshot(),
            "rolling_summary": summary_snapshot,
            "versions": {
                "prompt": nlg.prompt_version,
                "workflow": nlg.workflow_version,
            },
            "model": nlg.model,
            "decoding": deepcopy(DECODING_PRESETS),
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "database_reset_fingerprint": reset_fingerprint,
            "final_database_state": runtime_database_after,
            "final_database_fingerprint": hashlib.sha256(
                json.dumps(runtime_database_after, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "source_transcript": str(transcript_path.relative_to(RESULTS_DIR.parent)),
        }
        _write_json_atomic(snapshot_path, snapshot_payload)
        transcript["task2c_snapshot"] = str(snapshot_path.relative_to(RESULTS_DIR.parent))
        _write_json_atomic(transcript_path, transcript)
    manifest_entry = {
        key: transcript[key] for key in (
            "run_id", "purpose", "strategy", "session", "user_id", "status", "started_at",
            "finished_at", "expected_user_turns", "successfully_processed_user_turns",
            "failed_turns", "customer_facing_qwen_calls", "summary_generation_calls",
            "database_reset_fingerprint", "prompt_version", "workflow_version",
        )
    }
    manifest_entry["model"] = nlg.model
    manifest_entry["transcript"] = str(transcript_path.relative_to(RESULTS_DIR.parent))
    if snapshot_path is not None:
        manifest_entry["task2c_snapshot"] = str(snapshot_path.relative_to(RESULTS_DIR.parent))
    _write_manifest_entry(manifest_entry, purpose=purpose)
    print(
        f"[RecallBot replay {run_id}] {transcript['status']}: "
        f"{processed}/{len(source_turns)} turns; failed={failed}; fallback={fallback_turns}; "
        f"chat={len(chat_records)}; summary={len(summary_records)}; tools={len(tool_records)}"
    )
    return transcript


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a dataset session with one memory strategy")
    parser.add_argument("--session", default="A")
    parser.add_argument("--strategy", choices=("full", "trim", "summary"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--purpose", choices=("task2b", "task2c_snapshot"), default="task2b",
        help="Keep Task 2(c) snapshot runs separate from the historical Task 2(b) manifest.",
    )
    args = parser.parse_args()
    result = replay_session(
        session=args.session, strategy=args.strategy, run_id=args.run_id, purpose=args.purpose
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
