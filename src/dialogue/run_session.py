"""Run one scripted session (A or B) in its own process (Task 3 / Task 4).

    python -m src.dialogue.run_session --session A --config hybrid_mem --experiment t3_hm_01
    python -m src.dialogue.run_session --session B --config hybrid_mem --experiment t3_hm_01

Session A starts from an empty memory directory and the seed pickup database.
Session B refuses to start unless Session A completed; it loads memory *only*
from disk and records the SHA-256 of every file it loaded, which must equal
the hashes Session A saved, together with both process IDs. After the
scripted Session B turns, every Session B probe question is asked verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import logger as logger_module
from ..config import CHAT_MODEL, DATA_DIR, DB_DIR, MEMORY_DIR, RESULTS_DIR
from ..memory.episodic_memory import DEFAULT_WEIGHTING, WEIGHTINGS
from ..memory.memory_manager import EMBEDDING_MODEL, MemoryManager
from .dialogue_manager import TASK4_CONFIGS, DialogueManager
from .qwen_nlg import QwenNLG
from .replay_session import _write_json_atomic
from .state_tracker import DialogueStateTracker
from .tools import RecallBotToolGateway

SESSIONS_DIR = RESULTS_DIR / "task3_sessions"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _session_turns(dataset: dict[str, Any], session: str) -> list[dict[str, Any]]:
    bounds = dataset["session_boundary"][f"Session {session}"]
    turns = [t for t in dataset["turns"]
             if t["session"] == session and bounds["first_turn"] <= t["turn"] <= bounds["last_turn"]]
    return sorted(turns, key=lambda t: t["turn"])


def _row(turn: int, text: str, result: dict[str, Any], probe_id: str | None = None) -> dict[str, Any]:
    return {
        "turn": turn,
        "probe_id": probe_id,
        "user_message": text,
        "assistant_response": result["reply"],
        "intent": result["intent"],
        "nlg_status": result["nlg_status"],
        "nlg_error": result.get("nlg_error"),
        "tool_calls": result["tool_calls"],
        "sources": result.get("sources", []),
        "completed": result.get("completed", False),
        "dialogue_state": result["state"],
        "context_assembly": result.get("context_assembly"),
    }


def run_session(*, session: str, config: str, experiment: str, use_nlg: bool = True,
                ask_probes: bool = True, memory_root: Path | None = None, db_root: Path | None = None,
                sessions_dir: Path | None = None, memory_kwargs: dict[str, Any] | None = None,
                tools_kwargs: dict[str, Any] | None = None, weighting: str = DEFAULT_WEIGHTING) -> dict[str, Any]:
    """``*_root``/``*_kwargs`` exist for isolated tests; real runs use the defaults."""
    if session not in {"A", "B"}:
        raise ValueError("session must be A or B")
    if config not in TASK4_CONFIGS:
        raise ValueError(f"config must be one of {sorted(TASK4_CONFIGS)}")
    memory_dir = (memory_root or MEMORY_DIR / "runs") / experiment
    manifest_path = memory_dir / "session_manifest.json"
    sessions_dir = sessions_dir or SESSIONS_DIR
    transcript_path = sessions_dir / f"{experiment}_session_{session}.json"
    if transcript_path.exists():
        raise FileExistsError(f"Refusing to overwrite {transcript_path}")

    if session == "A":
        if memory_dir.exists() and any(memory_dir.iterdir()):
            raise FileExistsError(f"Session A needs an empty memory directory: {memory_dir}")
        manifest: dict[str, Any] = {"experiment": experiment, "config": config, "model": CHAT_MODEL,
                                    "embedding_model": EMBEDDING_MODEL, "sessions": {}}
        loaded_hashes: dict[str, str] = {}
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Session A manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        a_info = manifest.get("sessions", {}).get("A", {})
        if a_info.get("status") != "complete" or manifest.get("config") != config:
            raise RuntimeError("Session B requires a complete Session A of the same config")
        if "B" in manifest["sessions"]:
            raise FileExistsError("Session B already ran for this experiment")
        loaded_hashes = {}

    run_id = f"{experiment}_{session}"
    logger_module.set_run_id(run_id, reject_existing=True)
    dataset = json.loads((DATA_DIR / "test_conversation.json").read_text(encoding="utf-8"))
    user_id = dataset["user_id"]

    tools = RecallBotToolGateway(
        runtime_slots_path=(db_root or DB_DIR / "runs") / experiment / "runtime_pickup_slots.json",
        **(tools_kwargs or {}))
    if session == "A":
        tools.reset_runtime_database()
    memory = MemoryManager(memory_dir, config=config, user=user_id, weighting=weighting, **(memory_kwargs or {}))
    if session == "B":
        loaded_hashes = memory.file_hashes()
    # One BGE model instance serves both KB search and memory search.
    tools.embedder_factory = lambda _name: memory._get_embedder()

    tracker = DialogueStateTracker(session_id=session, user_id=user_id)
    manager = DialogueManager(tracker=tracker, tools=tools, user_id=user_id, config=config,
                              nlg=QwenNLG() if use_nlg else None, memory=memory)

    info: dict[str, Any] = {"run_id": run_id, "pid": os.getpid(), "parent_pid": os.getppid(),
                            "started_at": _now(), "status": "running", "use_nlg": use_nlg,
                            "episodic_weighting": memory.weights.name}
    if session == "B":
        saved = manifest["sessions"]["A"]["saved_hashes"]
        info["loaded_hashes"] = loaded_hashes
        info["loaded_hashes_match_session_a"] = loaded_hashes == saved
        info["session_a_pid"] = manifest["sessions"]["A"]["pid"]
        info["different_process_from_session_a"] = info["session_a_pid"] != info["pid"]
        if not info["loaded_hashes_match_session_a"]:
            raise RuntimeError("Memory on disk changed between Session A and Session B")

    transcript: dict[str, Any] = {"experiment": experiment, "config": config, "session": session,
                                  "run_id": run_id, "user_id": user_id, "model": CHAT_MODEL, "turns": []}
    turns = _session_turns(dataset, session)
    for source in turns:
        print(f"[run_session {run_id}] turn {source['turn']}: {source['text']}")
        result = manager.handle_turn(source["text"], turn=int(source["turn"]))
        transcript["turns"].append(_row(int(source["turn"]), source["text"], result))
        print(f"    -> [{result['intent']} | {result['nlg_status']}] {result['reply'][:160]}")
        _write_json_atomic(transcript_path, transcript)

    last_turn = int(turns[-1]["turn"])
    reflection = memory.write_reflection(session=session, last_turn=last_turn)
    transcript["reflection"] = reflection.to_dict() if reflection is not None else None

    if session == "B" and ask_probes:
        probes = json.loads((DATA_DIR / "probes.json").read_text(encoding="utf-8"))
        transcript["probe_turns"] = []
        for offset, probe in enumerate((p for p in probes if p["session"] == "B"), start=1):
            turn = last_turn + 1 + offset  # logical turns after the reflection timestamp
            result = manager.handle_turn(probe["question"], turn=turn)
            transcript["probe_turns"].append(_row(turn, probe["question"], result, probe_id=probe["probe_id"]))
            print(f"[probe {probe['probe_id']}] {probe['question']}\n    -> {result['reply'][:200]}")
            _write_json_atomic(transcript_path, transcript)

    info["saved_hashes"] = memory.save()
    info.update({"finished_at": _now(), "status": "complete", "turns": len(turns),
                 "memory_counts": {
                     "active_facts": len(memory.semantic.active_facts(user_id)),
                     "all_fact_records": len(memory.semantic.all_records(user_id)),
                     "episodes": len(memory.episodic.episodes(user_id)),
                     "reflections": len(memory.episodic.episodes(user_id, ["reflection"])),
                     "tombstones": len(memory.tombstones)}})
    try:
        info["transcript"] = str(transcript_path.relative_to(RESULTS_DIR.parent))
    except ValueError:
        info["transcript"] = str(transcript_path)
    transcript["session_info"] = deepcopy(info)
    _write_json_atomic(transcript_path, transcript)
    manifest["sessions"][session] = info
    _write_json_atomic(manifest_path, manifest)
    print(f"[run_session {run_id}] complete; memory saved to {memory_dir}")
    return transcript


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one RecallBot session in this process")
    parser.add_argument("--session", choices=("A", "B"), required=True)
    parser.add_argument("--config", choices=sorted(TASK4_CONFIGS), required=True)
    parser.add_argument("--experiment", required=True, help="Shared by Session A and B; names the memory directory")
    parser.add_argument("--disable-nlg", action="store_true", help="Deterministic replies (no chat model)")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--weighting", choices=sorted(WEIGHTINGS), default=DEFAULT_WEIGHTING,
                        help="Episodic retrieval weighting (recency/importance/relevance)")
    args = parser.parse_args()
    run_session(session=args.session, config=args.config, experiment=args.experiment,
                use_nlg=not args.disable_nlg, ask_probes=not args.skip_probes, weighting=args.weighting)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
