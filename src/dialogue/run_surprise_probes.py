"""Run the instructor's surprise probes S1-S5 as Session C (Task 4(c)).

    python -m src.dialogue.run_surprise_probes --part 1 --experiment sp_before_hybrid_mem
    python -m src.dialogue.run_surprise_probes --part 2 --experiment sp_before_hybrid_mem

Session C starts after Sessions A and B of the final hybrid_mem run. Part 1
copies that run's memory and pickup database into a new experiment folder (so
``memory/`` and the source run stay untouched), then asks S1 and plants the S2
instruction. Part 2 must run in a new process: it reloads memory from disk,
checks the hashes Part 1 saved, and asks the S2 booking turns, S3, S4 and S5.
Every call is logged with session="C" and config="hybrid_mem".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import logger as logger_module
from ..config import CHAT_MODEL, DATA_DIR, DB_DIR, MEMORY_DIR, RESULTS_DIR
from ..memory.memory_manager import MemoryManager
from .dialogue_manager import DialogueManager
from .qwen_nlg import QwenNLG
from .replay_session import _write_json_atomic
from .run_session import _row
from .state_tracker import DialogueStateTracker
from .tools import RecallBotToolGateway

CONFIG = "hybrid_mem"
SESSION = "C"
SOURCE_EXPERIMENT = "t4fix_hybrid_mem"
OUT_DIR = RESULTS_DIR / "surprise_probes"
FIRST_TURN = 60  # after Session B (turns 26-40) and its probe turns (42-55)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _turns(part: int) -> list[tuple[str, str]]:
    probes = {p["id"]: p for p in json.loads((DATA_DIR / "surprise_probes.json").read_text(encoding="utf-8"))}
    if part == 1:
        return [("S1", probes["S1"]["text"]), ("S2", probes["S2"]["plant_text"])]
    return ([("S2", text) for text in probes["S2"]["after_restart_turns"]]
            + [(pid, probes[pid]["text"]) for pid in ("S3", "S4", "S5")])


def run_part(part: int, experiment: str, source: str = SOURCE_EXPERIMENT) -> dict[str, Any]:
    memory_dir = MEMORY_DIR / "runs" / experiment
    db_path = DB_DIR / "runs" / experiment / "runtime_pickup_slots.json"
    manifest_path = memory_dir / "session_manifest.json"
    transcript_path = OUT_DIR / f"{experiment}_session_C{part}.json"
    if transcript_path.exists():
        raise FileExistsError(f"Refusing to overwrite {transcript_path}")

    if part == 1:
        if memory_dir.exists():
            raise FileExistsError(f"Session C part 1 needs a new experiment folder: {memory_dir}")
        src_manifest = json.loads((MEMORY_DIR / "runs" / source / "session_manifest.json").read_text(encoding="utf-8"))
        if src_manifest["config"] != CONFIG or src_manifest["sessions"].get("B", {}).get("status") != "complete":
            raise RuntimeError("Session C needs a hybrid_mem run whose Sessions A and B completed")
        shutil.copytree(MEMORY_DIR / "runs" / source, memory_dir)
        db_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(DB_DIR / "runs" / source / "runtime_pickup_slots.json", db_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["session_c"] = {"copied_from": source, "copied_at": _now(), "parts": {}}
        expected_hashes = src_manifest["sessions"]["B"]["saved_hashes"]
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        part1 = manifest.get("session_c", {}).get("parts", {}).get("1", {})
        if part1.get("status") != "complete":
            raise RuntimeError("Session C part 2 requires a complete part 1")
        if "2" in manifest["session_c"]["parts"]:
            raise FileExistsError("Session C part 2 already ran for this experiment")
        expected_hashes = part1["saved_hashes"]

    run_id = f"{experiment}_C{part}"
    logger_module.set_run_id(run_id, reject_existing=True)
    user_id = json.loads((DATA_DIR / "test_conversation.json").read_text(encoding="utf-8"))["user_id"]
    db_hash_before = _sha256(db_path)

    tools = RecallBotToolGateway(runtime_slots_path=db_path)
    memory = MemoryManager(memory_dir, config=CONFIG, user=user_id)
    loaded_hashes = memory.file_hashes()
    if loaded_hashes != expected_hashes:
        raise RuntimeError("Memory on disk does not match the previous session's saved hashes")
    tools.embedder_factory = lambda _name: memory._get_embedder()
    tracker = DialogueStateTracker(session_id=SESSION, user_id=user_id)
    manager = DialogueManager(tracker=tracker, tools=tools, user_id=user_id, config=CONFIG,
                              nlg=QwenNLG(), memory=memory)

    info: dict[str, Any] = {"run_id": run_id, "pid": os.getpid(), "started_at": _now(),
                            "loaded_hashes": loaded_hashes, "loaded_hashes_match": True}
    if part == 2:
        info["part1_pid"] = manifest["session_c"]["parts"]["1"]["pid"]
        info["different_process_from_part1"] = info["part1_pid"] != info["pid"]
    transcript: dict[str, Any] = {"experiment": experiment, "config": CONFIG, "session": SESSION, "part": part,
                                  "run_id": run_id, "user_id": user_id, "model": CHAT_MODEL, "turns": []}
    first = FIRST_TURN if part == 1 else manifest["session_c"]["parts"]["1"]["last_turn"] + 1
    turn = first - 1
    for turn, (probe_id, text) in enumerate(_turns(part), start=first):
        print(f"[{run_id}] turn {turn} ({probe_id}): {text}")
        result = manager.handle_turn(text, turn=turn)
        transcript["turns"].append(_row(turn, text, result, probe_id=probe_id))
        print(f"    -> [{result['intent']} | {result['nlg_status']}] {result['reply'][:240]}")
        _write_json_atomic(transcript_path, transcript)

    info["saved_hashes"] = memory.save()
    info.update({"finished_at": _now(), "status": "complete", "last_turn": turn,
                 "db_sha256_before": db_hash_before, "db_sha256_after": _sha256(db_path),
                 "memory_counts": {"active_facts": len(memory.semantic.active_facts(user_id)),
                                   "all_fact_records": len(memory.semantic.all_records(user_id)),
                                   "episodes": len(memory.episodic.episodes(user_id))}})
    info["db_unchanged"] = info["db_sha256_before"] == info["db_sha256_after"]
    transcript["session_info"] = deepcopy(info)
    _write_json_atomic(transcript_path, transcript)
    manifest["session_c"]["parts"][str(part)] = info
    _write_json_atomic(manifest_path, manifest)
    print(f"[{run_id}] complete; transcript {transcript_path}")
    return transcript


def main() -> int:
    parser = argparse.ArgumentParser(description="Run surprise probes S1-S5 as Session C")
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source", default=SOURCE_EXPERIMENT)
    args = parser.parse_args()
    run_part(args.part, args.experiment, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
