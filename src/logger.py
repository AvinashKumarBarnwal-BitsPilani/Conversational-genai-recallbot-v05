"""Append-only mandatory call logger for RecallBot.

Every LLM, embedding and tool invocation should call log_call exactly once after
it returns (or fails, with the measured latency and zero/known token counts).
Never edit or delete calls.jsonl by hand. Start a new run_id for a new experiment.
"""
from __future__ import annotations
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOG_DIR, VALID_CALL_TYPES, VALID_CONFIGS

RUN_ID = os.getenv("RECALLBOT_RUN_ID") or f"r-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
LOG_PATH = LOG_DIR / "calls.jsonl"


def log_call(*, config: str, user: str, session: str, turn: int,
             call_type: str, model: str, input_tokens: int = 0,
             output_tokens: int = 0, retrieved_tokens: int = 0,
             start_time: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write one measured call record and return the record written."""
    if config not in VALID_CONFIGS:
        raise ValueError(f"Invalid config {config!r}; expected one of {sorted(VALID_CONFIGS)}")
    if call_type not in VALID_CALL_TYPES:
        raise ValueError(f"Invalid call_type {call_type!r}; expected one of {sorted(VALID_CALL_TYPES)}")
    for name, value in {"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "retrieved_tokens": retrieved_tokens, "turn": turn}.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
    record: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "run_id": RUN_ID,
        "config": config,
        "user": user,
        "session": session,
        "turn": turn,
        "call_type": call_type,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_tokens": retrieved_tokens,
        "latency_ms": elapsed_ms,
    }
    if extra:
        # Keep only non-sensitive diagnostic metadata here; never log credentials or secrets.
        record["extra"] = extra
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def timer() -> float:
    """Capture immediately before an LLM, embedding or tool call."""
    return time.perf_counter()
