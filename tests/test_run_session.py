"""Session A and Session B as two real OS processes sharing only the disk."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).resolve().parent / "session_runner_child.py"
REAL_LOG = ROOT / "logs" / "calls.jsonl"


def _run(session: str, tmp: Path) -> None:
    subprocess.run([sys.executable, str(CHILD), session, str(tmp)], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def two_sessions(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sessions")
    before = hashlib.sha256(REAL_LOG.read_bytes()).hexdigest() if REAL_LOG.exists() else None
    _run("A", tmp)
    _run("B", tmp)
    after = hashlib.sha256(REAL_LOG.read_bytes()).hexdigest() if REAL_LOG.exists() else None
    load = lambda name: json.loads((tmp / name).read_text(encoding="utf-8"))  # noqa: E731
    return {
        "tmp": tmp, "real_log_unchanged": before == after,
        "manifest": load("memory/t3_test/session_manifest.json"),
        "A": load("sessions/t3_test_session_A.json"), "B": load("sessions/t3_test_session_B.json"),
        "log": [json.loads(line) for line in (tmp / "calls.jsonl").read_text(encoding="utf-8").splitlines()],
    }


def test_session_b_is_a_different_process_that_loaded_session_a_files(two_sessions):
    a, b = two_sessions["manifest"]["sessions"]["A"], two_sessions["manifest"]["sessions"]["B"]
    assert a["status"] == b["status"] == "complete"
    assert a["pid"] != b["pid"] and b["different_process_from_session_a"] is True
    assert b["loaded_hashes"] == a["saved_hashes"] and b["loaded_hashes_match_session_a"] is True
    assert two_sessions["real_log_unchanged"]


def test_session_a_books_with_a_real_reference_and_writes_a_reflection(two_sessions):
    turns = {row["turn"]: row for row in two_sessions["A"]["turns"]}
    assert turns[23]["completed"] is True and "RMA-" in turns[23]["assistant_response"]
    assert two_sessions["A"]["reflection"]["type"] == "reflection"
    assert "deleted your saved delivery note" in turns[24]["assistant_response"]


def test_session_b_recalls_from_disk_and_keeps_the_forgotten_fact_gone(two_sessions):
    turns = {row["turn"]: row for row in two_sessions["B"]["turns"]}
    assert "19:00-21:00" in turns[27]["assistant_response"]
    assert "deleted at your request" in turns[30]["assistant_response"]
    probes = {row["probe_id"]: row for row in two_sessions["B"]["probe_turns"]}
    assert "SMS" in probes["P03"]["assistant_response"]
    assert "Bengaluru" in probes["P02"]["assistant_response"]
    assert probes["P09"]["assistant_response"].startswith("I don't have that information")
    everything = json.dumps(two_sessions["B"]).casefold()
    assert "side door" not in everything
    for path in (two_sessions["tmp"] / "memory" / "t3_test").iterdir():
        assert "side door" not in path.read_bytes().decode("utf-8", "ignore").casefold(), path.name


def test_both_sessions_log_with_their_own_run_ids_and_task4_config(two_sessions):
    log = two_sessions["log"]
    assert {row["run_id"] for row in log} == {"t3_test_A", "t3_test_B"}
    assert {row["config"] for row in log} == {"hybrid_mem"}
    types_a = {row["call_type"] for row in log if row["run_id"] == "t3_test_A"}
    # embed rows are written by the real BGE embedder; this test injects a fake one.
    assert {"nlu", "tool", "extract", "importance", "reflect"} <= types_a
