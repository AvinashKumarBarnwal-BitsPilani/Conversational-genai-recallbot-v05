"""One-command demo of Task 3: long-term memory across two processes.

LIVE (runs Session A and Session B as two separate OS processes, then verifies):
    python -m src.dialogue.demo_memory

VERIFY ONLY (explains an existing run instantly; no chat-model calls):
    python -m src.dialogue.demo_memory --verify-only --experiment t3_hm_01

Sections follow the Task 6(a) video checklist: confirmed booking in Session A,
process restart, Session B recalling a fact and an episode, a forget request
honoured after restart, a cited knowledge-base answer, one failure case, and a
walkthrough of the episodic scoring function. Everything shown is read from the
run's transcripts, manifest and memory files; nothing is typed in.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, MEMORY_DIR, RESULTS_DIR

ROOT = Path(__file__).resolve().parents[2]
USER = "CUST-MAYA-1042"
WIDTH = 100


def banner(number: str, title: str, covers: str = "") -> None:
    print("\n" + "=" * WIDTH)
    print(f" {number}. {title}")
    if covers:
        print(f"    (video checklist: {covers})")
    print("=" * WIDTH)


def wrap(text: str, indent: str = "      ", width: int = WIDTH - 8) -> str:
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(indent + item for item in lines)


def run_session_process(session: str, experiment: str) -> None:
    """Start one session in a NEW Python process and stream its conversation."""
    command = [sys.executable, "-m", "src.dialogue.run_session", "--session", session,
               "--config", "hybrid_mem", "--experiment", experiment, "--weighting", "A"]
    print(f"\n$ {' '.join(command[1:])}\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=env)
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(("[run_session", "    ->", "[probe")) or "Traceback" in line or "Error" in line:
            print(line.rstrip())
    if process.wait() != 0:
        raise SystemExit(f"Session {session} failed (exit code {process.returncode}).")


def load_run(experiment: str) -> dict:
    sessions = RESULTS_DIR / "task3_sessions"
    memory_dir = MEMORY_DIR / "runs" / experiment
    return {
        "memory_dir": memory_dir,
        "manifest": json.loads((memory_dir / "session_manifest.json").read_text(encoding="utf-8")),
        "A": json.loads((sessions / f"{experiment}_session_A.json").read_text(encoding="utf-8")),
        "B": json.loads((sessions / f"{experiment}_session_B.json").read_text(encoding="utf-8")),
        "facts": json.loads((memory_dir / "semantic_facts.json").read_text(encoding="utf-8"))["records"],
        "episodes": json.loads((memory_dir / "episodes.json").read_text(encoding="utf-8"))["episodes"],
        "tombstones": json.loads((memory_dir / "forget_tombstones.json").read_text(encoding="utf-8"))["tombstones"],
        "probes": {p["probe_id"]: p for p in json.loads((DATA_DIR / "probes.json").read_text(encoding="utf-8"))},
    }


def show_session_a(run: dict) -> None:
    banner("1", "Session A: facts, events, a confirmed booking and a forget request",
           "Session A with a confirmed booking")
    turns = {row["turn"]: row for row in run["A"]["turns"]}
    for turn in (2, 7, 10, 17, 19, 20, 22, 23, 24):
        row = turns[turn]
        print(f"\n  Turn {turn:>2} USER : {row['user_message']}")
        print(wrap(f"BOT  : {row['assistant_response']}", indent="          "))
    booking = [c for c in turns[23]["tool_calls"] if c.get("tool") == "book_slot"]
    print(f"\n  -> book_slot ran only at turn 23, after the explicit 'yes': {booking}")


def show_restart(run: dict) -> None:
    banner("2", "Process restart: Session B is a different process that reads memory from disk",
           "a process restart")
    a, b = run["manifest"]["sessions"]["A"], run["manifest"]["sessions"]["B"]
    print(f"  Session A: PID {a['pid']:<7} {a['started_at']} -> {a['finished_at']}")
    print(f"  Session B: PID {b['pid']:<7} {b['started_at']} -> {b['finished_at']}")
    print(f"  Different process: {b['different_process_from_session_a']}")
    print(f"\n  Memory files saved by A (SHA-256)        loaded by B")
    for name, digest in sorted(a["saved_hashes"].items()):
        same = "same" if b["loaded_hashes"].get(name) == digest else "DIFFERENT"
        print(f"    {name:<26} {digest[:20]}...   {same}")
    print(f"\n  All loaded hashes match Session A: {b['loaded_hashes_match_session_a']}")


def show_memory(run: dict) -> None:
    banner("3", f"What is stored on disk: {run['memory_dir'].relative_to(ROOT)}",
           "semantic + episodic memory")
    print("  SEMANTIC MEMORY (keyed facts; an update supersedes, never duplicates)")
    for fact in run["facts"]:
        mark = "ACTIVE    " if fact["status"] == "active" else "superseded"
        by = f" -> {fact['superseded_by']}" if fact.get("superseded_by") else ""
        print(f"    {fact['memory_id']}  {mark}  turn {fact['source_turn']:>2}  {fact['key']:<26} = "
              f"{fact['value']:<18} ({fact['extraction_method']}){by}")
    print("\n  EPISODIC MEMORY (time-stamped events with rule-based importance 1-10)")
    for ep in run["episodes"]:
        text = ep["description"] if ep["type"] != "reflection" else "(reflection) " + ep["description"]
        print(f"    {ep['episode_id']}  {ep['timestamp'][:16].replace('T', ' ')}  imp {ep['importance']}  "
              f"{ep['type']:<17} {text[:58]}{'...' if len(text) > 58 else ''}")
    reflection = next((ep for ep in run["episodes"] if ep["type"] == "reflection"), None)
    if reflection:
        print("\n  END-OF-SESSION REFLECTION written at the end of Session A:")
        print(wrap(reflection["description"]))
    for stone in run["tombstones"]:
        print(f"\n  FORGET TOMBSTONE: key '{stone['key']}' forgotten at turn {stone['source_turn']}; "
              f"stores only a hash: {stone['fingerprints'][0]['sha256'][:24]}...")


def answer_of(run: dict, probe_id: str) -> dict:
    return next(row for row in run["B"]["probe_turns"] if row["probe_id"] == probe_id)


def show_session_b(run: dict) -> None:
    banner("4", "Session B (new process) answers from long-term memory and the knowledge base",
           "Session B recalls a fact and an episode; forget honoured after restart; KB answer with cited chunk")
    showcase = [
        ("P01", "Semantic recall, after an update (old 18:00-21:00 superseded)"),
        ("P03", "Updated fact (email -> SMS)"),
        ("P05", "Episodic recall of the Session A booking"),
        ("P06", "Episodic recall of the complaint"),
        ("P04", "FORGOTTEN fact asked after the restart"),
        ("P07", "Knowledge-base answer with cited chunks"),
        ("P09", "Never said and not in the KB: exact fallback sentence"),
    ]
    for probe_id, what in showcase:
        probe, row = run["probes"][probe_id], answer_of(run, probe_id)
        print(f"\n  [{probe_id}] {what}")
        print(f"      Q: {probe['question']}")
        print(wrap(f"A: {row['assistant_response']}"))
    leaks = [p.name for p in run["memory_dir"].iterdir() if p.is_file()
             and "side door" in p.read_bytes().decode("utf-8", "ignore").casefold()]
    in_b = "side door" in json.dumps(run["B"]).casefold()
    print(f"\n  Forgotten value 'side door' found in memory files: {leaks or 'none'}; "
          f"in Session B transcript: {'YES' if in_b else 'no'}")


FAILURE_NOTES = {
    "P11": [
        "Stage: knowledge-base retrieval, then the reply. 'What happens next after the pickup?' shares no",
        "words with the refund-lifecycle chunk KB-003-C128-0001 (ranked 12th). Task 4(b) added a query",
        "rewrite that now ranks it 1st, yet the 1.7B model still says the carrier inspects the item:",
        "the right text is in the prompt but misread (prompt/NLG fault). Evidence:",
        "results/task4_error_analysis.json and docs/task-evidence/Task-4-Evaluation-and-Error-Analysis.md.",
    ],
    "P13": [
        "Stage: memory write, surfacing as an episodic ranking error. The decision episode did not",
        "mention the 'USB-C charger', so the booking episode (importance 9) ranked first. Fixed in",
        "Task 4(b) by storing the customer's own words (run t4fix_hybrid_mem). Evidence: Q-E03 in",
        "results/task3_episodic_weight_comparison.csv and results/task4_error_analysis.json.",
    ],
}


def show_failure(run: dict, probe_id: str) -> None:
    banner("5", f"An honest failure case ({probe_id}) and where it comes from", "one failure case")
    probe, row = run["probes"][probe_id], answer_of(run, probe_id)
    print(f"  Q:        {probe['question']}")
    print(f"  Expected: {probe['expected']}")
    print(wrap(f"Got:      {row['assistant_response']}", indent="  "))
    print()
    for line in FAILURE_NOTES.get(probe_id, ["See docs/task-evidence/Task-4-Evaluation-and-Error-Analysis.md."]):
        print(f"  {line}")


def show_scoring(experiment: str, run: dict, query: str) -> None:
    banner("6", "How an episode is scored: recency + importance + relevance (src/memory/episodic_memory.py)",
           "walkthrough of the episodic scoring function")
    from .. import logger as logger_module
    from ..memory.clock import SessionClock
    from ..memory.episodic_memory import RECENCY_DECAY_PER_HOUR, WEIGHTINGS, score_episodes
    from ..memory.memory_manager import EPISODE_MIN_RELEVANCE_QUERY, MemoryManager

    logger_module.set_run_id(f"{experiment}_demo_{datetime.now().strftime('%Y%m%d%H%M%S')}", reject_existing=True)
    memory = MemoryManager(run["memory_dir"], config="hybrid_mem", user=USER, autosave=False)
    memory.set_call_context(user=USER, session="B", turn=28)
    now = SessionClock.for_session("B").iso_at(28)
    weights = WEIGHTINGS["A"]
    vector = memory._embed([query], is_query=True, purpose="demo_scoring_walkthrough")[0]
    relevance = memory.episodic.index.similarities(vector)
    ranked = score_episodes(memory.episodic.episodes(USER), relevance, now=now, weights=weights,
                            min_relevance=EPISODE_MIN_RELEVANCE_QUERY)
    print(f"  Query: \"{query}\"      now = {now} (Session B logical clock)")
    print(f"  R = {RECENCY_DECAY_PER_HOUR} ** hours_since   I = importance / 10   S = cosine(query, episode)")
    print(f"  Each signal is min-max normalised over the candidates, then")
    print(f"  score = {weights.recency:.2f} x R_norm + {weights.importance:.2f} x I_norm + {weights.relevance:.2f} x S_norm"
          f"   (weighting A, chosen on the E probes)\n")
    print(f"  {'rank':<5}{'episode':<10}{'type':<18}{'R':>7}{'R_norm':>8}{'I':>6}{'I_norm':>8}"
          f"{'S':>7}{'S_norm':>8}{'score':>8}")
    for rank, item in enumerate(ranked[:5], start=1):
        print(f"  {rank:<5}{item.episode.episode_id:<10}{item.episode.type:<18}{item.recency:>7.3f}"
              f"{item.recency_norm:>8.3f}{item.importance:>6.1f}{item.importance_norm:>8.3f}"
              f"{item.relevance:>7.3f}{item.relevance_norm:>8.3f}{item.score:>8.3f}")
    top = ranked[0].episode
    print(f"\n  Winner {top.episode_id} ({top.type}): {top.description[:85]}...")


def show_budget(run: dict) -> None:
    banner("7", "Twist E: every Session B prompt fits the 2,048-token window", "small context window")
    print(f"  {'turn':<9}{'intent':<32}{'input tokens':>14}{'budget':>8}  memories / KB chunks in the prompt")
    for row in run["B"]["turns"] + run["B"].get("probe_turns", []):
        record = row.get("context_assembly")
        if not record:
            continue
        ids = [i["id"] for i in record["included"] if i["component"] in ("long_term_memory", "retrieved_knowledge")]
        label = f"{row['turn']}" + (f" {row['probe_id']}" if row.get("probe_id") else "")
        print(f"  {label:<9}{row['intent']:<32}{record['total_input_tokens']:>14,}{record['input_budget']:>8,}  "
              f"{', '.join(ids)[:44]}")
    print("\n  2,048 window = up to 1,856 input + 160 reply + 32 safety. Items that do not fit are cut whole and")
    print("  logged; see results/task3_context_assembly.json (stress replay: 8 items cut at 1,836 / 1,856).")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Demonstrate Task 3 long-term memory")
    parser.add_argument("--verify-only", action="store_true", help="explain an existing run; no chat-model calls")
    parser.add_argument("--experiment", default=None, help="experiment to verify (default: a new demo run)")
    parser.add_argument("--no-scoring", action="store_true", help="skip section 6 (it makes 1 logged embed call)")
    parser.add_argument("--query", default="What did we arrange for my earbuds last time?")
    parser.add_argument("--failure", default="P11", choices=sorted(FAILURE_NOTES),
                        help="probe shown in section 5 (P11 still fails with the current code)")
    args = parser.parse_args()

    experiment = args.experiment or f"demo_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if not args.verify_only:
        print(f"Running a fresh demo experiment '{experiment}' (Session A, then Session B in a new process).")
        run_session_process("A", experiment)
        print("\n" + "-" * WIDTH + "\n Session A process has EXITED. Its memory now exists only on disk.\n" + "-" * WIDTH)
        run_session_process("B", experiment)
    elif not args.experiment:
        raise SystemExit("--verify-only needs --experiment (for example t3_hm_01)")

    run = load_run(experiment)
    show_session_a(run)
    show_restart(run)
    show_memory(run)
    show_session_b(run)
    show_failure(run, args.failure)
    if not args.no_scoring:
        show_scoring(experiment, run, args.query)
    show_budget(run)
    print("\nDemo complete. Evidence files: results/task3_sessions/, memory/runs/" + experiment + "/, logs/calls.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
