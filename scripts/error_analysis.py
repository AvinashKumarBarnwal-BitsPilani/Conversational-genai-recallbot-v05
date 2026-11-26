"""Task 4(b) error analysis evidence for three failed hybrid_mem probes (run t3_hm_01).

For each failure it rebuilds, from the run's own memory, index, transcript and
log: the ranked list with scores, the prompt excerpt (items that were in the
prompt, looked up by their logged IDs), the dialogue state, and a check of the
proposed fix at the failing stage. Embedding calls are logged (own run ID).

    python scripts/error_analysis.py --experiment t3_hm_01 --after t4_hybrid_mem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import logger as logger_module  # noqa: E402
from src.config import MEMORY_DIR, RESULTS_DIR  # noqa: E402
from src.memory.clock import SessionClock  # noqa: E402
from src.memory.episodic_memory import WEIGHTINGS, score_episodes  # noqa: E402
from src.memory.memory_manager import EPISODE_MIN_RELEVANCE_QUERY, MemoryManager  # noqa: E402
from src.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402

USER = "CUST-MAYA-1042"


def probe_row(transcript: dict, probe_id: str) -> dict:
    return next(r for r in transcript["probe_turns"] if r["probe_id"] == probe_id)


def prompt_items(row: dict, texts: dict[str, str]) -> list[str]:
    assembly = row.get("context_assembly") or {}
    return [f"[{i['id']}] ({i['component']}, {i['tokens']} tok) {texts.get(i['id'], '')[:220]}"
            for i in assembly.get("included", []) if i["component"] in ("long_term_memory", "retrieved_knowledge")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="t3_hm_01")
    parser.add_argument("--after", default="t4_hybrid_mem", help="later hybrid_mem run used as 'after'")
    parser.add_argument("--run-id", default="t4_error_analysis")
    args = parser.parse_args()
    logger_module.set_run_id(args.run_id, reject_existing=True)
    sessions = RESULTS_DIR / "task3_sessions"
    before = json.loads((sessions / f"{args.experiment}_session_B.json").read_text(encoding="utf-8"))
    after_path = sessions / f"{args.after}_session_B.json"
    after = json.loads(after_path.read_text(encoding="utf-8")) if after_path.exists() else None
    probes = {p["probe_id"]: p for p in json.loads((ROOT / "data" / "probes.json").read_text(encoding="utf-8"))}
    memory = MemoryManager(MEMORY_DIR / "runs" / args.experiment, config="hybrid_mem", user=USER, autosave=False)
    chunks = {c["chunk_id"]: c for c in json.loads((ROOT / "index" / "dense_128" / "chunks.json")
                                                   .read_text(encoding="utf-8"))}
    texts = {cid: c["content"] for cid, c in chunks.items()}
    texts.update({f.memory_id: f"{f.key}: {f.value}" for f in memory.semantic.all_records(USER)})
    texts.update({e.episode_id: f"{e.type}: {e.description}" for e in memory.episodic.episodes(USER)})
    report: dict = {"experiment": args.experiment, "after_run": args.after if after else None, "failures": {}}

    # ---------------- P13: episodic retrieval / memory write ----------------
    row = probe_row(before, "P13")
    memory.set_call_context(user=USER, session="B", turn=row["turn"])
    now = SessionClock.for_session("B").iso_at(row["turn"])
    query = probes["P13"]["question"]
    vector = memory._embed([query], is_query=True, purpose="error_analysis_P13")[0]
    episodes = memory.episodic.episodes(USER)
    relevance = memory.episodic.index.similarities(vector)
    ranked = score_episodes(episodes, relevance, now=now, weights=WEIGHTINGS["A"],
                            min_relevance=EPISODE_MIN_RELEVANCE_QUERY)
    decision = next(e for e in episodes if e.type == "decision")
    enriched = (decision.description.rstrip(".") +
                " (the customer's words: 'I also chose an exchange rather than a refund for my USB-C charger "
                "order ORD-66310 last month.')")
    enriched_vec = memory._embed([f"decision: {enriched}"], is_query=False, purpose="error_analysis_P13_fix")[0]
    fixed_relevance = dict(relevance)
    fixed_relevance[decision.episode_id] = float(enriched_vec @ vector)
    reranked = score_episodes(episodes, fixed_relevance, now=now, weights=WEIGHTINGS["A"],
                              min_relevance=EPISODE_MIN_RELEVANCE_QUERY)
    report["failures"]["P13"] = {
        "question": query, "expected": probes["P13"]["expected"], "answer": row["assistant_response"],
        "stage": "memory write (episode text lacked the product), surfacing as an episodic ranking error",
        "dialogue_state": {k: row["dialogue_state"].get(k) for k in ("current_intent", "active_task", "task_status")},
        "ranked_list": [s.to_dict() for s in ranked[:6]],
        "decision_episode": decision.to_dict(),
        "prompt_excerpt": prompt_items(row, texts),
        "fix_check": {
            "change": "store the customer's own words (redacted) in decision/complaint episodes",
            "decision_relevance_before": round(relevance[decision.episode_id], 4),
            "decision_relevance_after": round(fixed_relevance[decision.episode_id], 4),
            "decision_rank_before": 1 + [s.episode.episode_id for s in ranked].index(decision.episode_id),
            "decision_rank_after": 1 + [s.episode.episode_id for s in reranked].index(decision.episode_id),
            "ranked_after": [s.to_dict() for s in reranked[:4]],
        },
    }

    # ---------------- P11: knowledge-base retrieval ----------------
    from src.dialogue.tools import RecallBotToolGateway
    row = probe_row(before, "P11")
    tools = RecallBotToolGateway(chunk_size=128)
    tools.embedder_factory = lambda _name: memory._get_embedder()
    tools.set_call_context(user=USER, session="B", turn=row["turn"], config="hybrid_mem")
    tools._load_retrieval_components()

    def ranked_lists(text: str, purpose: str) -> dict:
        qv = tools._embedder.encode([text], purpose=purpose, is_query=True, user=USER, session="B",
                                    turn=row["turn"], config="hybrid_mem")[0]
        dense = tools._dense_index.search(qv, top_k=20)
        bm25 = tools._bm25_index.search(text, top_k=20)
        fused = reciprocal_rank_fusion([[h.chunk_id for h in dense], [h.chunk_id for h in bm25]], k=60)
        return {
            "dense_top8": [(h.chunk_id, round(h.score, 4)) for h in dense[:8]],
            "bm25_top8": [(h.chunk_id, round(h.score, 3)) for h in bm25[:8]],
            "rrf_top8": [(cid, round(score, 5)) for cid, score in fused[:8]],
            "kb003_ranks": {
                cid: {"dense": next((i for i, h in enumerate(dense, 1) if h.chunk_id == cid), None),
                      "bm25": next((i for i, h in enumerate(bm25, 1) if h.chunk_id == cid), None),
                      "rrf": next((i for i, (c, _) in enumerate(fused, 1) if c == cid), None)}
                for cid in chunks if cid.startswith("KB-003")},
        }

    original = probes["P11"]["question"]
    booking = next(e for e in memory.episodic.episodes(USER) if e.type == "booking")
    rewritten = f"{original} Context: return pickup for a defective item; returns centre, inspection, refund."
    report["failures"]["P11"] = {
        "question": original, "expected": probes["P11"]["expected"], "answer": row["assistant_response"],
        "stage": "knowledge-base retrieval (refund-stage chunk of KB-003 not retrieved)",
        "dialogue_state": {k: row["dialogue_state"].get(k) for k in ("current_intent", "active_task", "task_status")},
        "prompt_excerpt": prompt_items(row, texts),
        "ranked_lists": ranked_lists(original, "error_analysis_P11"),
        "fix_check": {
            "change": ("state-aware query rewrite for a follow-up with no active task: add the return-lifecycle "
                       "context of the customer's latest booking episode before retrieval"),
            "booking_episode_used": booking.episode_id,
            "rewritten_query": rewritten,
            "ranked_lists_after": ranked_lists(rewritten, "error_analysis_P11_fix"),
        },
    }

    # ---------------- P15: NLG / prompt ----------------
    row = probe_row(before, "P15")
    turn39 = next(r for r in before["turns"] if r["turn"] == 39)
    log = [json.loads(line) for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
           if f'"{args.experiment}_B"' in line]
    chat = next(r for r in log if r["call_type"] == "chat" and r["turn"] == row["turn"])
    report["failures"]["P15"] = {
        "question": probes["P15"]["question"], "expected": probes["P15"]["expected"],
        "answer": row["assistant_response"],
        "stage": "NLG / prompt (the deterministic draft was correct; the model followed the question instead)",
        "dialogue_state": {k: row["dialogue_state"].get(k) for k in ("current_intent", "active_task", "task_status")},
        "same_draft_other_question": {"turn": 39, "question": turn39["user_message"],
                                      "answer": turn39["assistant_response"]},
        "chat_call": {k: chat[k] for k in ("turn", "input_tokens", "output_tokens", "latency_ms")}
                     | {"status": chat["extra"].get("status"),
                        "prompt_tokens_by_component": [
                            (i["id"], i["tokens"]) for i in chat["extra"]["context_assembly"]["included"]]},
        "fix": "context-budget answers are emitted verbatim by the dialogue manager (no model call)",
    }
    if after:
        for probe_id in ("P11", "P13", "P15"):
            report["failures"][probe_id]["after_run_answer"] = probe_row(after, probe_id)["assistant_response"]

    out = RESULTS_DIR / "task4_error_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
