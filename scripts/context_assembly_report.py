"""Task 3(e) context-assembly tables and the Twist E budget stress replay.

1. Official tables: copied from the official Session B transcript (the exact
   assembly records logged with the chat calls) for selected turns.
2. Stress replay: Session B turn 31 rebuilt with real components but a larger
   load - 512-token KB chunks, top-5 hybrid results, memory retrieved at the
   memory-question thresholds and the full Session B history (no trim) - so
   the 1,856-token input budget must cut items. Embedding/tool calls are logged.

    python scripts/context_assembly_report.py --experiment t3_hm_01
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
from src.context.context_assembler import ContextAssembler, assembly_record, build_turn_items  # noqa: E402
from src.dialogue.dialogue_manager import DialogueManager  # noqa: E402
from src.dialogue.qwen_nlg import DECODING_PRESETS  # noqa: E402
from src.dialogue.tools import RecallBotToolGateway  # noqa: E402
from src.memory.memory_manager import (  # noqa: E402
    EPISODE_MIN_RELEVANCE_QUERY, FACT_MIN_SCORE_QUERY, MemoryManager,
)

USER = "CUST-MAYA-1042"
STRESS_TURN = 31


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="t3_hm_01")
    parser.add_argument("--turns", default="27,31")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    transcript = json.loads((RESULTS_DIR / "task3_sessions" / f"{args.experiment}_session_B.json")
                            .read_text(encoding="utf-8"))
    rows = {row["turn"]: row for row in transcript["turns"]}
    official = {}
    for turn in (int(t) for t in args.turns.split(",")):
        record = rows[turn]["context_assembly"]
        official[str(turn)] = {"user_message": rows[turn]["user_message"], "intent": rows[turn]["intent"],
                               "assistant_response": rows[turn]["assistant_response"], **record}

    # ----- stress replay (logged under its own run ID) -----
    logger_module.set_run_id(args.run_id or f"{args.experiment}_budget_stress", reject_existing=True)
    source = rows[STRESS_TURN]
    memory = MemoryManager(MEMORY_DIR / "runs" / args.experiment, config="hybrid_mem", user=USER, autosave=False)
    memory.set_call_context(user=USER, session="B", turn=STRESS_TURN, config="hybrid_mem")
    tools = RecallBotToolGateway(chunk_size=512)
    tools.embedder_factory = lambda _name: memory._get_embedder()
    tools.set_call_context(user=USER, session="B", turn=STRESS_TURN, config="hybrid_mem")
    candidates = list(tools.search_kb(source["user_message"], top_k=10))
    # Same rule as the dialogue manager for Task 4 configs: superseded chunks are
    # excluded unless the question asks about an older period.
    kb_hits = [h for h in candidates if DialogueManager._resolve_policy_status(h) != "superseded"][:5]
    retrieved = memory.retrieve_for_turn(source["user_message"], fact_top_k=3, episode_top_k=3,
                                         fact_min_score=FACT_MIN_SCORE_QUERY,
                                         episode_min_relevance=EPISODE_MIN_RELEVANCE_QUERY)
    history = []
    for row in transcript["turns"]:
        if row["turn"] < STRESS_TURN:
            history += [{"role": "user", "content": row["user_message"]},
                        {"role": "assistant", "content": row["assistant_response"]}]
    assembler = ContextAssembler()
    items, profile = build_turn_items(
        intent=source["intent"],
        system_prompt=(ROOT / "skills" / "system_prompt_v2.txt").read_text(encoding="utf-8").strip(),
        system_prompt_version="system_prompt_v2", workflow_step=None,
        user_message=source["user_message"],
        state={"session_id": "B", "turn": STRESS_TURN, "current_intent": source["intent"]},
        draft=("Answer the customer's policy question using only the KNOWLEDGE BASE chunks. For a current-policy "
               "question use only active chunks. Cite chunk IDs in brackets. If a detail is absent, say it is "
               "not specified."),
        facts=retrieved["facts"], episodes=retrieved["episodes"], kb_hits=kb_hits, history=history,
        truncate=assembler.truncate, max_chunk_tokens=512,
    )
    offered = sum(assembler.count(item.text) for item in items)
    assembled = assembler.assemble(items, output_reserve=int(DECODING_PRESETS["task"]["num_predict"]),
                                   profile=profile)
    stress = {
        "kb_candidates_before_version_filter": [h["chunk_id"] for h in candidates],
        "description": "Session B turn 31 rebuilt with 512-token chunks, top-5 active KB hits, memory at the "
                       "memory-question thresholds and the full Session B history (no trim).",
        "user_message": source["user_message"], "content_tokens_offered": offered,
        **assembly_record(assembled),
    }
    out = RESULTS_DIR / "task3_context_assembly.json"
    out.write_text(json.dumps({"experiment": args.experiment, "official_turns": official, "stress_replay": stress},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, record in [*((f"official turn {k}", v) for k, v in official.items()), ("stress turn 31", stress)]:
        print(f"\n== {label}: {record['total_input_tokens']} / {record['input_budget']} input tokens "
              f"(+{record['output_reserve']} reserved for the reply)")
        for row in record["component_table"]:
            print(f"   {row['component']:<36} {row['tokens']:>5}  {row['memory_type']:<22} {row['home']}  {row['ids']}")
        for item in record.get("dropped", []):
            print(f"   CUT {item['id']} ({item['component']}, {item['tokens']} tokens): {item['reason']}")
        for item in record.get("dropped_by_trim", []) or []:
            print(f"   CUT {item['id']} ({item['tokens']} tokens): {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
