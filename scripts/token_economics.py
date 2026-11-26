"""Task 5(a) token economics from measured tokens (logs/calls.jsonl) and the CS3 price table.

Scenario (assignment): 25,000 daily users, one conversation per user per day,
40 turns per conversation (the test conversation), 30-day month.

Measured per turn = tokens of BILLABLE calls (model calls: chat, summary, LLM
extraction, reflection) divided by the scripted turns of the run. Rule-based
NLU/extraction/importance, tools and the local BGE embedder cost nothing.

Prices: CS3 Lecture 3, slide 17 ("2026 Pricing Landscape - per 1M tokens"):
  GPT-4o (OpenAI)  input $2.50, output $10.00, cached input $1.25   <- reported row
  Claude Haiku 5.5 input $0.10, output $0.50,  cached input ~$0.01  <- comparison
Caching rules: CS3 slide 18 - minimum cacheable prompt 1,024 tokens (Claude
Sonnet/Opus and OpenAI); 90% discount Anthropic, 50% OpenAI; Anthropic cache
write costs 1.25x; TTL 5 minutes, refreshed on every hit.

    python scripts/token_economics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context.context_assembler import TOOLS_TEXT  # noqa: E402
from src.dialogue.qwen_nlg import QwenNLG  # noqa: E402
from src.retrieval.tokenizer import load_tokenizer  # noqa: E402

DAILY_USERS = 25_000
TURNS_PER_CONVERSATION = 40
DAYS_PER_MONTH = 30
CACHE_MIN_TOKENS = 1_024          # CS3 slide 18
CACHE_TTL_MINUTES = 5             # CS3 slide 18
PRICES = {                        # USD per 1M tokens, CS3 slide 17
    "GPT-4o (OpenAI)": {"input": 2.50, "output": 10.00, "cached": 1.25, "write_multiplier": 1.0,
                        "min_note": "1,024 tokens (CS3 slide 18, OpenAI)"},
    "Claude Haiku 5.5": {"input": 0.10, "output": 0.50, "cached": 0.01, "write_multiplier": 1.25,
                         "min_note": "not stated on CS3; 1,024 assumed (stated for Sonnet/Opus)"},
}
BILLABLE = {"chat", "summary", "reflect", "judge"}   # plus extract when an LLM produced it
RUNS = {
    "trim": {"runs": ["task2b_trim_run2"], "turns": 25,
             "note": "Task 2 official trim run (Session A, turns 1-25; system_prompt_v1 + workflow JSON)"},
    "hybrid_mem": {"runs": ["t4_hybrid_mem_A", "t4_hybrid_mem_B"], "turns": 40,
                   "note": "Task 4 ablation run (Sessions A+B, turns 1-40; system_prompt_v2 + tools)"},
    # Reference only: trim history with the compact v2 prompt, to separate the prompt effect
    # from the memory effect when comparing trim and hybrid_mem.
    "no_rag (reference)": {"runs": ["t4_no_rag_A", "t4_no_rag_B"], "turns": 40,
                           "note": "Reference: trim k=3 history with system_prompt_v2, no KB or long-term memory "
                                   "(Task 4 ablation run, Sessions A+B)"},
}


def billable(record: dict) -> bool:
    if record["call_type"] == "extract":
        return record["model"] != "rule-based"
    return record["call_type"] in BILLABLE and record.get("extra", {}).get("status", "success") == "success"


def prefix_tokens(tokenizer, system_text: str) -> int:
    """Tokens of the templated system message - the fixed, cacheable prefix."""
    return len(tokenizer.encode(f"<|im_start|>system\n{system_text}<|im_end|>\n", add_special_tokens=False))


def money(tokens: float, price_per_million: float) -> float:
    return tokens * price_per_million / 1_000_000


def main() -> int:
    log = [json.loads(line) for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    tokenizer = load_tokenizer(local_files_only=True)
    v2 = (ROOT / "skills" / "system_prompt_v2.txt").read_text(encoding="utf-8").strip()
    prefixes = {
        "trim": prefix_tokens(tokenizer, QwenNLG(logger=None)._system_message()),
        "hybrid_mem": prefix_tokens(tokenizer, f"{v2}\n\n{TOOLS_TEXT}"),
    }
    prefixes["no_rag (reference)"] = prefixes["hybrid_mem"]
    turns_per_month = DAILY_USERS * TURNS_PER_CONVERSATION * DAYS_PER_MONTH
    turns_per_day = DAILY_USERS * TURNS_PER_CONVERSATION
    # A shared prefix stays warm while traffic keeps arriving within the TTL; at most one cold
    # (re)write per TTL window per day, so the cache hit rate is close to 1.
    cache_writes_per_day = 24 * 60 // CACHE_TTL_MINUTES
    report = {"assumptions": {
        "daily_users": DAILY_USERS, "turns_per_conversation": TURNS_PER_CONVERSATION,
        "days_per_month": DAYS_PER_MONTH, "turns_per_month": turns_per_month,
        "cache_min_tokens": CACHE_MIN_TOKENS, "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "cache_writes_per_day_upper_bound": cache_writes_per_day,
        "cache_hit_rate": round(1 - cache_writes_per_day / turns_per_day, 6),
        "token_counts": "Qwen3 tokenizer counts reported by Ollama; GPT-4o's tokenizer would differ slightly",
        "embeddings": "local BAAI/bge-small-en-v1.5, $0 (not an API cost)"}, "configs": {}, "scenarios": []}

    for config, spec in RUNS.items():
        records = [r for r in log if r["run_id"] in spec["runs"] and 1 <= r["turn"] <= spec["turns"]]
        bill = [r for r in records if billable(r)]
        chat = [r for r in bill if r["call_type"] == "chat"]
        per_turn_in = sum(r["input_tokens"] for r in bill) / spec["turns"]
        per_turn_out = sum(r["output_tokens"] for r in bill) / spec["turns"]
        chat_calls_per_turn = len(chat) / spec["turns"]
        report["configs"][config] = {
            "source": spec["note"], "runs": spec["runs"], "scripted_turns": spec["turns"],
            "billable_calls": {t: sum(r["call_type"] == t for r in bill) for t in sorted({r["call_type"] for r in bill})},
            "input_tokens_per_turn": round(per_turn_in, 1), "output_tokens_per_turn": round(per_turn_out, 1),
            "chat_calls_per_turn": round(chat_calls_per_turn, 3),
            "fixed_prefix_tokens": prefixes[config],
            "prefix_cacheable": prefixes[config] >= CACHE_MIN_TOKENS,
        }
        for row, price in PRICES.items():
            input_month = per_turn_in * turns_per_month
            output_month = per_turn_out * turns_per_month
            base = money(input_month, price["input"]) + money(output_month, price["output"])
            cacheable = prefixes[config] >= CACHE_MIN_TOKENS
            cached_month = (prefixes[config] * chat_calls_per_turn * turns_per_month * report["assumptions"]["cache_hit_rate"]
                            if cacheable else 0.0)
            writes_month = prefixes[config] * cache_writes_per_day * DAYS_PER_MONTH if cacheable else 0.0
            with_cache = (money(input_month - cached_month, price["input"]) + money(cached_month, price["cached"])
                          + money(writes_month, price["input"] * (price["write_multiplier"] - 1))
                          + money(output_month, price["output"]))
            report["scenarios"].append({
                "config": config, "price_row": row, "monthly_input_tokens": round(input_month),
                "monthly_output_tokens": round(output_month), "monthly_cached_tokens": round(cached_month),
                "cost_no_cache_usd": round(base, 2), "cost_with_cache_usd": round(with_cache, 2),
                "saving_pct": round(100 * (base - with_cache) / base, 1) if base else 0.0,
                "cache_applies": cacheable,
            })

    # What-if: pad hybrid_mem's prefix up to the minimum so it becomes cacheable.
    hm = report["configs"]["hybrid_mem"]
    pad = CACHE_MIN_TOKENS - hm["fixed_prefix_tokens"]
    calls_month = hm["chat_calls_per_turn"] * turns_per_month
    padding = []
    for row, price in PRICES.items():
        today = money(hm["fixed_prefix_tokens"] * calls_month, price["input"])
        padded = money(CACHE_MIN_TOKENS * calls_month, price["cached"])
        padding.append({"price_row": row, "padding_tokens_per_call": pad,
                        "prefix_cost_today_usd": round(today, 2), "prefix_cost_if_padded_and_cached_usd": round(padded, 2),
                        "monthly_change_usd": round(padded - today, 2),
                        "input_budget_used_by_padding_pct": round(100 * pad / 1856, 1)})
    report["padding_what_if"] = padding

    out = ROOT / "results" / "task5_token_economics.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["configs"], indent=2))
    print(f"\n{'config':<11}{'price row':<20}{'no cache $/month':>18}{'with cache':>14}{'saving':>8}")
    for s in report["scenarios"]:
        print(f"{s['config']:<11}{s['price_row']:<20}{s['cost_no_cache_usd']:>18,.2f}{s['cost_with_cache_usd']:>14,.2f}"
              f"{s['saving_pct']:>7.1f}%")
    print("\nPadding hybrid_mem's prefix to 1,024 tokens:")
    for p in padding:
        print(f"  {p['price_row']:<20} today ${p['prefix_cost_today_usd']:,.2f} -> padded+cached "
              f"${p['prefix_cost_if_padded_and_cached_usd']:,.2f} ({p['monthly_change_usd']:+,.2f}); "
              f"uses {p['input_budget_used_by_padding_pct']}% of the 1,856-token input budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
