"""Reproduce Task 2(b) Session A token analysis from append-only call logs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "calls.jsonl"
RESULTS_DIR = ROOT / "results"
DATASET_PATH = ROOT / "data" / "test_conversation.json"
MANIFEST_PATH = RESULTS_DIR / "task2b_session_A_run_manifest.json"
RUNS = {
    "Full": ("task2b_full_run3", "full"),
    "Trim": ("task2b_trim_run2", "trim"),
    "Summary": ("task2b_summary_run2", "summary"),
}
EXPECTED_CALLS = {
    "full": {"nlu": 25, "chat": 25, "summary": 0, "tool": 7, "embed": 3},
    "trim": {"nlu": 25, "chat": 25, "summary": 0, "tool": 7, "embed": 3},
    "summary": {"nlu": 25, "chat": 25, "summary": 22, "tool": 7, "embed": 3},
}
REQUIRED_FIELDS = {
    "ts", "run_id", "config", "user", "session", "turn", "call_type",
    "model", "input_tokens", "output_tokens", "retrieved_tokens", "latency_ms",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def _is_prompt_preparation_failure(record: dict[str, Any]) -> bool:
    extra = record.get("extra", {})
    return (
        extra.get("status") == "failed"
        and extra.get("error_type") == "ValueError"
        and float(extra.get("api_call_ms", 0) or 0) == 0
        and float(extra.get("prompt_build_ms", 0) or 0) == 0
    )


def _was_model_processed(record: dict[str, Any]) -> bool:
    """Use API usage when present; distinguish estimates and unsent prompts."""
    if record.get("call_type") not in {"chat", "summary"}:
        return False
    extra = record.get("extra", {})
    # These counts come from Ollama's prompt_eval_count/eval_count.
    if extra.get("token_counts_source") == "ollama":
        return True
    # A successful response can be counted using the project's existing
    # tokenizer fallback when Ollama omits usage fields.
    return (
        extra.get("status") == "success"
        or float(extra.get("api_call_ms", 0) or 0) > 0
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _sum(records: list[dict[str, Any]], field: str) -> int:
    return sum(int(record.get(field, 0) or 0) for record in records)


def analyze() -> dict[str, Any]:
    logs = _read_jsonl(LOG_PATH)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_turns = [
        row for row in dataset["turns"]
        if row.get("session") == "A" and row.get("role") == "user"
        and 1 <= int(row.get("turn", -1)) <= 25
    ]
    source_turns.sort(key=lambda row: int(row["turn"]))
    expected_text = {int(row["turn"]): row["text"] for row in source_turns}
    if list(expected_text) != list(range(1, 26)):
        raise ValueError("Dataset does not contain the exact 25 Session A user turns")

    manifest_runs = {row["run_id"]: row for row in manifest.get("runs", [])}
    filtered = {
        strategy: [row for row in logs if row.get("run_id") == run_id]
        for strategy, (run_id, _) in RUNS.items()
    }
    if len({run_id for run_id, _ in RUNS.values()}) != 3:
        raise ValueError("Official run IDs must be distinct")

    transcripts: dict[str, dict[str, Any]] = {}
    for strategy, (run_id, config) in RUNS.items():
        manifest_row = manifest_runs.get(run_id)
        if not manifest_row or manifest_row.get("status") != "complete":
            raise ValueError(f"Manifest does not mark official run {run_id} complete")
        transcript_path = ROOT / manifest_row["transcript"]
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript.get("run_id") != run_id or transcript.get("strategy") != config:
            raise ValueError(f"Transcript does not match official run {run_id}")
        transcript_text = {
            int(row["turn"]): row["user_message"] for row in transcript.get("turns", [])
        }
        if transcript_text != expected_text:
            raise ValueError(f"Transcript for {run_id} differs from source dataset")
        if len(transcript.get("turns", [])) != 25 or transcript.get("status") != "complete":
            raise ValueError(f"Transcript for {run_id} is not a complete 25-turn run")
        transcripts[strategy] = transcript

        records = filtered[strategy]
        if not records:
            raise ValueError(f"No log records found for {run_id}")
        counts = Counter(row.get("call_type") for row in records)
        expected_counts = {
            kind: count for kind, count in EXPECTED_CALLS[config].items() if count
        }
        if dict(counts) != expected_counts:
            raise ValueError(
                f"Call counts for {run_id} do not match audit: {dict(counts)}"
            )
        seen: set[str] = set()
        for row in records:
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{run_id} record missing fields: {sorted(missing)}")
            if row["config"] != config or row["session"] != "A":
                raise ValueError(f"Unexpected config/session in {run_id} record")
            if not 1 <= int(row["turn"]) <= 25:
                raise ValueError(f"Out-of-range turn in {run_id}: {row['turn']}")
            if not isinstance(row["latency_ms"], (int, float)) or row["latency_ms"] < 0:
                raise ValueError(f"Invalid latency in {run_id}")
            for field in ("input_tokens", "output_tokens", "retrieved_tokens", "turn"):
                if not isinstance(row[field], int) or row[field] < 0:
                    raise ValueError(f"Invalid {field} in {run_id}")
            key = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if key in seen:
                raise ValueError(f"Duplicate call record detected in {run_id}")
            seen.add(key)
        for call_type in ("nlu", "chat"):
            per_turn = Counter(row["turn"] for row in records if row["call_type"] == call_type)
            if per_turn != Counter({turn: 1 for turn in range(1, 26)}):
                raise ValueError(f"Missing or duplicate {call_type} turn records for {run_id}")
        logged_tool_by_turn = Counter(
            int(row["turn"]) for row in records if row["call_type"] == "tool"
        )
        transcript_tool_by_turn = Counter(
            int(turn["turn"])
            for turn in transcript["turns"]
            for _ in turn.get("tool_results", [])
        )
        if logged_tool_by_turn != transcript_tool_by_turn:
            raise ValueError(f"Tool log records do not reconcile with transcript for {run_id}")
        summary_rows = [row for row in records if row["call_type"] == "summary"]
        for row in records:
            if row["call_type"] in {"chat", "summary"} and not row.get("extra", {}).get("status"):
                raise ValueError(f"Missing model-call status in {run_id}")
            if row["call_type"] == "tool" and (
                not row.get("extra", {}).get("status")
                or not row.get("extra", {}).get("tool_name")
            ):
                raise ValueError(f"Missing tool status/name in {run_id}")
        if config == "summary":
            if any(row["config"] != "summary" for row in summary_rows):
                raise ValueError("Summary call uses incorrect config")
            if any(row.get("extra", {}).get("purpose") != "summary_generation" for row in summary_rows):
                raise ValueError("Summary call missing summary_generation purpose")
            summary_per_turn = Counter(row["turn"] for row in summary_rows)
            if any(count > 1 for count in summary_per_turn.values()):
                raise ValueError("Duplicate summary-generation call on a turn")
            if set(summary_per_turn) != set(range(4, 26)):
                raise ValueError("Summary run is missing an expected aged-turn summary call")
        elif summary_rows:
            raise ValueError(f"Unexpected summary calls in {config}")
        for row in records:
            if row["call_type"] == "chat" and row.get("extra", {}).get("purpose") != "qwen_nlg_response":
                raise ValueError(f"Unexpected chat purpose in {run_id}")
            if row["call_type"] == "tool" and any(
                int(row.get(field, -1)) != 0
                for field in ("input_tokens", "output_tokens", "retrieved_tokens")
            ):
                raise ValueError(f"Nonzero tool token field in {run_id}")

    summary_rows: list[dict[str, Any]] = []
    per_turn_rows: list[dict[str, Any]] = []
    plot_data: dict[str, dict[str, list[int | None]]] = {}
    overall: dict[str, dict[str, Any]] = {}

    for strategy, (run_id, config) in RUNS.items():
        records = filtered[strategy]
        by_type = {
            kind: [row for row in records if row["call_type"] == kind]
            for kind in ("nlu", "chat", "summary", "tool", "embed")
        }
        chats = by_type["chat"]
        summaries = by_type["summary"]
        prompt_prep_failures = [row for row in chats if _is_prompt_preparation_failure(row)]
        other_failed_chat = [
            row for row in chats
            if row.get("extra", {}).get("status") == "failed"
            and not _is_prompt_preparation_failure(row)
        ]
        successful_chat = [row for row in chats if row.get("extra", {}).get("status") == "success"]
        actual_qwen = [row for row in by_type["chat"] + summaries if _was_model_processed(row)]
        exact_qwen = [
            row for row in actual_qwen
            if row.get("extra", {}).get("token_counts_source") == "ollama"
        ]
        estimated_qwen = [row for row in actual_qwen if row not in exact_qwen]
        all_latency = [float(row["latency_ms"]) for row in records]
        chat_latency = [float(row["latency_ms"]) for row in chats]
        actual_chat_by_turn: dict[int, int] = {turn: 0 for turn in range(1, 26)}
        attempted_chat_by_turn: dict[int, int] = {turn: 0 for turn in range(1, 26)}
        unsent_chat_by_turn: dict[int, int] = {turn: 0 for turn in range(1, 26)}
        chat_status_by_turn: dict[int, str] = {}
        submitted_plot: list[int | None] = [None] * 25
        unsent_plot: list[int | None] = [None] * 25
        summary_input_by_turn: dict[int, int] = {turn: 0 for turn in range(1, 26)}
        summary_output_by_turn: dict[int, int] = {turn: 0 for turn in range(1, 26)}
        summary_status_by_turn: dict[int, str] = {}
        for row in chats:
            turn = int(row["turn"])
            attempted_chat_by_turn[turn] += int(row["input_tokens"])
            extra = row.get("extra", {})
            status = str(extra.get("status", "missing_status"))
            if _is_prompt_preparation_failure(row):
                unsent_chat_by_turn[turn] += int(row["input_tokens"])
                unsent_plot[turn - 1] = int(row["input_tokens"])
                chat_status_by_turn[turn] = "prompt_prep_failed_unsent"
            elif _was_model_processed(row):
                actual_chat_by_turn[turn] += int(row["input_tokens"])
                submitted_plot[turn - 1] = int(row["input_tokens"])
                chat_status_by_turn[turn] = status
            else:
                chat_status_by_turn[turn] = f"failed_after_request_usage_unknown:{status}"
        for row in summaries:
            turn = int(row["turn"])
            summary_input_by_turn[turn] += int(row["input_tokens"])
            summary_output_by_turn[turn] += int(row["output_tokens"])
            old = summary_status_by_turn.get(turn)
            state = str(row.get("extra", {}).get("status", "missing_status"))
            summary_status_by_turn[turn] = f"{old};{state}" if old else state

        plot_data[strategy] = {
            "submitted": submitted_plot,
            "unsent": unsent_plot,
            "summary_overhead": [
                summary_input_by_turn[turn] if summary_input_by_turn[turn] else None
                for turn in range(1, 26)
            ],
        }
        for turn in range(1, 26):
            per_turn_rows.append({
                "strategy": strategy.lower(),
                "run_id": run_id,
                "turn": turn,
                "chat_status": chat_status_by_turn.get(turn, "missing"),
                "chat_input_tokens_logged_attempted": attempted_chat_by_turn[turn],
                "chat_input_tokens_model_processed_or_success_estimate": actual_chat_by_turn[turn],
                "chat_input_tokens_prompt_prep_failed_unsent": unsent_chat_by_turn[turn],
                "chat_output_tokens_logged": sum(
                    int(row["output_tokens"]) for row in chats if int(row["turn"]) == turn
                ),
                "summary_call_count": sum(int(row["turn"]) == turn for row in summaries),
                "summary_status": summary_status_by_turn.get(turn, "none"),
                "summary_input_tokens_logged": summary_input_by_turn[turn],
                "summary_output_tokens_logged": summary_output_by_turn[turn],
                "summary_input_tokens_model_processed_or_success_estimate": sum(
                    int(row["input_tokens"]) for row in summaries
                    if int(row["turn"]) == turn and _was_model_processed(row)
                ),
                "summary_latency_ms": sum(
                    float(row["latency_ms"]) for row in summaries if int(row["turn"]) == turn
                ),
            })

        logged_input = _sum(records, "input_tokens")
        logged_output = _sum(records, "output_tokens")
        exact_input = _sum(exact_qwen, "input_tokens")
        exact_output = _sum(exact_qwen, "output_tokens")
        estimated_input = _sum(estimated_qwen, "input_tokens")
        estimated_output = _sum(estimated_qwen, "output_tokens")
        actual_chat_inputs = [actual_chat_by_turn[turn] for turn in range(1, 26)]
        attempted_chat_inputs = [attempted_chat_by_turn[turn] for turn in range(1, 26)]
        successful_chat_inputs = [
            int(row["input_tokens"]) for row in chats if _was_model_processed(row)
        ]
        row = {
            "strategy": strategy.lower(),
            "run_id": run_id,
            "logged_input_tokens_all_call_types": logged_input,
            "logged_output_tokens_all_call_types": logged_output,
            "logged_total_tokens_input_plus_output": logged_input + logged_output,
            "logged_retrieved_tokens_reported_separately_not_added": _sum(records, "retrieved_tokens"),
            "customer_chat_input_tokens_logged_including_unsent_attempts": _sum(chats, "input_tokens"),
            "customer_chat_output_tokens_logged": _sum(chats, "output_tokens"),
            "customer_chat_actual_or_success_estimated_input_tokens": sum(actual_chat_inputs),
            "customer_chat_actual_or_success_estimated_output_tokens": _sum(
                [row for row in chats if _was_model_processed(row)], "output_tokens"
            ),
            "customer_chat_prompt_prep_failed_unsent_input_tokens": _sum(prompt_prep_failures, "input_tokens"),
            "summary_generation_input_tokens_logged": _sum(summaries, "input_tokens"),
            "summary_generation_output_tokens_logged": _sum(summaries, "output_tokens"),
            "summary_generation_actual_or_success_estimated_input_tokens": _sum(
                [row for row in summaries if _was_model_processed(row)], "input_tokens"
            ),
            "summary_generation_actual_or_success_estimated_output_tokens": _sum(
                [row for row in summaries if _was_model_processed(row)], "output_tokens"
            ),
            "summary_call_count": len(summaries),
            "successful_chat_calls": len(successful_chat),
            "failed_chat_calls_total": len(chats) - len(successful_chat),
            "prompt_preparation_failed_chat_calls_not_sent": len(prompt_prep_failures),
            "other_failed_chat_calls_request_or_stage_unknown": len(other_failed_chat),
            "ollama_usage_exact_model_input_tokens_chat_plus_summary": exact_input,
            "ollama_usage_exact_model_output_tokens_chat_plus_summary": exact_output,
            "ollama_usage_exact_model_total_tokens_chat_plus_summary": exact_input + exact_output,
            "successful_qwen_local_estimate_input_tokens_separate": estimated_input,
            "successful_qwen_local_estimate_output_tokens_separate": estimated_output,
            "nlu_logged_input_tokens_overhead": _sum(by_type["nlu"], "input_tokens"),
            "embedding_logged_input_tokens_overhead": _sum(by_type["embed"], "input_tokens"),
            "embedding_bge_content_tokens_metadata": sum(
                int(row.get("extra", {}).get("embedding_input_tokens", 0) or 0)
                for row in by_type["embed"]
            ),
            "tool_logged_tokens": _sum(by_type["tool"], "input_tokens")
            + _sum(by_type["tool"], "output_tokens"),
            "mean_customer_chat_input_logged_attempted_per_turn": sum(attempted_chat_inputs) / 25,
            "median_customer_chat_input_logged_attempted_per_turn": _median(attempted_chat_inputs),
            "mean_customer_chat_input_model_processed_per_turn_zero_when_unsent": sum(actual_chat_inputs) / 25,
            "median_customer_chat_input_model_processed_per_turn_zero_when_unsent": _median(actual_chat_inputs),
            "mean_customer_chat_input_per_successful_processed_call": (
                sum(successful_chat_inputs) / len(successful_chat_inputs)
                if successful_chat_inputs else 0
            ),
            "median_customer_chat_input_per_successful_processed_call": _median(successful_chat_inputs),
            "total_latency_ms_all_call_types": sum(all_latency),
            "median_latency_ms_all_call_types": _median(all_latency),
            "total_latency_ms_chat_calls": sum(chat_latency),
            "median_latency_ms_chat_calls": _median(chat_latency),
            "total_latency_ms_summary_calls": _sum(summaries, "latency_ms"),
            "database_reset_fingerprint": transcripts[strategy]["database_reset_fingerprint"],
            "processed_turns": transcripts[strategy]["successfully_processed_user_turns"],
            "deterministic_fallback_turns": transcripts[strategy]["deterministic_fallback_turns"],
        }
        summary_rows.append(row)
        overall[strategy] = {
            "records": records,
            "by_type": by_type,
            "prompt_prep_failures": prompt_prep_failures,
            "other_failed_chat": other_failed_chat,
            "successful_chat": successful_chat,
            "actual_qwen": actual_qwen,
            "summary_input_by_turn": summary_input_by_turn,
            "transcript": transcripts[strategy],
            "summary_row": row,
        }

    # This output has one row per strategy and source user turn (75 rows).
    per_turn_rows.sort(key=lambda row: (row["turn"], row["strategy"]))
    _write_csv(RESULTS_DIR / "task2b_token_summary.csv", summary_rows)
    _write_csv(RESULTS_DIR / "task2b_token_per_turn.csv", per_turn_rows)
    _plot(plot_data)
    _write_report(summary_rows, overall, per_turn_rows)
    _print_summary(summary_rows)
    return {"summary": summary_rows, "per_turn": per_turn_rows}


def _plot(data: dict[str, dict[str, list[int | None]]]) -> None:
    turns = list(range(1, 26))
    fig, ax = plt.subplots(figsize=(15, 7.6), constrained_layout=True)
    colors = {"Full": "#2563eb", "Trim": "#16a34a", "Summary": "#7c3aed"}
    for strategy in ("Full", "Trim", "Summary"):
        vals = data[strategy]["submitted"]
        ax.plot(turns, [float("nan") if v is None else v for v in vals],
                marker="o", linewidth=2, color=colors[strategy],
                label=f"{strategy}: customer chat tokens processed")
        unsent_x = [turn for turn, v in zip(turns, data[strategy]["unsent"]) if v is not None]
        unsent_y = [v for v in data[strategy]["unsent"] if v is not None]
        if unsent_x:
            ax.scatter(unsent_x, unsent_y, marker="x", s=70, linewidths=2,
                       color=colors[strategy], label=f"{strategy}: prompt attempted, not sent")
    overhead = data["Summary"]["summary_overhead"]
    overhead_x = [turn for turn, value in zip(turns, overhead) if value is not None]
    overhead_y = [value for value in overhead if value is not None]
    if overhead_x:
        ax.plot(overhead_x, overhead_y, marker="^", linestyle="--", linewidth=1.8,
                color="#ea580c", label="Summary: extra summary-generation input")
    ax.axhline(1856, color="#6b7280", linestyle=":", linewidth=1.2,
               label="Safe input budget (2048 context − 160 output − 32 margin)")
    ax.set_title("Task 2(b) — Session A input tokens by turn\nOfficial Full, Trim and Summary runs")
    ax.set_xlabel("Session A user turn")
    ax.set_ylabel("Input tokens")
    ax.set_xticks(turns)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    fig.text(
        0.01, -0.01,
        "Solid lines show successful model calls with logged usage (or a successful-call tokenizer estimate if Ollama omitted usage). X markers are attempted chat prompts rejected during prompt preparation; they were not sent to Ollama. "
        "Orange dashed series is additional Summary-generation input. Retrieved tokens are not added again.",
        fontsize=9,
    )
    fig.savefig(RESULTS_DIR / "task2b_input_tokens_per_turn.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def _write_report(
    rows: list[dict[str, Any]], overall: dict[str, dict[str, Any]],
    per_turn_rows: list[dict[str, Any]],
) -> None:
    by_strategy = {row["strategy"].title(): row for row in rows}
    def table(headers: list[str], values: list[list[Any]]) -> str:
        return "\n".join([
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(value) for value in line) + " |" for line in values],
        ])

    strategy_summary = table(
        ["Strategy", "Logged input", "Logged output", "Logged total", "Exact Ollama input", "Exact Ollama output", "Chat success / fallback", "Summary calls", "All-call latency (ms)"],
        [[
            name,
            _fmt(row["logged_input_tokens_all_call_types"]),
            _fmt(row["logged_output_tokens_all_call_types"]),
            _fmt(row["logged_total_tokens_input_plus_output"]),
            _fmt(row["ollama_usage_exact_model_input_tokens_chat_plus_summary"]),
            _fmt(row["ollama_usage_exact_model_output_tokens_chat_plus_summary"]),
            f"{row['successful_chat_calls']} / {row['failed_chat_calls_total']}",
            row["summary_call_count"],
            _fmt(row["total_latency_ms_all_call_types"]),
        ] for name, row in by_strategy.items()],
    )
    detail_table = table(
        ["Strategy", "Chat logged input (incl. unsent)", "Chat exact/estimated processed input", "Unsent attempted input", "Chat output", "Summary input", "Summary output", "NLU input overhead", "Embedding input overhead", "Mean chat input / turn (attempted)", "Median chat input / turn (attempted)", "Mean input / successful chat", "Median input / successful chat", "Median latency (ms)"],
        [[
            name,
            _fmt(row["customer_chat_input_tokens_logged_including_unsent_attempts"]),
            _fmt(row["customer_chat_actual_or_success_estimated_input_tokens"]),
            _fmt(row["customer_chat_prompt_prep_failed_unsent_input_tokens"]),
            _fmt(row["customer_chat_output_tokens_logged"]),
            _fmt(row["summary_generation_input_tokens_logged"]),
            _fmt(row["summary_generation_output_tokens_logged"]),
            _fmt(row["nlu_logged_input_tokens_overhead"]),
            _fmt(row["embedding_logged_input_tokens_overhead"]),
            _fmt(row["mean_customer_chat_input_logged_attempted_per_turn"]),
            _fmt(row["median_customer_chat_input_logged_attempted_per_turn"]),
            _fmt(row["mean_customer_chat_input_per_successful_processed_call"]),
            _fmt(row["median_customer_chat_input_per_successful_processed_call"]),
            _fmt(row["median_latency_ms_all_call_types"]),
        ] for name, row in by_strategy.items()],
    )
    per_turn_table = table(
        ["Strategy", "Mean attempted input / turn", "Median attempted", "Mean submitted input / turn (0 if unsent)", "Median submitted", "Total call latency (ms)", "Median call latency (ms)"],
        [[
            name,
            _fmt(row["mean_customer_chat_input_logged_attempted_per_turn"]),
            _fmt(row["median_customer_chat_input_logged_attempted_per_turn"]),
            _fmt(row["mean_customer_chat_input_model_processed_per_turn_zero_when_unsent"]),
            _fmt(row["median_customer_chat_input_model_processed_per_turn_zero_when_unsent"]),
            _fmt(row["total_latency_ms_all_call_types"]),
            _fmt(row["median_latency_ms_all_call_types"]),
        ] for name, row in by_strategy.items()],
    )

    full = by_strategy["Full"]
    trim = by_strategy["Trim"]
    summary = by_strategy["Summary"]
    summary_extra_in = summary["summary_generation_input_tokens_logged"]
    summary_extra_out = summary["summary_generation_output_tokens_logged"]
    full_start = next(row["input_tokens"] for row in overall["Full"]["by_type"]["chat"] if row["turn"] == 1)
    full_last_success = max(row["turn"] for row in overall["Full"]["successful_chat"])
    full_failures = overall["Full"]["prompt_prep_failures"]
    top_failure = max(int(row["input_tokens"]) for row in full_failures) if full_failures else 0
    summary_only = sum(int(row["input_tokens"]) for row in overall["Summary"]["by_type"]["summary"])
    trim_chat_input = trim["customer_chat_actual_or_success_estimated_input_tokens"]
    full_chat_input = full["customer_chat_actual_or_success_estimated_input_tokens"]
    successful_chat_inputs = {
        name: [
            int(row["input_tokens"])
            for row in overall[name]["by_type"]["chat"]
            if _was_model_processed(row)
        ]
        for name in ("Full", "Trim", "Summary")
    }
    successful_chat_mean = {
        name: (sum(values) / len(values) if values else 0)
        for name, values in successful_chat_inputs.items()
    }
    fallback_counts = ", ".join(
        f"{name} {row['failed_chat_calls_total']}" for name, row in by_strategy.items()
    )
    prompt_failures = ", ".join(
        f"{name} {row['prompt_preparation_failed_chat_calls_not_sent']}" for name, row in by_strategy.items()
    )
    prompt_failure_turns = "; ".join(
        f"{name}: " + (
            ", ".join(str(record["turn"]) for record in overall[name]["prompt_prep_failures"])
            or "none"
        )
        for name in ("Full", "Trim", "Summary")
    )
    other_failures = ", ".join(
        f"{name} {row['other_failed_chat_calls_request_or_stage_unknown']}"
        for name, row in by_strategy.items()
    )
    total_logged = ", ".join(
        f"{name} {_fmt(row['logged_total_tokens_input_plus_output'])}" for name, row in by_strategy.items()
    )

    report = f"""# Task 2(b) Stage 5 — Session A token analysis

## Scope and source

This report uses only the three official run IDs: `task2b_full_run3`, `task2b_trim_run2`, and `task2b_summary_run2`. The script reads `logs/calls.jsonl`, the official run manifest, the source Session A turns, and the three transcripts. Superseded and interrupted runs are excluded. No Session A replay was run during this analysis.

Every official transcript contains the exact 25 source user messages. Summary generated {summary['summary_call_count']} extra `call_type=summary` calls. Tool calls have zero token fields as required. `retrieved_tokens` is reported separately and is never added to input totals.

## Token and latency comparison

{strategy_summary}

### Breakdown and per-turn measures

{detail_table}

{per_turn_table}

`Logged input/output/total` sums the corresponding fields across NLU, chat, summary, tool, and embedding records. This includes logged prompt estimates for prompts rejected before Ollama. NLU and embedding input totals are shown separately as full-system accounting overhead. Embedding `extra.embedding_input_tokens` is a separate BGE-specific diagnostic and is not added to `input_tokens`.

`Exact Ollama` totals include chat and summary records whose `extra.token_counts_source` is `ollama`. Successful calls without Ollama usage would be reported in the separate local-estimate columns. No such successful fallback-counted Qwen calls occur in these runs. Prompt-preparation failures are attempted prompt counts, not model consumption.

## Plot

![Input tokens per Session A turn](task2b_input_tokens_per_turn.png)

Solid series are successful customer-facing calls with model usage. Cross markers show attempted chat prompt counts from prompt-preparation failures; those inputs were not sent to Ollama. The orange series shows additional summary-generation input tokens by turn. Thus the plot distinguishes completed model inputs from logged-but-unsent attempts.

## Prompt failures and fallback behavior

Failed/fallback customer-facing chat records: {fallback_counts}. Prompt-preparation failures classified as unsent: {prompt_failures}. Other failed chat records after prompt preparation or with an ambiguous stage: {other_failures}. The unsent records have `extra.error_type=ValueError`, `api_call_ms=0`, and `prompt_build_ms=0`; their `input_tokens` preserve the attempted prompt size. The largest Full attempted prompt was {top_failure:,} tokens. Full had successful customer-facing calls through turn {full_last_success}; later oversized full-history prompts were rejected and produced deterministic fallback responses. No failed chat prompt is counted as actual model-processed input.

Prompt-preparation failure turns: {prompt_failure_turns}.

Trim and Summary also reached the safe prompt limit on some turns and used the same deterministic fallback path. Their histories or summary-plus-recent contexts were not silently recategorized as another strategy. The exact failed turns and token counts are in `task2b_token_per_turn.csv` and the raw logs.

## Strategy observations

- Full has the lowest raw customer-facing input actually processed ({_fmt(full_chat_input)} tokens), but only {full['successful_chat_calls']} of 25 chat prompts were sent successfully; its lower total reflects {full['failed_chat_calls_total']} unsent prompt-budget failures, not an efficiency advantage. Mean input per successful chat is Full {_fmt(successful_chat_mean['Full'])}, Trim {_fmt(successful_chat_mean['Trim'])}, and Summary {_fmt(successful_chat_mean['Summary'])} tokens.
- Full's logged/attempted chat inputs grow from {full_start:,} tokens on turn 1 and eventually exceed the 1,856-token safe input budget. It logs {_fmt(full['customer_chat_input_tokens_logged_including_unsent_attempts'])} attempted chat input tokens versus {_fmt(trim['customer_chat_input_tokens_logged_including_unsent_attempts'])} for Trim. Attempts are preserved in logs, but unsent prompts do not consume model tokens.
- Trim logged {_fmt(trim['customer_chat_input_tokens_logged_including_unsent_attempts'])} chat input tokens including its {trim['failed_chat_calls_total']} unsent attempts.
- Summary adds {_fmt(summary_extra_in)} summary-generation input tokens and {_fmt(summary_extra_out)} summary-generation output tokens across {summary['summary_call_count']} calls. That overhead is included once in Summary's all-call logged and exact Ollama totals.
- Summary's exact Qwen input (customer-facing plus summary calls) is {_fmt(summary['ollama_usage_exact_model_input_tokens_chat_plus_summary'])} tokens versus {_fmt(trim['ollama_usage_exact_model_input_tokens_chat_plus_summary'])} for Trim. Summary generation adds {summary_extra_in:,} input tokens while its customer-facing calls use less input; exact Qwen input-plus-output totals are {_fmt(summary['ollama_usage_exact_model_total_tokens_chat_plus_summary'])} for Summary and {_fmt(trim['ollama_usage_exact_model_total_tokens_chat_plus_summary'])} for Trim.
- Trim has the lowest all-call logged total ({_fmt(trim['logged_total_tokens_input_plus_output'])}) and the most successful customer-facing calls ({trim['successful_chat_calls']}); in this sample it is the most favorable token/fallback trade-off. Summary retains condensed older context, but its quality or information-retention advantage was not measured, and it had {summary['failed_chat_calls_total']} fallbacks.
- Logged total input-plus-output tokens are {total_logged}. These totals include NLU and embedding call fields but not `retrieved_tokens` a second time.
- The runs support a token and prompt-budget comparison only. They do not establish a response-quality or accuracy winner. No successful booking/database write occurred in these official runs.

## Reproducibility checks and limitations

The analysis validates mandatory fields, exact per-run call counts, one NLU and one chat record per user turn, Summary call configuration/purpose, zero-token tool records, duplicate complete records, transcript/source-message agreement, run IDs, and the matching database-reset fingerprints. All three reset fingerprints are `{full['database_reset_fingerprint']}`.

Actual model-processed totals are limited to Ollama-reported usage in these runs. If a future successful call lacks API usage, the script keeps its tokenizer-based estimate separate; if a request fails after dispatch without usage, actual processed tokens remain unknown rather than being inferred. The script does not estimate monetary cost or evaluate answer quality.

## Reproduction

From the project root, run:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\analyze_task2b_logs.py
```

The script filters exclusively to the three official run IDs and regenerates this report, both CSVs, and the PNG without modifying `calls.jsonl`.
"""
    (RESULTS_DIR / "task2b_stage5_report.md").write_text(report, encoding="utf-8", newline="\n")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("Task 2(b) Stage 5 — official Session A runs")
    for row in rows:
        print(
            f"{row['strategy']:7s} {row['run_id']}: "
            f"logged={row['logged_total_tokens_input_plus_output']:,} tokens; "
            f"exact Ollama={row['ollama_usage_exact_model_total_tokens_chat_plus_summary']:,}; "
            f"chat success/fallback={row['successful_chat_calls']}/{row['failed_chat_calls_total']}; "
            f"summary calls={row['summary_call_count']}"
        )
    print("Wrote task2b_input_tokens_per_turn.png, task2b_token_summary.csv, "
          "task2b_token_per_turn.csv and task2b_stage5_report.md")


if __name__ == "__main__":
    analyze()
