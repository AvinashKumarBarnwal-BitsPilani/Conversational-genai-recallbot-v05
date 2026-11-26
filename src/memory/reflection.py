"""End-of-session reflection (Task 3(d)).

One Qwen call (call_type=reflect) turns the session's notable episodes and the
customer's active facts into 1-3 higher-level insight sentences, which are
written back into the episodic stream as a ``reflection`` episode.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .llm_calls import LoggedLLM
from .records import Episode, SemanticFact, key_label

REFLECTION_MIN_IMPORTANCE = 4
MAX_REFLECTION_CHARS = 700

REFLECTION_SYSTEM_PROMPT = (
    "You write a short reflection for an e-commerce support assistant's long-term memory. "
    "Using only the listed events and saved facts, write 1 to 3 sentences of higher-level insight "
    "that will help in a future session: open issues, commitments made, and stable preferences. "
    "Copy order IDs, SKUs, dates and booking or RMA references exactly. Do not invent anything. "
    "Return only the sentences."
)


def reflection_inputs(
    episodes: Sequence[Episode], facts: Sequence[SemanticFact]
) -> tuple[list[Episode], dict[str, Any]]:
    selected = [
        ep for ep in episodes
        if ep.type != "reflection" and ep.importance >= REFLECTION_MIN_IMPORTANCE
    ]
    payload = {
        "events": [
            {"id": ep.episode_id, "type": ep.type, "turn": ep.source_turn, "description": ep.description}
            for ep in selected
        ],
        "saved_facts": [
            {"id": fact.memory_id, "fact": key_label(fact.key), "value": fact.value} for fact in facts
        ],
    }
    return selected, payload


def generate_reflection(
    llm: LoggedLLM,
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> str:
    text = llm.call(
        [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        call_type="reflect",
        purpose="end_of_session_reflection",
        context=context,
        options={"num_predict": 160},
        extra={"input_event_ids": [event["id"] for event in payload.get("events", [])]},
    )
    text = " ".join(text.split())
    if len(text) > MAX_REFLECTION_CHARS:
        text = text[:MAX_REFLECTION_CHARS].rsplit(" ", 1)[0]
    return text
