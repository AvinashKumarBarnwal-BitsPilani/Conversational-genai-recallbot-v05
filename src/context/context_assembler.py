"""Budgeted context assembly for the V05 2,048-token window (Task 3(e), Twist E).

The context manager, not the model, decides what fits:

    input budget = context window - reserved output tokens - safety margin
                 = 2,048 - 160 - 32 = 1,856 tokens (task turns)

Items are counted with the Qwen3 tokenizer, using the model's real chat
template when the tokenizer is cached locally. Mandatory items (system prompt,
current request, dialogue state, draft) are always kept. Optional items
(long-term memories, KB chunks, history exchanges) are added best-first in an
intent-dependent priority order; whatever does not fit is dropped *whole* and
recorded with its size and reason, which is the evidence of "what gets cut".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..config import CONTEXT_WINDOW_TOKENS, TOKENIZER_ID

FALLBACK_SENTENCE = (
    "I don't have that information in my saved details or the ShopSphere knowledge base, so I won't guess."
)

# Measured on pilot run t3_hm_pilot_B: Ollama's prompt_eval_count exceeded the
# Hugging Face Qwen3 chat-template count by exactly 8 tokens on all 26 chat
# calls (the empty <think></think> block Ollama inserts when think=False).
OLLAMA_TEMPLATE_OFFSET = 8

TOOLS_TEXT = (
    "TOOLS (run by the dialogue manager, not by you): search_kb(query) hybrid KB search; "
    "get_order(order_id) verified order read; check_pickup_slots(date, window) availability read; "
    "book_slot(details) pickup write, only after the customer's explicit yes."
)

SECTION_ORDER = ("CURRENT REQUEST", "DIALOGUE STATE", "DRAFT", "SAVED FACTS", "PAST EVENTS", "KNOWLEDGE BASE")

# Base priority per component for each intent profile (lower is kept first).
PRIORITY_PROFILES: dict[str, dict[str, int]] = {
    "memory": {"long_term_memory": 10, "retrieved_knowledge": 30, "conversation_history": 40},
    "knowledge": {"retrieved_knowledge": 10, "long_term_memory": 30, "conversation_history": 40},
    "task": {"long_term_memory": 20, "retrieved_knowledge": 20, "conversation_history": 30},
}
MEMORY_INTENTS = {"semantic_memory_query", "episodic_memory_query", "forgotten_memory_query"}
KNOWLEDGE_INTENTS = {"knowledge_base_query", "state_followup", "order_cancel_eligibility_query"}


def priority_profile(intent: str) -> str:
    if intent in MEMORY_INTENTS:
        return "memory"
    if intent in KNOWLEDGE_INTENTS:
        return "knowledge"
    return "task"


@dataclass
class ContextItem:
    item_id: str
    component: str      # system | instructions | tools | current_task | long_term_memory | retrieved_knowledge | conversation_history
    memory_type: str    # procedural | working | semantic | episodic | semantic_kb | short_term
    home: str           # model-native | external
    text: str
    section: str        # system | history | one of SECTION_ORDER
    priority: int = 0
    mandatory: bool = False
    messages: list[dict[str, str]] = field(default_factory=list)  # history items only
    tokens: int = 0


@dataclass
class AssembledContext:
    messages: list[dict[str, str]]
    included: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    total_input_tokens: int
    input_budget: int
    output_reserve: int
    context_window: int
    retrieved_tokens: int
    overhead_tokens: int
    token_count_method: str
    profile: str

    def summary(self) -> dict[str, Any]:
        """Compact form for the chat call's ``extra.context_assembly`` log field."""
        return {
            "profile": self.profile,
            "total_input_tokens": self.total_input_tokens,
            "input_budget": self.input_budget,
            "output_reserve": self.output_reserve,
            "context_window": self.context_window,
            "retrieved_tokens": self.retrieved_tokens,
            "overhead_tokens": self.overhead_tokens,
            "token_count_method": self.token_count_method,
            "included": [{k: item[k] for k in ("id", "component", "tokens")} for item in self.included],
            "dropped": [{k: item[k] for k in ("id", "component", "tokens", "reason")} for item in self.dropped],
        }

    def component_table(self) -> list[dict[str, Any]]:
        """Rows in the style of the CS5 context-assembly example."""
        rows: dict[str, dict[str, Any]] = {}
        for item in self.included:
            row = rows.setdefault(item["component"], {
                "component": item["component"], "memory_type": set(), "home": set(), "tokens": 0, "ids": [],
            })
            row["memory_type"].add(item["memory_type"])
            row["home"].add(item["home"])
            row["tokens"] += item["tokens"]
            row["ids"].append(item["id"])
        table = [
            {**row, "memory_type": "/".join(sorted(row["memory_type"])), "home": "/".join(sorted(row["home"]))}
            for row in rows.values()
        ]
        table.append({"component": "chat template and section headings", "memory_type": "-", "home": "model-native",
                      "tokens": self.overhead_tokens, "ids": []})
        return table


class ContextAssembler:
    def __init__(
        self,
        *,
        context_window: int = CONTEXT_WINDOW_TOKENS,
        safety_margin: int = 32,
        tokenizer: Any | None = None,
        token_counter: Callable[[str], int] | None = None,
        per_message_overhead: int = 5,
    ) -> None:
        self.context_window = int(context_window)
        self.safety_margin = int(safety_margin)
        self.per_message_overhead = int(per_message_overhead)
        self._tokenizer = tokenizer
        self._tokenizer_attempted = tokenizer is not None or token_counter is not None
        self._token_counter = token_counter

    # ------------------------------------------------------------- counting
    def _get_tokenizer(self) -> Any | None:
        if not self._tokenizer_attempted:
            self._tokenizer_attempted = True
            try:
                from ..retrieval.tokenizer import load_tokenizer
                self._tokenizer = load_tokenizer(TOKENIZER_ID, local_files_only=True)
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def count(self, text: str) -> int:
        if self._token_counter is not None:
            return int(self._token_counter(text))
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return (len(text) + 2) // 3 if text else 0

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, str]:
        tokenizer = None if self._token_counter is not None else self._get_tokenizer()
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                [dict(m) for m in messages], tokenize=True, add_generation_prompt=True,
            )
            # transformers 5.x returns a BatchEncoding (input_ids, attention_mask);
            # older versions return a plain list of IDs. len() of the former is 2.
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return len(ids) + OLLAMA_TEMPLATE_OFFSET, "qwen3_chat_template+ollama_offset"
        total = sum(self.count(str(m["content"])) for m in messages)
        return total + self.per_message_overhead * (len(messages) + 1), "content_plus_overhead_estimate"

    def truncate(self, text: str, max_tokens: int) -> str:
        tokenizer = None if self._token_counter is not None else self._get_tokenizer()
        if tokenizer is not None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens]).strip() + " ..."
        words = text.split()
        while words and self.count(" ".join(words)) > max_tokens:
            words = words[: max(1, len(words) * 7 // 8 - 1)] if len(words) > 1 else []
        return " ".join(words) + (" ..." if " ".join(words) != text else "")

    # ------------------------------------------------------------- assembly
    @staticmethod
    def build_messages(items: Sequence[ContextItem]) -> list[dict[str, str]]:
        system_parts = [item.text for item in items if item.section == "system"]
        messages: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for item in items:
            if item.section == "history":
                messages.extend(dict(m) for m in item.messages)
        blocks: list[str] = []
        for heading in SECTION_ORDER:
            texts = [item.text for item in items if item.section == heading]
            if texts:
                blocks.append(f"{heading}:\n" + "\n".join(texts))
        messages.append({"role": "user", "content": "\n\n".join(blocks)})
        return messages

    def assemble(self, items: Sequence[ContextItem], *, output_reserve: int, profile: str) -> AssembledContext:
        budget = self.context_window - int(output_reserve) - self.safety_margin
        for item in items:
            item.tokens = self.count(item.text)
        order = {id(item): position for position, item in enumerate(items)}
        mandatory = [item for item in items if item.mandatory]
        optional = sorted((item for item in items if not item.mandatory), key=lambda i: (i.priority, order[id(i)]))

        def in_display_order(selected: list[ContextItem]) -> list[ContextItem]:
            return sorted(selected, key=lambda i: order[id(i)])

        total, method = self.count_messages(self.build_messages(in_display_order(mandatory)))
        if total > budget:
            raise ValueError(f"Mandatory context needs {total} tokens, above the input budget {budget}")
        selected = list(mandatory)
        dropped: list[dict[str, Any]] = []
        for item in optional:
            candidate = in_display_order(selected + [item])
            candidate_total, method = self.count_messages(self.build_messages(candidate))
            if candidate_total <= budget:
                selected.append(item)
                total = candidate_total
            else:
                dropped.append({**self._describe(item), "reason": f"budget: would reach {candidate_total} > {budget}"})
        final_items = in_display_order(selected)
        messages = self.build_messages(final_items)
        total, method = self.count_messages(messages)
        included = [self._describe(item) for item in final_items]
        retrieved = sum(i["tokens"] for i in included if i["component"] in {"long_term_memory", "retrieved_knowledge"})
        return AssembledContext(
            messages=messages, included=included, dropped=dropped, total_input_tokens=total,
            input_budget=budget, output_reserve=int(output_reserve), context_window=self.context_window,
            retrieved_tokens=retrieved, overhead_tokens=total - sum(i["tokens"] for i in included),
            token_count_method=method, profile=profile,
        )

    @staticmethod
    def _describe(item: ContextItem) -> dict[str, Any]:
        return {"id": item.item_id, "component": item.component, "memory_type": item.memory_type,
                "home": item.home, "tokens": item.tokens, "priority": item.priority, "mandatory": item.mandatory}


# ---------------------------------------------------------------- item builders
def build_turn_items(
    *,
    intent: str,
    system_prompt: str,
    system_prompt_version: str,
    workflow_step: str | None,
    user_message: str,
    state: Mapping[str, Any],
    draft: str,
    facts: Sequence[tuple[Any, float]] = (),
    episodes: Sequence[Any] = (),
    kb_hits: Sequence[Mapping[str, Any]] = (),
    history: Sequence[Mapping[str, str]] = (),
    truncate: Callable[[str, int], str] = lambda text, n: text,
    max_chunk_tokens: int = 180,
) -> tuple[list[ContextItem], str]:
    """Turn one dialogue turn's material into budget items. Returns (items, profile)."""
    profile = priority_profile(intent)
    base = PRIORITY_PROFILES[profile]
    items: list[ContextItem] = [
        ContextItem(system_prompt_version, "system", "procedural", "external file skills/ -> prompt",
                    system_prompt, "system", mandatory=True),
        ContextItem("TOOLS", "tools", "procedural", "model-native (prompt)", TOOLS_TEXT, "system", mandatory=True),
    ]
    if workflow_step:
        items.append(ContextItem("WORKFLOW-STEP", "instructions", "procedural", "external file skills/ -> prompt",
                                 f"CURRENT WORKFLOW STEP: {workflow_step}", "system", mandatory=True))
    items += [
        ContextItem("USER-TURN", "current_task", "working", "model-native (prompt)", user_message,
                    "CURRENT REQUEST", mandatory=True),
        ContextItem("STATE", "current_task", "working", "model-native (prompt)",
                    json.dumps(state, ensure_ascii=False, default=str, separators=(",", ":")),
                    "DIALOGUE STATE", mandatory=True),
        ContextItem("DRAFT", "current_task", "working", "model-native (prompt)", draft, "DRAFT", mandatory=True),
    ]
    for rank, (fact, score) in enumerate(facts):
        items.append(ContextItem(
            fact.memory_id, "long_term_memory", "semantic", "external (semantic store + vector index)",
            f"[{fact.memory_id}] {fact.key.replace('_', ' ')}: {fact.value} (saved {fact.date[:10]}, turn {fact.source_turn})",
            "SAVED FACTS", priority=base["long_term_memory"] + rank,
        ))
    for rank, scored in enumerate(episodes):
        ep = scored.episode
        items.append(ContextItem(
            ep.episode_id, "long_term_memory", "episodic", "external (episodic stream + vector index)",
            f"[{ep.episode_id}] {ep.timestamp[:16].replace('T', ' ')} {ep.type}: {ep.description}",
            "PAST EVENTS", priority=base["long_term_memory"] + len(facts) + rank,
        ))
    for rank, hit in enumerate(kb_hits):
        content = truncate(str(hit.get("content", "")).strip(), max_chunk_tokens)
        label = f"{hit.get('doc_id')} v{hit.get('version')} {hit.get('status')}, effective {hit.get('effective_from')}"
        items.append(ContextItem(
            str(hit.get("chunk_id")), "retrieved_knowledge", "semantic_kb", "external (KB index, hybrid RRF)",
            f"[{hit.get('chunk_id')}] ({label}) {content}", "KNOWLEDGE BASE",
            priority=base["retrieved_knowledge"] + rank,
        ))
    exchanges = [list(history[i:i + 2]) for i in range(0, len(history) - 1, 2)]
    for age, exchange in enumerate(reversed(exchanges)):
        items.append(ContextItem(
            f"HISTORY-{len(exchanges) - age}", "conversation_history", "short_term", "model-native (prompt)",
            "\n".join(str(m["content"]) for m in exchange), "history",
            priority=base["conversation_history"] + age, messages=[dict(m) for m in exchange],
        ))
    # History items must appear in chronological order in the prompt.
    history_items = sorted((i for i in items if i.section == "history"), key=lambda i: int(i.item_id.split("-")[1]))
    items = [i for i in items if i.section != "history"] + history_items
    # Keep the system/history/payload display order stable: system first, then history, then payload.
    items = ([i for i in items if i.section == "system"] + [i for i in items if i.section == "history"]
             + [i for i in items if i.section not in {"system", "history"}])
    return items, profile


def assembly_record(assembled: AssembledContext) -> dict[str, Any]:
    """Full record (with the component table) for results files."""
    return {**assembled.summary(), "component_table": assembled.component_table()}
