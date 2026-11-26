"""Turn user utterances into keyed semantic facts (Task 3(c)).

Pass 1 (rules) maps the rule-based NLU entities onto the canonical keys. No
rule was added for any specific scripted sentence; the mapping reuses the
Day 3 NLU entities only.

Pass 2 (Qwen fallback) runs only for a *fact-like* turn where the rules found
nothing. Its JSON output is validated before anything is stored. Filler turns
(greeting, small talk, closing, confirmations) never reach either pass.

Every attempted pass writes one ``call_type=extract`` row to calls.jsonl.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..logger import log_call, timer
from .llm_calls import LoggedLLM, default_token_counter
from .records import CANONICAL_FACT_KEYS

FACT_INTENTS = frozenset({"provide_user_fact", "update_user_fact"})
NEVER_EXTRACT_INTENTS = frozenset({
    "greeting", "small_talk", "closing", "confirm_action", "cancel_action",
    "forget_request", "forgotten_memory_query", "semantic_memory_query",
    "episodic_memory_query", "context_budget_query",
})
# First-person preference cues used only to decide whether a turn the NLU did
# not label as a fact is still worth one fallback extraction attempt.
FACT_CUE_RE = re.compile(
    r"\b(i prefer|i'd prefer|i would prefer|my (?:preferred|usual|favourite|favorite)|"
    r"i usually|i always|please always|from now on|going forward|call me|my name is|"
    r"i live in|i'm based in|i am based in|remember that i|i'd like you to remember)\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(r"\b(password|otp|one[- ]time password|cvv|pin|card number)\b", re.IGNORECASE)
_SNAKE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ExtractedFact:
    key: str
    value: str
    method: str  # "rule" | "llm"
    rule_id: str | None = None


@dataclass
class ExtractionResult:
    facts: list[ExtractedFact] = field(default_factory=list)
    attempted: bool = False
    fallback_used: bool = False
    status: str = "skipped"  # skipped | rules | llm | none | llm_rejected | llm_failed
    rejected: list[dict[str, Any]] = field(default_factory=list)


def _rule_facts(nlu: Mapping[str, Any]) -> list[ExtractedFact]:
    """Map NLU entities to canonical semantic keys."""
    entities = nlu.get("entities") or {}
    facts: list[ExtractedFact] = []

    def add(key: str, value: Any, rule_id: str) -> None:
        if value not in (None, ""):
            facts.append(ExtractedFact(key, str(value), "rule", rule_id))

    add("preferred_name", entities.get("preferred_name"), "FX-NAME")
    add("notification_channel", entities.get("notification_channel"), "FX-CHANNEL")
    add("preferred_city", entities.get("city"), "FX-CITY")
    add("delivery_note", entities.get("delivery_note"), "FX-DELIVERY-NOTE")
    method = entities.get("return_method")
    add("return_method_preference", str(method).replace("_", " ") if method else None, "FX-RETURN-METHOD")
    add("packaging_preference", entities.get("packaging_preference"), "FX-PACKAGING")
    window = entities.get("time_window")
    if isinstance(window, Mapping) and window.get("start") and window.get("end"):
        # Only a bounded window on a fact turn is a delivery-window preference;
        # an open "after 6 pm" belongs to the pickup slot-filling task.
        add("preferred_delivery_window", f"{window['start']}-{window['end']}", "FX-TIME-WINDOW")
    return facts


def is_fact_like(text: str, nlu: Mapping[str, Any]) -> bool:
    intent = str(nlu.get("intent", "unknown"))
    if intent in NEVER_EXTRACT_INTENTS:
        return False
    if intent in FACT_INTENTS:
        return True
    return intent == "unknown" and bool(FACT_CUE_RE.search(text))


_LLM_SYSTEM_PROMPT = (
    "You extract durable customer profile facts for an e-commerce support assistant. "
    "Return JSON only, in the form {\"facts\": [{\"key\": \"...\", \"value\": \"...\"}]}. "
    "Use one of these keys when it fits: " + ", ".join(CANONICAL_FACT_KEYS) + ". "
    "Only if none fits, use a short snake_case key. Extract a fact only when the customer states "
    "a lasting preference or profile detail about themselves. The value must be short and taken "
    "from the message. Never extract passwords, OTPs or payment details. "
    "If there is no such fact, return {\"facts\": []}."
)


class FactExtractor:
    def __init__(
        self,
        *,
        llm: LoggedLLM | None = None,
        enable_llm_fallback: bool = True,
        logger: Callable[..., Any] | None = log_call,
        token_counter: Callable[[str], int] = default_token_counter,
    ) -> None:
        self.enable_llm_fallback = enable_llm_fallback
        self.llm = llm if llm is not None else (LoggedLLM(logger=logger) if enable_llm_fallback else None)
        self.logger = logger
        self.token_counter = token_counter

    def extract(self, text: str, nlu: Mapping[str, Any], *, context: Mapping[str, Any]) -> ExtractionResult:
        if not is_fact_like(text, nlu) or _SECRET_RE.search(text):
            return ExtractionResult(status="skipped")
        result = ExtractionResult(attempted=True)
        started = timer()
        result.facts = _rule_facts(nlu)
        result.status = "rules" if result.facts else "none"
        self._log_rule_pass(text, nlu, result, context, started)
        if result.facts or not (self.enable_llm_fallback and self.llm is not None):
            return result

        result.fallback_used = True
        try:
            raw = self.llm.call(
                [{"role": "system", "content": _LLM_SYSTEM_PROMPT}, {"role": "user", "content": text}],
                call_type="extract",
                purpose="semantic_fact_extraction_llm_fallback",
                context=context,
                json_mode=True,
                options={"num_predict": 120},
            )
        except Exception:
            result.status = "llm_failed"
            return result
        accepted, rejected = self._validate_llm_output(raw, text)
        result.facts, result.rejected = accepted, rejected
        result.status = "llm" if accepted else ("llm_rejected" if rejected else "none")
        return result

    def _log_rule_pass(
        self, text: str, nlu: Mapping[str, Any], result: ExtractionResult,
        context: Mapping[str, Any], started: float,
    ) -> None:
        if self.logger is None:
            return
        self.logger(
            config=str(context.get("config", "hybrid_mem")),
            user=str(context.get("user", "UNKNOWN")),
            session=str(context.get("session", "A")),
            turn=int(context.get("turn", 0)),
            call_type="extract",
            model="rule-based",
            input_tokens=self.token_counter(text),
            output_tokens=0,
            retrieved_tokens=0,
            start_time=started,
            extra={
                "purpose": "semantic_fact_extraction_rules",
                "intent": nlu.get("intent"),
                "facts": [{"key": f.key, "rule_id": f.rule_id} for f in result.facts],
            },
        )

    @staticmethod
    def _validate_llm_output(raw: str, text: str) -> tuple[list[ExtractedFact], list[dict[str, Any]]]:
        """Accept only well-formed, grounded, non-secret facts."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return [], [{"reason": "invalid_json"}]
        items = payload.get("facts", []) if isinstance(payload, Mapping) else []
        if not isinstance(items, list):
            return [], [{"reason": "facts_not_a_list"}]
        source_words = set(_WORD_RE.findall(text.casefold()))
        accepted: list[ExtractedFact] = []
        rejected: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                rejected.append({"reason": "not_an_object"})
                continue
            key = str(item.get("key", "")).strip().casefold()
            value = " ".join(str(item.get("value", "")).split())
            reason = None
            if not (key in CANONICAL_FACT_KEYS or _SNAKE_KEY_RE.match(key)):
                reason = "invalid_key"
            elif not value or len(value) > 80:
                reason = "invalid_value_length"
            elif _SECRET_RE.search(value) or _SECRET_RE.search(key.replace("_", " ")):
                reason = "secret"
            elif not set(_WORD_RE.findall(value.casefold())) & source_words:
                reason = "value_not_grounded_in_message"
            if reason:
                rejected.append({"key": key, "reason": reason})
            else:
                accepted.append(ExtractedFact(key, value, "llm", "FX-LLM"))
        return accepted, rejected
