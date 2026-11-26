"""Rule-based NLU for RecallBot V05 (Task 2a).

The NLU is deliberately deterministic and explainable. It extracts a small,
project-specific set of entities and assigns one primary intent. The dialogue
manager will use the current dialogue state to interpret state-dependent turns
such as "And for Saturday?"; the NLU does not pretend to resolve that turn in
isolation.

No LLM, LangChain, LlamaIndex or external classification service is used here.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# --- Entity patterns -------------------------------------------------------
_ORDER_ID_RE = re.compile(r"\bORD-\d{4,}\b", re.IGNORECASE)
_SKU_RE = re.compile(r"\bSKU-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)
# Some scripted utterances use the short product code (e.g. AUR-220) rather
# than its inventory form (SKU-AUR-220).
_SHORT_PRODUCT_CODE_RE = re.compile(r"\b(?:AUR|PHC|CHG|BAG|MUG)-[A-Z0-9]+\b", re.IGNORECASE)
_DOMAIN_CODE_RE = re.compile(
    r"\b(?:ERR-[A-Z0-9]+(?:-[A-Z0-9]+)*|RMA-[A-Z0-9]+(?:-[A-Z0-9]+)*|"
    r"RET-[A-Z0-9]+(?:-[A-Z0-9]+)*|PAY-[A-Z0-9]+(?:-[A-Z0-9]+)*|"
    r"SHP-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
# Exact 24-hour slot labels are shown by the pickup tool (e.g. 18:00–20:00).
# Accept the same format when the user selects one of those displayed options.
_CLOCK_TIME_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start_h>\d{1,2}):(?P<start_m>\d{2})"
    r"\s*(?:to|[-–])\s*"
    r"(?P<end_h>\d{1,2}):(?P<end_m>\d{2})(?!\d)",
    re.IGNORECASE,
)
_TIME_RANGE_RE = re.compile(
    r"(?:between\s+)?"
    r"(?P<start_h>\d{1,2})(?::(?P<start_m>\d{2}))?\s*(?P<start_ap>am|pm)"
    r"\s*(?:to|and|[-–])\s*"
    r"(?P<end_h>\d{1,2})(?::(?P<end_m>\d{2}))?\s*(?P<end_ap>am|pm)\b",
    re.IGNORECASE,
)
_AFTER_TIME_RE = re.compile(
    r"\bafter\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    """Normalise whitespace and case for matching; preserve original elsewhere."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _strip_outer_quotes(text: str) -> str:
    """Remove one or more balanced pairs of wrapping quotes before rule matching.

    Users often copy a suggested utterance including its surrounding quotation
    marks. Without this cleanup, anchored rules such as confirm_action fail to
    match a message beginning with a quote character.
    """
    cleaned = text.strip()
    quote_pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`")}
    while len(cleaned) >= 2 and (cleaned[0], cleaned[-1]) in quote_pairs:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _to_24_hour(hour_text: str, ampm: str, minute_text: str | None = None) -> str:
    """Convert a 12-hour clock value to HH:MM."""
    hour = int(hour_text)
    minute = int(minute_text or 0)
    marker = ampm.casefold()
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        raise ValueError("Invalid 12-hour time")
    if marker == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return f"{hour:02d}:{minute:02d}"


def _normalise_24_hour(hour_text: str, minute_text: str) -> str:
    """Validate an already-24-hour clock value and return HH:MM."""
    hour, minute = int(hour_text), int(minute_text)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Invalid 24-hour time")
    return f"{hour:02d}:{minute:02d}"


def extract_entities(text: str) -> dict[str, Any]:
    """Extract entities used by the e-commerce order/return workflow.

    Values are returned in normalised forms where practical. The raw user
    utterance remains available to the dialogue manager for clarification.
    """
    entities: dict[str, Any] = {}

    order_ids = list(dict.fromkeys(m.upper() for m in _ORDER_ID_RE.findall(text)))
    if order_ids:
        entities["order_ids"] = order_ids

    skus = list(dict.fromkeys(m.upper() for m in _SKU_RE.findall(text)))
    short_codes = [m.upper() for m in _SHORT_PRODUCT_CODE_RE.findall(text)]
    for code in short_codes:
        canonical = code if code.startswith("SKU-") else f"SKU-{code}"
        if canonical not in skus:
            skus.append(canonical)
    if skus:
        entities["skus"] = skus

    domain_codes = list(dict.fromkeys(m.upper() for m in _DOMAIN_CODE_RE.findall(text)))
    if domain_codes:
        entities["domain_codes"] = domain_codes

    date_match = _DATE_RE.search(text)
    if date_match:
        parsed = datetime.strptime(
            f"{date_match.group('day')} {date_match.group('month')} {date_match.group('year')}",
            "%d %B %Y",
        )
        entities["date"] = parsed.strftime("%Y-%m-%d")

    clock_range_match = _CLOCK_TIME_RANGE_RE.search(text)
    range_match = _TIME_RANGE_RE.search(text)
    if clock_range_match:
        # Parse exact values copied from the options displayed by RecallBot.
        entities["time_window"] = {
            "start": _normalise_24_hour(clock_range_match.group("start_h"), clock_range_match.group("start_m")),
            "end": _normalise_24_hour(clock_range_match.group("end_h"), clock_range_match.group("end_m")),
            "raw": clock_range_match.group(0).strip(),
        }
    elif range_match:
        entities["time_window"] = {
            "start": _to_24_hour(range_match.group("start_h"), range_match.group("start_ap"), range_match.group("start_m")),
            "end": _to_24_hour(range_match.group("end_h"), range_match.group("end_ap"), range_match.group("end_m")),
            "raw": range_match.group(0).strip(),
        }
    else:
        after_match = _AFTER_TIME_RE.search(text)
        if after_match:
            entities["time_window"] = {
                "start": _to_24_hour(after_match.group("hour"), after_match.group("ampm"), after_match.group("minute")),
                "end": None,
                "raw": after_match.group(0).strip(),
            }

    preferred_name_match = re.search(r"\bplease\s+call\s+me\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
    if preferred_name_match:
        entities["preferred_name"] = preferred_name_match.group(1).title()
    else:
        name_match = re.search(r"\bmy name is\s+([A-Z][a-z]+)", text, re.IGNORECASE)
        if name_match:
            entities["preferred_name"] = name_match.group(1).title()

    if re.search(r"\bSMS\b", text, re.IGNORECASE):
        entities["notification_channel"] = "SMS"
    elif re.search(r"\bemail\b", text, re.IGNORECASE):
        entities["notification_channel"] = "email"

    city_match = re.search(r"\b(Bengaluru|Bangalore)\b", text, re.IGNORECASE)
    if city_match:
        entities["city"] = "Bengaluru" if city_match.group(1).casefold() in {"bengaluru", "bangalore"} else city_match.group(1)

    if re.search(r"\bside door\b", text, re.IGNORECASE):
        entities["delivery_note"] = "side door"
    if re.search(r"\bhome pickup\b", text, re.IGNORECASE):
        entities["return_method"] = "home_pickup"
    if re.search(r"\brecyclable packaging\b", text, re.IGNORECASE):
        entities["packaging_preference"] = "recyclable"

    # Extract a structured return reason so the state tracker does not need
    # to reinterpret free text after the NLU stage.
    if re.search(
        r"\bleft earbud\b.*\b(does not charge|doesn't charge|not charging)\b|"
        r"\b(does not charge|doesn't charge|not charging)\b.*\bleft earbud\b",
        text,
        re.IGNORECASE,
    ):
        entities["return_reason"] = "defective_left_earbud"
    elif re.search(r"\b(defective|faulty|not working|stopped working)\b", text, re.IGNORECASE):
        entities["return_reason"] = "defective"
    elif re.search(r"\b(damaged|broken|cracked)\b", text, re.IGNORECASE):
        entities["return_reason"] = "damaged"

    return entities


# A question either ends with "?" or opens with a question word/auxiliary.
_QUESTION = r"(?:.*\?\s*$|\s*(?:what|which|who|when|where|how|do|does|did|can|could|would|is|are|have|has)\b)"
_IS_QUESTION = rf"^(?={_QUESTION})"
_NOT_QUESTION = rf"^(?!{_QUESTION})"

# Each rule is intentionally small and readable. Order matters: specific
# action/confirmation patterns must run before broad policy or fact patterns.
_INTENT_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("forgotten_memory_query", re.compile(
        r"\b(what was|what is)\b.*\b(asked you to forget|told you to forget|forgotten)\b"
        rf"|{_IS_QUESTION}.*\b(?:ask(?:ed)? you to|told you to)\s+(?:forget|delete|remove)\b", re.I | re.S),
        "R01_FORGOTTEN_QUERY"),
    # A question about a forget request must never be treated as a new forget
    # request: that would delete data in response to a question.
    ("forget_request", re.compile(
        rf"{_NOT_QUESTION}.*(?:\b(forget|delete|remove)\b.*\b(saved|memory|remember|note|fact|preference)\b|\bforget\b)",
        re.I | re.S), "R02_FORGET"),
    ("cancel_action", re.compile(r"^\s*(?:no,?\s*)?(?:cancel (?:the|this|that) (?:pickup|booking|return)|cancel that|do not book|don't book|do not proceed|don't proceed|stop the booking|never mind|nevermind)\b", re.I), "R03A_CANCEL"),
    ("confirm_action", re.compile(r"^\s*(yes|yep|i confirm|confirmed|go ahead|please proceed)\b", re.I), "R03_CONFIRM"),
    ("request_task_summary", re.compile(r"\b(repeat|show|read back)\b.*\b(summary|details|pickup|booking)\b|\bcomplete .*summary\b", re.I), "R04_TASK_SUMMARY"),
    ("clarification_required", re.compile(r"\b(other item|another item|which item|which order|not sure which)\b", re.I), "R05_CLARIFICATION"),
    ("greeting", re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b", re.I), "R06_GREETING"),
    ("closing", re.compile(r"\b(that's all for now|thats all for now|bye|goodbye|have a good day)\b", re.I), "R07_CLOSING"),
    ("context_budget_query", re.compile(r"\b(token limit|context window|context budget|configured to respect|\d[\d,]*[- ]tokens?)\b", re.I), "R08_CONTEXT_BUDGET"),
    # Questions about what happened before (events), then about saved
    # profile details (facts). Both require a question form.
    ("episodic_memory_query", re.compile(
        r"\b(what did we arrange|what problem did i ask you to register|what did i book|what happened last time)\b"
        rf"|{_IS_QUESTION}.*\b(?:did|have) (?:we|i|you)\b.*\b(?:arrange|book|schedule|register|report|complain|"
        r"choose|chose|decide|make|start|ask you to (?:register|record|note|log))\w*\b"
        rf"|{_IS_QUESTION}.*\b(?:last time|last session|earlier session|previous(?:ly)? (?:session|visit|chat))\b",
        re.I | re.S), "R09_EPISODIC_QUERY"),
    ("semantic_memory_query", re.compile(
        r"\b(what .* do i currently prefer|what do i prefer|what is my current preference)\b"
        rf"|{_IS_QUESTION}.*\b(?:my|i|me)\b.*\b(?:prefer\w*|usual(?:ly)?|name|city|channel|"
        r"delivery (?:window|note|time|preference)|packaging|saved|save|recall|remember)\b"
        rf"|{_IS_QUESTION}.*\b(?:routine updates|notification channel)\b.*\b(?:now|currently)\b",
        re.I | re.S), "R10_SEMANTIC_QUERY"),
    ("start_return", re.compile(r"\b(need|want|would like) to return\b|\breturn the .+ from order\b", re.I), "R11_START_RETURN"),
    ("provide_return_reason", re.compile(r"\b(does not charge|doesn't charge|not charging|defective|damaged|not working|stopped working)\b", re.I), "R12_RETURN_REASON"),
    ("provide_pickup_date", _DATE_RE, "R13_PICKUP_DATE"),
    ("state_followup", re.compile(r"^\s*and\b.*\b(saturday|sunday|next|what happens|after the pickup|pickup)\b", re.I), "R14_STATE_FOLLOWUP"),
    ("update_user_fact", re.compile(r"\b(changed to|no longer|from now on|going forward|instead of email|instead of my previous|use\s+\w+\s+instead)\b", re.I), "R21_FACT_UPDATE"),
    ("provide_user_fact", re.compile(r"\b(my name is|call me|best way to send me|i usually place orders|if i am not home|my preferred|i prefer|please use)\b", re.I), "R22_USER_FACT"),
    ("provide_pickup_window", re.compile(r"\b\d{1,2}:\d{2}\s*(?:to|[-–])\s*\d{1,2}:\d{2}\b|\b(after|between|from)\s+\d{1,2}\s*(?::\d{2})?\s*(am|pm)\b|\b\d{1,2}\s*(?::\d{2})?\s*(am|pm)\s*(to|and|[-–])\s*\d{1,2}\s*(?::\d{2})?\s*(am|pm)\b", re.I), "R15_PICKUP_WINDOW"),
    ("order_cancel_eligibility_query", re.compile(r"\b(cancel|cancellation)\b.*\border\b|\border\b.*\b(cancel|cancellation)\b", re.I), "R16_CANCEL_CHECK"),
    ("order_status_query", re.compile(r"\b(status|track|where is|latest status|check the latest)\b.*\border\b|\border\b.*\b(status|track)\b", re.I), "R17_ORDER_STATUS"),
    ("report_complaint", re.compile(r"\b(frustrating|complaint|late delivery|missed delivery|bad experience)\b", re.I), "R18_COMPLAINT"),
    ("report_decision", re.compile(r"\b(chose|decided|decision)\b.*\b(exchange|refund)\b|\b(exchange rather than a refund)\b", re.I), "R19_DECISION"),
    ("knowledge_base_query", re.compile(r"\b(what is|what does|what happens|how many days|how long|policy|return window|refund|code)\b", re.I), "R20_KB_QUERY"),
    ("small_talk", re.compile(r"\b(thanks|thank you|appreciate|perfect|got it|that's helpful|thats helpful)\b", re.I), "R23_SMALL_TALK"),
)


def analyze_utterance(text: str) -> dict[str, Any]:
    """Return the primary intent, extracted entities and matched rule.

    If no rule matches, intent is ``unknown``. State-dependent interpretations
    (e.g. slot filling from "And for Saturday?") remain the dialogue manager's
    responsibility, using the returned intent and entities together with the
    current state tracker.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "intent": "unknown",
            "entities": {},
            "confidence": 0.0,
            "rule_id": "R00_EMPTY_OR_INVALID",
        }

    # Match against the utterance without balanced wrapper quotes. Entity
    # extraction still uses the original content so the user's actual values
    # remain unchanged.
    match_text = _strip_outer_quotes(text)

    for intent, pattern, rule_id in _INTENT_RULES:
        if pattern.search(match_text):
            # Confidence here represents deterministic rule-match certainty,
            # not a statistically calibrated probability.
            return {
                "intent": intent,
                "entities": extract_entities(text),
                "confidence": 1.0,
                "rule_id": rule_id,
            }

    return {
        "intent": "unknown",
        "entities": extract_entities(text),
        "confidence": 0.0,
        "rule_id": "R99_NO_MATCH",
    }
