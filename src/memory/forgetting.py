"""Forget-request helpers: key resolution, value fingerprints and redaction.

A forgotten value must not survive anywhere on disk, so tombstones keep only a
SHA-256 fingerprint of each normalised surface form plus its word count. That
is enough to detect (and redact) the value inside any later text by hashing
word n-grams, without ever storing the value itself.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from .records import key_label

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# NLU entity name -> semantic key (same mapping as the rule extractor).
ENTITY_TO_KEY: dict[str, str] = {
    "preferred_name": "preferred_name",
    "notification_channel": "notification_channel",
    "city": "preferred_city",
    "delivery_note": "delivery_note",
    "return_method": "return_method_preference",
    "packaging_preference": "packaging_preference",
    "time_window": "preferred_delivery_window",
}
# Generic wording a customer may use to name a saved detail.
KEY_ALIASES: dict[str, re.Pattern[str]] = {
    "delivery_note": re.compile(r"\bdelivery (?:note|instruction)s?\b", re.I),
    "preferred_delivery_window": re.compile(r"\bdelivery (?:window|time|slot)\b", re.I),
    "notification_channel": re.compile(r"\b(?:notification|contact) (?:channel|preference)\b|\broutine updates\b", re.I),
    "preferred_city": re.compile(r"\b(?:delivery )?city\b", re.I),
    "preferred_name": re.compile(r"\b(?:my|preferred) name\b", re.I),
    "packaging_preference": re.compile(r"\bpackaging\b", re.I),
    "return_method_preference": re.compile(r"\breturn (?:method|preference)\b", re.I),
}


def resolve_fact_keys(text: str, nlu: Mapping[str, Any]) -> list[str]:
    """Identify which saved fact(s) a request or question names explicitly."""
    return resolve_forget_keys(text, nlu)


def resolve_forget_keys(text: str, nlu: Mapping[str, Any]) -> list[str]:
    """Identify which saved fact(s) a forget request refers to."""
    keys: list[str] = []
    for entity, key in ENTITY_TO_KEY.items():
        if (nlu.get("entities") or {}).get(entity) and key not in keys:
            keys.append(key)
    for key, pattern in KEY_ALIASES.items():
        if pattern.search(text) and key not in keys:
            keys.append(key)
    return keys


def normalised_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def fingerprint(value: str) -> dict[str, Any]:
    words = normalised_words(value)
    return {
        "sha256": hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest(),
        "words": len(words),
    }


def _matching_spans(text: str, fingerprints: Iterable[Mapping[str, Any]]) -> list[tuple[int, int]]:
    tokens = list(_WORD_RE.finditer(text))
    lowered = [match.group(0).lower() for match in tokens]
    spans: list[tuple[int, int]] = []
    for fp in fingerprints:
        size = int(fp.get("words", 0))
        if size <= 0:
            continue
        for start in range(0, len(tokens) - size + 1):
            gram = " ".join(lowered[start:start + size])
            if hashlib.sha256(gram.encode("utf-8")).hexdigest() == fp["sha256"]:
                spans.append((tokens[start].start(), tokens[start + size - 1].end()))
    return spans


def contains_fingerprint(text: str, fingerprints: Iterable[Mapping[str, Any]]) -> bool:
    return bool(_matching_spans(text, list(fingerprints)))


def redact(text: str, fingerprints: Iterable[Mapping[str, Any]], label: str) -> str:
    """Replace every fingerprinted span with a neutral placeholder."""
    spans = sorted(_matching_spans(text, list(fingerprints)), reverse=True)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end >= merged[-1][0]:
            merged[-1] = (min(start, merged[-1][0]), max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    for start, end in merged:
        text = text[:start] + label + text[end:]
    return text


def placeholder_for(key: str) -> str:
    return f"[forgotten {key_label(key)}]"
