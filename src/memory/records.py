"""Record types for RecallBot's external long-term memory (Task 3(c), 3(d))."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Canonical semantic keys. They match data/test_conversation.json
# planted_user_facts so extraction precision/recall can be measured exactly.
CANONICAL_FACT_KEYS: dict[str, str] = {
    "preferred_name": "preferred name",
    "notification_channel": "notification channel for routine updates",
    "preferred_city": "usual delivery city",
    "delivery_note": "delivery note",
    "preferred_delivery_window": "preferred delivery window",
    "return_method_preference": "preferred return method",
    "packaging_preference": "packaging preference",
}


def key_label(key: str) -> str:
    """Human-readable label used for embeddings, prompts and replies."""
    return CANONICAL_FACT_KEYS.get(key, key.replace("_", " "))


@dataclass
class SemanticFact:
    """One keyed user fact (CS5 fact notebook)."""

    memory_id: str
    user: str
    key: str
    value: str
    source_turn: int
    session: str
    date: str
    status: str = "active"  # active | superseded (forgotten facts are deleted)
    supersedes: str | None = None
    superseded_by: str | None = None
    extraction_method: str = "rule"
    rule_id: str | None = None

    def embedding_text(self) -> str:
        return f"{key_label(self.key)}: {self.value}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticFact":
        return cls(**value)


@dataclass
class Episode:
    """One time-stamped event in the episodic memory stream."""

    episode_id: str
    user: str
    session: str
    timestamp: str
    source_turn: int
    type: str
    description: str
    importance: int
    importance_method: str = "rule"
    entities: dict[str, Any] = field(default_factory=dict)
    linked_memory_ids: list[str] = field(default_factory=list)
    redacted: bool = False

    def embedding_text(self) -> str:
        return f"{self.type.replace('_', ' ')}: {self.description}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Episode":
        return cls(**value)
