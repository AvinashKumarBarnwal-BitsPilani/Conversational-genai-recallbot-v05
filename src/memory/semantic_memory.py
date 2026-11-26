"""Semantic memory: keyed user facts plus a vector index (Task 3(c)).

Records are (user, key, value, source_turn, date, status). An update marks the
previous active record ``superseded`` and removes its vector, so a stale value
can never be retrieved (the same fix as the CS4 stale-chunk failure). A forget
request deletes every record of the key, active or superseded, from both the
record store and the index. Embedding and logging are done by MemoryManager;
this class is pure state so it can be unit tested without a model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .records import SemanticFact
from .vector_store import MemoryVectorIndex


def normalise_value(value: str) -> str:
    return " ".join(str(value).casefold().split())


class SemanticMemoryStore:
    def __init__(self) -> None:
        self._records: list[SemanticFact] = []
        self._next_id = 1
        self.index = MemoryVectorIndex()

    # ----- inspection -------------------------------------------------
    def all_records(self, user: str | None = None) -> list[SemanticFact]:
        return [r for r in self._records if user is None or r.user == user]

    def active_facts(self, user: str) -> list[SemanticFact]:
        return [r for r in self._records if r.user == user and r.status == "active"]

    def get_active(self, user: str, key: str) -> SemanticFact | None:
        return next((r for r in self.active_facts(user) if r.key == key), None)

    def get(self, memory_id: str) -> SemanticFact | None:
        return next((r for r in self._records if r.memory_id == memory_id), None)

    def would_change(self, user: str, key: str, value: str) -> bool:
        current = self.get_active(user, key)
        return current is None or normalise_value(current.value) != normalise_value(value)

    # ----- writes -----------------------------------------------------
    def upsert(
        self,
        *,
        user: str,
        key: str,
        value: str,
        source_turn: int,
        session: str,
        date: str,
        vector: np.ndarray | None,
        extraction_method: str = "rule",
        rule_id: str | None = None,
    ) -> tuple[str, SemanticFact, SemanticFact | None]:
        """Create or supersede a fact. Returns (action, record, superseded_record).

        action is ``created``, ``updated`` or ``unchanged``. A repeated value is
        a no-op so a fact restated later does not create duplicates.
        """
        if not key or not str(value).strip():
            raise ValueError("A semantic fact needs a key and a non-empty value")
        current = self.get_active(user, key)
        if current is not None and normalise_value(current.value) == normalise_value(value):
            return "unchanged", current, None
        record = SemanticFact(
            memory_id=f"SM-{self._next_id:04d}",
            user=user,
            key=key,
            value=str(value).strip(),
            source_turn=int(source_turn),
            session=session,
            date=date,
            status="active",
            supersedes=current.memory_id if current else None,
            extraction_method=extraction_method,
            rule_id=rule_id,
        )
        self._next_id += 1
        if current is not None:
            current.status = "superseded"
            current.superseded_by = record.memory_id
            # A superseded value must not be retrievable by similarity search.
            self.index.remove(current.memory_id)
        self._records.append(record)
        if vector is not None:
            self.index.add(record.memory_id, vector)
        return ("updated" if current else "created"), record, current

    def forget_key(self, user: str, key: str) -> list[SemanticFact]:
        """Delete every record (active or superseded) of the key, plus vectors."""
        removed = [r for r in self._records if r.user == user and r.key == key]
        removed_ids = {r.memory_id for r in removed}
        self._records = [r for r in self._records if r.memory_id not in removed_ids]
        for memory_id in removed_ids:
            self.index.remove(memory_id)
        return removed

    # ----- retrieval --------------------------------------------------
    def search(
        self,
        user: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[SemanticFact, float]]:
        """Rank this user's active facts by cosine similarity to the query."""
        active = {r.memory_id: r for r in self.active_facts(user)}
        scores = self.index.similarities(query_vector, allowed_ids=set(active))
        ranked = sorted(
            ((active[memory_id], score) for memory_id, score in scores.items() if score >= min_score),
            key=lambda item: (-item[1], item[0].memory_id),
        )
        return ranked[:top_k]

    # ----- persistence ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"next_id": self._next_id, "records": [r.to_dict() for r in self._records]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: MemoryVectorIndex | None = None) -> "SemanticMemoryStore":
        store = cls()
        store._records = [SemanticFact.from_dict(item) for item in payload.get("records", [])]
        store._next_id = int(payload.get("next_id", len(store._records) + 1))
        store.index = index or MemoryVectorIndex()
        known = {r.memory_id for r in store._records if r.status == "active"}
        stale = [memory_id for memory_id in store.index.ids if memory_id not in known]
        if stale:
            raise ValueError(f"Semantic index holds vectors for non-active records: {stale}")
        return store
