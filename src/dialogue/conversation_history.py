"""Raw completed conversation exchanges for Task 2(b) memory strategies.

This component is deliberately independent of DialogueStateTracker. Each
exchange contains exactly one user message followed by its assistant reply.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..config import TRIM_WINDOW_K_TURNS


class ConversationHistory:
    """Store complete exchanges and select full or V05 last-k history."""

    STRATEGIES = frozenset({"full", "trim"})

    def __init__(self, *, trim_k: int = TRIM_WINDOW_K_TURNS) -> None:
        if not isinstance(trim_k, int) or isinstance(trim_k, bool) or trim_k <= 0:
            raise ValueError("trim_k must be a positive integer")
        self._trim_k = trim_k
        self._exchanges: list[tuple[dict[str, Any], dict[str, Any]]] = []

    @property
    def completed_exchange_count(self) -> int:
        """Number of complete user/assistant exchanges recorded."""
        return len(self._exchanges)

    def to_snapshot(self) -> dict[str, Any]:
        """Return the complete raw exchange transcript for a replay snapshot."""
        return {
            "trim_k": self._trim_k,
            "exchanges": deepcopy([
                {"user": user, "assistant": assistant}
                for user, assistant in self._exchanges
            ]),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "ConversationHistory":
        """Restore an independent history instance from a serialized snapshot."""
        if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("exchanges"), list):
            raise ValueError("conversation history snapshot must contain an exchanges list")
        history = cls(trim_k=snapshot.get("trim_k", TRIM_WINDOW_K_TURNS))
        for exchange in snapshot["exchanges"]:
            if not isinstance(exchange, Mapping):
                raise ValueError("each history snapshot exchange must be a mapping")
            history.record_exchange(exchange.get("user"), exchange.get("assistant"))
        return history

    def record_exchange(
        self,
        user_message: Mapping[str, Any],
        assistant_message: Mapping[str, Any],
    ) -> None:
        """Append a validated user/assistant exchange to the raw transcript."""
        user = self._validated_message(user_message, expected_role="user")
        assistant = self._validated_message(assistant_message, expected_role="assistant")
        self._exchanges.append((user, assistant))

    def get_history(self, strategy: str) -> list[dict[str, Any]]:
        """Return copied role/content messages in chronological order.

        The current incoming user message is intentionally not accepted here;
        the prompt caller adds it separately, preventing accidental duplication.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(self.STRATEGIES)}")
        exchanges = self._exchanges if strategy == "full" else self._exchanges[-self._trim_k:]
        return deepcopy([message for exchange in exchanges for message in exchange])

    @staticmethod
    def _validated_message(message: Mapping[str, Any], *, expected_role: str) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            raise TypeError(f"{expected_role}_message must be a mapping")
        if message.get("role") != expected_role:
            raise ValueError(f"message role must be {expected_role!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{expected_role}_message content must be a non-empty string")
        return deepcopy(dict(message))
