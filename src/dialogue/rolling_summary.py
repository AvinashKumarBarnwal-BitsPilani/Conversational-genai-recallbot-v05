"""Transcript-only rolling summary memory for Task 2(b)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .conversation_history import ConversationHistory

SummaryGenerator = Callable[
    [str, Sequence[tuple[dict[str, Any], dict[str, Any]]]], str
]


class RollingSummaryMemory:
    """Maintain a summary of aged exchanges plus a safe recent context window.

    Summary updates are transactional: the stored summary and cursor advance
    only after the supplied callback returns a non-empty string.
    """

    RECENT_EXCHANGES = 2

    def __init__(self, history: ConversationHistory) -> None:
        if not isinstance(history, ConversationHistory):
            raise TypeError("history must be a ConversationHistory")
        self._history = history
        self._summary = ""
        self._last_summarized_exchange_count = 0
        self._redactions: list[str] = []

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def last_summarized_exchange_count(self) -> int:
        return self._last_summarized_exchange_count

    @property
    def summarized_exchange_count(self) -> int:
        return self._last_summarized_exchange_count

    def to_snapshot(self) -> dict[str, Any]:
        """Export the actual summary, cursor and forget-redaction state."""
        return {
            "summary": self._summary,
            "last_summarized_exchange_count": self._last_summarized_exchange_count,
            "redactions": list(self._redactions),
        }

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore summary state after validating it against the raw transcript."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("rolling summary snapshot must be a mapping")
        summary = snapshot.get("summary", "")
        cursor = snapshot.get("last_summarized_exchange_count", 0)
        redactions = snapshot.get("redactions", [])
        exchange_count = self._history.completed_exchange_count
        if not isinstance(summary, str):
            raise ValueError("rolling summary snapshot summary must be a string")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or not 0 <= cursor <= exchange_count:
            raise ValueError("rolling summary cursor is outside the restored transcript")
        if not isinstance(redactions, list) or not all(
            isinstance(item, str) and item.strip() for item in redactions
        ):
            raise ValueError("rolling summary redactions must be non-empty strings")
        self._summary = summary
        self._last_summarized_exchange_count = cursor
        self._redactions = list(dict.fromkeys(redactions))

    def add_redaction(self, text: str) -> None:
        """Register transcript text to redact from future summary inputs/context."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("redaction text must be a non-empty string")
        if text not in self._redactions:
            self._redactions.append(text)
        self._summary = self._redact(self._summary)

    def redact_text(self, text: str) -> str:
        """Apply registered short-term redactions to text before prompt use."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._redact(text)

    def get_context(self) -> dict[str, Any]:
        """Return summary plus retained transcript messages as defensive copies.

        If summary generation previously failed, all exchanges beyond the
        successfully summarized cursor are included, not merely the recent two.
        """
        exchanges = self._completed_exchanges()
        if self._last_summarized_exchange_count < len(exchanges):
            visible = exchanges[self._last_summarized_exchange_count:]
        else:
            visible = exchanges[-self.RECENT_EXCHANGES:]
        messages = [
            self._redact_message(message)
            for exchange in visible
            for message in exchange
        ]
        return {"summary": self._summary, "messages": deepcopy(messages)}

    def update_summary(self, summary_generator: SummaryGenerator) -> bool:
        """Summarize newly aged exchanges; return False when none are aged."""
        if not callable(summary_generator):
            raise TypeError("summary_generator must be callable")
        exchanges = self._completed_exchanges()
        aged_count = max(0, len(exchanges) - self.RECENT_EXCHANGES)
        if aged_count <= self._last_summarized_exchange_count:
            return False

        newly_aged = [
            self._redact_exchange(exchange)
            for exchange in exchanges[self._last_summarized_exchange_count:aged_count]
        ]
        updated = summary_generator(self._summary, deepcopy(newly_aged))
        if not isinstance(updated, str) or not updated.strip():
            raise ValueError("summary_generator must return a non-empty string")

        self._summary = self._redact(updated.strip())
        self._last_summarized_exchange_count = aged_count
        return True

    def _redact(self, text: str) -> str:
        for fragment in self._redactions:
            text = text.replace(fragment, "[redacted]")
        return text

    def _completed_exchanges(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        messages = self._history.get_history("full")
        return [
            (messages[index], messages[index + 1])
            for index in range(0, len(messages), 2)
        ]

    def _redact_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        copied = deepcopy(dict(message))
        content = copied.get("content")
        if isinstance(content, str):
            copied["content"] = self._redact(content)
        return copied

    def _redact_exchange(
        self, exchange: tuple[dict[str, Any], dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return tuple(self._redact_message(message) for message in exchange)  # type: ignore[return-value]
