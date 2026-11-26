"""Logical session clock for memory timestamps and recency (Task 3(d)).

A scripted replay finishes within minutes, so wall-clock timestamps would make
the recency term almost identical for every episode. Each session therefore
has a declared logical start time and every user turn advances the clock by a
fixed number of minutes. The values are stated in the Task 3 design document.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# Session A follows the dataset's date_context (2026-11-21); Session B is the
# customer's return visit three days later.
SESSION_STARTS: dict[str, datetime] = {
    "A": datetime(2026, 11, 21, 10, 0, tzinfo=IST),
    "B": datetime(2026, 11, 24, 10, 0, tzinfo=IST),
}
SESSION_FIRST_TURNS: dict[str, int] = {"A": 1, "B": 26}
MINUTES_PER_TURN = 2


@dataclass(frozen=True)
class SessionClock:
    """Map a dialogue turn number to a deterministic logical timestamp."""

    session: str
    start: datetime
    first_turn: int
    minutes_per_turn: int = MINUTES_PER_TURN

    @classmethod
    def for_session(cls, session: str) -> "SessionClock":
        if session not in SESSION_STARTS:
            raise ValueError(f"No logical start time is declared for session {session!r}")
        return cls(session, SESSION_STARTS[session], SESSION_FIRST_TURNS[session])

    def at(self, turn: int) -> datetime:
        offset = max(0, int(turn) - self.first_turn) * self.minutes_per_turn
        return self.start + timedelta(minutes=offset)

    def iso_at(self, turn: int) -> str:
        return self.at(turn).isoformat(timespec="seconds")


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=IST)


def hours_between(earlier: str | datetime, later: str | datetime) -> float:
    start = parse_timestamp(earlier) if isinstance(earlier, str) else earlier
    end = parse_timestamp(later) if isinstance(later, str) else later
    return max(0.0, (end - start).total_seconds() / 3600.0)
