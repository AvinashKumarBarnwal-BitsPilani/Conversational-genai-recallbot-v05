"""Episodic memory stream with Generative-Agents style retrieval (Task 3(d)).

score = w_r * R_hat + w_i * I_hat + w_s * S_hat

R = 0.995 ** hours_since_episode   (recency decay, logical session clock)
I = importance / 10                (rule-based importance, 1-10)
S = cosine(query, episode)         (same BGE model as the knowledge base)

Each component is min-max normalised over the candidate set before weighting,
as in Generative Agents. A candidate must also clear a minimum *raw* cosine
relevance so that a recent but unrelated episode is never returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .clock import hours_between
from .records import Episode
from .vector_store import MemoryVectorIndex

RECENCY_DECAY_PER_HOUR = 0.995

# Rule-based importance (FAQ 6 allows rules). Bookings and complaints score
# high; preference bookkeeping scores low; small talk is never written.
IMPORTANCE_BY_TYPE: dict[str, int] = {
    "booking": 9,
    "complaint": 8,
    "decision": 7,
    "reflection": 6,
    "task_event": 5,
    "preference_update": 4,
    "preference_event": 3,
    "forget_event": 3,
}
SMALL_TALK_IMPORTANCE = 1
WRITE_THRESHOLD = 3


def rule_importance(episode_type: str) -> int:
    return IMPORTANCE_BY_TYPE.get(episode_type, SMALL_TALK_IMPORTANCE)


@dataclass(frozen=True)
class EpisodicWeights:
    name: str
    recency: float
    importance: float
    relevance: float


WEIGHTINGS: dict[str, EpisodicWeights] = {
    "A": EpisodicWeights("A_relevance_focused", 0.20, 0.30, 0.50),
    "B": EpisodicWeights("B_recency_focused", 0.40, 0.30, 0.30),
    "C": EpisodicWeights("C_equal", 1 / 3, 1 / 3, 1 / 3),
}
# Provisional default; the kept weighting is chosen from the E-probe and
# episodic-query comparison in Iteration 6 and recorded in the evidence doc.
DEFAULT_WEIGHTING = "A"


@dataclass(frozen=True)
class ScoredEpisode:
    episode: Episode
    recency: float
    importance: float
    relevance: float
    recency_norm: float
    importance_norm: float
    relevance_norm: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "type": self.episode.type,
            "source_turn": self.episode.source_turn,
            "recency": round(self.recency, 6),
            "importance": round(self.importance, 6),
            "relevance": round(self.relevance, 6),
            "recency_norm": round(self.recency_norm, 6),
            "importance_norm": round(self.importance_norm, 6),
            "relevance_norm": round(self.relevance_norm, 6),
            "score": round(self.score, 6),
        }


def _min_max(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        # A constant component carries no ranking information.
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def score_episodes(
    episodes: list[Episode],
    relevance: dict[str, float],
    *,
    now: str,
    weights: EpisodicWeights,
    min_relevance: float = 0.0,
) -> list[ScoredEpisode]:
    """Rank episodes with the combined score; highest first, ties by ID."""
    candidates = [ep for ep in episodes if relevance.get(ep.episode_id, -1.0) >= min_relevance]
    if not candidates:
        return []
    recency = [RECENCY_DECAY_PER_HOUR ** hours_between(ep.timestamp, now) for ep in candidates]
    importance = [ep.importance / 10.0 for ep in candidates]
    similarity = [relevance[ep.episode_id] for ep in candidates]
    r_norm, i_norm, s_norm = _min_max(recency), _min_max(importance), _min_max(similarity)
    scored = [
        ScoredEpisode(
            episode=ep,
            recency=recency[i], importance=importance[i], relevance=similarity[i],
            recency_norm=r_norm[i], importance_norm=i_norm[i], relevance_norm=s_norm[i],
            score=weights.recency * r_norm[i] + weights.importance * i_norm[i] + weights.relevance * s_norm[i],
        )
        for i, ep in enumerate(candidates)
    ]
    return sorted(scored, key=lambda item: (-item.score, item.episode.episode_id))


class EpisodicMemoryStream:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []
        self._next_id = 1
        self.index = MemoryVectorIndex()

    def episodes(self, user: str | None = None, types: Iterable[str] | None = None) -> list[Episode]:
        allowed = set(types) if types is not None else None
        return [
            ep for ep in self._episodes
            if (user is None or ep.user == user) and (allowed is None or ep.type in allowed)
        ]

    def get(self, episode_id: str) -> Episode | None:
        return next((ep for ep in self._episodes if ep.episode_id == episode_id), None)

    def add(
        self,
        *,
        user: str,
        session: str,
        timestamp: str,
        source_turn: int,
        episode_type: str,
        description: str,
        vector: np.ndarray | None,
        importance: int | None = None,
        importance_method: str = "rule",
        entities: dict[str, Any] | None = None,
        linked_memory_ids: list[str] | None = None,
    ) -> Episode | None:
        """Append an episode; returns None when it is below the write threshold."""
        score = rule_importance(episode_type) if importance is None else int(importance)
        if not 1 <= score <= 10:
            raise ValueError("importance must be between 1 and 10")
        if score < WRITE_THRESHOLD:
            return None
        episode = Episode(
            episode_id=f"EP-{self._next_id:04d}",
            user=user,
            session=session,
            timestamp=timestamp,
            source_turn=int(source_turn),
            type=episode_type,
            description=description.strip(),
            importance=score,
            importance_method=importance_method,
            entities=dict(entities or {}),
            linked_memory_ids=list(linked_memory_ids or []),
        )
        self._next_id += 1
        self._episodes.append(episode)
        if vector is not None:
            self.index.add(episode.episode_id, vector)
        return episode

    def delete(self, episode_ids: Iterable[str]) -> list[Episode]:
        targets = set(episode_ids)
        removed = [ep for ep in self._episodes if ep.episode_id in targets]
        self._episodes = [ep for ep in self._episodes if ep.episode_id not in targets]
        for episode_id in targets:
            self.index.remove(episode_id)
        return removed

    def replace_text(self, episode_id: str, description: str, vector: np.ndarray | None) -> Episode:
        """Redact an episode's text; its old vector is replaced so it cannot match."""
        episode = self.get(episode_id)
        if episode is None:
            raise KeyError(episode_id)
        episode.description = description
        episode.redacted = True
        self.index.remove(episode_id)
        if vector is not None:
            self.index.add(episode_id, vector)
        return episode

    def retrieve(
        self,
        user: str,
        query_vector: np.ndarray,
        *,
        now: str,
        weights: EpisodicWeights,
        top_k: int = 3,
        min_relevance: float = 0.0,
        types: Iterable[str] | None = None,
    ) -> list[ScoredEpisode]:
        candidates = self.episodes(user, types)
        relevance = self.index.similarities(query_vector, allowed_ids={ep.episode_id for ep in candidates})
        return score_episodes(candidates, relevance, now=now, weights=weights, min_relevance=min_relevance)[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {"next_id": self._next_id, "episodes": [ep.to_dict() for ep in self._episodes]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: MemoryVectorIndex | None = None) -> "EpisodicMemoryStream":
        stream = cls()
        stream._episodes = [Episode.from_dict(item) for item in payload.get("episodes", [])]
        stream._next_id = int(payload.get("next_id", len(stream._episodes) + 1))
        stream.index = index or MemoryVectorIndex()
        known = {ep.episode_id for ep in stream._episodes}
        stale = [episode_id for episode_id in stream.index.ids if episode_id not in known]
        if stale:
            raise ValueError(f"Episodic index holds vectors for deleted episodes: {stale}")
        return stream
