from __future__ import annotations

import math

import numpy as np
import pytest

from src.memory.clock import SessionClock, hours_between
from src.memory.episodic_memory import (
    RECENCY_DECAY_PER_HOUR, WEIGHTINGS, EpisodicMemoryStream, EpisodicWeights, rule_importance, score_episodes,
)
from src.memory.records import Episode
from memory_fakes import FakeEmbedder

EMBED = FakeEmbedder()
USER = "CUST-MAYA-1042"


def _episode(episode_id, timestamp, importance, episode_type="decision"):
    return Episode(episode_id=episode_id, user=USER, session="A", timestamp=timestamp, source_turn=1,
                   type=episode_type, description=episode_id, importance=importance)


def test_logical_clock_spaces_turns_and_sessions():
    a, b = SessionClock.for_session("A"), SessionClock.for_session("B")
    assert a.iso_at(1) == "2026-11-21T10:00:00+05:30"
    assert a.iso_at(23) == "2026-11-21T10:44:00+05:30"
    assert b.iso_at(26) == "2026-11-24T10:00:00+05:30"
    # 10:44 on 21 Nov to 10:04 on 24 Nov = 72 h - 40 min.
    assert hours_between(a.iso_at(23), b.iso_at(28)) == pytest.approx(71 + 1 / 3)


def test_rule_importance_ranks_events_and_drops_small_talk():
    assert rule_importance("booking") > rule_importance("complaint") > rule_importance("decision")
    assert rule_importance("decision") > rule_importance("preference_update") > rule_importance("preference_event") - 1
    stream = EpisodicMemoryStream()
    kept = stream.add(user=USER, session="A", timestamp="2026-11-21T10:00:00+05:30", source_turn=3,
                      episode_type="small_talk", description="Customer said thanks.", vector=None)
    assert kept is None and stream.episodes() == []


def test_score_uses_min_max_normalised_components_and_declared_decay():
    episodes = [
        _episode("EP-OLD-IMPORTANT", "2026-11-21T10:00:00+05:30", 9),
        _episode("EP-NEW-MINOR", "2026-11-24T09:00:00+05:30", 3),
    ]
    relevance = {"EP-OLD-IMPORTANT": 0.40, "EP-NEW-MINOR": 0.60}
    ranked = score_episodes(episodes, relevance, now="2026-11-24T10:00:00+05:30", weights=WEIGHTINGS["C"])
    by_id = {item.episode.episode_id: item for item in ranked}
    assert by_id["EP-OLD-IMPORTANT"].recency == pytest.approx(RECENCY_DECAY_PER_HOUR ** 72)
    assert by_id["EP-NEW-MINOR"].recency == pytest.approx(RECENCY_DECAY_PER_HOUR ** 1)
    assert (by_id["EP-OLD-IMPORTANT"].recency_norm, by_id["EP-NEW-MINOR"].recency_norm) == (0.0, 1.0)
    assert (by_id["EP-OLD-IMPORTANT"].importance_norm, by_id["EP-NEW-MINOR"].importance_norm) == (1.0, 0.0)
    assert by_id["EP-NEW-MINOR"].score == pytest.approx(2 / 3)


def test_weightings_can_rank_the_same_candidates_differently():
    episodes = [
        _episode("EP-RELEVANT-OLD", "2026-11-21T10:00:00+05:30", 7),
        _episode("EP-RECENT-LESS-RELEVANT", "2026-11-24T09:30:00+05:30", 7),
    ]
    relevance = {"EP-RELEVANT-OLD": 0.80, "EP-RECENT-LESS-RELEVANT": 0.55}
    now = "2026-11-24T10:00:00+05:30"
    first_a = score_episodes(episodes, relevance, now=now, weights=WEIGHTINGS["A"])[0].episode.episode_id
    first_b = score_episodes(episodes, relevance, now=now, weights=EpisodicWeights("recency_only", 1, 0, 0))[0]
    assert first_a == "EP-RELEVANT-OLD"
    assert first_b.episode.episode_id == "EP-RECENT-LESS-RELEVANT"


def test_min_relevance_uses_raw_cosine_and_excludes_unrelated_recent_episodes():
    episodes = [_episode("EP-RECENT", "2026-11-24T09:59:00+05:30", 9), _episode("EP-MATCH", "2026-11-21T10:00:00+05:30", 3)]
    ranked = score_episodes(episodes, {"EP-RECENT": 0.05, "EP-MATCH": 0.70},
                            now="2026-11-24T10:00:00+05:30", weights=WEIGHTINGS["B"], min_relevance=0.3)
    assert [item.episode.episode_id for item in ranked] == ["EP-MATCH"]


def test_stream_retrieval_delete_and_redaction_keep_index_consistent(tmp_path):
    stream = EpisodicMemoryStream()
    texts = {
        "complaint": "Late delivery reported for order ORD-88216.",
        "decision": "Customer chose an exchange rather than a refund for ORD-66310.",
        "booking": "Return pickup booked for SKU-AUR-220 in ORD-74021 on 2026-11-28.",
    }
    for turn, (kind, text) in enumerate(texts.items(), start=10):
        stream.add(user=USER, session="A", timestamp=SessionClock.for_session("A").iso_at(turn), source_turn=turn,
                   episode_type=kind, description=text, vector=EMBED.encode([f"{kind}: {text}"], purpose="t")[0])
    query = EMBED.encode(["what did I choose for the charger exchange refund ORD-66310"], purpose="t", is_query=True)[0]
    top = stream.retrieve(USER, query, now="2026-11-24T10:00:00+05:30", weights=WEIGHTINGS["A"], top_k=1)
    assert top[0].episode.type == "decision"

    booking = stream.episodes(types=["booking"])[0]
    stream.replace_text(booking.episode_id, "Return pickup booked for [redacted].", EMBED.encode(["x"], purpose="t")[0])
    assert stream.get(booking.episode_id).redacted is True
    stream.delete([top[0].episode.episode_id])
    assert len(stream.index) == len(stream.episodes()) == 2

    stream.index.save(tmp_path, "episodic")
    from src.memory.vector_store import MemoryVectorIndex
    restored = EpisodicMemoryStream.from_dict(stream.to_dict(), MemoryVectorIndex.load(tmp_path, "episodic"))
    assert [ep.to_dict() for ep in restored.episodes()] == [ep.to_dict() for ep in stream.episodes()]
    nxt = restored.add(user=USER, session="B", timestamp="2026-11-24T10:00:00+05:30", source_turn=26,
                       episode_type="decision", description="New", vector=None)
    assert nxt.episode_id == "EP-0004"  # IDs are never reused after a delete
