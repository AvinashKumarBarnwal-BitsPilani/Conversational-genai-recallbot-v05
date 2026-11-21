"""Reciprocal Rank Fusion implemented directly from the CS4 formula."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked document-ID lists using RRF(d) = sum(1 / (k + rank)).

    Ranks are one-based. A document absent from a ranking contributes zero.
    Ties are sorted by document ID to make results repeatable across runs.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        # If an ID appears more than once in a ranking, its first occurrence wins.
        seen: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if document_id in seen:
                continue
            seen.add(document_id)
            scores[document_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
