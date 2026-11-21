"""BM25 sparse retrieval written from first principles (no rank_bm25)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import KnowledgeChunk, SearchHit

# Keep hyphenated/underscored product and policy codes as a single token,
# e.g. RET-E14, ERR-RMA-04, SHIP-EXP-02. This helps exact-ID retrieval.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase lexical tokens while preserving codes such as RET-E14."""
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def bm25_term_score(
    *,
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    document_frequency: int,
    corpus_size: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calculate one BM25 term contribution using the standard Robertson IDF."""
    if corpus_size <= 0:
        return 0.0
    if term_frequency <= 0 or document_frequency <= 0:
        return 0.0
    if document_frequency > corpus_size:
        raise ValueError("document_frequency cannot exceed corpus_size")
    if average_document_length <= 0:
        raise ValueError("average_document_length must be positive")
    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError("k1 must be non-negative and b must be between 0 and 1")

    # The +1 inside the logarithm keeps IDF non-negative for common terms.
    idf = math.log(1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))
    length_norm = 1.0 - b + b * (document_length / average_document_length)
    tf_weight = (term_frequency * (k1 + 1.0)) / (term_frequency + k1 * length_norm)
    return idf * tf_weight


def bm25_score_from_statistics(
    query_terms: Sequence[str],
    document_terms: Sequence[str],
    *,
    corpus_size: int,
    average_document_length: float,
    document_frequencies: dict[str, int],
    document_length: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Score one document when corpus statistics are supplied explicitly.

    This function lets our unit test reproduce the exact numerical example on
    the professor's CS4 slides without manufacturing a 1,000-document corpus.
    """
    term_counts = Counter(document_terms)
    # Allow an explicit length for hand-calculated lecture examples where the
    # slide supplies corpus statistics independently of the displayed text.
    doc_len = len(document_terms) if document_length is None else document_length
    score = 0.0
    for term in dict.fromkeys(query_terms):  # repeated query terms count once
        score += bm25_term_score(
            term_frequency=term_counts[term],
            document_length=doc_len,
            average_document_length=average_document_length,
            document_frequency=document_frequencies.get(term, 0),
            corpus_size=corpus_size,
            k1=k1,
            b=b,
        )
    return score


class BM25Index:
    """In-memory inverted statistics for exact BM25 ranking over chunks."""

    def __init__(self, chunks: Iterable[KnowledgeChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []

        for chunk in self.chunks:
            terms = tokenize(chunk.content)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            # Document frequency counts a term once per chunk, not once per occurrence.
            self.document_frequencies.update(frequencies.keys())

        self.corpus_size = len(self.chunks)
        self.average_document_length = (
            sum(self.document_lengths) / self.corpus_size if self.corpus_size else 0.0
        )

    def score_document(self, query: str, index: int) -> float:
        """Return BM25 score for one indexed chunk."""
        query_terms = dict.fromkeys(tokenize(query))
        frequencies = self.term_frequencies[index]
        score = 0.0
        for term in query_terms:
            score += bm25_term_score(
                term_frequency=frequencies[term],
                document_length=self.document_lengths[index],
                average_document_length=self.average_document_length,
                document_frequency=self.document_frequencies.get(term, 0),
                corpus_size=self.corpus_size,
                k1=self.k1,
                b=self.b,
            )
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Rank all chunks by BM25 score in descending order."""
        if top_k <= 0:
            return []
        scored = [(i, self.score_document(query, i)) for i in range(self.corpus_size)]
        # Stable tie-breaking follows the order in which chunks were indexed.
        scored.sort(key=lambda item: item[1], reverse=True)
        # Do not fill the result list with arbitrary zero-score chunks when the
        # query has no lexical overlap with the corpus.
        scored = [(index, score) for index, score in scored if score > 0.0]
        hits: list[SearchHit] = []
        for rank, (index, score) in enumerate(scored[:top_k], start=1):
            chunk = self.chunks[index]
            hits.append(SearchHit(chunk.chunk_id, chunk.doc_id, float(score), rank,
                                  chunk.content, chunk.section))
        return hits
