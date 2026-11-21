"""Flat dense retrieval implemented with NumPy cosine similarity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import KnowledgeChunk, SearchHit


class DenseIndex:
    """An exact flat vector index; normalized vectors make dot = cosine."""

    def __init__(self, chunks: list[KnowledgeChunk], vectors: np.ndarray):
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2-D matrix")
        if len(chunks) != matrix.shape[0]:
            raise ValueError("There must be exactly one vector per chunk")
        # Normalise every document vector once. Search can then use a fast dot
        # product while retaining cosine-similarity semantics.
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(row_norms == 0.0):
            raise ValueError("The index cannot contain a zero vector")
        self.chunks = chunks
        self.vectors = matrix / row_norms

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """Return top-k chunks ranked by cosine similarity, highest first."""
        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query vector dimension does not match the index")
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError("Cannot search with a zero query vector")
        query = query / norm
        scores = self.vectors @ query
        # Stable sorting makes equal-score results deterministic for evaluation.
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [
            SearchHit(
                chunk_id=self.chunks[i].chunk_id,
                doc_id=self.chunks[i].doc_id,
                score=float(scores[i]),
                rank=rank,
                content=self.chunks[i].content,
                section=self.chunks[i].section,
            )
            for rank, i in enumerate(order, start=1)
        ]

    def save(self, directory: Path) -> None:
        """Persist vectors and metadata separately for inspectability."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        metadata_path = directory / "chunks.json"
        metadata_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        """Load a previously generated flat index from disk."""
        chunks_data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [KnowledgeChunk.from_dict(item) for item in chunks_data]
        vectors = np.load(directory / "vectors.npy")
        return cls(chunks, vectors)
