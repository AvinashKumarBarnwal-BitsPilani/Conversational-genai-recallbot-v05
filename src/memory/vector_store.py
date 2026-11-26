"""Persisted flat cosine index for memory records, with deletion.

The knowledge-base DenseIndex is immutable after build. Memory needs inserts,
replacements and hard deletes (supersede and forget), so this small index is
keyed by memory ID. Vectors are L2-normalised, so a dot product is cosine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MemoryVectorIndex:
    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension
        self._ids: list[str] = []
        self._vectors = np.empty((0, dimension or 0), dtype=np.float32)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._ids

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        row = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            raise ValueError("A memory vector cannot be the zero vector")
        return row / norm

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        """Insert a vector, replacing any existing vector for the same ID."""
        row = self._normalise(vector)
        if not self._ids:
            # An empty index adopts the dimension of its first vector.
            self.dimension = row.shape[0]
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        if row.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {row.shape[0]} does not match index dimension {self.dimension}")
        self.remove(memory_id)
        self._ids.append(memory_id)
        self._vectors = np.vstack([self._vectors, row[None, :]])

    def remove(self, memory_id: str) -> bool:
        if memory_id not in self._ids:
            return False
        position = self._ids.index(memory_id)
        del self._ids[position]
        self._vectors = np.delete(self._vectors, position, axis=0)
        return True

    def similarities(self, query_vector: np.ndarray, allowed_ids: set[str] | None = None) -> dict[str, float]:
        """Cosine similarity of the query against every (allowed) stored vector."""
        if not self._ids:
            return {}
        query = self._normalise(query_vector)
        if query.shape[0] != self.dimension:
            raise ValueError("Query vector dimension does not match the memory index")
        scores = self._vectors @ query
        return {
            memory_id: float(score)
            for memory_id, score in zip(self._ids, scores)
            if allowed_ids is None or memory_id in allowed_ids
        }

    def save(self, directory: Path, name: str) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        vectors_path = directory / f"{name}_vectors.npy"
        ids_path = directory / f"{name}_index.json"
        np.save(vectors_path, self._vectors.astype(np.float32))
        ids_path.write_text(
            json.dumps({"dimension": self.dimension, "ids": self._ids}, indent=2), encoding="utf-8"
        )
        return [vectors_path, ids_path]

    @classmethod
    def load(cls, directory: Path, name: str) -> "MemoryVectorIndex":
        ids_path = directory / f"{name}_index.json"
        vectors_path = directory / f"{name}_vectors.npy"
        if not ids_path.exists():
            return cls()
        meta = json.loads(ids_path.read_text(encoding="utf-8"))
        index = cls(meta.get("dimension"))
        vectors = np.load(vectors_path) if vectors_path.exists() else np.empty((0, 0), dtype=np.float32)
        ids = list(meta.get("ids", []))
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"Memory index {name!r} has {len(ids)} IDs but {vectors.shape[0]} vectors")
        index._ids = ids
        index._vectors = vectors.astype(np.float32) if ids else np.empty((0, index.dimension or 0), dtype=np.float32)
        return index
