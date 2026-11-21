"""Dense search tests use tiny vectors; no embedding model download is needed."""
import numpy as np

from src.retrieval.dense import DenseIndex
from src.retrieval.models import KnowledgeChunk


def _chunk(chunk_id: str, doc_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id, doc_id, "Title", "Section", "1.0",
                          "2026-01-01", "open", "active", "doc.md",
                          128, 19, 3, "example content")


def test_dense_index_returns_highest_cosine_similarity_first():
    chunks = [_chunk("c1", "KB-1"), _chunk("c2", "KB-2"), _chunk("c3", "KB-3")]
    # The constructor normalises rows; c2 is the closest direction to the query.
    vectors = np.array([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32)
    index = DenseIndex(chunks, vectors)
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[1].chunk_id == "c2"
    assert hits[0].score == 1.0


def test_dense_index_rejects_zero_document_vector():
    import pytest

    with pytest.raises(ValueError, match="zero vector"):
        DenseIndex([_chunk("c1", "KB-1")], np.array([[0.0, 0.0]], dtype=np.float32))
