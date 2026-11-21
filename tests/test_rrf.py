"""RRF unit tests against the exact CS4 ranked-list example."""
import pytest

from src.retrieval.rrf import reciprocal_rank_fusion


def test_cs4_rrf_example_order_and_tie_with_k_60():
    dense = ["A", "C", "B", "E", "D"]
    sparse = ["B", "A", "D", "F", "C"]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [doc_id for doc_id, _score in fused]
    scores = dict(fused)

    assert ids == ["A", "B", "C", "D", "E", "F"]
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 62 + 1 / 65)
    assert scores["D"] == pytest.approx(1 / 65 + 1 / 63)
    assert scores["E"] == pytest.approx(1 / 64)
    assert scores["F"] == pytest.approx(1 / 64)
    assert scores["E"] == scores["F"]


def test_rrf_ignores_duplicate_id_inside_one_ranking():
    fused = reciprocal_rank_fusion([["A", "A", "B"]], k=60)
    assert dict(fused)["A"] == pytest.approx(1 / 61)
    assert dict(fused)["B"] == pytest.approx(1 / 63)
