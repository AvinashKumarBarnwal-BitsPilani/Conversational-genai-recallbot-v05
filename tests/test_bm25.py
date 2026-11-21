"""BM25 unit tests, including the worked calculation from the CS4 slides."""
import pytest

from src.retrieval.bm25 import bm25_score_from_statistics, tokenize
from src.retrieval.models import KnowledgeChunk
from src.retrieval.bm25 import BM25Index


def test_cs4_bm25_worked_example_rounds_to_4_160():
    # CS4 example: N=1000, avgdl=50, dl=12; df(machine)=300,
    # df(learning)=400; f(machine)=2, f(learning)=3; k1=1.5, b=0.75.
    document_terms = tokenize(
        "Machine learning is a subset of machine learning and deep learning"
    )
    # The slide explicitly supplies dl=12. Its printed document string has
    # 11 whitespace-delimited words, so we pass the slide's stated dl directly.
    assert len(document_terms) == 11
    score = bm25_score_from_statistics(
        query_terms=["machine", "learning"],
        document_terms=document_terms,
        corpus_size=1000,
        average_document_length=50,
        document_frequencies={"machine": 300, "learning": 400},
        document_length=12,
        k1=1.5,
        b=0.75,
    )
    assert score == pytest.approx(4.1595405186, abs=1e-9)
    assert round(score, 3) == 4.160


def test_tokenizer_preserves_hyphenated_codes():
    assert "ret-e14" in tokenize("The rule is RET-E14.")
    assert "err-rma-04" in tokenize("Reason code ERR-RMA-04")


def test_bm25_index_ranks_exact_match_first():
    chunks = [
        KnowledgeChunk("c1", "KB-1", "A", "", "1", "", "", "active", "a.md", 128, 19, 4,
                       "RET-E14 permits fourteen calendar days for electronics."),
        KnowledgeChunk("c2", "KB-2", "B", "", "1", "", "", "active", "b.md", 128, 19, 4,
                       "Most products have a thirty day return period."),
    ]
    results = BM25Index(chunks).search("RET-E14", top_k=2)
    assert len(results) == 1  # No zero-score chunks are returned.
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0
