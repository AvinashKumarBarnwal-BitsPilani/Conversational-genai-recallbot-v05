"""Unit tests for the Task 3(b) ranked-evidence CSV helper."""
from src.retrieval.evaluate import build_ranking_rows
from src.retrieval.models import KnowledgeChunk


def _chunk(chunk_id: str, doc_id: str, start: int, end: int, section: str = "Section"):
    """Create a minimal chunk with source character offsets for ranking tests."""
    return KnowledgeChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title="ShopSphere KB",
        section=section,
        version="2026",
        effective_from="2026-01-01",
        effective_to="open",
        status="active",
        source_file="doc.md",
        chunk_size=128,
        overlap_tokens=19,
        token_count=end - start,
        content="test excerpt",
        start_char=start,
        end_char=end,
    )


def test_ranked_rows_include_scores_categories_and_cumulative_span_hit():
    source = "prefix GOLD ANSWER suffix"
    gold = "GOLD ANSWER"
    gold_start = source.index(gold)
    split = gold_start + 5
    gold_end = gold_start + len(gold)

    unrelated = _chunk("c-unrelated", "KB-1", 0, gold_start, "Introduction")
    first_half = _chunk("c-first", "KB-1", gold_start, split, "Answer part 1")
    second_half = _chunk("c-second", "KB-1", split, gold_end, "Answer part 2")
    wrong_doc = _chunk("c-other", "KB-OTHER", gold_start, gold_end, "Other document")

    question = {
        "question_id": "Q-TEST",
        "question": "What is the answer?",
        "gold_answer_span": gold,
        "document_id": "KB-1",
        "paraphrased_no_answer_keyword": True,
    }
    ranked = {
        "dense": [(unrelated, 0.91), (first_half, 0.80), (second_half, 0.70)],
        "bm25": [(wrong_doc, 8.5), (first_half, 7.2), (second_half, 6.1)],
        "hybrid": [(first_half, 0.0325), (second_half, 0.0312)],
    }

    rows = build_ranking_rows(
        question=question,
        chunk_size=128,
        ranked_results=ranked,
        source_text=source,
        gold_start=gold_start,
        max_k=5,
    )

    dense_rows = [row for row in rows if row["method"] == "dense"]
    bm25_rows = [row for row in rows if row["method"] == "bm25"]
    hybrid_rows = [row for row in rows if row["method"] == "hybrid"]

    assert [row["rank"] for row in dense_rows] == [1, 2, 3]
    assert dense_rows[0]["chunk_answer_span_overlap"] == 0
    assert dense_rows[1]["chunk_answer_span_overlap"] == 1
    assert dense_rows[1]["cumulative_answer_span_hit"] == 0
    assert dense_rows[2]["cumulative_answer_span_hit"] == 1

    # A full-span chunk from another document must never count as relevant.
    assert bm25_rows[0]["chunk_answer_span_overlap"] == 0
    assert bm25_rows[0]["cumulative_answer_span_hit"] == 0
    assert bm25_rows[2]["cumulative_answer_span_hit"] == 1

    assert hybrid_rows[0]["category"] == "paraphrased_no_answer_keyword"
    assert hybrid_rows[0]["score"] == "0.0325000000"
    assert hybrid_rows[1]["cumulative_answer_span_hit"] == 1


def test_question_category_prioritises_exact_code_flag():
    from src.retrieval.evaluate import _question_category

    assert _question_category({"contains_exact_code_or_id": True}) == "exact_code_or_id"
    assert _question_category({"paraphrased_no_answer_keyword": True}) == "paraphrased_no_answer_keyword"
    assert _question_category({}) == "other"
