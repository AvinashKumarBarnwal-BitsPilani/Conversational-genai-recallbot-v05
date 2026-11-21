"""Strict retrieval-hit tests: a document match alone is not sufficient."""
from src.retrieval.evaluate import answer_span_is_covered
from src.retrieval.models import KnowledgeChunk


def _chunk(chunk_id: str, doc_id: str, start: int, end: int) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id, doc_id, "Title", "Section", "1.0", "2026-01-01", "open",
        "active", "doc.md", 128, 19, end - start, "source excerpt", start, end,
    )


def test_wrong_section_in_correct_document_is_not_a_hit():
    source = "Opening section. The electronics return window is fourteen days. Final section."
    answer = "The electronics return window is fourteen days."
    start = source.index(answer)
    end = start + len(answer)
    wrong_section = _chunk("c1", "KB-1", 0, source.index("The electronics"))

    assert not answer_span_is_covered(
        source_text=source, gold_start=start, gold_end=end,
        document_id="KB-1", retrieved_chunks=[wrong_section],
    )


def test_two_retrieved_chunks_can_jointly_cover_a_split_answer_span():
    source = "prefix The electronics return window is fourteen days suffix"
    answer = "The electronics return window is fourteen days"
    start = source.index(answer)
    end = start + len(answer)
    split = start + 25
    first = _chunk("c1", "KB-1", start, split)
    second = _chunk("c2", "KB-1", split, end)

    assert answer_span_is_covered(
        source_text=source, gold_start=start, gold_end=end,
        document_id="KB-1", retrieved_chunks=[first, second],
    )


def test_chunks_from_another_document_do_not_count_as_a_hit():
    source = "prefix gold answer suffix"
    start = source.index("gold answer")
    end = start + len("gold answer")
    other_document = _chunk("c1", "KB-OTHER", start, end)

    assert not answer_span_is_covered(
        source_text=source, gold_start=start, gold_end=end,
        document_id="KB-1", retrieved_chunks=[other_document],
    )
