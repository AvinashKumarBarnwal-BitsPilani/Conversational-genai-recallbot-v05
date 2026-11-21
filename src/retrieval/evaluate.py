"""Evaluate dense, BM25 and RRF retrieval on labelled KB answer spans.

The official hit_rate.csv uses strict answer-span coverage: the retrieved
chunks must collectively contain the complete gold answer span in the source
file. A document-level match alone is not counted as a successful retrieval.

This evaluator also writes retrieval_rankings.csv. That file preserves the
ranked chunk IDs and scores needed to explain, with evidence, when hybrid
retrieval fixes a failure and when it does not.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

from ..config import DATA_DIR, INDEX_DIR, RESULTS_DIR
from .bm25 import BM25Index
from .dense import DenseIndex
from .embeddings import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder
from .models import KnowledgeChunk
from .rrf import reciprocal_rank_fusion


RANKING_METHODS = ("dense", "bm25", "hybrid")


def _normalise(text: str) -> str:
    """Normalise whitespace and case for source-span validation."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _question_category(question: dict) -> str:
    """Assign the evaluation category recorded in kb_questions.json."""
    if question.get("contains_exact_code_or_id"):
        return "exact_code_or_id"
    if question.get("paraphrased_no_answer_keyword"):
        return "paraphrased_no_answer_keyword"
    return "other"


def validate_gold_spans(
    questions: list[dict],
    kb_dir: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Validate each gold span and return source text keyed by document ID."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = {item["doc_id"]: item for item in manifest}
    source_by_doc: dict[str, str] = {}

    for question in questions:
        doc_id = question["document_id"]
        doc = docs.get(doc_id)
        if doc is None:
            raise ValueError(f"{question['question_id']}: unknown document_id {doc_id}")

        if doc_id not in source_by_doc:
            source_by_doc[doc_id] = (kb_dir / doc["file"]).read_text(encoding="utf-8")

        source = source_by_doc[doc_id]
        gold_span = question["gold_answer_span"]
        if gold_span not in source and _normalise(gold_span) not in _normalise(source):
            raise ValueError(
                f"{question['question_id']}: gold answer span not found in {doc['file']}"
            )

        # Exact offsets are required for interval-based answer-span evaluation.
        if source.find(gold_span) < 0:
            raise ValueError(
                f"{question['question_id']}: gold span only matches after whitespace "
                "normalisation; store the exact source substring so offsets remain unambiguous"
            )

    return source_by_doc


def answer_span_is_covered(
    *,
    source_text: str,
    gold_start: int,
    gold_end: int,
    document_id: str,
    retrieved_chunks: Iterable[KnowledgeChunk],
) -> bool:
    """Return True only if retrieved chunks cover the complete gold answer span.

    Multiple adjacent/overlapping retrieved chunks may jointly cover a span.
    Whitespace-only gaps are allowed because chunk content is trimmed at its
    outer edges. A chunk from the correct document but a different section
    does not count as a retrieval hit.
    """
    intervals: list[tuple[int, int]] = []
    for chunk in retrieved_chunks:
        if chunk.doc_id != document_id:
            continue
        start = max(gold_start, int(chunk.start_char))
        end = min(gold_end, int(chunk.end_char))
        if start < end:
            intervals.append((start, end))

    if not intervals:
        return False

    intervals.sort()
    covered_until = gold_start
    for start, end in intervals:
        if end <= covered_until:
            continue
        # Ignore gaps containing only spaces/newlines removed during chunk trimming.
        if start > covered_until and source_text[covered_until:start].strip():
            return False
        covered_until = max(covered_until, end)
        if covered_until >= gold_end:
            return True

    return False


def _chunk_overlaps_answer_span(
    chunk: KnowledgeChunk,
    *,
    gold_start: int,
    gold_end: int,
    document_id: str,
) -> bool:
    """Return True when this individual chunk contains any part of the gold span."""
    if chunk.doc_id != document_id:
        return False
    return max(gold_start, int(chunk.start_char)) < min(gold_end, int(chunk.end_char))


def build_ranking_rows(
    *,
    question: dict,
    chunk_size: int,
    ranked_results: dict[str, list[tuple[KnowledgeChunk, float]]],
    source_text: str,
    gold_start: int,
    max_k: int,
) -> list[dict]:
    """Create top-ranked evidence rows, including cumulative answer-span hits.

    `chunk_answer_span_overlap` says whether one chunk overlaps any part of the
    gold span. `cumulative_answer_span_hit` says whether all chunks retrieved
    from rank 1 through this rank jointly cover the complete answer span.
    """
    rows: list[dict] = []
    gold_end = gold_start + len(question["gold_answer_span"])
    expected_doc_id = question["document_id"]
    category = _question_category(question)

    for method in RANKING_METHODS:
        # Some sparse queries may have fewer than max_k lexical matches. We do
        # not add artificial zero-score rows because they are not retrieved hits.
        ranked = ranked_results.get(method, [])[:max_k]
        retrieved_prefix: list[KnowledgeChunk] = []

        for rank, (chunk, score) in enumerate(ranked, start=1):
            retrieved_prefix.append(chunk)
            individual_overlap = _chunk_overlaps_answer_span(
                chunk,
                gold_start=gold_start,
                gold_end=gold_end,
                document_id=expected_doc_id,
            )
            cumulative_hit = answer_span_is_covered(
                source_text=source_text,
                gold_start=gold_start,
                gold_end=gold_end,
                document_id=expected_doc_id,
                retrieved_chunks=retrieved_prefix,
            )

            rows.append({
                "question_id": question["question_id"],
                "category": category,
                "chunk_size": chunk_size,
                "method": method,
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.doc_id,
                "section": chunk.section,
                "version": chunk.version,
                # Keep enough precision to inspect close RRF scores and ties.
                "score": f"{float(score):.10f}",
                "chunk_answer_span_overlap": int(individual_overlap),
                "cumulative_answer_span_hit": int(cumulative_hit),
                "gold_document_id": expected_doc_id,
            })

    return rows


def evaluate(chunk_sizes: list[int], embedding_model: str, ks: list[int]) -> Path:
    """Run dense, BM25 and hybrid evaluation and write all evidence CSVs."""
    if not chunk_sizes or any(size <= 0 for size in chunk_sizes):
        raise ValueError("chunk_sizes must contain positive values")
    if not ks or any(value <= 0 for value in ks):
        raise ValueError("ks must contain positive values")

    questions = json.loads((DATA_DIR / "kb_questions.json").read_text(encoding="utf-8"))
    source_by_doc = validate_gold_spans(
        questions,
        DATA_DIR / "kb",
        DATA_DIR / "kb" / "manifest.json",
    )
    embedder = SentenceTransformerEmbedder(embedding_model)

    out_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    ranking_rows: list[dict] = []
    max_k = max(ks)

    for size in chunk_sizes:
        directory = INDEX_DIR / f"dense_{size}"
        index = DenseIndex.load(directory)

        # Old indexes without source offsets cannot support strict answer-span evaluation.
        if not index.chunks or not all(chunk.end_char > chunk.start_char for chunk in index.chunks):
            raise ValueError(
                f"Index {directory} does not contain source character offsets. "
                "Rebuild it with: python -m src.retrieval.build_index"
            )

        bm25 = BM25Index(index.chunks)
        question_texts = [item["question"] for item in questions]

        # Apply the BGE query instruction only to query embeddings, not KB passages.
        query_vectors = embedder.encode(
            question_texts,
            purpose=f"evaluate_queries_{size}",
            is_query=True,
            show_progress_bar=True,
        )

        chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}

        for question, vector in zip(questions, query_vectors):
            dense_hits = index.search(vector, top_k=max_k)
            sparse_hits = bm25.search(question["question"], top_k=max_k)

            # RRF consumes ranked IDs; it deliberately does not combine raw scores.
            fused_pairs = reciprocal_rank_fusion(
                [
                    [hit.chunk_id for hit in dense_hits],
                    [hit.chunk_id for hit in sparse_hits],
                ],
                k=60,
            )

            # Retain each method's actual score alongside the corresponding chunk.
            # Dense/BM25 scores are native scores; hybrid scores are RRF scores.
            ranked_results: dict[str, list[tuple[KnowledgeChunk, float]]] = {
                "dense": [
                    (chunk_by_id[hit.chunk_id], float(hit.score))
                    for hit in dense_hits
                ],
                "bm25": [
                    (chunk_by_id[hit.chunk_id], float(hit.score))
                    for hit in sparse_hits
                ],
                "hybrid": [
                    (chunk_by_id[chunk_id], float(rrf_score))
                    for chunk_id, rrf_score in fused_pairs[:max_k]
                ],
            }

            # Derive the chunk-only lists used by the official hit-rate calculation.
            hit_lists = {
                method: [chunk for chunk, _score in results]
                for method, results in ranked_results.items()
            }

            gold_span = question["gold_answer_span"]
            expected_doc_id = question["document_id"]
            source = source_by_doc[expected_doc_id]
            gold_start = source.find(gold_span)
            gold_end = gold_start + len(gold_span)

            # Export top-five (or configured max_k) ranked evidence for Task 3(b).
            ranking_rows.extend(build_ranking_rows(
                question=question,
                chunk_size=size,
                ranked_results=ranked_results,
                source_text=source,
                gold_start=gold_start,
                max_k=max_k,
            ))

            for method, hits in hit_lists.items():
                for k in ks:
                    retrieved = hits[:k]
                    span_hit = int(answer_span_is_covered(
                        source_text=source,
                        gold_start=gold_start,
                        gold_end=gold_end,
                        document_id=expected_doc_id,
                        retrieved_chunks=retrieved,
                    ))
                    # Document-level matching is retained only as a diagnostic.
                    document_hit = int(any(
                        chunk.doc_id == expected_doc_id for chunk in retrieved
                    ))

                    # Preserve the exact five-column schema required by the professor.
                    out_rows.append({
                        "method": method,
                        "chunk_size": size,
                        "k": k,
                        "question_id": question["question_id"],
                        "hit": span_hit,
                    })
                    diagnostic_rows.append({
                        "method": method,
                        "chunk_size": size,
                        "k": k,
                        "question_id": question["question_id"],
                        "answer_span_hit": span_hit,
                        "document_hit": document_hit,
                    })

    # All result files are generated from the same actual retrieval run.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Official result file: do not add columns to the required schema.
    output_path = RESULTS_DIR / "hit_rate.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "chunk_size", "k", "question_id", "hit"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    # Diagnostic comparison of strict answer-span hits vs document-level hits.
    diagnostics_path = RESULTS_DIR / "hit_rate_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method", "chunk_size", "k", "question_id",
                "answer_span_hit", "document_hit",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    # Detailed rankings: the key evidence for Task 3(b)'s fixed/unfixed examples.
    rankings_path = RESULTS_DIR / "retrieval_rankings.csv"
    ranking_fields = [
        "question_id", "category", "chunk_size", "method", "rank",
        "chunk_id", "document_id", "section", "version", "score",
        "chunk_answer_span_overlap", "cumulative_answer_span_hit",
        "gold_document_id",
    ]
    with rankings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields)
        writer.writeheader()
        writer.writerows(ranking_rows)

    # Aggregate the official answer-span hit metric overall and by question type.
    question_by_id = {item["question_id"]: item for item in questions}
    categories = {
        "overall": lambda item: True,
        "exact_code_or_id": lambda item: bool(item.get("contains_exact_code_or_id")),
        "paraphrased_no_answer_keyword": lambda item: bool(
            item.get("paraphrased_no_answer_keyword")
        ),
    }
    summary_rows: list[dict] = []

    for size in chunk_sizes:
        for method in RANKING_METHODS:
            for k in ks:
                matching = [
                    row for row in out_rows
                    if row["chunk_size"] == size
                    and row["method"] == method
                    and row["k"] == k
                ]
                for category, predicate in categories.items():
                    selected = [
                        row for row in matching
                        if predicate(question_by_id[row["question_id"]])
                    ]
                    if not selected:
                        continue

                    hits = sum(row["hit"] for row in selected)
                    rate = hits / len(selected)
                    summary_rows.append({
                        "method": method,
                        "chunk_size": size,
                        "k": k,
                        "category": category,
                        "question_count": len(selected),
                        "hits": hits,
                        "hit_rate": f"{rate:.6f}",
                    })
                    print(
                        f"size={size:>3} method={method:<6} @{k} {category:<30} "
                        f"answer_span_hit_rate={rate:.3f} ({hits}/{len(selected)})"
                    )

    summary_path = RESULTS_DIR / "hit_rate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method", "chunk_size", "k", "category",
                "question_count", "hits", "hit_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(out_rows)} answer-span hit rows to {output_path}")
    print(f"Wrote document-vs-span diagnostics to {diagnostics_path}")
    print(f"Wrote {len(ranking_rows)} ranked evidence rows to {rankings_path}")
    print(f"Wrote category aggregates to {summary_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RecallBot retrieval methods")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    if not args.chunk_sizes or any(value <= 0 for value in args.chunk_sizes):
        parser.error("chunk sizes must be positive")
    if not args.ks or any(value <= 0 for value in args.ks):
        parser.error("k values must be positive")

    evaluate(args.chunk_sizes, args.embedding_model, args.ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
