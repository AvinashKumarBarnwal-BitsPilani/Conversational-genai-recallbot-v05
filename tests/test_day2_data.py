"""Day 2 source-data checks; no model calls or logs are produced here."""
import json
import re

from src.config import DATA_DIR, KB_DIR
from src.retrieval.chunker import load_knowledge_documents, chunk_document


class ToyOffsetTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_all_kb_documents_have_version_and_effective_date_metadata():
    documents = load_knowledge_documents(KB_DIR, KB_DIR / "manifest.json")
    assert len(documents) >= 15
    assert all(doc.get("version") for doc in documents)
    assert all(doc.get("effective_from") for doc in documents)


def test_plain_text_kb_section_heading_is_captured():
    document = {
        "doc_id": "KB-X", "title": "Policy", "version": "1.0",
        "effective_from": "2026-01-01", "effective_to": "open", "status": "active",
        "file": "x.md",
        "content": "Policy title\nDocument ID: KB-X | Version: 1.0\n\nReturn windows\nOne two three four five six seven eight nine ten.",
    }
    chunks = chunk_document(document, ToyOffsetTokenizer(), chunk_size=6, overlap_fraction=0.15)
    assert chunks
    # Chunk 1 begins in the document header; a later chunk begins inside the
    # body section and should carry that section's title.
    assert any(chunk.section == "Return windows" for chunk in chunks[1:])


def test_all_gold_answer_spans_exist_in_their_labelled_source_document():
    questions = json.loads((DATA_DIR / "kb_questions.json").read_text(encoding="utf-8"))
    documents = load_knowledge_documents(KB_DIR, KB_DIR / "manifest.json")
    by_id = {document["doc_id"]: document for document in documents}

    def normalise(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    for question in questions:
        source = by_id[question["document_id"]]["content"]
        assert normalise(question["gold_answer_span"]) in normalise(source), question["question_id"]
