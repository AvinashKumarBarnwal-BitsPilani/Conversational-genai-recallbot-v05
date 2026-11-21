"""Offline chunker tests using a tiny deterministic tokenizer (no model download)."""
from src.retrieval.chunker import chunk_document


class ToyOffsetTokenizer:
    """One whitespace-delimited token per item, with character offsets."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        for match in __import__("re").finditer(r"\S+", text):
            offsets.append((match.start(), match.end()))
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_chunker_respects_token_window_and_overlap():
    document = {
        "doc_id": "KB-TEST", "title": "Test policy", "version": "1",
        "effective_from": "2026-01-01", "effective_to": "open", "status": "active",
        "file": "test.md", "content": "one two three four five six seven eight nine ten",
    }
    chunks = chunk_document(document, ToyOffsetTokenizer(), chunk_size=4, overlap_fraction=0.25)
    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]
    assert chunks[0].content.endswith("four")
    assert chunks[1].content.startswith("four")  # one-token overlap
    assert all(chunk.doc_id == "KB-TEST" for chunk in chunks)
    # Offsets point to the trimmed content in the original source document.
    assert document["content"][chunks[0].start_char:chunks[0].end_char] == chunks[0].content
    assert document["content"][chunks[1].start_char:chunks[1].end_char] == chunks[1].content
