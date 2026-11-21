"""Embedding segmentation tests use a tiny fake tokenizer; no model download."""
from src.retrieval.embeddings import split_text_for_embedding


class ToyEmbeddingTokenizer:
    """A simple one-token-per-word tokenizer for testing safe segmentation."""
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(token_ids)


def test_short_text_is_not_split():
    tokenizer = ToyEmbeddingTokenizer()
    text = "one two three"
    assert split_text_for_embedding(text, tokenizer, max_content_tokens=5) == [text]


def test_long_text_is_split_without_losing_tokens_or_exceeding_limit():
    tokenizer = ToyEmbeddingTokenizer()
    text = "one two three four five six seven eight nine ten eleven"
    segments = split_text_for_embedding(text, tokenizer, max_content_tokens=4)

    assert all(len(tokenizer.encode(segment)) <= 4 for segment in segments)
    reconstructed_tokens = [token for segment in segments for token in tokenizer.encode(segment)]
    assert reconstructed_tokens == text.split()
    assert len(segments) == 3
