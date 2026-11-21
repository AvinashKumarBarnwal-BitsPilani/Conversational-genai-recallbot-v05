"""Tokenizer helpers shared by chunking and token-cost accounting.

RecallBot uses the Qwen3 tokenizer because Qwen3:1.7B is the chat model selected
for this project. Hugging Face downloads tokenizer files only; it does not need
to download the full model weights for token counting.
"""
from __future__ import annotations

from functools import lru_cache
from transformers import AutoTokenizer

from ..config import TOKENIZER_ID


@lru_cache(maxsize=2)
def load_tokenizer(model_name: str = TOKENIZER_ID):
    """Load and cache a fast Hugging Face tokenizer for offset-aware chunking."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer {model_name!r} is not a fast tokenizer; "
            "the chunker needs character offset mappings."
        )
    return tokenizer


def count_tokens(text: str, tokenizer=None) -> int:
    """Count model tokens without adding BOS/EOS or other special tokens."""
    tokenizer = tokenizer or load_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))
