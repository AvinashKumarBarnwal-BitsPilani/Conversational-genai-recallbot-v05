"""Deterministic test doubles for memory tests (no model download, no Ollama)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "my", "i", "to", "of", "for", "and", "is", "their", "customer", "did", "what", "which"}


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words have higher cosine."""

    dimension = 256

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts, *, purpose, is_query=False, **context):
        self.calls.append({"texts": list(texts), "purpose": purpose, "is_query": is_query, **context})
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for word in _WORD_RE.findall(text.casefold()):
                if word in _STOP:
                    continue
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vector[bucket] += 1.0
            if not vector.any():
                vector[0] = 1.0
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


class FakeOllamaClient:
    """Returns queued contents and records every chat request."""

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.requests: list[dict[str, Any]] = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        content = self.contents.pop(0) if self.contents else json.dumps({"facts": []})
        return {"message": {"content": content}, "prompt_eval_count": 40, "eval_count": 12}


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def __call__(self, **record):
        self.records.append(record)
        return record

    def of_type(self, call_type: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["call_type"] == call_type]
