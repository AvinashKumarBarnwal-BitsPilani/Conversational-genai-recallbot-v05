"""Persistence helpers for the two chunk-size retrieval indexes."""
from __future__ import annotations

import json
from pathlib import Path

from .models import KnowledgeChunk


def save_chunks(chunks: list[KnowledgeChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeChunk.from_dict(item) for item in data]
