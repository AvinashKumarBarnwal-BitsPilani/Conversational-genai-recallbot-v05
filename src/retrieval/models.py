"""Small typed records used by the custom retrieval implementations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    """A token-bounded piece of a KB document with provenance metadata."""
    chunk_id: str
    doc_id: str
    title: str
    section: str
    version: str
    effective_from: str
    effective_to: str
    status: str
    source_file: str
    chunk_size: int
    overlap_tokens: int
    token_count: int
    content: str
    # Character offsets locate this exact chunk within its source document.
    # Defaults keep older serialized chunks loadable, but new indexes always populate them.
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeChunk":
        return cls(**value)


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk and the score produced by one retrieval method."""
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    content: str
    section: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
