"""Token-aware chunking for the ShopSphere knowledge base."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import KnowledgeChunk


def load_knowledge_documents(kb_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Load markdown files and join each one to its manifest metadata."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    for item in manifest:
        path = kb_dir / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"KB manifest points to missing file: {path}")
        content = path.read_text(encoding="utf-8")
        # The Day 1 manifest tracks effective dates/status; the document header
        # carries its explicit version number (for example, Version: 2.0).
        version_match = re.search(r"\bVersion:\s*([A-Za-z0-9._-]+)", content, flags=re.IGNORECASE)
        documents.append({
            **item,
            "version": version_match.group(1) if version_match else "unspecified",
            "content": content,
            "path": path,
        })
    return documents


def _section_at(text: str, char_position: int, fallback: str) -> str:
    """Return the latest Markdown or plain-text section heading before a chunk.

    The supplied ShopSphere KB files use plain-text section labels rather than
    Markdown hash headings, so both formats are recognised here.
    """
    heading = fallback
    prefix = text[:char_position]
    for line in prefix.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        markdown_heading = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
        if markdown_heading:
            heading = markdown_heading.group(1).strip()
            continue
        # A short standalone line without sentence punctuation is treated as
        # a section label. Metadata lines containing separators are excluded.
        if (len(stripped) <= 80 and "|" not in stripped
                and not stripped.lower().startswith("document id:")
                and not stripped.endswith((".", "?", "!", ":"))):
            heading = stripped
    return heading


def chunk_document(
    document: dict[str, Any],
    tokenizer,
    chunk_size: int,
    overlap_fraction: float = 0.15,
) -> list[KnowledgeChunk]:
    """Split one document into overlapping windows measured in model tokens.

    Fast-tokenizer character offsets let us slice the original Markdown text,
    rather than decoding token IDs and risking altered punctuation or codes.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")

    text = document["content"]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids, offsets = token_ids[0], offsets[0]
    if not token_ids:
        return []
    if len(token_ids) != len(offsets):
        raise ValueError("Tokenizer returned mismatched token IDs and offsets")

    overlap_tokens = int(round(chunk_size * overlap_fraction))
    step = chunk_size - overlap_tokens
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk_size")

    chunks: list[KnowledgeChunk] = []
    start_token = 0
    while start_token < len(token_ids):
        end_token = min(start_token + chunk_size, len(token_ids))
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        raw_content = text[char_start:char_end]
        # Keep offsets aligned with the trimmed content saved in the chunk.
        leading_whitespace = len(raw_content) - len(raw_content.lstrip())
        content = raw_content.strip()
        content_start = char_start + leading_whitespace
        content_end = content_start + len(content)

        if content:
            # Recount after trimming boundary whitespace so token_count matches
            # the actual text stored in the chunk as closely as the tokenizer allows.
            actual_token_count = len(tokenizer.encode(content, add_special_tokens=False))
            chunk_number = len(chunks) + 1
            doc_id = document["doc_id"]
            size_label = f"{chunk_size:03d}"
            section = _section_at(text, char_start, document.get("title", doc_id))
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{doc_id}-C{size_label}-{chunk_number:04d}",
                    doc_id=doc_id,
                    title=document.get("title", doc_id),
                    section=section,
                    version=str(document.get("version", "")),
                    effective_from=str(document.get("effective_from", "")),
                    effective_to=str(document.get("effective_to", "")),
                    status=str(document.get("status", "")),
                    source_file=str(document.get("file", "")),
                    chunk_size=chunk_size,
                    overlap_tokens=overlap_tokens,
                    token_count=actual_token_count,
                    content=content,
                    # These offsets let evaluation check whether the retrieved
                    # chunk(s) cover the actual gold answer span, not just its document.
                    start_char=content_start,
                    end_char=content_end,
                )
            )

        if end_token == len(token_ids):
            break
        start_token += step

    return chunks


def chunk_corpus(
    documents: Iterable[dict[str, Any]],
    tokenizer,
    chunk_size: int,
    overlap_fraction: float = 0.15,
) -> list[KnowledgeChunk]:
    """Chunk all supplied documents, keeping each chunk's source metadata."""
    all_chunks: list[KnowledgeChunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, tokenizer, chunk_size, overlap_fraction))
    return all_chunks
