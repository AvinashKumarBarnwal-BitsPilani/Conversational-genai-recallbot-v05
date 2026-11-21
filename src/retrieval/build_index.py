"""Build dense vector files and metadata for both required chunk sizes.

Run from the project root:
    python -m src.retrieval.build_index
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..config import INDEX_DIR, KB_DIR, TOKENIZER_ID
from .chunker import chunk_corpus, load_knowledge_documents
from .dense import DenseIndex
from .embeddings import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder
from .io import save_chunks
from .tokenizer import load_tokenizer


def build_indexes(chunk_sizes: list[int], overlap_fraction: float, embedding_model: str) -> None:
    """Create one self-contained flat dense index for each chunk size."""
    manifest_path = KB_DIR / "manifest.json"
    documents = load_knowledge_documents(KB_DIR, manifest_path)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    embedder = SentenceTransformerEmbedder(embedding_model)

    print(f"Loaded {len(documents)} KB documents")
    print(f"Chunk tokenizer: {TOKENIZER_ID}")
    print(f"Embedding model: {embedding_model} (dimension={embedder.dimension})")
    print("Pooling: BGE CLS pooling; token-weighted mean across subsegments only when needed")

    for size in chunk_sizes:
        print(f"\nBuilding {size}-token chunks with {overlap_fraction:.0%} overlap...")
        chunks = chunk_corpus(documents, tokenizer, size, overlap_fraction)
        print(f"Created {len(chunks)} chunks")

        # This single encode call is measured and recorded by the embedder.
        vectors = embedder.encode(
            [chunk.content for chunk in chunks],
            purpose=f"index_kb_chunks_{size}",
            show_progress_bar=True,
        )
        index = DenseIndex(chunks, vectors)
        output_dir = INDEX_DIR / f"dense_{size}"
        index.save(output_dir)

        # Keep a small manifest beside the arrays so later runs can identify
        # exactly how this index was produced without guessing from filenames.
        index_manifest = {
            "index_type": "numpy_flat_exact",
            "chunk_size_tokens": size,
            "overlap_fraction": overlap_fraction,
            "overlap_tokens": int(round(size * overlap_fraction)),
            "chunk_tokenizer": TOKENIZER_ID,
            "embedding_model": embedding_model,
            "embedding_dimension": embedder.dimension,
            "pooling": "CLS",
            "embedding_max_sequence_length": embedder.max_seq_length,
            "long_chunk_strategy": (
                "split with embedding-model tokenizer; token-weighted mean of segment CLS vectors; L2 normalize"
            ),
            "normalization": "L2",
            "similarity": "cosine (dot product over normalized vectors)",
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        (output_dir / "index_manifest.json").write_text(
            json.dumps(index_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(chunks)} vectors to {output_dir.relative_to(Path.cwd())}")

    print("\nIndex build complete. No chat LLM was used.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RecallBot dense retrieval indexes")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--overlap", type=float, default=0.15, help="Overlap fraction, e.g. 0.15")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    if any(size <= 0 for size in args.chunk_sizes):
        parser.error("chunk sizes must be positive")
    if not 0.10 <= args.overlap <= 0.20:
        parser.error("Mini Project 1 requires 10–20 percent chunk overlap")
    build_indexes(args.chunk_sizes, args.overlap, args.embedding_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
