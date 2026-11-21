"""Sentence-transformer embeddings with explicit sequence-length handling.

BGE-small-en-v1.5 produces 384-dimensional embeddings and supports a maximum
sequence length of 512 model tokens. The project's chunks are measured with
the Qwen tokenizer, so a Qwen-sized chunk may be longer under BGE's own
WordPiece tokenizer. This module splits such inputs into safe BGE-token
segments, embeds each segment, and combines the segment vectors back into one
vector per original input. No chunk tail is silently truncated.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import TOKENIZER_ID
from ..logger import log_call, timer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE recommends this instruction for short-query-to-passage retrieval.
# It is applied only to queries, never to knowledge-base passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _content_token_ids(text: str, tokenizer: Any) -> list[int]:
    """Return content token IDs without adding model special tokens.

    Calling the Hugging Face ``tokenize()`` wrapper can emit a maximum-length
    warning when we inspect an original chunk that is intentionally longer
    than the embedding model limit. The underlying fast-tokenizer backend
    performs the same tokenization without that wrapper-level length warning.

    Lightweight test tokenizers may not expose ``backend_tokenizer``; those
    continue to use their simple ``encode()`` implementation.
    """
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)

    if backend_tokenizer is not None:
        # Direct backend call: content token IDs only, with no [CLS]/[SEP].
        encoding = backend_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return list(encoding.ids)

    # Compatibility fallback for minimal test doubles such as ToyEmbeddingTokenizer.
    return list(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def _content_token_count(text: str, tokenizer: Any) -> int:
    """Count content tokens using the same warning-free path as segmentation."""
    return len(_content_token_ids(text, tokenizer))


def split_text_for_embedding(
    text: str,
    tokenizer: Any,
    max_content_tokens: int,
) -> list[str]:
    """Split text into decoded segments that fit the embedding model's limit.

    The project chunker measures chunks with Qwen's tokenizer. This helper
    checks them again with BGE's tokenizer because token counts differ between
    models. Content-token counting uses the fast-tokenizer backend directly, so inspecting
    an oversized original chunk does not trigger the Transformers wrapper warning.

    Each decoded segment is tokenized again before being accepted. This is
    important because decoding and re-tokenizing a token slice can sometimes
    change token boundaries.
    """
    if max_content_tokens <= 0:
        raise ValueError("max_content_tokens must be positive")

    # Convert the original text to content token IDs without preparing a model
    # input. For Hugging Face tokenizers this avoids the long-sequence warning;
    # minimal test tokenizers use their simpler encode() implementation.
    token_ids = _content_token_ids(text, tokenizer)

    # Most chunks fit in one model input and do not need any further splitting.
    if len(token_ids) <= max_content_tokens:
        return [text]

    segments: list[str] = []
    start = 0

    while start < len(token_ids):
        # Try the largest permitted token window first.
        end = min(start + max_content_tokens, len(token_ids))
        segment = ""

        # If decoding changes token boundaries and creates an oversized piece,
        # shorten the slice until its re-tokenized content fits the limit.
        while end > start:
            token_slice = token_ids[start:end]
            segment = tokenizer.decode(
                token_slice,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if segment and _content_token_count(segment, tokenizer) <= max_content_tokens:
                break

            end -= 1

        if end <= start or not segment:
            raise ValueError(
                "Could not split text safely for the embedding model tokenizer"
            )

        segments.append(segment)
        start = end

    return segments


class SentenceTransformerEmbedder:
    """Generate normalized BGE embeddings without silent input truncation.

    BGE-small-en-v1.5 uses CLS pooling in its Sentence-Transformers model
    configuration. If an original project chunk requires multiple BGE-sized
    segments, their normalized vectors are combined using a content-token-
    weighted mean and normalized again. The returned array therefore contains
    exactly one embedding vector for every original input text.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        # Import lazily: pure unit tests for the splitting helper should not
        # require downloading a model or importing the PyTorch backend.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

        # Use the current Sentence-Transformers API for the output dimension.
        self.dimension = int(self.model.get_embedding_dimension())

        # Keep BGE's tokenizer separate from Qwen's project tokenizer.
        self.embedding_tokenizer = self.model.tokenizer
        self.max_seq_length = int(self.model.max_seq_length)

        # Reserve space for BGE's special tokens (typically [CLS] and [SEP]).
        special_tokens = int(
            self.embedding_tokenizer.num_special_tokens_to_add(pair=False)
        )
        self.max_content_tokens = self.max_seq_length - special_tokens

        if self.max_content_tokens <= 0:
            raise ValueError(
                "Embedding model sequence length is too small for special tokens"
            )

        # Load Qwen's tokenizer only when a real embedder is created. It is used
        # for the project's chat-model token accounting in the mandatory logs.
        from .tokenizer import load_tokenizer

        self.chat_tokenizer = load_tokenizer(TOKENIZER_ID)

    def encode(
        self,
        texts: list[str],
        *,
        purpose: str,
        is_query: bool = False,
        user: str = "SYSTEM",
        session: str = "A",
        turn: int = 0,
        config: str = "hybrid",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts and append a mandatory log row for the embedding call.

        Long inputs are split with BGE's tokenizer, embedded independently,
        and aggregated back into one vector per original text. The maximum
        actual BGE sequence length (including special tokens) is measured and
        recorded, allowing us to verify that no model input exceeds its limit.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # These arrays maintain the relationship between original texts and
        # their one-or-many embedding segments.
        segments: list[str] = []
        segment_weights: list[int] = []
        segment_ranges: list[tuple[int, int]] = []
        model_inputs: list[str] = []

        # Audit metric: longest actual input sent to the BGE model, including
        # special tokens. It must never exceed self.max_seq_length.
        max_segment_sequence_length = 0

        # Process each original text/chunk independently.
        for original_text in texts:
            # Add BGE's recommended instruction only for search queries.
            model_text = (
                f"{QUERY_INSTRUCTION}{original_text}"
                if is_query
                else original_text
            )
            model_inputs.append(model_text)

            # Split the text into BGE-token windows that fit the content budget.
            pieces = split_text_for_embedding(
                model_text,
                self.embedding_tokenizer,
                self.max_content_tokens,
            )

            # Record where this original text's segments begin in the flat list.
            first_segment = len(segments)

            # Validate and collect every segment generated for this text.
            for piece in pieces:
                # Count the full model input, including special tokens.
                sequence_length = len(
                    self.embedding_tokenizer.encode(
                        piece,
                        add_special_tokens=True,
                    )
                )

                # Fail fast instead of allowing Sentence-Transformers to truncate.
                if sequence_length > self.max_seq_length:
                    raise ValueError(
                        "Embedding input exceeds model limit: "
                        f"{sequence_length} > {self.max_seq_length}"
                    )

                # Retain the maximum observed sequence length for the audit log.
                max_segment_sequence_length = max(
                    max_segment_sequence_length,
                    sequence_length,
                )

                segments.append(piece)

                # Use content-token count as the segment's aggregation weight.
                weight = _content_token_count(piece, self.embedding_tokenizer)
                segment_weights.append(max(weight, 1))

            # This line belongs OUTSIDE the inner loop: it closes the segment
            # range for one original text after all its pieces have been added.
            segment_ranges.append((first_segment, len(segments)))

        # Count input tokens with Qwen's chat tokenizer for consistent project-
        # wide token accounting. Separately count BGE tokens sent to embeddings.
        input_tokens = sum(
            len(self.chat_tokenizer.encode(text, add_special_tokens=False))
            for text in model_inputs
        )
        embedding_input_tokens = sum(
            _content_token_count(text, self.embedding_tokenizer)
            for text in segments
        )

        start_time = timer()
        try:
            # Sentence-Transformers adds the model's special tokens here. Every
            # segment was already checked against the same model-token limit.
            segment_vectors = self.model.encode(
                segments,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            )
            segment_vectors = np.asarray(segment_vectors, dtype=np.float32)

            # Aggregate one-or-more segment vectors into one vector per input.
            output_vectors: list[np.ndarray] = []
            for first, last in segment_ranges:
                weights = np.asarray(
                    segment_weights[first:last],
                    dtype=np.float32,
                )
                vectors = segment_vectors[first:last]

                # Longer segments contribute proportionally more to the result.
                combined = np.average(vectors, axis=0, weights=weights)
                norm = float(np.linalg.norm(combined))

                if norm == 0.0:
                    raise ValueError("Embedding aggregation produced a zero vector")

                output_vectors.append(combined / norm)

            result = np.vstack(output_vectors).astype(np.float32)

            # Log successful embedding call and the observed model input limit.
            log_call(
                config=config,
                user=user,
                session=session,
                turn=turn,
                call_type="embed",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=0,
                retrieved_tokens=0,
                start_time=start_time,
                extra={
                    "purpose": purpose,
                    "role": "query" if is_query else "passage",
                    "query_instruction_applied": is_query,
                    "texts_count": len(texts),
                    "embedding_segments": len(segments),
                    "texts_split_for_limit": sum(
                        last - first > 1 for first, last in segment_ranges
                    ),
                    "embedding_input_tokens": embedding_input_tokens,
                    "max_sequence_length": self.max_seq_length,
                    "max_content_tokens_per_segment": self.max_content_tokens,
                    "max_segment_sequence_length": max_segment_sequence_length,
                    "dimension": self.dimension,
                    "pooling": (
                        "CLS; token-weighted mean across segments "
                        "when an input exceeds the model limit"
                    ),
                },
            )

            return result

        except Exception as exc:
            # Keep failed embedding attempts in the append-only audit log too.
            log_call(
                config=config,
                user=user,
                session=session,
                turn=turn,
                call_type="embed",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=0,
                retrieved_tokens=0,
                start_time=start_time,
                extra={
                    "purpose": purpose,
                    "role": "query" if is_query else "passage",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "max_sequence_length": self.max_seq_length,
                    "max_segment_sequence_length": max_segment_sequence_length,
                },
            )
            raise
