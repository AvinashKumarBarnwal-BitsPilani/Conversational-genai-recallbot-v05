"""Logged Qwen/Ollama calls used by memory components (extract, reflect).

Every attempted call writes exactly one calls.jsonl row, including failures.
Token counts come from Ollama's usage fields when present; otherwise the local
Qwen tokenizer (or a conservative estimate) is used and labelled as such.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from ..config import CHAT_MODEL, OLLAMA_HOST
from ..logger import log_call, timer

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def default_token_counter(text: str) -> int:
    """Qwen tokenizer from the local cache only; conservative estimate otherwise."""
    if not text:
        return 0
    try:
        from ..retrieval.tokenizer import count_tokens, load_tokenizer
        return count_tokens(text, tokenizer=load_tokenizer(local_files_only=True))
    except Exception:
        return (len(text) + 2) // 3


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class LoggedLLM:
    """Thin wrapper over the Ollama client that always logs the call."""

    def __init__(
        self,
        *,
        model: str = CHAT_MODEL,
        host: str = OLLAMA_HOST,
        client: Any | None = None,
        logger: Callable[..., Any] | None = log_call,
        token_counter: Callable[[str], int] = default_token_counter,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model = model
        self.host = host
        self.client = client
        self.logger = logger
        self.token_counter = token_counter
        self.timeout_seconds = timeout_seconds

    def _get_client(self) -> Any:
        if self.client is None:
            from ollama import Client
            self.client = Client(host=self.host, timeout=self.timeout_seconds)
        return self.client

    def call(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        call_type: str,
        purpose: str,
        context: Mapping[str, Any],
        json_mode: bool = False,
        options: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Run one chat completion and return its text content."""
        started = timer()
        request_options = {"temperature": 0.0, "top_p": 0.9, "num_ctx": 2048, "num_predict": 200}
        request_options.update(options or {})
        response: Any = None
        content = ""
        status = "success"
        error_type: str | None = None
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [dict(message) for message in messages],
                "options": request_options,
                "think": False,
                "stream": False,
            }
            if json_mode:
                kwargs["format"] = "json"
            response = self._get_client().chat(**kwargs)
            message = _field(response, "message", {})
            content = _THINK_RE.sub("", str(_field(message, "content", "") or "")).strip()
            if not content:
                raise RuntimeError(f"Ollama returned an empty {call_type} response")
            return content
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            if self.logger is not None:
                raw_in = _field(response, "prompt_eval_count", None) if response is not None else None
                raw_out = _field(response, "eval_count", None) if response is not None else None
                joined = "\n".join(str(m.get("content", "")) for m in messages)
                record_extra: dict[str, Any] = {
                    "purpose": purpose,
                    "status": status,
                    "decoding": request_options,
                    "json_mode": json_mode,
                    "token_counts_source": "ollama" if raw_in is not None and raw_out is not None else "local_estimate",
                }
                if error_type:
                    record_extra["error_type"] = error_type
                record_extra.update(extra or {})
                self.logger(
                    config=str(context.get("config", "hybrid_mem")),
                    user=str(context.get("user", "UNKNOWN")),
                    session=str(context.get("session", "A")),
                    turn=int(context.get("turn", 0)),
                    call_type=call_type,
                    model=self.model,
                    input_tokens=max(0, int(raw_in) if raw_in is not None else self.token_counter(joined)),
                    output_tokens=max(0, int(raw_out) if raw_out is not None else self.token_counter(content)),
                    retrieved_tokens=0,
                    start_time=started,
                    extra=record_extra,
                )
