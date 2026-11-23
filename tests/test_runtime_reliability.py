import json
import sys
import types

import httpx
import pytest

from src.dialogue.dialogue_manager import _default_token_counter
from src.dialogue.qwen_nlg import QwenNLG
from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker
from src.retrieval.embeddings import SentenceTransformerEmbedder


def test_nlu_counter_falls_back_without_cached_tokenizer(monkeypatch):
    from src.retrieval import tokenizer
    def missing(*args, **kwargs):
        raise OSError("not cached")
    monkeypatch.setattr(tokenizer.AutoTokenizer, "from_pretrained", missing)
    assert _default_token_counter("a sentence") == 4


def test_nlu_counter_requests_local_files_only(monkeypatch):
    from src.retrieval import tokenizer
    calls = []
    monkeypatch.setattr(tokenizer.AutoTokenizer, "from_pretrained", lambda *a, **kw: calls.append(kw) or types.SimpleNamespace(is_fast=True, encode=lambda text, **k: [1, 2]))
    assert _default_token_counter("hello") == 2
    assert calls and calls[0]["local_files_only"] is True


def test_nlu_log_written_with_fallback_and_no_hf_request(tmp_path, monkeypatch, request):
    from src.retrieval import tokenizer
    from src import logger as logger_module

    # Other tests may have populated this specific shared loader cache key.
    # Snapshot and restore the cache so this regression always exercises loading.
    tokenizer.load_tokenizer.cache_clear()
    request.addfinalizer(tokenizer.load_tokenizer.cache_clear)
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("not cached")
    monkeypatch.setattr(tokenizer.AutoTokenizer, "from_pretrained", unavailable)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "calls.jsonl")

    class Tools:
        pass

    manager = DialogueManager(
        tracker=DialogueStateTracker(session_id="test", user_id="customer"),
        tools=Tools(),
        logger=logger_module.log_call,
    )
    nlu = {"intent": "unknown", "rule_id": "R99_NO_MATCH"}
    manager._log_nlu("What is this?", nlu, turn=1, start_time=logger_module.timer())

    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["call_type"] == "nlu"
    assert rows[0]["input_tokens"] > 0
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["use_fast"] is True


def test_nlg_local_tokenizer_fallback(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"; prompt.write_text("prompt")
    workflow = tmp_path / "w.json"; workflow.write_text('{"version":"1"}')
    fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: (_ for _ in ()).throw(OSError("missing"))))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    nlg = QwenNLG(system_prompt_path=prompt, workflow_path=workflow, client=object(), logger=None)
    assert nlg._count_tokens("abcdef") == 2
    assert nlg._nlg_tokenizer is None


def _make_truncation_nlg(tmp_path, tokenizer):
    prompt = tmp_path / "truncate_prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "truncate_workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=object(),
        logger=None,
        token_counter=None,
    )
    nlg._nlg_tokenizer = tokenizer
    nlg._nlg_tokenizer_load_attempted = True
    return nlg


class _ToyTokenIDTokenizer:
    def __init__(self):
        self.encode_inputs = []
        self.decode_inputs = []

    def encode(self, text, add_special_tokens=False):
        self.encode_inputs.append(text)
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_inputs.append(list(token_ids))
        return " ".join(token_ids)


def test_truncation_preserves_text_within_limit(tmp_path):
    tok = _ToyTokenIDTokenizer()
    nlg = _make_truncation_nlg(tmp_path, tok)
    text = "one two three"
    assert nlg._truncate_to_tokens(text, 3) == text
    assert tok.encode_inputs == [text]
    assert tok.decode_inputs == []


def test_truncation_uses_token_ids_and_respects_limit(tmp_path):
    tok = _ToyTokenIDTokenizer()
    nlg = _make_truncation_nlg(tmp_path, tok)
    result = nlg._truncate_to_tokens("one two three four five", 3)
    assert result == "one two three"
    assert len(tok.encode(result, add_special_tokens=False)) <= 3
    assert tok.decode_inputs == [["one", "two", "three"]]


def test_truncation_fallback_and_boundaries(tmp_path):
    nlg = _make_truncation_nlg(tmp_path, None)
    nlg._nlg_tokenizer_load_attempted = True
    assert nlg._truncate_to_tokens("short", 2) == "short"
    assert nlg._truncate_to_tokens("abcdefghij", 2) == "abcdef"
    assert nlg._truncate_to_tokens("", 2) == ""
    assert nlg._truncate_to_tokens("x", 1) == "x"
    with pytest.raises(ValueError, match="zero-token"):
        nlg._truncate_to_tokens("x", 0)


def test_custom_counter_truncation_respects_counter_limit(tmp_path):
    nlg = _make_truncation_nlg(tmp_path, None)
    nlg._nlg_tokenizer_load_attempted = True
    nlg.token_counter = len
    result = nlg._truncate_to_tokens("0123456789", 4)
    assert result == "0123"
    assert nlg._count_tokens(result) <= 4


def test_tokenizer_truncation_boundary_uses_prompt_counter_convention(tmp_path):
    class BoundaryTokenizer(_ToyTokenIDTokenizer):
        def encode(self, text, add_special_tokens=False):
            self.encode_inputs.append(text)
            # Simulate decode/re-encode boundary merging: selected prefix IDs
            # can produce a decoded string that counts over the requested cap.
            return list(text)

        def decode(self, token_ids, skip_special_tokens=True):
            self.decode_inputs.append(list(token_ids))
            return "".join(token_ids)

    tok = BoundaryTokenizer()
    nlg = _make_truncation_nlg(tmp_path, tok)
    result = nlg._truncate_to_tokens("abcdefgh", 4)
    assert nlg._count_tokens(result) <= 4


def _build_prompt_nlg(tmp_path, counter, *, max_evidence_chunks=3, per_chunk=180):
    prompt = tmp_path / "build_prompt.txt"
    prompt.write_text("system", encoding="utf-8")
    workflow = tmp_path / "build_workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    return QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=object(),
        logger=None,
        token_counter=counter,
        context_window=2048,
        max_evidence_chunks=max_evidence_chunks,
        max_evidence_tokens_per_chunk=per_chunk,
    )


def _build_prompt(nlg, evidence=None, conversation_history=None, conversation_summary=None):
    return nlg._build_messages(
        draft_response="Draft", user_message="Question", intent="knowledge_base_query",
        state={}, evidence=evidence, sources=[], turn_type="task",
        conversation_history=conversation_history,
        conversation_summary=conversation_summary,
    )


def test_prompt_build_without_evidence(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text) // 100 + 1)
    messages, evidence = _build_prompt(nlg)
    assert len(messages) == 2 and evidence == []


def test_prompt_build_formats_three_evidence_chunks(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text) // 100 + 1)
    hits = [{"content": f"passage {i}", "doc_id": f"KB-{i}"} for i in range(3)]
    _, evidence = _build_prompt(nlg, hits)
    assert len(evidence) == 3


@pytest.mark.parametrize("strategy", ["full", "trim"])
def test_prompt_trimming_preserves_history_and_rebuilds_current_payload(tmp_path, strategy):
    nlg = _build_prompt_nlg(tmp_path, len)
    nlg._system_message = lambda: "s" * 1070
    exchange_count = 2 if strategy == "full" else 3
    history = [
        message
        for index in range(exchange_count)
        for message in (
            {"role": "user", "content": f"prior user {index} " + "u" * 10},
            {"role": "assistant", "content": f"prior assistant {index} " + "a" * 10},
        )
    ]

    messages, evidence = _build_prompt(
        nlg,
        [{"content": "e" * 1000, "doc_id": "KB-TRIM"}],
        conversation_history=history,
    )

    assert messages[1 : 1 + len(history)] == history
    assert messages[1]["content"] == history[0]["content"]
    current_payload_index = 1 + len(history)
    assert messages[current_payload_index]["role"] == "user"
    payload = json.loads(messages[current_payload_index]["content"].split("\n\n", 1)[1])
    assert payload["current_user_message"] == "Question"
    assert messages[current_payload_index]["content"].count('"current_user_message"') == 1
    assert len(evidence[0]["content"]) < 180
    assert payload["retrieved_evidence"] == evidence
    budget = 2048 - 160 - 32
    assert nlg._count_tokens("\n".join(message["content"] for message in messages)) <= budget


def test_prompt_over_budget_trims_evidence_and_makes_progress(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text))
    nlg._system_message = lambda: "s" * 300
    hit = {"content": "e" * 1200, "doc_id": "KB-X"}
    _, evidence = _build_prompt(nlg, [hit])
    assert evidence[0]["content"] and len(evidence[0]["content"]) < 540


def test_prompt_trimming_no_progress_raises(tmp_path, monkeypatch):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text))
    nlg._system_message = lambda: "s" * 1800
    monkeypatch.setattr(nlg, "_truncate_to_tokens", lambda text, limit: text)
    with pytest.raises(RuntimeError, match="did not shorten"):
        _build_prompt(nlg, [{"content": "e" * 400}])


def test_prompt_trimming_rejects_different_but_longer_evidence(tmp_path, monkeypatch):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text))
    nlg._system_message = lambda: "s" * 1800
    monkeypatch.setattr(nlg, "_truncate_to_tokens", lambda text, limit: text + "x")
    with pytest.raises(RuntimeError, match="did not shorten"):
        _build_prompt(nlg, [{"content": "e" * 400}])


def test_prompt_trimming_enforces_derived_iteration_limit(tmp_path, monkeypatch):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text))
    nlg._system_message = lambda: "s" * 1900
    monkeypatch.setattr(nlg, "_truncate_to_tokens", lambda text, limit: text[:-1])
    with pytest.raises(RuntimeError, match="derived limit of 12"):
        _build_prompt(nlg, [{"content": "e" * 500}])


def test_system_prompt_alone_over_budget_raises_final_budget_error(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text))
    nlg._system_message = lambda: "s" * 2100
    with pytest.raises(ValueError, match="above the safe input budget"):
        _build_prompt(nlg)


def test_conversation_summary_is_included_in_prompt_budget(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, len)
    nlg._system_message = lambda: "short system prompt"
    with pytest.raises(ValueError, match="above the safe input budget"):
        _build_prompt(nlg, conversation_summary="summary " * 2500)


def test_embedder_requests_local_model_and_clear_missing_error(monkeypatch):
    captured = {}
    st = types.ModuleType("sentence_transformers")
    def constructor(*args, **kwargs):
        captured.update(kwargs)
        raise OSError("missing")
    st.SentenceTransformer = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    with pytest.raises(RuntimeError, match="unavailable"):
        SentenceTransformerEmbedder()
    assert captured["local_files_only"] is True


def test_tools_token_accounting_uses_cached_tokenizer_only(monkeypatch):
    from src.dialogue.tools import RecallBotToolGateway
    from src.retrieval import tokenizer

    calls = []
    monkeypatch.setattr(
        tokenizer,
        "load_tokenizer",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or types.SimpleNamespace(encode=lambda text, **kw: [1, 2]),
    )
    gateway = RecallBotToolGateway(logger=None, token_counter=None)
    assert gateway._count_tokens("tool log") == 2
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True


def test_embedder_qwen_log_tokenizer_is_cache_only_and_optional(monkeypatch):
    from src.retrieval import tokenizer

    calls = []

    class FakeEmbeddingTokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.tokenizer = FakeEmbeddingTokenizer()
            self.max_seq_length = 512

        def get_embedding_dimension(self):
            return 384

    def no_cached_chat_tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("Qwen tokenizer is not cached")

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr(tokenizer, "load_tokenizer", no_cached_chat_tokenizer)

    embedder = SentenceTransformerEmbedder()
    assert embedder.chat_tokenizer is None
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True


def test_ollama_timeout_is_logged_and_manager_can_fallback(tmp_path):
    prompt = tmp_path / "p.txt"; prompt.write_text("prompt")
    workflow = tmp_path / "w.json"; workflow.write_text('{"version":"1"}')
    logged = []
    class Client:
        def chat(self, **kwargs):
            raise httpx.ReadTimeout("timeout")
    nlg = QwenNLG(system_prompt_path=prompt, workflow_path=workflow, client=Client(), logger=lambda **kw: logged.append(kw), token_counter=lambda _: 1)
    with pytest.raises(httpx.ReadTimeout):
        nlg.generate_reply("draft", user_message="x", intent="unknown")
    assert logged[-1]["extra"]["error_type"] == "OllamaTimeout"
    assert logged[-1]["extra"]["status"] == "failed"


def test_successful_request_preserves_chat_parameters(tmp_path):
    prompt = tmp_path / "p.txt"; prompt.write_text("prompt")
    workflow = tmp_path / "w.json"; workflow.write_text('{"version":"1"}')
    class Client:
        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"message":{"content":"draft"}}
    client = Client()
    nlg = QwenNLG(system_prompt_path=prompt, workflow_path=workflow, client=client, logger=None, token_counter=lambda _: 1)
    assert nlg.generate_reply("draft", user_message="x", intent="unknown") == "draft"
    assert client.kwargs["think"] is False and client.kwargs["stream"] is False
    assert client.kwargs["options"]["num_ctx"] == 2048


def test_full_history_prompt_overflow_logs_attempted_input_tokens(tmp_path):
    nlg = _build_prompt_nlg(tmp_path, lambda text: len(text.split()))
    nlg._system_message = lambda: "system"
    logged = []

    class Client:
        called = False

        def chat(self, **kwargs):
            self.called = True
            raise AssertionError("Ollama must not receive an over-budget full-history prompt")

    client = Client()
    nlg.client = client
    nlg.logger = lambda **event: logged.append(event)
    nlg.set_call_context(user="customer", session="A", turn=25, config="full")
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "word " * 40}
        for index in range(80)
    ]

    with pytest.raises(ValueError, match="above the safe input budget") as error:
        nlg.generate_reply(
            "draft", user_message="current", intent="unknown",
            conversation_history=history,
        )

    attempted_messages = error.value.prompt_messages
    attempted_tokens = nlg._count_tokens(
        "\n".join(message["content"] for message in attempted_messages)
    )
    assert client.called is False
    assert logged[0]["input_tokens"] == attempted_tokens
    assert logged[0]["input_tokens"] > 1856
    assert logged[0]["extra"]["status"] == "failed"


def test_customer_response_output_token_fallback_counts_generated_content(tmp_path):
    prompt = tmp_path / "p.txt"; prompt.write_text("prompt")
    workflow = tmp_path / "w.json"; workflow.write_text('{"version":"1"}')
    events = []

    class Client:
        def chat(self, **kwargs):
            return {"message": {"content": "one two three"}}

    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=Client(),
        logger=lambda **event: events.append(event),
        token_counter=lambda text: len(text.split()),
    )
    nlg.generate_reply("draft", user_message="question", intent="unknown")
    assert events[0]["output_tokens"] == 3
