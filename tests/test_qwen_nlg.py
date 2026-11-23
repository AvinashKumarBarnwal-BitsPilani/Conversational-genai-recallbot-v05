from __future__ import annotations

from pathlib import Path

import pytest

from src.dialogue.qwen_nlg import QwenNLG
from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker

ROOT = Path(__file__).resolve().parents[1]


class FakeOllamaClient:
    def __init__(self, content="I can help with that.", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {
            "message": {"content": self.content},
            "prompt_eval_count": 77,
            "eval_count": 18,
        }


def make_nlg(tmp_path, *, content="I can help with that.", logger=None, error=None, token_counter=None):
    client = FakeOllamaClient(content=content, error=error)
    events = []
    nlg = QwenNLG(
        model="qwen3:1.7b",
        system_prompt_path=ROOT / "skills" / "system_prompt_v1.txt",
        workflow_path=ROOT / "skills" / "return_pickup_workflow_v1.json",
        client=client,
        logger=logger if logger is not None else lambda **event: events.append(event),
        token_counter=token_counter or (lambda text: max(1, len(text.split())) if text else 0),
        context_window=2048,
    )
    nlg.set_call_context(user="CUST-MAYA-1042", session="A", turn=7, config="full")
    return nlg, client, events


def test_qwen_nlg_loads_versioned_procedural_files_at_runtime(tmp_path):
    nlg, _, _ = make_nlg(tmp_path)
    assert "You are RecallBot" in nlg.system_prompt
    assert nlg.workflow["skill"] == "return_pickup_workflow"
    assert nlg.workflow_version == "1.0"
    assert nlg.prompt_version == "system_prompt_v1"


def test_task_turn_uses_deterministic_decoding_and_logs_actual_usage(tmp_path):
    nlg, client, events = make_nlg(tmp_path, content="Please share the order ID.")
    reply = nlg.generate_reply(
        "Please share the order ID.",
        user_message="I want to return an item.",
        intent="start_return",
        state={"active_task": "return_pickup", "task_status": "collecting_slots"},
        turn_type="task",
    )
    assert reply == "Please share the order ID."
    options = client.calls[0]["options"]
    assert options["temperature"] == 0.0
    assert options["top_p"] == 0.9
    assert options["num_ctx"] == 2048
    assert events[0]["call_type"] == "chat"
    assert events[0]["model"] == "qwen3:1.7b"
    assert events[0]["input_tokens"] == 77
    assert events[0]["output_tokens"] == 18
    assert events[0]["extra"]["purpose"] == "qwen_nlg_response"
    assert events[0]["extra"]["turn_type"] == "task"


def test_open_turn_uses_open_decoding_preset(tmp_path):
    nlg, client, _ = make_nlg(tmp_path, content="Hello! How can I help?")
    nlg.generate_reply(
        "Hello! I'm RecallBot.",
        user_message="Hi",
        intent="greeting",
        turn_type="open",
    )
    options = client.calls[0]["options"]
    assert options["temperature"] == 0.3
    assert options["top_p"] == 0.9
    assert options["num_ctx"] == 2048


def test_nlg_preserves_critical_order_and_rma_identifiers(tmp_path):
    nlg, _, _ = make_nlg(tmp_path, content="Your pickup is confirmed.")
    draft = "Your pickup has been booked. Reference: RMA-20261122-ABC12345. Status: confirmed."
    reply = nlg.generate_reply(
        draft,
        user_message="Did you book it?",
        intent="confirm_action",
        turn_type="task",
    )
    # The model omitted the protected RMA identifier, so the safe draft wins.
    assert reply == draft


def test_nlg_appends_source_citation_from_retrieval_metadata(tmp_path):
    nlg, _, _ = make_nlg(tmp_path, content="Eligible products can be returned within the stated window.")
    source = "KB-002 / KB-002-C128-0001 (version 2.0)"
    reply = nlg.generate_reply(
        "Use the current policy evidence to answer concisely.",
        user_message="What is the return policy?",
        intent="knowledge_base_query",
        evidence=[{"doc_id": "KB-002", "chunk_id": "KB-002-C128-0001", "version": "2.0", "content": "Eligible products can be returned within 30 days."}],
        sources=[source],
        turn_type="task",
    )
    assert source in reply


def test_nlg_strips_qwen_thinking_block(tmp_path):
    nlg, _, _ = make_nlg(tmp_path, content="<think>internal reasoning</think>\nHello, Maya.")
    reply = nlg.generate_reply("Hello, Maya.", user_message="Hi", intent="greeting", turn_type="open")
    assert reply == "Hello, Maya."
    assert "internal reasoning" not in reply


def test_failed_ollama_call_is_logged_and_raises_for_manager_fallback(tmp_path):
    nlg, _, events = make_nlg(tmp_path, error=ConnectionError("Ollama unavailable"))
    with pytest.raises(ConnectionError):
        nlg.generate_reply("Please try again.", user_message="hello", intent="unknown")
    assert len(events) == 1
    assert events[0]["call_type"] == "chat"
    assert events[0]["extra"]["status"] == "failed"
    assert events[0]["extra"]["error_type"] == "ConnectionError"


def test_dialogue_manager_uses_injected_qwen_nlg_and_keeps_deterministic_fallback(tmp_path):
    class FakeTools:
        def get_order(self, order_id, customer_id=None):
            raise LookupError("not used")

        def search_kb(self, query, top_k=3):
            return []

        def check_pickup_slots(self, pickup_date, requested_window, address_summary):
            return []

        def book_slot(self, arguments):
            raise AssertionError("not used")

    class FakeNLG:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail
            self.calls = []

        def set_call_context(self, **context):
            self.context = context

        def generate_reply(self, draft, **kwargs):
            self.calls.append((draft, kwargs))
            if self.should_fail:
                raise ConnectionError("offline")
            return "Hello from Qwen."

    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    nlg = FakeNLG()
    manager = DialogueManager(tracker=tracker, tools=FakeTools(), user_id="CUST-MAYA-1042", logger=None, nlg=nlg)
    result = manager.handle_turn("Hi", turn=1)
    assert result["reply"] == "Hello from Qwen."
    assert result["nlg_status"] == "qwen"
    assert nlg.context["turn"] == 1
    assert nlg.calls[0][1]["turn_type"] == "open"

    failing_nlg = FakeNLG(should_fail=True)
    tracker2 = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager2 = DialogueManager(tracker=tracker2, tools=FakeTools(), user_id="CUST-MAYA-1042", logger=None, nlg=failing_nlg)
    fallback = manager2.handle_turn("Hi", turn=1)
    assert "Hello" in fallback["reply"]
    assert fallback["nlg_status"] == "fallback:ConnectionError"
