import json

import pytest

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.qwen_nlg import DECODING_PRESETS, QwenNLG
from src.dialogue.rolling_summary import RollingSummaryMemory
from src.dialogue.state_tracker import DialogueStateTracker


class Tools:
    def set_call_context(self, **context):
        self.context = context


class CapturingNLG:
    def __init__(self, *, summary_error=None, summary_result=None):
        self.summary_calls = []
        self.reply_calls = []
        self.summary_error = summary_error
        self.summary_result = summary_result

    def set_call_context(self, **context):
        self.context = context

    def generate_summary(self, previous, aged):
        self.summary_calls.append((previous, aged, self.context.copy()))
        if self.summary_error:
            raise self.summary_error
        if self.summary_result is not None:
            return self.summary_result
        details = "; ".join(exchange[0]["content"] for exchange in aged)
        return f"{previous}; {details}".strip("; ")

    def generate_reply(self, draft, **kwargs):
        self.reply_calls.append((draft, kwargs))
        return f"reply {len(self.reply_calls)}"


def make_manager(nlg, strategy="summary"):
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    manager = DialogueManager(
        tracker=tracker,
        tools=Tools(),
        user_id="CUST-MAYA-1042",
        memory_strategy=strategy,
        nlg=nlg,
        logger=None,
        token_counter=lambda text: len(text.split()),
    )
    return manager, tracker


def test_summary_strategy_initializes_and_summarizes_only_newly_aged_exchanges():
    nlg = CapturingNLG()
    manager, _ = make_manager(nlg)
    assert isinstance(manager.rolling_summary_memory, RollingSummaryMemory)

    for turn in range(1, 6):
        manager.handle_turn(f"Hi {turn}", turn=turn)

    assert [call[2]["turn"] for call in nlg.summary_calls] == [4, 5]
    assert [message[0]["content"] for message in nlg.summary_calls[0][1]] == ["Hi 1"]
    assert nlg.summary_calls[0][0] == ""
    assert [message[0]["content"] for message in nlg.summary_calls[1][1]] == ["Hi 2"]
    assert "Hi 1" in nlg.summary_calls[1][0]
    assert manager.rolling_summary_memory.last_summarized_exchange_count == 2

    summary, context = nlg.reply_calls[-1][1]["conversation_summary"], nlg.reply_calls[-1][1]["conversation_history"]
    assert "Hi 1" in summary and "Hi 2" in summary
    assert [message["content"] for message in context] == ["Hi 3", "reply 3", "Hi 4", "reply 4"]
    assert "Hi 5" not in [message["content"] for message in context]
    assert nlg.reply_calls[-1][1]["user_message"] == "Hi 5"


def test_summary_failure_retains_unsummarized_context_and_cursor_for_retry():
    nlg = CapturingNLG(summary_error=RuntimeError("summary unavailable"))
    manager, _ = make_manager(nlg)
    for turn in range(1, 5):
        manager.handle_turn(f"Hi {turn}", turn=turn)

    assert manager.rolling_summary_memory.last_summarized_exchange_count == 0
    supplied = nlg.reply_calls[-1][1]
    assert supplied["conversation_summary"] == ""
    assert [m["content"] for m in supplied["conversation_history"]] == [
        value for turn in range(1, 4) for value in (f"Hi {turn}", f"reply {turn}")
    ]

    nlg.summary_error = None
    manager.handle_turn("Hi 5", turn=5)
    assert [message[0]["content"] for message in nlg.summary_calls[-1][1]] == ["Hi 1", "Hi 2"]
    assert manager.rolling_summary_memory.last_summarized_exchange_count == 2


def test_summary_redaction_filters_summary_transcript_and_current_forget_request():
    nlg = CapturingNLG()
    manager, _ = make_manager(nlg)
    manager.handle_turn(
        "If I am not home, leave small parcels at the side door when the carrier permits it.",
        turn=1,
    )
    manager.handle_turn("Hi 2", turn=2)
    manager.handle_turn("Hi 3", turn=3)
    manager.handle_turn(
        "Please forget my saved delivery note about leaving parcels at the side door.",
        turn=4,
    )

    previous, newly_aged, _ = nlg.summary_calls[-1]
    assert "side door" not in previous
    assert "side door" not in json.dumps(newly_aged).casefold()
    summary, context = nlg.reply_calls[-1][1]["conversation_summary"], nlg.reply_calls[-1][1]["conversation_history"]
    assert "side door" not in summary.casefold()
    assert "side door" not in json.dumps(context).casefold()
    assert "side door" not in nlg.reply_calls[-1][1]["user_message"].casefold()
    assert "side door" not in manager.rolling_summary_memory.get_context()["summary"].casefold()


def test_tracker_and_tool_decisions_match_across_all_three_strategies():
    results = []
    for strategy in ("full", "trim", "summary"):
        nlg = CapturingNLG()
        tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
        tools = Tools()
        observed = []

        def get_order(order_id, customer_id=None):
            observed.append((order_id, customer_id))
            return {"order_id": order_id, "order_status": "In transit"}

        tools.get_order = get_order
        manager = DialogueManager(
            tracker=tracker,
            tools=tools,
            user_id="CUST-MAYA-1042",
            memory_strategy=strategy,
            nlg=nlg,
            logger=None,
        )
        outcomes = [
            manager.handle_turn(text, turn=turn)
            for turn, text in enumerate(
                ("Check the status of order ORD-74021.", "Hi 2", "Thanks again", "Hi 4"),
                start=1,
            )
        ]
        results.append((tracker.snapshot(), observed, [item["tool_calls"] for item in outcomes]))

    assert results[0] == results[1] == results[2]
    assert results[0][1] == [("ORD-74021", "CUST-MAYA-1042")]


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_summary_output_is_logged_as_failure_and_cursor_does_not_advance(tmp_path, content):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    events = []

    class Client:
        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"message": {"content": content}, "prompt_eval_count": 25, "eval_count": 0}

    client = Client()
    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=client,
        logger=lambda **event: events.append(event),
        token_counter=lambda text: len(text.split()),
    )
    nlg.set_call_context(user="customer", session="A", turn=3, config="summary")
    with pytest.raises(RuntimeError, match="empty summary"):
        nlg.generate_summary("prior", [({"role": "user", "content": "aged"}, {"role": "assistant", "content": "reply"})])

    assert client.kwargs["model"] == "qwen3:1.7b"
    assert client.kwargs["options"] == DECODING_PRESETS["task"]
    assert client.kwargs["think"] is False and client.kwargs["stream"] is False
    assert events[0]["config"] == "summary"
    assert events[0]["call_type"] == "summary"
    assert events[0]["turn"] == 3
    assert events[0]["input_tokens"] == 25
    assert events[0]["output_tokens"] == 0
    assert events[0]["extra"]["status"] == "failed"
    assert events[0]["extra"]["purpose"] == "summary_generation"


def test_summary_and_response_are_separate_logged_calls_and_tokens_add_once(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    events = []

    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            count = len(self.calls)
            return {
                "message": {"content": "summary text" if count == 1 else "customer reply"},
                "prompt_eval_count": 31 if count == 1 else 71,
                "eval_count": 9 if count == 1 else 14,
            }

    client = Client()
    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=client,
        logger=lambda **event: events.append(event),
        token_counter=lambda text: len(text.split()),
    )
    nlg.set_call_context(user="customer", session="A", turn=8, config="summary")
    result = nlg.generate_summary(
        "old summary",
        [({"role": "user", "content": "aged only"}, {"role": "assistant", "content": "answer"})],
    )
    nlg.generate_reply(
        "draft",
        user_message="current question",
        intent="unknown",
        conversation_summary=result,
        conversation_history=[{"role": "user", "content": "recent user"}],
    )

    assert len(client.calls) == 2
    summary_input = json.loads(client.calls[0]["messages"][1]["content"])
    assert summary_input["previous_summary"] == "old summary"
    assert "aged only" in json.dumps(summary_input)
    chat_payload = json.loads(client.calls[1]["messages"][-1]["content"].split("\n\n", 1)[1])
    assert chat_payload["conversation_summary"] == result
    assert chat_payload["current_user_message"] == "current question"
    assert events[0]["config"] == events[1]["config"] == "summary"
    assert [event["call_type"] for event in events] == ["summary", "chat"]
    assert [event["extra"]["purpose"] for event in events] == ["summary_generation", "qwen_nlg_response"]
    assert sum(event["input_tokens"] + event["output_tokens"] for event in events) == 125


def test_oversized_summary_prompt_fails_and_is_logged_without_calling_ollama(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    events = []

    class Client:
        called = False

        def chat(self, **kwargs):
            self.called = True

    client = Client()
    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=client,
        logger=lambda **event: events.append(event),
        token_counter=len,
    )
    nlg.set_call_context(user="customer", session="A", turn=4, config="summary")
    with pytest.raises(ValueError, match="Summary prompt needs"):
        nlg.generate_summary("x" * 3000, [])

    assert client.called is False
    assert events[0]["call_type"] == "summary"
    assert events[0]["extra"]["status"] == "failed"
    assert events[0]["input_tokens"] > 2048 - 160 - 32


def test_failed_summary_request_logs_error_and_estimated_input_tokens(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    events = []

    class Client:
        def chat(self, **kwargs):
            raise ConnectionError("Ollama unavailable")

    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=Client(),
        logger=lambda **event: events.append(event),
        token_counter=lambda text: len(text.split()),
    )
    nlg.set_call_context(user="customer", session="A", turn=9, config="summary")
    with pytest.raises(ConnectionError, match="Ollama unavailable"):
        nlg.generate_summary("prior", [])

    assert len(events) == 1
    assert events[0]["config"] == "summary"
    assert events[0]["call_type"] == "summary"
    assert events[0]["model"] == "qwen3:1.7b"
    assert events[0]["input_tokens"] > 0
    assert events[0]["output_tokens"] == 0
    assert events[0]["extra"]["status"] == "failed"
    assert events[0]["extra"]["error_type"] == "ConnectionError"


def test_summary_output_token_fallback_counts_generated_content(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"version":"1"}', encoding="utf-8")
    events = []

    class Client:
        def chat(self, **kwargs):
            return {"message": {"content": "updated useful summary"}}

    nlg = QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=Client(),
        logger=lambda **event: events.append(event),
        token_counter=lambda text: len(text.split()),
    )
    nlg.set_call_context(user="customer", session="A", turn=4, config="summary")
    assert nlg.generate_summary("", []) == "updated useful summary"
    assert events[0]["output_tokens"] == 3


@pytest.mark.parametrize("strategy", ["full", "trim"])
def test_existing_strategies_still_pass_their_selected_history(tmp_path, strategy):
    from src.dialogue.conversation_history import ConversationHistory

    nlg = CapturingNLG()
    manager, _ = make_manager(nlg, strategy)
    for turn in range(1, 6):
        manager.handle_turn(f"Hi {turn}", turn=turn)
    selected = nlg.reply_calls[-1][1]["conversation_history"]
    expected_turns = range(1, 5) if strategy == "full" else range(2, 5)
    assert [m["content"] for m in selected] == [
        value for turn in expected_turns for value in (f"Hi {turn}", f"reply {turn}")
    ]
    assert nlg.reply_calls[-1][1]["conversation_summary"] == ""
    assert isinstance(manager.conversation_history, ConversationHistory)
