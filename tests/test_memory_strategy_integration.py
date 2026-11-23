import pytest

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.state_tracker import DialogueStateTracker


class Tools:
    def __init__(self): self.calls = []
    def set_call_context(self, **context): self.context = context


class CapturingNLG:
    def __init__(self, reply="Qwen response"):
        self.reply = reply
        self.contexts = []
    def set_call_context(self, **context): self.call_context = context
    def generate_reply(self, draft, **kwargs):
        self.contexts.append(kwargs)
        return self.reply


def manager(strategy="full", *, nlg=None):
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    tools = Tools()
    return DialogueManager(
        tracker=tracker, tools=tools, user_id="CUST-MAYA-1042",
        memory_strategy=strategy, nlg=nlg, logger=None,
        token_counter=lambda text: len(text.split()),
    ), tracker, tools


def test_full_history_is_passed_to_qwen_and_current_message_is_once():
    nlg = CapturingNLG()
    dm, _, _ = manager(nlg=nlg)
    dm.handle_turn("Hi there!", turn=1)
    dm.handle_turn("Thanks again", turn=2)
    # Greeting/small-talk both invoke NLG. Second prompt has only completed exchange 1.
    ctx = nlg.contexts[-1]
    assert ctx["conversation_history"] == [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Qwen response"},
    ]
    assert ctx["user_message"] == "Thanks again"
    assert sum(m["content"] == "Thanks again" for m in ctx["conversation_history"]) == 0


def test_trim_passes_exactly_last_three_complete_exchanges():
    nlg = CapturingNLG()
    dm, _, _ = manager("trim", nlg=nlg)
    for i in range(1, 6):
        dm.handle_turn(f"Hi {i}", turn=i)
    selected = nlg.contexts[-1]["conversation_history"]
    assert [m["content"] for m in selected] == [
        value for i in range(2, 5) for value in (f"Hi {i}", "Qwen response")
    ]
    assert nlg.contexts[-1]["user_message"] == "Hi 5"


def test_early_return_and_deterministic_fallback_are_recorded_once():
    dm, _, _ = manager()
    reply = dm.handle_turn("Hi there!", turn=1)["reply"]
    assert dm.conversation_history.completed_exchange_count == 1
    assert dm.conversation_history.get_history("full") == [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": reply},
    ]
    dm.handle_turn("unknown request", turn=2)
    assert dm.conversation_history.completed_exchange_count == 2


def test_failed_incomplete_turn_is_not_recorded(monkeypatch):
    dm, _, _ = manager()
    import src.dialogue.dialogue_manager as module
    monkeypatch.setattr(module, "analyze_utterance", lambda _: (_ for _ in ()).throw(RuntimeError("abort")))
    with pytest.raises(RuntimeError, match="abort"):
        dm.handle_turn("broken", turn=1)
    assert dm.conversation_history.completed_exchange_count == 0


def test_tracker_input_and_state_are_strategy_invariant():
    managers = [manager(strategy)[0] for strategy in ("full", "trim")]
    inputs = ["Hi", "My name is Maya", "I prefer email for updates"]
    states = []
    for dm in managers:
        for turn, text in enumerate(inputs, 1):
            dm.handle_turn(text, turn=turn)
        states.append(dm.tracker.snapshot())
    assert states[0] == states[1]
    assert [dm.conversation_history.completed_exchange_count for dm in managers] == [3, 3]


def test_tracker_state_and_tool_decisions_match_with_controlled_order_tool():
    outcomes = []
    for strategy in ("full", "trim"):
        dm, tracker, tools = manager(strategy)
        observed_tool_inputs = []

        def get_order(order_id, customer_id=None):
            observed_tool_inputs.append((order_id, customer_id))
            return {"order_id": order_id, "order_status": "In transit"}

        tools.get_order = get_order
        result = dm.handle_turn("Check the status of order ORD-74021.", turn=1)
        outcomes.append((tracker.snapshot(), observed_tool_inputs, result["tool_calls"], result["reply"]))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] == [("ORD-74021", "CUST-MAYA-1042")]


def test_full_history_overflow_uses_fallback_without_trimming_history():
    from src.dialogue.qwen_nlg import QwenNLG

    class NeverCalledClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise AssertionError("Ollama must not be called for an over-budget prompt")

    client = NeverCalledClient()
    nlg = QwenNLG(client=client, logger=None, token_counter=len)
    nlg._system_message = lambda: "system"
    dm, _, _ = manager("full", nlg=nlg)
    for i in range(40):
        dm.conversation_history.record_exchange(
            {"role": "user", "content": f"user-{i}-" + "u" * 180},
            {"role": "assistant", "content": f"assistant-{i}-" + "a" * 180},
        )

    result = dm.handle_turn("Hi there!", turn=41)

    assert result["nlg_status"].startswith("fallback:ValueError")
    assert client.calls == 0
    assert dm.conversation_history.completed_exchange_count == 41
    history = dm.conversation_history.get_history("full")
    assert len(history) == 82
    assert history[0]["content"] == "user-0-" + "u" * 180
    assert history[-2]["content"] == "Hi there!"


def test_full_history_is_never_trimmed_by_history_selection():
    dm, _, _ = manager("full")
    for i in range(6):
        dm.conversation_history.record_exchange(
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        )
    assert len(dm.conversation_history.get_history("full")) == 12


def test_default_strategy_remains_full_and_invalid_strategy_rejected():
    tracker = DialogueStateTracker(session_id="A", user_id="CUST-MAYA-1042")
    dm = DialogueManager(tracker=tracker, tools=Tools(), logger=None)
    assert dm.memory_strategy == "full"
    summary, _, _ = manager("summary")
    assert summary.memory_strategy == "summary"
    assert summary.rolling_summary_memory is not None
    with pytest.raises(ValueError, match="memory_strategy"):
        manager("invalid")
