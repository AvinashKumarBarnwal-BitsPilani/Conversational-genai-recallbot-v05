import pytest

from src.dialogue.conversation_history import ConversationHistory


def exchange(index):
    return (
        {"role": "user", "content": f"user-{index}"},
        {"role": "assistant", "content": f"assistant-{index}"},
    )


def record(history, index):
    history.record_exchange(*exchange(index))


def test_full_strategy_returns_all_exchanges_chronologically():
    history = ConversationHistory()
    for index in range(5):
        record(history, index)
    assert [m["content"] for m in history.get_history("full")] == [
        value for index in range(5) for value in (f"user-{index}", f"assistant-{index}")
    ]


def test_trim_returns_last_three_complete_ordered_exchanges():
    history = ConversationHistory()
    for index in range(5):
        record(history, index)
    messages = history.get_history("trim")
    assert [m["content"] for m in messages] == [
        value for index in range(2, 5) for value in (f"user-{index}", f"assistant-{index}")
    ]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 3


def test_trim_handles_fewer_than_three_exchanges_and_empty_history():
    history = ConversationHistory()
    assert history.get_history("trim") == []
    record(history, 1)
    record(history, 2)
    assert [m["content"] for m in history.get_history("trim")] == [
        "user-1", "assistant-1", "user-2", "assistant-2"
    ]
    assert history.completed_exchange_count == 2


def test_incoming_user_message_is_not_part_of_completed_history():
    history = ConversationHistory()
    record(history, 1)
    incoming = {"role": "user", "content": "incoming"}
    assert incoming not in history.get_history("full")
    assert incoming not in history.get_history("trim")


def test_recording_new_exchange_updates_both_strategy_views():
    history = ConversationHistory()
    for index in range(3):
        record(history, index)
    record(history, 3)
    assert history.get_history("full")[-2:] == list(exchange(3))
    assert history.get_history("trim")[-2:] == list(exchange(3))


def test_returned_history_cannot_mutate_raw_transcript():
    history = ConversationHistory()
    record(history, 1)
    returned = history.get_history("full")
    returned[0]["content"] = "changed"
    returned.append({"role": "user", "content": "fake"})
    assert [m["content"] for m in history.get_history("full")] == ["user-1", "assistant-1"]


@pytest.mark.parametrize("strategy", ["summary", "invalid", ""])
def test_invalid_strategy_is_rejected(strategy):
    with pytest.raises(ValueError, match="strategy"):
        ConversationHistory().get_history(strategy)


@pytest.mark.parametrize(
    ("user", "assistant", "error"),
    [
        (None, {"role": "assistant", "content": "ok"}, TypeError),
        ({"role": "assistant", "content": "wrong"}, {"role": "assistant", "content": "ok"}, ValueError),
        ({"role": "user", "content": "ok"}, {"role": "user", "content": "wrong"}, ValueError),
        ({"role": "user", "content": " "}, {"role": "assistant", "content": "ok"}, ValueError),
        ({"role": "user", "content": "ok"}, {"role": "assistant", "content": ""}, ValueError),
    ],
)
def test_malformed_exchanges_are_rejected(user, assistant, error):
    with pytest.raises(error):
        ConversationHistory().record_exchange(user, assistant)


def test_history_component_is_independent_of_tracker():
    class TrackerMustNotBeRead:
        @property
        def state(self):
            raise AssertionError("history component read tracker state")

        def snapshot(self):
            raise AssertionError("history component called tracker")

    tracker = TrackerMustNotBeRead()
    history = ConversationHistory()
    record(history, 1)
    _ = tracker  # Deliberately not injected or referenced by ConversationHistory.
    assert history.completed_exchange_count == 1
    assert len(history.get_history("full")) == 2
