import pytest

from src.dialogue.conversation_history import ConversationHistory
from src.dialogue.rolling_summary import RollingSummaryMemory


def add(history, n):
    history.record_exchange(
        {"role": "user", "content": f"u{n}"},
        {"role": "assistant", "content": f"a{n}"},
    )


def join_summary(previous, aged):
    prior = f"{previous}|" if previous else ""
    return prior + ",".join(exchange[0]["content"] for exchange in aged)


def test_empty_summary_and_no_update_until_third_exchange():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    calls = []
    assert memory.summary == ""
    assert memory.update_summary(lambda p, e: calls.append((p, e)) or "s") is False
    add(history, 1)
    assert memory.update_summary(lambda p, e: calls.append((p, e)) or "s") is False
    add(history, 2)
    assert memory.update_summary(lambda p, e: calls.append((p, e)) or "s") is False
    assert calls == [] and memory.last_summarized_exchange_count == 0


def test_rolling_summary_updates_newly_aged_exchanges_once():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    seen = []
    for n in range(1, 6):
        add(history, n)
        memory.update_summary(lambda p, e: seen.append((p, [x[0]["content"] for x in e])) or join_summary(p, e))
    assert seen == [("", ["u1"]), ("u1", ["u2"]), ("u1|u2", ["u3"])]
    assert memory.summary == "u1|u2|u3"
    assert memory.last_summarized_exchange_count == 3
    assert memory.summarized_exchange_count == 3
    assert [m["content"] for m in memory.get_context()["messages"]] == ["u4", "a4", "u5", "a5"]


def test_only_newly_aged_exchanges_and_previous_summary_are_passed():
    history = ConversationHistory()
    for n in range(1, 5):
        add(history, n)
    memory = RollingSummaryMemory(history)
    calls = []
    memory.update_summary(lambda previous, aged: calls.append((previous, aged)) or "first")
    add(history, 5)
    memory.update_summary(lambda previous, aged: calls.append((previous, aged)) or "second")
    assert calls[0][0] == ""
    assert [e[0]["content"] for e in calls[0][1]] == ["u1", "u2"]
    assert calls[1][0] == "first"
    assert [e[0]["content"] for e in calls[1][1]] == ["u3"]


def test_failed_summary_keeps_cursor_and_unsummarized_exchanges_in_context():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    for n in range(1, 5):
        add(history, n)
    with pytest.raises(RuntimeError, match="summary failed"):
        memory.update_summary(lambda previous, aged: (_ for _ in ()).throw(RuntimeError("summary failed")))
    assert memory.last_summarized_exchange_count == 0
    assert [m["content"] for m in memory.get_context()["messages"]] == [
        value for n in range(1, 5) for value in (f"u{n}", f"a{n}")
    ]


def test_failed_incremental_update_preserves_pending_exchanges_for_retry():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    for n in range(1, 5):
        add(history, n)

    successful_calls = []
    assert memory.update_summary(
        lambda previous, aged: successful_calls.append(
            (previous, [exchange[0]["content"] for exchange in aged])
        ) or join_summary(previous, aged)
    ) is True
    saved_summary = memory.summary
    saved_cursor = memory.last_summarized_exchange_count
    assert successful_calls == [("", ["u1", "u2"])]
    assert [message["content"] for message in memory.get_context()["messages"]] == [
        "u3", "a3", "u4", "a4"
    ]

    add(history, 5)
    with pytest.raises(RuntimeError, match="temporary summary failure"):
        memory.update_summary(
            lambda previous, aged: (_ for _ in ()).throw(
                RuntimeError("temporary summary failure")
            )
        )

    assert memory.summary == saved_summary
    assert memory.last_summarized_exchange_count == saved_cursor
    pending = [message["content"] for message in memory.get_context()["messages"]]
    assert pending == ["u3", "a3", "u4", "a4", "u5", "a5"]

    retry_calls = []
    assert memory.update_summary(
        lambda previous, aged: retry_calls.append(
            (previous, [exchange[0]["content"] for exchange in aged])
        ) or join_summary(previous, aged)
    ) is True
    assert retry_calls == [(saved_summary, ["u3"])]
    assert memory.last_summarized_exchange_count == 3
    assert [message["content"] for message in memory.get_context()["messages"]] == [
        "u4", "a4", "u5", "a5"
    ]


@pytest.mark.parametrize("result", ["", "  ", None, 42])
def test_invalid_summary_result_does_not_advance_cursor(result):
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    for n in range(3):
        add(history, n)
    with pytest.raises(ValueError, match="non-empty string"):
        memory.update_summary(lambda previous, aged: result)
    assert memory.summary == ""
    assert memory.last_summarized_exchange_count == 0


def test_cursor_prevents_duplicate_summary_calls():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    for n in range(3):
        add(history, n)
    calls = []
    assert memory.update_summary(lambda p, e: calls.append(e) or "summary") is True
    assert memory.update_summary(lambda p, e: calls.append(e) or "again") is False
    assert len(calls) == 1


def test_context_is_defensive_and_incoming_message_is_separate():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    add(history, 1)
    incoming = {"role": "user", "content": "incoming"}
    context = memory.get_context()
    assert incoming not in context["messages"]
    context["messages"][0]["content"] = "mutated"
    context["summary"] = "mutated summary"
    assert memory.get_context() == {"summary": "", "messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]}


def test_forget_redaction_applies_to_aged_transcript_and_existing_summary():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    history.record_exchange({"role": "user", "content": "leave side door"}, {"role": "assistant", "content": "noted"})
    add(history, 2)
    add(history, 3)
    memory.update_summary(join_summary)
    assert "side door" in memory.summary
    memory.add_redaction("side door")
    assert "side door" not in memory.summary
    for n in (4, 5, 6):
        add(history, n)
    supplied = []
    memory.update_summary(lambda p, e: supplied.append(e) or (p + " update"))
    assert all("side door" not in str(exchange) for exchange in supplied)


def test_component_does_not_use_tracker():
    history = ConversationHistory()
    memory = RollingSummaryMemory(history)
    add(history, 1)
    class TrackerBomb:
        def __getattr__(self, name):
            raise AssertionError("tracker must not be accessed")
    _tracker = TrackerBomb()
    assert memory.get_context()["summary"] == ""
    assert memory.update_summary(lambda p, e: "summary") is False
