import json

from src.dialogue import replay_session as replay_module
from src.dialogue.replay_session import _booking_write_validation, _load_session_turns


def test_session_a_loader_reads_exact_dataset_turns_without_retyping():
    dataset, turns = _load_session_turns("A")
    source_turns = [
        turn for turn in dataset["turns"]
        if turn["session"] == "A" and 1 <= turn["turn"] <= 25
    ]
    assert len(turns) == 25
    assert [turn["turn"] for turn in turns] == list(range(1, 26))
    assert [turn["text"] for turn in turns] == [turn["text"] for turn in source_turns]


def test_no_booking_attempt_is_reported_as_not_tested():
    assert _booking_write_validation([], [], []) == (False, None, None)


def test_booking_write_validation_requires_explicit_confirmation():
    event = {"request": {"explicit_confirmation": True}}
    record = {
        "turn": 4,
        "extra": {"tool_name": "book_slot"},
    }
    assert _booking_write_validation([event], [record], [3]) == (True, True, True)
    assert _booking_write_validation(
        [{"request": {"explicit_confirmation": False}}], [record], [3]
    ) == (True, False, True)
    assert _booking_write_validation([event], [record], []) == (True, True, False)


def test_task2c_snapshot_manifest_is_separate_from_task2b(tmp_path, monkeypatch):
    task2b_path = tmp_path / "task2b_session_A_run_manifest.json"
    original = {"runs": [{"run_id": "task2b_historical"}]}
    task2b_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path)

    replay_module._write_manifest_entry(
        {"run_id": "task2c_new", "purpose": "task2c_snapshot"},
        purpose="task2c_snapshot",
    )

    assert json.loads(task2b_path.read_text(encoding="utf-8")) == original
    task2c = json.loads((tmp_path / "task2c_session_A_run_manifest.json").read_text(encoding="utf-8"))
    assert task2c["runs"] == [{"run_id": "task2c_new", "purpose": "task2c_snapshot"}]
