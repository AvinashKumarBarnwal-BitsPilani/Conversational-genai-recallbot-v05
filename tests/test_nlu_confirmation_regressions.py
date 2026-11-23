from src.dialogue.nlu import analyze_utterance


def test_confirmation_with_ascii_double_quotes_is_recognised():
    result = analyze_utterance('"Yes, I confirm the pickup booking."')
    assert result["intent"] == "confirm_action"


def test_confirmation_with_curly_quotes_is_recognised():
    result = analyze_utterance("“Yes, I confirm the pickup booking.”")
    assert result["intent"] == "confirm_action"


def test_exact_displayed_time_range_is_extracted():
    result = analyze_utterance("18:00–20:00")
    assert result["intent"] == "provide_pickup_window"
    assert result["entities"]["time_window"]["start"] == "18:00"
    assert result["entities"]["time_window"]["end"] == "20:00"
