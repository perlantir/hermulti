"""Unit tests for agent.outcome_signals.infer_outcome_from_turn."""

from agent.outcome_signals import infer_outcome_from_turn


def test_positive_thanks():
    assert infer_outcome_from_turn("thanks, that worked!") == "positive"


def test_positive_perfect():
    assert infer_outcome_from_turn("Perfect, exactly what I needed") == "positive"


def test_negative_wrong():
    assert infer_outcome_from_turn("no, that's wrong") == "negative"


def test_negative_undo():
    assert infer_outcome_from_turn("undo that change") == "negative"


def test_neutral_unknown():
    assert infer_outcome_from_turn("can you also add logging?") is None


def test_empty_returns_none():
    assert infer_outcome_from_turn("") is None
    assert infer_outcome_from_turn(None) is None


def test_negative_wins_over_positive():
    # Mixed: negative takes precedence.
    assert infer_outcome_from_turn("thanks but that's wrong") == "negative"


def test_case_insensitive():
    assert infer_outcome_from_turn("THANKS!!!") == "positive"
    assert infer_outcome_from_turn("WRONG answer") == "negative"
