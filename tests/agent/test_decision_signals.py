import pytest
from agent.outcome_signals import extract_decision_signals, DecisionSignal


def test_extracts_explicit_decision():
    text = "I'll use PostgreSQL for the database because it supports JSONB and we need complex queries."
    signals = extract_decision_signals(text)
    assert len(signals) >= 1
    assert any("postgresql" in s.title.lower() or "database" in s.title.lower() for s in signals)


def test_extracts_rejection():
    text = "Rejected MongoDB because we need ACID transactions for the payment flow."
    signals = extract_decision_signals(text)
    assert len(signals) >= 1
    assert signals[0].confidence in ("high", "medium", "low")


def test_no_false_positives_on_plain_text():
    text = "The weather is nice today. Here is a summary of the results."
    signals = extract_decision_signals(text)
    assert len(signals) == 0


def test_caps_at_five_signals():
    text = (
        "I'll use Redis. We decided on Python. Going with FastAPI. "
        "Choosing PostgreSQL. Opted for Docker. Selected Nginx."
    )
    signals = extract_decision_signals(text)
    assert len(signals) <= 5


def test_high_confidence_detection():
    text = "We definitely must use TLS everywhere - this is absolutely required."
    signals = extract_decision_signals(text)
    assert any(s.confidence == "high" for s in signals)
