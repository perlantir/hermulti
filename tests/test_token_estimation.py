"""Centralized token-estimation helper sanity checks.

Phase 9 of the audit plan routed all `len(x) // 4` token sites through
`agent.model_metadata.estimate_{tokens,messages_tokens}_rough`.  These
tests pin the helper's behavior so drift in one call-site can't silently
distort capacity/compression math.
"""

from agent.model_metadata import (
    estimate_tokens_rough,
    estimate_messages_tokens_rough,
)


def test_estimate_tokens_rough_empty_and_none():
    assert estimate_tokens_rough("") == 0
    assert estimate_tokens_rough(None) == 0


def test_estimate_tokens_rough_in_sensible_range():
    # "hello world " * 100 == 1200 chars -> ~300 rough tokens
    text = "hello world " * 100
    tokens = estimate_tokens_rough(text)
    # Rough 4-char/token estimate: allow generous band around ~300.
    assert 200 <= tokens <= 400


def test_estimate_messages_tokens_rough_empty():
    assert estimate_messages_tokens_rough([]) == 0


def test_estimate_messages_tokens_rough_fixture():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I'm doing well, thanks for asking."},
    ]
    tokens = estimate_messages_tokens_rough(messages)
    # Combined str(msg) length is ~250 chars -> ~60 rough tokens.
    # Assert a sensible range rather than an exact value.
    assert 30 <= tokens <= 150


def test_estimate_messages_tokens_rough_is_monotonic():
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "x" * 4000}]
    assert estimate_messages_tokens_rough(big) > estimate_messages_tokens_rough(small)


def test_estimate_messages_tokens_rough_handles_tool_calls():
    """Tool-call messages with content=None still contribute tokens."""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "1", "function": {"name": "terminal", "arguments": "{}"}}
        ],
    }
    assert estimate_messages_tokens_rough([msg]) > 0
