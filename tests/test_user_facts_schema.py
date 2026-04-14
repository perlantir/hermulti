"""Strict schema validation for user_facts rendering."""

from __future__ import annotations

import logging

from agent.hipp0_memory_provider import CompiledContext


def test_valid_key_renders() -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "t"}],
        user_facts=[{"key": "name", "value": "Bob"}],
    )
    out = ctx.as_prompt_block()
    assert "- **name**: Bob" in out


def test_missing_key_drops_entry(caplog) -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "t"}],
        user_facts=[{"value": "orphan"}],
    )
    with caplog.at_level(logging.WARNING):
        out = ctx.as_prompt_block()
    assert "orphan" not in out
    assert "User Facts" not in out
    assert any("user_fact missing 'key'" in r.getMessage() for r in caplog.records)


def test_legacy_fact_key_no_longer_accepted(caplog) -> None:
    """The old `fact_key` fallback must be gone — such entries drop."""
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "t"}],
        user_facts=[{"fact_key": "legacy", "fact_value": "v"}],
    )
    with caplog.at_level(logging.WARNING):
        out = ctx.as_prompt_block()
    assert "legacy" not in out
    assert "User Facts" not in out
    assert any("user_fact missing 'key'" in r.getMessage() for r in caplog.records)


def test_mixed_valid_and_invalid_filters_invalid(caplog) -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "t"}],
        user_facts=[
            {"key": "name", "value": "Bob"},
            {"fact_key": "legacy", "fact_value": "v"},
            {"key": "", "value": "empty-key"},
            {"key": "city", "value": "NYC"},
        ],
    )
    with caplog.at_level(logging.WARNING):
        out = ctx.as_prompt_block()
    assert "- **name**: Bob" in out
    assert "- **city**: NYC" in out
    assert "legacy" not in out
    assert "empty-key" not in out
    # Header count reflects only the rendered entries.
    assert "## User Facts (2)" in out


def test_non_string_key_drops_entry(caplog) -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "t"}],
        user_facts=[{"key": 123, "value": "int-key"}],
    )
    with caplog.at_level(logging.WARNING):
        out = ctx.as_prompt_block()
    assert "int-key" not in out
