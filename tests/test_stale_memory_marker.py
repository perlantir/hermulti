"""Tests for the stale-memory marker rendered by CompiledContext."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from agent.hipp0_memory_provider import (
    CompiledContext,
    Hipp0MemoryProvider,
    Hipp0UnavailableError,
)


def test_as_prompt_block_omits_marker_when_fresh() -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "y"}],
        stale_minutes=None,
    )
    out = ctx.as_prompt_block()
    assert "STALE MEMORY" not in out


def test_as_prompt_block_prepends_marker_when_stale() -> None:
    ctx = CompiledContext(
        decisions=[{"id": "x", "text": "y"}],
        stale_minutes=42,
    )
    out = ctx.as_prompt_block()
    first_line = out.splitlines()[0]
    assert first_line == "[STALE MEMORY: last successful compile 42m ago]"


@pytest.mark.asyncio
async def test_degraded_compile_emits_marker_when_never_succeeded(tmp_path) -> None:
    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        memory_md_path=tmp_path / "MEMORY.md",
    )
    provider._post_json = AsyncMock(side_effect=Hipp0UnavailableError("boom"))  # type: ignore[assignment]
    ctx = await provider.compile("task")
    assert ctx.degraded is True
    assert ctx.stale_minutes is not None
    block = ctx.as_prompt_block()
    assert "[STALE MEMORY:" in block
    await provider.aclose()


@pytest.mark.asyncio
async def test_breaker_open_short_circuit_emits_marker(tmp_path) -> None:
    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        memory_md_path=tmp_path / "MEMORY.md",
    )
    # Simulate a prior successful compile 45m ago, then trip breaker.
    provider._last_compile_success_ts = time.time() - 45 * 60
    for _ in range(3):
        provider._compile_breaker.record_failure()

    ctx = await provider.compile("task")
    assert ctx.degraded is True
    assert ctx.stale_minutes is not None
    assert ctx.stale_minutes >= 45
    assert "[STALE MEMORY:" in ctx.as_prompt_block()
    await provider.aclose()
