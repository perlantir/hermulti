"""Unit tests for the compile circuit breaker on Hipp0MemoryProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.hipp0_memory_provider import (
    Hipp0MemoryProvider,
    Hipp0UnavailableError,
    _CompileCircuitBreaker,
)


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_closed_to_open_trips_after_three_failures_within_window() -> None:
    clock = _FakeClock()
    cb = _CompileCircuitBreaker(clock=clock)

    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "CLOSED"
    clock.advance(10)
    cb.record_failure()
    assert cb.state == "CLOSED"
    clock.advance(10)
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.allow() is False


def test_failures_outside_window_do_not_trip() -> None:
    clock = _FakeClock()
    cb = _CompileCircuitBreaker(clock=clock)

    cb.record_failure()
    clock.advance(30)
    cb.record_failure()
    clock.advance(61)  # first failure now outside 60s window
    cb.record_failure()
    # Only 2 failures inside the window; breaker stays closed.
    assert cb.state == "CLOSED"


def test_open_transitions_to_half_open_after_cooldown() -> None:
    clock = _FakeClock()
    cb = _CompileCircuitBreaker(clock=clock)

    for _ in range(3):
        cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.allow() is False

    clock.advance(119)
    assert cb.state == "OPEN"
    clock.advance(2)
    assert cb.state == "HALF_OPEN"
    assert cb.allow() is True


def test_half_open_success_closes_breaker() -> None:
    clock = _FakeClock()
    cb = _CompileCircuitBreaker(clock=clock)
    for _ in range(3):
        cb.record_failure()
    clock.advance(121)
    assert cb.state == "HALF_OPEN"
    cb.record_success()
    assert cb.state == "CLOSED"


def test_half_open_failure_reopens_breaker() -> None:
    clock = _FakeClock()
    cb = _CompileCircuitBreaker(clock=clock)
    for _ in range(3):
        cb.record_failure()
    clock.advance(121)
    assert cb.state == "HALF_OPEN"
    cb.record_failure()
    assert cb.state == "OPEN"
    # Fresh cooldown — still open right after re-trip.
    clock.advance(60)
    assert cb.state == "OPEN"


@pytest.mark.asyncio
async def test_provider_compile_short_circuits_when_breaker_open(tmp_path) -> None:
    """When the breaker is OPEN, compile() returns degraded without HTTP."""
    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        memory_md_path=tmp_path / "MEMORY.md",
    )
    # Spy on _post_json to assert it is not called while OPEN.
    post_spy = AsyncMock()
    provider._post_json = post_spy  # type: ignore[assignment]

    # Force breaker OPEN.
    for _ in range(3):
        provider._compile_breaker.record_failure()
    assert provider._compile_breaker.state == "OPEN"

    ctx = await provider.compile("task")
    assert ctx.degraded is True
    assert post_spy.await_count == 0
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_records_trip_on_three_unavailable_errors(tmp_path) -> None:
    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        memory_md_path=tmp_path / "MEMORY.md",
    )
    provider._post_json = AsyncMock(  # type: ignore[assignment]
        side_effect=Hipp0UnavailableError("boom"),
    )

    for _ in range(3):
        ctx = await provider.compile("task")
        assert ctx.degraded is True

    assert provider._compile_breaker.state == "OPEN"
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_success_closes_half_open(tmp_path) -> None:
    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        memory_md_path=tmp_path / "MEMORY.md",
    )
    # Trip to OPEN and fast-forward past cooldown.
    clock = _FakeClock()
    provider._compile_breaker._clock = clock  # type: ignore[attr-defined]
    for _ in range(3):
        provider._compile_breaker.record_failure()
    clock.advance(121)
    assert provider._compile_breaker.state == "HALF_OPEN"

    provider._post_json = AsyncMock(return_value={"decisions": []})  # type: ignore[assignment]
    ctx = await provider.compile("task")
    assert ctx.degraded is False
    assert provider._compile_breaker.state == "CLOSED"
    await provider.aclose()
