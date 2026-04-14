"""Concurrency regression test for async-in-sync blocking bugs.

Reproduces the failure modes of Phase 2:

* Vision-fallback ``_describe_image_for_anthropic_fallback`` used to call
  ``asyncio.run()`` inside whatever thread the gateway picked — which
  raises ``RuntimeError`` when the thread already owns a running loop.
* ``cron.reflection.gather_reflection_input`` used to call
  ``asyncio.get_event_loop().run_until_complete(...)`` inside a
  coroutine, which under Python 3.12 raises "event loop already running"
  (or a deprecation-turned-error) when the cron tick fires while the
  gateway loop is live.

The test drives both paths under concurrency and asserts neither
raises.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Vision fallback: 10 concurrent "gateway handle_message" calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_fallback_safe_under_running_loop():
    """Simulate 10 gateway threads dispatching the Anthropic image
    fallback while the asyncio loop is live.

    Before the fix, ``asyncio.run(vision_analyze_tool(...))`` on
    ``run_agent.py:5587`` would raise ``RuntimeError: asyncio.run()
    cannot be called from a running event loop`` as soon as the caller's
    thread acquired the loop.
    """
    from run_agent import AIAgent

    async def fake_vision(image_url: str, user_prompt: str) -> str:
        # Yield once so the coroutine actually needs a loop.
        await asyncio.sleep(0)
        return '{"analysis": "a test image"}'

    # Minimal shim that exposes the attrs the bound method reads.
    shim = types.SimpleNamespace(
        _anthropic_image_fallback_cache={},
        _materialize_data_url_for_vision=AIAgent._materialize_data_url_for_vision,
    )

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=fake_vision):
        # Call the sync method directly from inside a running loop — this
        # is the failure mode: the gateway's async handler invokes sync
        # adapter code that hits the Anthropic fallback. Before the fix
        # this raised "asyncio.run() cannot be called from a running
        # event loop". Fire 10 in parallel via asyncio.to_thread to
        # stress the running-loop guard.
        results: List[Any] = await asyncio.gather(
            *(
                asyncio.to_thread(
                    AIAgent._describe_image_for_anthropic_fallback,
                    shim,
                    f"https://example.com/img-{i}.png",
                    "user",
                )
                for i in range(10)
            ),
            return_exceptions=True,
        )
        # And also one direct in-loop invocation, which is the harder
        # case: the sync method runs on the thread that owns the loop.
        direct = AIAgent._describe_image_for_anthropic_fallback(
            shim, "https://example.com/direct.png", "user"
        )

    # No exception should escape — in particular no RuntimeError about a
    # running loop.
    for r in results:
        assert not isinstance(r, BaseException), f"unexpected failure: {r!r}"
        assert "Image analysis failed" not in r, r
        assert "a test image" in r
    assert "a test image" in direct, direct


# ---------------------------------------------------------------------------
# Reflection tick while the gateway loop is running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_tick_during_gateway_loop(tmp_path, monkeypatch):
    """Simulate the cron ticker firing a reflection cycle while the
    gateway's event loop is running.

    Before the fix, ``gather_reflection_input`` called
    ``asyncio.get_event_loop().run_until_complete(...)`` which raises
    under a live loop (and is deprecated outright in 3.12).
    """
    import cron.reflection as reflection
    from hermes_state import SessionDB

    # Point reflection at a throw-away state DB and seed enough sessions
    # that the reflection cycle doesn't early-exit.
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(reflection, "_state_db_path", lambda: db_path)
    sdb = SessionDB(db_path=db_path)
    for i in range(5):
        sid = f"sess-{i}"
        sdb.create_session(sid, source="cli", agent_name="agent-x")
        sdb.record_outcome(sid, "positive", "turn_heuristic", None)

    # Agent directory stub so _read_text_file and _list_skills work.
    agent_dir = tmp_path / "agent-x"
    agent_dir.mkdir()
    (agent_dir / "MEMORY.md").write_text("", encoding="utf-8")
    (agent_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(reflection, "_agent_dir", lambda name: agent_dir)

    # Skip the real LLM + compile fetch.
    async def noop_compile(_name):
        return None

    monkeypatch.setattr(reflection, "_try_compile_context", noop_compile)

    # Direct await: this is the concurrency scenario — the reflection
    # coroutine is scheduled on the same loop that handles the gateway.
    rin = await reflection.gather_reflection_input("agent-x", lookback_days=7)
    assert rin.agent_name == "agent-x"
    assert rin.compiled_context is None
    # No RuntimeError / "event loop already running" — the await path
    # now works from inside a live loop.


@pytest.mark.asyncio
async def test_gateway_and_reflection_concurrent(tmp_path, monkeypatch):
    """End-to-end concurrency: 10 vision-fallback calls dispatched from
    gateway worker threads plus a reflection gather on the main loop.

    Assert no RuntimeError / "event loop already running" escapes.
    """
    from run_agent import AIAgent
    import cron.reflection as reflection
    from hermes_state import SessionDB

    # --- reflection setup (as above) --------------------------------------
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(reflection, "_state_db_path", lambda: db_path)
    sdb = SessionDB(db_path=db_path)
    for i in range(5):
        sid = f"sess-{i}"
        sdb.create_session(sid, source="cli", agent_name="agent-y")
        sdb.record_outcome(sid, "positive", "turn_heuristic", None)

    agent_dir = tmp_path / "agent-y"
    agent_dir.mkdir()
    (agent_dir / "MEMORY.md").write_text("", encoding="utf-8")
    (agent_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(reflection, "_agent_dir", lambda name: agent_dir)

    async def noop_compile(_name):
        return None

    monkeypatch.setattr(reflection, "_try_compile_context", noop_compile)

    # --- vision setup -----------------------------------------------------
    async def fake_vision(image_url: str, user_prompt: str) -> str:
        await asyncio.sleep(0)
        return '{"analysis": "ok"}'

    shim = types.SimpleNamespace(
        _anthropic_image_fallback_cache={},
        _materialize_data_url_for_vision=AIAgent._materialize_data_url_for_vision,
    )

    loop = asyncio.get_running_loop()

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=fake_vision):
        vision_tasks = [
            loop.run_in_executor(
                None,
                AIAgent._describe_image_for_anthropic_fallback,
                shim,
                f"https://example.com/concurrent-{i}.png",
                "user",
            )
            for i in range(10)
        ]
        reflection_task = asyncio.create_task(
            reflection.gather_reflection_input("agent-y", lookback_days=7)
        )

        results = await asyncio.gather(
            *vision_tasks, reflection_task, return_exceptions=True
        )

    # Separate the reflection result (last) from the vision results.
    *vision_results, reflection_result = results

    for r in vision_results:
        assert not isinstance(r, BaseException), f"vision call failed: {r!r}"
        assert "ok" in r

    assert not isinstance(reflection_result, BaseException), (
        f"reflection gather failed: {reflection_result!r}"
    )
    assert reflection_result.agent_name == "agent-y"
