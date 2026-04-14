"""Unit tests for :func:`tools.persistent_delegate_tool.classify_task`.

Pure, no I/O. Verifies the routing table that decides compile mode
for a delegate invocation.
"""

from __future__ import annotations

import pytest

from tools.persistent_delegate_tool import classify_task


class TestSelfContained:
    def test_hello_world_from_scratch_skips(self):
        r = classify_task("write a hello world from scratch")
        assert r == {"skip_compile": True}

    def test_write_simple_pure_function_skips(self):
        r = classify_task("write a simple function that adds two numbers")
        assert r == {"skip_compile": True}

    def test_self_contained_with_proper_noun_falls_through(self):
        # "Q3" / "Kickoff" trigger the proper-noun guard, so the task
        # is NOT treated as self-contained even if it says "from scratch".
        r = classify_task("write a Q3 kickoff email from scratch")
        assert "skip_compile" not in r


class TestTechnical:
    @pytest.mark.parametrize("task", [
        "fix the crash in the login handler",
        "why does this throw a TypeError exception",
        "how to debug a stack trace in asyncio",
        "the build has a bug in CI",
    ])
    def test_technical_tasks_route_to_technical_namespace(self, task):
        r = classify_task(task)
        assert r["namespace"] == "technical"
        assert r["fast_mode"] is False


class TestUser:
    @pytest.mark.parametrize("task", [
        "remember my preference for dark mode",
        "what's my favourite editor style",
        "remind me what I like for breakfast",
    ])
    def test_user_tasks_route_to_user_namespace(self, task):
        r = classify_task(task)
        assert r["namespace"] == "user"
        assert r["fast_mode"] is True


class TestDefault:
    def test_empty_task_is_default(self):
        assert classify_task("") == {"namespace": None, "fast_mode": True}

    def test_generic_task_is_default(self):
        r = classify_task("summarize the last meeting")
        assert r == {"namespace": None, "fast_mode": True}

    def test_technical_takes_precedence_over_user(self):
        r = classify_task("fix my preference handler crash")
        # "fix" + "crash" dominate; route to technical.
        assert r["namespace"] == "technical"
