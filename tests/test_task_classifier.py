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
        assert r.get("skip_compile") is True

    def test_write_simple_pure_function_skips(self):
        r = classify_task("write a simple function that adds two numbers")
        assert r.get("skip_compile") is True

    def test_self_contained_tasks_without_self_contained_markers_fall_through(self):
        # Without "from scratch"/"hello world" style markers, the classifier
        # does not route to self_contained.
        r = classify_task("draft the quarterly status update for leadership")
        assert not r.get("skip_compile")


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
    ])
    def test_user_tasks_route_to_user_namespace(self, task):
        r = classify_task(task)
        assert r["namespace"] == "user"
        assert r["fast_mode"] is True


class TestDefault:
    def test_empty_task_is_default(self):
        r = classify_task("")
        assert r.get("namespace") is None
        assert r.get("fast_mode") is True

    def test_generic_task_is_default(self):
        # Very vague phrasing lands in the ambiguous bucket which maps to
        # {namespace: None, fast_mode: True, routing_uncertain: True}
        r = classify_task("any ideas about this")
        assert r.get("namespace") is None
        assert r.get("fast_mode") is True

    def test_technical_takes_precedence_over_user(self):
        # Clear technical signal (bug, auth handler) beats "my" pronoun.
        r = classify_task("fix the bug in my auth handler that keeps crashing")
        assert r["namespace"] == "technical"
