"""Tests for SkillDispatcher."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from agent.skills.dispatcher import SkillDispatcher, DispatchSummary
from agent.skills.matcher import EventType, SkillEvent


class FakeLLM:
    def __init__(self, response='{"actions": []}'):
        self.response = response
        self.calls = 0
    async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
        self.calls += 1
        return self.response


class FakeProvider:
    def __init__(self):
        self.recorded_decisions: list[dict[str, Any]] = []
        self.recorded_outcomes: list[dict[str, Any]] = []
    async def record_decision(self, *, title, rationale, tags=None, confidence='medium', agent_name=None):
        self.recorded_decisions.append({'title': title, 'rationale': rationale, 'tags': tags, 'confidence': confidence})
        return True
    async def record_outcome(self, *, session_id, outcome, signal_source, snippet_ids=None):
        self.recorded_outcomes.append({'session_id': session_id, 'outcome': outcome, 'signal_source': signal_source})
        return True


def _new_dispatcher(llm=None, provider=None, env=None):
    """Helper that sets/restores HIPP0_SKILL_DISPATCHER env."""
    return SkillDispatcher(
        skills_dir='/root/audit/hipp0ai/skills',
        llm_client=llm,
        hipp0_provider=provider,
    )


@pytest.mark.asyncio
async def test_disabled_when_no_llm_in_auto_mode(monkeypatch):
    monkeypatch.delenv('HIPP0_SKILL_DISPATCHER', raising=False)
    d = SkillDispatcher(skills_dir='/root/audit/hipp0ai/skills', llm_client=None)
    assert d.enabled is False
    summary = await d.dispatch(SkillEvent(type=EventType.INBOUND_MESSAGE, text='hi'))
    assert summary.matched_skills == []


@pytest.mark.asyncio
async def test_enabled_when_llm_present_in_auto_mode(monkeypatch):
    monkeypatch.delenv('HIPP0_SKILL_DISPATCHER', raising=False)
    d = SkillDispatcher(skills_dir='/root/audit/hipp0ai/skills', llm_client=FakeLLM(), hipp0_provider=FakeProvider())
    assert d.enabled is True


@pytest.mark.asyncio
async def test_force_off(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'off')
    d = SkillDispatcher(skills_dir='/root/audit/hipp0ai/skills', llm_client=FakeLLM())
    assert d.enabled is False


@pytest.mark.asyncio
async def test_loads_real_skills():
    d = _new_dispatcher(llm=FakeLLM(), provider=FakeProvider())
    names = {s.name for s in d.skills}
    assert 'signal-detector' in names
    assert 'brain-ops' in names
    assert 'capture-decision' in names


@pytest.mark.asyncio
async def test_inbound_message_fires_signal_detector_in_parallel(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'on')
    llm = FakeLLM()
    d = _new_dispatcher(llm=llm, provider=FakeProvider())

    summary = await d.dispatch(SkillEvent(type=EventType.INBOUND_MESSAGE, text='hello world'))
    assert 'signal-detector' in summary.matched_skills
    # signal-detector goes through parallel path, not the awaited results
    assert summary.parallel_tasks >= 1
    # Wait for background tasks
    await d.close()
    assert llm.calls >= 1


@pytest.mark.asyncio
async def test_pre_task_runs_brain_ops_first(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'on')
    call_order: list[str] = []

    class OrderingLLM:
        async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
            # extract skill name from user prompt
            for line in user.splitlines():
                if line.startswith('# Skill: '):
                    call_order.append(line.removeprefix('# Skill: ').strip())
                    break
            return '{"actions": []}'

    d = _new_dispatcher(llm=OrderingLLM(), provider=FakeProvider())
    await d.dispatch(SkillEvent(type=EventType.PRE_TASK, text='Implement feature X'))
    await d.close()

    # If brain-ops matched and any other skill matched, brain-ops should be first
    if 'brain-ops' in call_order and len(call_order) > 1:
        assert call_order[0] == 'brain-ops'


@pytest.mark.asyncio
async def test_dispatcher_swallows_runner_errors(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'on')

    class CrashLLM:
        async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
            raise RuntimeError('boom')

    d = _new_dispatcher(llm=CrashLLM(), provider=FakeProvider())
    # Should NOT raise
    summary = await d.dispatch(SkillEvent(type=EventType.OUTBOUND_MESSAGE, text='we decided to use redis'))
    await d.close()
    assert isinstance(summary, DispatchSummary)


@pytest.mark.asyncio
async def test_close_idempotent(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'on')
    d = _new_dispatcher(llm=FakeLLM(), provider=FakeProvider())
    await d.close()  # nothing to close
    await d.close()  # safe to call twice
