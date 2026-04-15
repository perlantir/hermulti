"""Tests for SkillRunner with mocked LLM and provider."""
from __future__ import annotations

from typing import Any

import pytest

from agent.skills.loader import Skill
from agent.skills.matcher import EventType, SkillEvent
from agent.skills.runner import SkillRunner


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []
    async def call(self, system: str, user: str, *, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        return self.response


class FakeProvider:
    def __init__(self):
        self.recorded_decisions: list[dict[str, Any]] = []
        self.recorded_outcomes: list[dict[str, Any]] = []
        self.fail_decisions = False
    async def record_decision(self, *, title, rationale, tags=None, confidence='medium', agent_name=None):
        if self.fail_decisions:
            return False
        self.recorded_decisions.append({
            'title': title, 'rationale': rationale, 'tags': tags or [],
            'confidence': confidence, 'agent_name': agent_name,
        })
        return True
    async def record_outcome(self, *, session_id, outcome, signal_source, snippet_ids=None):
        self.recorded_outcomes.append({
            'session_id': session_id, 'outcome': outcome,
            'signal_source': signal_source, 'snippet_ids': snippet_ids or [],
        })
        return True


def _skill(name='test-skill', body='Do the thing.', triggers=None, mutating=True):
    return Skill(
        name=name, version='1.0', description='Test',
        triggers=triggers or [], mutating=mutating, tools=[], body=body, path='/tmp/x',
    )


@pytest.mark.asyncio
async def test_runs_record_decision_action():
    llm = FakeLLM('{"actions": [{"type": "record_decision", "args": {"title": "Use Redis", "rationale": "Speed", "tags": ["cache"], "confidence": "high"}}]}')
    provider = FakeProvider()
    runner = SkillRunner(llm, provider)
    event = SkillEvent(type=EventType.OUTBOUND_MESSAGE, text='we decided to use redis')

    result = await runner.run(_skill(), event)

    assert result.actions_attempted == 1
    assert result.actions_succeeded == 1
    assert result.actions_failed == 0
    assert len(provider.recorded_decisions) == 1
    assert provider.recorded_decisions[0]['title'] == 'Use Redis'
    assert provider.recorded_decisions[0]['confidence'] == 'high'


@pytest.mark.asyncio
async def test_handles_log_action():
    llm = FakeLLM('{"actions": [{"type": "log", "args": {"message": "noted"}}]}')
    runner = SkillRunner(llm, FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='hi'))
    assert result.actions_attempted == 1
    assert result.actions_succeeded == 1


@pytest.mark.asyncio
async def test_handles_noop():
    llm = FakeLLM('{"actions": [{"type": "noop", "args": {"reason": "irrelevant"}}]}')
    runner = SkillRunner(llm, FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='x'))
    assert result.actions_succeeded == 1


@pytest.mark.asyncio
async def test_handles_empty_actions_array():
    llm = FakeLLM('{"actions": []}')
    runner = SkillRunner(llm, FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='x'))
    assert result.actions_attempted == 0
    assert result.actions_succeeded == 0


@pytest.mark.asyncio
async def test_handles_malformed_json():
    llm = FakeLLM('not json at all')
    runner = SkillRunner(llm, FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='x'))
    assert result.actions_attempted == 0
    assert result.error is None  # Malformed JSON is not a runner error, just zero actions


@pytest.mark.asyncio
async def test_handles_llm_failure():
    class FailingLLM:
        async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
            raise RuntimeError('LLM is down')
    runner = SkillRunner(FailingLLM(), FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='x'))
    assert result.actions_attempted == 0
    assert result.error is not None
    assert 'LLM' in result.error


@pytest.mark.asyncio
async def test_provider_failure_counted_as_failed_action():
    llm = FakeLLM('{"actions": [{"type": "record_decision", "args": {"title": "X", "rationale": "Y"}}]}')
    provider = FakeProvider()
    provider.fail_decisions = True
    runner = SkillRunner(llm, provider)
    result = await runner.run(_skill(), SkillEvent(type=EventType.OUTBOUND_MESSAGE, text='x'))
    assert result.actions_attempted == 1
    assert result.actions_succeeded == 0
    assert result.actions_failed == 1


@pytest.mark.asyncio
async def test_no_llm_returns_error():
    runner = SkillRunner(None, FakeProvider())
    result = await runner.run(_skill(), SkillEvent(type=EventType.INBOUND_MESSAGE, text='x'))
    assert result.error is not None
    assert result.actions_attempted == 0
