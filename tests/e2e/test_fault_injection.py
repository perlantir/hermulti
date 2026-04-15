"""E2E: fault injection, verify graceful degradation."""
from __future__ import annotations

import asyncio
import os

import pytest


def test_provider_unreachable_is_nonfatal():
    """record_decision against a nonexistent hipp0 returns False, does not raise."""
    from agent.hipp0_memory_provider import Hipp0MemoryProvider
    provider = Hipp0MemoryProvider(
        base_url='http://localhost:1',  # nothing listens here
        api_key='',
        project_id='nonexistent',
        agent_name='e2e',
        agent_id='e2e-agent',
    )
    result = asyncio.run(provider.record_decision(
        title='Fault injection test',
        rationale='hipp0 unreachable',
        tags=['fault'],
        confidence='low',
        agent_name='e2e',
    ))
    assert result is False, 'should return False, not raise, when hipp0 unreachable'


def test_llm_failure_does_not_crash_dispatcher():
    """When LLM raises, SkillDispatcher logs the error but does not crash."""
    from agent.skills.dispatcher import SkillDispatcher
    from agent.skills.matcher import SkillEvent, EventType

    class FailingLLM:
        async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
            raise RuntimeError('LLM is on fire')

    class FakeProvider:
        async def record_decision(self, **kwargs): return True
        async def record_outcome(self, **kwargs): return True

    os.environ['HIPP0_SKILL_DISPATCHER'] = 'on'
    d = SkillDispatcher(llm_client=FailingLLM(), hipp0_provider=FakeProvider(), agent_name='e2e')
    # Should NOT raise
    summary = asyncio.run(d.dispatch(SkillEvent(
        type=EventType.OUTBOUND_MESSAGE,
        text='we decided to blow everything up',
    )))
    asyncio.run(d.close())
    assert summary is not None
