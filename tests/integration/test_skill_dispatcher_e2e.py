"""End-to-end integration test for SkillDispatcher.

Wires a fake LLM that returns a record_decision action, dispatches an
OUTBOUND_MESSAGE, and verifies the action propagates all the way to
hipp0_provider.record_decision() with the expected payload.

This is the Python-side companion to the hipp0 e2e scenarios; it lives
under tests/integration/ so the fast unit loop can skip it.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from agent.skills.dispatcher import SkillDispatcher
from agent.skills.matcher import EventType, SkillEvent


SKILLS_DIR = '/root/audit/hipp0ai/skills'


class RecordingLLM:
    """Fake LLM that returns a canned record_decision action.

    Mirrors the OpenAI-compatible fake-llm-server.ts fixture
    (e2e/fixtures/llm/record-decision.json).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._response = (
            '{"actions": [{"type": "record_decision", "args": '
            '{"title": "Use Redis", "rationale": "Pub/sub + TTL", '
            '"tags": ["cache", "redis"], "confidence": "high"}}]}'
        )

    async def call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append((system, user))
        return self._response


class RecordingProvider:
    def __init__(self) -> None:
        self.recorded_decisions: list[dict[str, Any]] = []

    async def record_decision(
        self,
        *,
        title: str,
        rationale: str,
        tags: list[str] | None = None,
        confidence: str = 'medium',
        agent_name: str | None = None,
    ) -> bool:
        self.recorded_decisions.append(
            {
                'title': title,
                'rationale': rationale,
                'tags': list(tags or []),
                'confidence': confidence,
                'agent_name': agent_name,
            }
        )
        return True


@pytest.mark.skipif(
    not os.path.isdir(SKILLS_DIR),
    reason=f'skills dir not available at {SKILLS_DIR}',
)
@pytest.mark.asyncio
async def test_outbound_message_triggers_record_decision(monkeypatch):
    monkeypatch.setenv('HIPP0_SKILL_DISPATCHER', 'on')

    llm = RecordingLLM()
    provider = RecordingProvider()
    dispatcher = SkillDispatcher(
        skills_dir=SKILLS_DIR,
        llm_client=llm,
        hipp0_provider=provider,
    )
    assert dispatcher.enabled is True

    event = SkillEvent(
        type=EventType.OUTBOUND_MESSAGE,
        text='we decided to use Redis for our cache tier.',
    )
    summary = await dispatcher.dispatch(event)

    # Drain background tasks scheduled by signal-detector / capture-decision.
    await dispatcher.close()

    # At least one skill must have matched. signal-detector fires on every
    # message, capture-decision matches "decided".
    assert summary.matched_skills, f'No skills matched: {summary}'

    # The LLM must have been invoked with the skill body in the prompt.
    assert llm.calls, 'LLM was never called'
    _, user_prompt = llm.calls[0]
    assert '# Skill: ' in user_prompt

    # And the record_decision action must have reached the provider.
    assert provider.recorded_decisions, (
        f'provider.record_decision was never called. '
        f'Matched skills: {summary.matched_skills}, '
        f'LLM calls: {len(llm.calls)}'
    )
    first = provider.recorded_decisions[0]
    assert first['title'] == 'Use Redis'
    assert first['confidence'] == 'high'
    assert 'cache' in first['tags']
