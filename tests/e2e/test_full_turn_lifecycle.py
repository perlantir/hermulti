"""
E2E: Full hermulti turn lifecycle with fake LLM and real hipp0 HTTP.

Requires:
  - hipp0 server running at HIPP0_BASE_URL (default http://localhost:3001)
  - fake LLM server running at OPENAI_BASE_URL
  - A seeded project_id in HIPP0_SEED_FILE

Skips if not reachable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import httpx


HIPP0_BASE_URL = os.environ.get('HIPP0_BASE_URL', 'http://localhost:3001')
HIPP0_SEED_FILE = os.environ.get('HIPP0_SEED_FILE')


def _server_reachable() -> bool:
    try:
        r = httpx.get(f'{HIPP0_BASE_URL}/api/health', timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f'hipp0 server not reachable at {HIPP0_BASE_URL}',
)


@pytest.fixture(scope='module')
def seed() -> dict:
    if not HIPP0_SEED_FILE or not Path(HIPP0_SEED_FILE).exists():
        pytest.skip('no HIPP0_SEED_FILE set')
    return json.loads(Path(HIPP0_SEED_FILE).read_text())


def test_hipp0_memory_provider_can_record_decision(seed):
    """Direct HTTP test: Hipp0MemoryProvider.record_decision actually writes to hipp0."""
    from agent.hipp0_memory_provider import Hipp0MemoryProvider

    provider = Hipp0MemoryProvider(
        base_url=HIPP0_BASE_URL,
        api_key=os.environ.get('HIPP0_API_KEY', ''),
        project_id=seed['project_id'],
        agent_name='e2e',
        agent_id=seed.get('agent_id', 'e2e-agent'),
    )
    # Note: async method, need to run in an event loop
    import asyncio
    result = asyncio.run(provider.record_decision(
        title='E2E test decision',
        rationale='Placed by test_full_turn_lifecycle to verify connectivity.',
        tags=['e2e', 'test'],
        confidence='medium',
        agent_name='e2e',
    ))
    assert result is True, 'record_decision should succeed against live hipp0'

    # Verify it's visible
    listing = httpx.get(
        f'{HIPP0_BASE_URL}/api/projects/{seed["project_id"]}/decisions',
        timeout=5,
    )
    assert listing.status_code == 200
    decisions = listing.json()
    # decisions may be wrapped in {decisions: [...]} or raw list
    items = decisions if isinstance(decisions, list) else decisions.get('decisions', [])
    titles = [d.get('title', '') for d in items]
    assert any('E2E test decision' in t for t in titles)


def test_skill_dispatcher_fires_on_outbound_message(seed):
    """The skill dispatcher, when wired, should dispatch a SkillEvent on OUTBOUND_MESSAGE."""
    import asyncio
    from agent.skills.dispatcher import SkillDispatcher
    from agent.skills.matcher import SkillEvent, EventType

    # Capture LLM calls
    llm_calls: list[tuple[str, str]] = []

    class CaptureLLM:
        async def call(self, system, user, *, max_tokens=1500, temperature=0.2):
            llm_calls.append((system, user))
            return '{"actions": [{"type": "record_decision", "args": {"title": "E2E decided PostgreSQL", "rationale": "triggered from E2E test", "tags": ["e2e", "postgres"], "confidence": "high"}}]}'

    recorded: list[dict] = []

    class CaptureProvider:
        async def record_decision(self, *, title, rationale, tags=None, confidence='medium', agent_name=None):
            recorded.append({'title': title, 'rationale': rationale, 'tags': tags or [], 'confidence': confidence})
            return True

        async def record_outcome(self, *, session_id=None, outcome=None, signal_source=None, snippet_ids=None):
            return True

    os.environ['HIPP0_SKILL_DISPATCHER'] = 'on'
    dispatcher = SkillDispatcher(
        llm_client=CaptureLLM(),
        hipp0_provider=CaptureProvider(),
        agent_name='e2e',
    )
    assert dispatcher.enabled, 'dispatcher should be enabled'

    summary = asyncio.run(dispatcher.dispatch(SkillEvent(
        type=EventType.OUTBOUND_MESSAGE,
        text='We decided to use PostgreSQL because of JSONB and transactions.',
    )))
    # Signal-detector runs in parallel, wait for it
    asyncio.run(dispatcher.close())

    assert len(llm_calls) >= 1, 'LLM should have been called by signal-detector or capture-decision'
    # At least one recorded decision (from our deterministic LLM response)
    assert any('E2E decided PostgreSQL' in r['title'] for r in recorded), \
        f'Expected decision from LLM action but got: {[r["title"] for r in recorded]}'
