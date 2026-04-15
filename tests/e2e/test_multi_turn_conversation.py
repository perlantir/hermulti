"""E2E: Multi-turn conversation with outcome signal."""
from __future__ import annotations

import os

import pytest
import httpx


HIPP0_BASE_URL = os.environ.get('HIPP0_BASE_URL', 'http://localhost:3001')


def _server_reachable() -> bool:
    try:
        return httpx.get(f'{HIPP0_BASE_URL}/api/health', timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_reachable(), reason='hipp0 not reachable')


def test_session_end_records_outcome():
    """POST /api/hermes/session/end should record an outcome and attribute it."""
    # Create a fresh project for this test
    proj = httpx.post(
        f'{HIPP0_BASE_URL}/api/projects',
        json={'name': 'e2e-multi-turn-session'},
        timeout=5,
    )
    assert proj.status_code in (200, 201), proj.text
    project_id = proj.json()['id']

    # Register a hermes agent - /api/hermes/session/start requires it.
    agent_name = 'e2e-multi-turn-agent'
    reg = httpx.post(
        f'{HIPP0_BASE_URL}/api/hermes/register',
        json={
            'project_id': project_id,
            'agent_name': agent_name,
            'soul': '# Soul\nE2E multi-turn agent.',
            'config': {'model': 'gpt-4o-mini', 'platform_access': ['web']},
        },
        timeout=5,
    )
    assert reg.status_code in (200, 201), reg.text

    # Record a decision (hipp0 expects `description`, not `content`).
    httpx.post(
        f'{HIPP0_BASE_URL}/api/projects/{project_id}/decisions',
        json={
            'made_by': 'architect',
            'title': 'Multi-turn test decision',
            'description': 'Placed during multi-turn E2E test.',
            'tags': ['e2e'],
            'confidence': 'high',
        },
        timeout=5,
    )

    # Start a real session to get a UUID session_id. hipp0 enforces
    # `session_id must be a valid UUID` on /session/end.
    start = httpx.post(
        f'{HIPP0_BASE_URL}/api/hermes/session/start',
        json={
            'project_id': project_id,
            'agent_name': agent_name,
            'platform': 'web',
        },
        timeout=5,
    )
    assert start.status_code in (200, 201), start.text
    session_id = start.json()['session_id']

    # End the session with a positive outcome
    end = httpx.post(
        f'{HIPP0_BASE_URL}/api/hermes/session/end',
        json={
            'session_id': session_id,
            'outcome': {
                'rating': 'positive',
                'signal_source': 'user_feedback',
                'snippet_ids': [],
            },
        },
        timeout=5,
    )
    # The route may return 200 even when auth is off
    assert end.status_code in (200, 201, 204), end.text
