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

    # Record a decision
    httpx.post(
        f'{HIPP0_BASE_URL}/api/projects/{project_id}/decisions',
        json={
            'made_by': 'architect',
            'title': 'Multi-turn test decision',
            'content': 'Placed during multi-turn E2E test.',
            'tags': ['e2e'],
            'confidence': 'high',
        },
        timeout=5,
    )

    # End the session with a positive outcome
    end = httpx.post(
        f'{HIPP0_BASE_URL}/api/hermes/session/end',
        json={
            'project_id': project_id,
            'session_id': 'e2e-multi-turn-session-1',
            'ended_at': '2026-04-15T12:00:00Z',
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
