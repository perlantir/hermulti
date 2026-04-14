"""Unit tests for the reflection skill-creation eval gate.

A candidate skill proposal must be anchored to at least one NEGATIVE-outcome
session whose first-user-message mentions a topic token from the proposal.
Otherwise the candidate is rejected (logged as ``skill_eval_gate_failed``).
"""

from __future__ import annotations

import cron.reflection as reflection
from cron.reflection import (
    ReflectionInput,
    _score_skill_candidate,
    _skill_topic_tokens,
)


def test_topic_tokens_drops_short_tokens():
    toks = _skill_topic_tokens({
        "name": "db-migration-helper",
        "content_hint": "a tool to run pg migrations",
    })
    assert "migration" in toks
    assert "helper" in toks
    # 2-char tokens dropped.
    assert "pg" not in toks
    assert "a" not in toks


def test_gate_passes_when_negative_session_matches_topic():
    rin = ReflectionInput(
        agent_name="agent-a",
        negative_sessions=[
            {"id": "s1",
             "first_user_message": "the migration broke the users table again"},
        ],
    )
    proposal = {"name": "migration-guard", "reason": "prevent migration breakage"}
    result = _score_skill_candidate("agent-a", proposal, rin)
    assert result["passed"] is True
    assert result["matches"] == 1
    assert "migration" in result["tokens"]


def test_gate_rejects_when_no_negative_evidence():
    rin = ReflectionInput(
        agent_name="agent-a",
        negative_sessions=[
            {"id": "s1", "first_user_message": "something completely unrelated"},
        ],
        positive_sessions=[
            {"id": "s2", "first_user_message": "fix the migration please"},
        ],
    )
    proposal = {"name": "migration-guard", "reason": ""}
    result = _score_skill_candidate("agent-a", proposal, rin)
    assert result["passed"] is False
    assert result["reason"] == "no_prior_negative"


def test_gate_rejects_when_no_topic_tokens():
    rin = ReflectionInput(agent_name="a")
    proposal = {"name": "xx", "reason": "", "content_hint": ""}
    result = _score_skill_candidate("a", proposal, rin)
    assert result["passed"] is False
    assert result["reason"] == "no_topic_tokens"


def test_skill_cap_is_one():
    # Smoke check — phase 5 commit (a) contract.
    assert reflection.MAX_SKILLS_PER_CYCLE == 1
