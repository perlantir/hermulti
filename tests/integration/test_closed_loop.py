"""End-to-end integration test for the outcome -> reflection pipeline.

Verifies the three pieces Phase 1 added actually compose:

1. ``infer_outcome_from_turn`` fires on a positive/negative user message.
2. ``SessionDB.record_outcome`` persists the signal to the sessions row
   (simulating the new turn-boundary hook in ``run_agent.run_conversation``).
3. ``cron.reflection._query_sessions`` backfills aged NULL-outcome sessions
   via ``_backfill_aged_null_outcomes`` and the row ends up in the
   positive/negative bucket on the next reflection cycle.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.outcome_signals import infer_outcome_from_turn
from hermes_state import SessionDB


@pytest.fixture()
def state_db(tmp_path, monkeypatch):
    """Redirect the reflection module's state.db lookup to a fresh DB."""
    db_path = tmp_path / "state.db"
    # Patch reflection's resolver to use our temp DB.
    import cron.reflection as reflection
    monkeypatch.setattr(reflection, "_state_db_path", lambda: db_path)
    return SessionDB(db_path=db_path)


def test_turn_heuristic_persists_outcome(state_db):
    """Commit 1 + 2 path: infer -> record_outcome writes the sessions row."""
    sid = "sess-pos-1"
    state_db.create_session(sid, source="cli", agent_name="agent-a")

    inferred = infer_outcome_from_turn("thanks, that worked perfectly!")
    assert inferred == "positive"

    state_db.record_outcome(sid, inferred, "turn_heuristic", None)

    row = state_db._conn.execute(
        "SELECT outcome, outcome_source FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["outcome"] == "positive"
    assert row["outcome_source"] == "turn_heuristic"


def test_reflection_backfills_aged_null_outcomes(state_db, tmp_path):
    """Commit 3 path: NULL sessions older than 3 days get a heuristic label."""
    import cron.reflection as reflection

    aged_neg = "sess-aged-neg"
    aged_neutral = "sess-aged-neutral"
    fresh_null = "sess-fresh-null"

    # Three sessions for the same agent, all with NULL outcome.
    for sid in (aged_neg, aged_neutral, fresh_null):
        state_db.create_session(sid, source="cli", agent_name="agent-a")

    # Backdate the two "aged" sessions past the 3-day cutoff.
    old_ts = time.time() - 5 * 86400
    state_db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id IN (?, ?)",
        (old_ts, old_ts, aged_neg, aged_neutral),
    )

    # Last user message content drives the heuristic.
    state_db.append_message(aged_neg, role="user", content="no, that's wrong")
    state_db.append_message(aged_neutral, role="user", content="add more logging please")
    state_db.append_message(fresh_null, role="user", content="no, that's wrong")

    result = reflection._query_sessions("agent-a", lookback_days=30)

    by_id = {s["id"]: s for s in result["all"]}
    # Aged + negative-feedback message: backfilled as negative.
    assert by_id[aged_neg]["outcome"] == "negative"
    assert by_id[aged_neg]["outcome_source"] == "reflection_backfill"
    # Aged but neutral message: heuristic returns None -> still NULL.
    assert by_id[aged_neutral]["outcome"] is None
    # Fresh (within 3 days): left untouched even though it would classify.
    assert by_id[fresh_null]["outcome"] is None

    # And the buckets reflect the backfill.
    assert any(s["id"] == aged_neg for s in result["negative"])
    assert any(s["id"] == aged_neutral for s in result["neutral"])
    assert any(s["id"] == fresh_null for s in result["neutral"])


def test_full_closed_loop(state_db):
    """Simulate the full pipeline: turn -> record_outcome -> reflection reads it."""
    import cron.reflection as reflection

    sid = "sess-e2e"
    state_db.create_session(sid, source="cli", agent_name="agent-a")
    state_db.append_message(sid, role="user", content="Perfect, exactly what I wanted")

    # Step 1: turn boundary fires.
    inferred = infer_outcome_from_turn("Perfect, exactly what I wanted")
    assert inferred == "positive"
    state_db.record_outcome(sid, inferred, "turn_heuristic", None)

    # Step 2: reflection picks it up on the next cycle.
    result = reflection._query_sessions("agent-a", lookback_days=7)
    assert any(s["id"] == sid for s in result["positive"])
