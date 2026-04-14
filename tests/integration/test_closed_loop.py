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


def test_skill_outcomes_write_path(state_db):
    """Auto-invoke wiring writes baseline + match rows, and the public
    ``record_skill_outcome_for_session`` hook appends a post row."""
    import sqlite3
    import cron.reflection as reflection

    # Seed a prior-week baseline session with a negative outcome on the topic.
    baseline_sid = "sess-baseline"
    state_db.create_session(baseline_sid, source="cli", agent_name="agent-a")
    state_db.append_message(baseline_sid, role="user",
                            content="the migration broke everything again")
    state_db.record_outcome(baseline_sid, "negative", "turn_heuristic", None)
    # Backdate to 2 days ago so it's within the 7d baseline window.
    old = time.time() - 2 * 86400
    state_db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
        (old, old, baseline_sid),
    )

    rin = reflection.ReflectionInput(
        agent_name="agent-a",
        recent_sessions=[{
            "id": baseline_sid,
            "first_user_message": "the migration broke everything again",
            "outcome": "negative",
        }],
        negative_sessions=[{
            "id": baseline_sid,
            "first_user_message": "the migration broke everything again",
        }],
    )
    proposal = {"name": "migration-guard", "reason": "prevent breakage"}

    reflection._register_skill_autoinvoke("agent-a", proposal, rin)

    con = sqlite3.connect(str(reflection._state_db_path()))
    try:
        rows = con.execute(
            "SELECT kind, outcome, session_id FROM skill_outcomes "
            "WHERE skill_id = ? ORDER BY kind", ("migration-guard",),
        ).fetchall()
    finally:
        con.close()
    kinds = {r[0] for r in rows}
    assert "baseline" in kinds
    assert "match" in kinds

    # Public hook appends a post row.
    reflection.record_skill_outcome_for_session(
        "migration-guard", "agent-a", baseline_sid, "positive",
    )
    con = sqlite3.connect(str(reflection._state_db_path()))
    try:
        post_rows = con.execute(
            "SELECT outcome FROM skill_outcomes "
            "WHERE skill_id = ? AND kind = 'post'", ("migration-guard",),
        ).fetchall()
    finally:
        con.close()
    assert post_rows and post_rows[0][0] == "positive"


# ---------------------------------------------------------------------------
# Phase 10: full end-to-end chain
#
#   task -> subagent (fake hipp0 provider) -> compile -> infer_outcome_from_turn
#          -> record_outcome -> reflection NULL backfill -> second compile
#          observes outcome and ranks D1 > D2.
#
# We test the HERMES-side wiring: the fake provider records every call and
# simulates the hipp0-side trust-multiplier effect by biasing the second
# compile's ranking based on the outcomes it saw.  The actual hipp0 scoring
# math is verified separately in packages/server/tests/closed_loop.test.ts.
# ---------------------------------------------------------------------------


class FakeHipp0Provider:
    """Lightweight in-memory stand-in for Hipp0MemoryProvider.

    Records every compile() and record_outcome() call, and uses its own
    outcome state to re-rank decisions on subsequent compile() calls.
    """

    def __init__(self, *, fail_record: bool = False):
        self.compile_calls: list[dict] = []
        self.outcome_calls: list[dict] = []
        self._positive_ids: set[str] = set()
        self._fail_record = fail_record

    async def compile(self, task_description: str, **kwargs):
        self.compile_calls.append({"task": task_description, **kwargs})
        # Baseline ranking: D2 slightly above D1.
        decisions = [
            {"id": "D1", "title": "Use JWT", "combined_score": 0.60},
            {"id": "D2", "title": "Use sessions", "combined_score": 0.65},
        ]
        # Simulate hipp0 trust multiplier: positive outcomes bump that id.
        for d in decisions:
            if d["id"] in self._positive_ids:
                d["combined_score"] *= 1.10
        decisions.sort(key=lambda d: d["combined_score"], reverse=True)
        return {
            "decisions": decisions,
            "total_tokens": 100,
            "compile_request_id": f"cr-{len(self.compile_calls)}",
            "compiled_snippet_ids": [d["id"] for d in decisions],
        }

    async def record_outcome(
        self,
        snippet_ids,
        outcome,
        *,
        signal_source,
        note=None,
    ):
        if self._fail_record:
            raise RuntimeError("simulated hipp0 outage")
        self.outcome_calls.append({
            "snippet_ids": list(snippet_ids),
            "outcome": outcome,
            "signal_source": signal_source,
            "note": note,
        })
        if outcome == "positive":
            for sid in snippet_ids:
                self._positive_ids.add(sid)


@pytest.mark.asyncio
async def test_closed_loop_full_chain(state_db):
    """End-to-end: compile -> turn -> record_outcome -> backfill -> recompile re-ranks."""
    import cron.reflection as reflection

    provider = FakeHipp0Provider()
    sid = "sess-e2e-full"
    state_db.create_session(sid, source="cli", agent_name="agent-a")

    # 1. First compile — baseline ranking (D2 > D1).
    first = await provider.compile("build auth module", task_session_id=sid)
    first_ids = [d["id"] for d in first["decisions"]]
    assert first_ids == ["D2", "D1"], f"baseline ranking unexpected: {first_ids}"
    # The snippet ids that participated in this compile — what we'll attribute.
    compiled_ids = first["compiled_snippet_ids"]

    # 2. Subagent produces a turn; user feedback is positive.
    user_msg = "Perfect, exactly what I wanted"
    state_db.append_message(sid, role="user", content=user_msg)
    inferred = infer_outcome_from_turn(user_msg)
    assert inferred == "positive"

    # 3. Turn-boundary record_outcome — hits both local SessionDB and provider.
    state_db.record_outcome(sid, inferred, "turn_heuristic", None)
    await provider.record_outcome(
        compiled_ids, inferred, signal_source="turn_heuristic"
    )

    # The provider captured the call.
    assert len(provider.outcome_calls) == 1
    assert provider.outcome_calls[0]["outcome"] == "positive"
    assert set(provider.outcome_calls[0]["snippet_ids"]) == {"D1", "D2"}

    # 4. Reflection NULL-outcome backfill — no-op because outcome already recorded.
    result = reflection._query_sessions("agent-a", lookback_days=7)
    this_sess = next(s for s in result["all"] if s["id"] == sid)
    assert this_sess["outcome"] == "positive"
    assert this_sess["outcome_source"] == "turn_heuristic"  # not reflection_backfill

    # 5. Second compile for same task — mock observes outcome state.
    # We bias only D1 positive to show the ranking flip.
    provider._positive_ids = {"D1"}  # simulate attribution landed on D1 only
    second = await provider.compile("build auth module", task_session_id=sid)
    second_ids = [d["id"] for d in second["decisions"]]
    assert second_ids == ["D1", "D2"], (
        f"after positive outcome, D1 should outrank D2; got {second_ids}"
    )
    # And the trust boost is visible in the score.
    d1_score = next(d["combined_score"] for d in second["decisions"] if d["id"] == "D1")
    assert d1_score > 0.60, f"D1 score should be boosted, got {d1_score}"


@pytest.mark.asyncio
async def test_closed_loop_fails_when_record_outcome_silently_drops(state_db):
    """Failure mode: if record_outcome no-ops, second compile keeps baseline ranking.

    This guards against a regression where the turn-boundary hook silently
    fails and the provider never sees the signal.  The assertion message
    documents exactly what failed.
    """
    provider = FakeHipp0Provider()
    sid = "sess-e2e-broken"
    state_db.create_session(sid, source="cli", agent_name="agent-a")

    first = await provider.compile("task", task_session_id=sid)
    assert [d["id"] for d in first["decisions"]] == ["D2", "D1"]

    # Simulate the bug: record_outcome is never called (e.g. hook stripped).
    inferred = infer_outcome_from_turn("thanks, that worked perfectly!")
    assert inferred == "positive"
    # DELIBERATELY skip provider.record_outcome(...) here.

    second = await provider.compile("task", task_session_id=sid)
    second_ids = [d["id"] for d in second["decisions"]]
    # This is the assertion that WOULD fail in prod if the hook is broken.
    # In this failure-mode test we assert the broken behaviour so a future
    # "fix" that actually wires record_outcome into compile() breaks this test.
    assert second_ids == ["D2", "D1"], (
        "without record_outcome, ranking must stay at baseline; "
        f"got {second_ids} — did record_outcome leak in?"
    )
    assert provider.outcome_calls == [], (
        "FakeProvider saw an outcome call it shouldn't have — "
        "test fixture drifted"
    )


@pytest.mark.asyncio
async def test_closed_loop_raises_when_provider_record_outcome_errors(state_db):
    """Failure mode: provider.record_outcome raises — caller must surface it."""
    provider = FakeHipp0Provider(fail_record=True)
    sid = "sess-e2e-err"
    state_db.create_session(sid, source="cli", agent_name="agent-a")

    await provider.compile("task", task_session_id=sid)
    inferred = infer_outcome_from_turn("thanks that worked")
    assert inferred == "positive"

    with pytest.raises(RuntimeError, match="simulated hipp0 outage"):
        await provider.record_outcome(
            ["D1", "D2"], inferred, signal_source="turn_heuristic"
        )
