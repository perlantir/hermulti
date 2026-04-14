"""Test for GET /admin/routing-quality.

Exercises the aggregation function directly (handler-free) since the full
api_server harness is heavyweight. The handler is a thin wrapper around
``tools.routing_outcomes.aggregate`` — if the aggregation is correct, the
handler's shape test below covers the JSON envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.routing_outcomes import (
    ClassAggregate,
    aggregate,
    positive_rate,
    record_decision,
    record_outcome,
)


def test_aggregate_handler_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "routing.jsonl"

    # Seed a mixed-outcome log.
    for _ in range(3):
        record_decision("fix the crash", decided_class="technical", score=0.4, margin=0.1, uncertain=False, log_path=log)
        record_outcome("fix the crash", outcome="positive", log_path=log)
    record_decision("remember style", decided_class="user", score=0.4, margin=0.1, uncertain=False, log_path=log)
    record_outcome("remember style", outcome="negative", log_path=log)

    monkeypatch.setenv("HERMES_ROUTING_OUTCOMES_LOG", str(log))
    agg = aggregate()

    # Shape that /admin/routing-quality would serialize.
    serialized = {
        cls: {
            "decision_count": data.count,
            "outcomes": dict(data.outcomes),
            "positive_rate": positive_rate(data),
        }
        for cls, data in agg.items()
    }
    assert "technical" in serialized
    assert serialized["technical"]["decision_count"] >= 1
    assert serialized["technical"]["positive_rate"] == pytest.approx(1.0)
    assert serialized["user"]["positive_rate"] == pytest.approx(0.0)
    # Must be JSON-serializable end-to-end.
    assert json.loads(json.dumps(serialized)) == serialized


def test_aggregate_empty_log_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_ROUTING_OUTCOMES_LOG", str(tmp_path / "missing.jsonl"))
    agg = aggregate()
    assert agg == {}
