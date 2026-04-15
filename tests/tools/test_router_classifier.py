"""Tests for the similarity-based task classifier + routing-outcomes log."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.router_classifier import (
    DEFAULT_MARGIN,
    classify,
    decision_to_classify_task_hint,
)
from tools.routing_outcomes import (
    aggregate,
    positive_rate,
    record_decision,
    record_outcome,
    task_hash,
)


# ------------------------------------------------------------------
#  Classifier


def test_classifier_routes_obvious_technical() -> None:
    d = classify("the database connection keeps crashing in production")
    assert d.cls == "technical"
    assert not d.uncertain


def test_classifier_routes_user_preference() -> None:
    d = classify("remember that I prefer MLA citation style")
    assert d.cls == "user"
    assert not d.uncertain


def test_classifier_routes_self_contained() -> None:
    d = classify("write a hello world program from scratch")
    assert d.cls == "self_contained"
    assert not d.uncertain


def test_classifier_flags_oblique_phrasing_as_ambiguous() -> None:
    # The v1 keyword router missed all of these — it had no keyword hit,
    # so it silently routed to the default "full fast compile". The new
    # classifier either lands on the ambiguous class (explicit "I don't
    # know") or marks the call uncertain. Both outcomes are acceptable;
    # both tell the caller this is not a confident technical/user route.
    for oblique in ["hmm", "thoughts?", "take a look", "something is off"]:
        d = classify(oblique)
        assert d.cls == "ambiguous" or d.uncertain, (
            f"expected ambiguous or uncertain for oblique: {oblique!r}, got {d}"
        )


def test_classifier_empty_task_is_ambiguous() -> None:
    d = classify("")
    assert d.cls == "ambiguous"
    assert d.uncertain


def test_classifier_margin_controls_uncertainty() -> None:
    # Nearly identical scores for two classes should flag uncertainty.
    d = classify("preference bug", margin=0.5)
    assert d.uncertain


def test_hint_mapping_preserves_contract() -> None:
    d = classify("debug the traceback from the failing tests")
    hint = decision_to_classify_task_hint(d)
    assert hint.get("namespace") == "technical"
    assert hint.get("fast_mode") is False

    d = classify("remember my style is concise names")
    hint = decision_to_classify_task_hint(d)
    assert hint.get("namespace") == "user"


# ------------------------------------------------------------------
#  Routing-outcomes log


def test_log_records_and_aggregates(tmp_path: Path) -> None:
    log = tmp_path / "routing.jsonl"

    # 2 technical, 1 user. 2 of the technical get positive outcomes; the user one negative.
    record_decision("fix the crash", decided_class="technical", score=0.4, margin=0.1, uncertain=False, log_path=log)
    record_outcome("fix the crash", outcome="positive", log_path=log)

    record_decision("debug the traceback", decided_class="technical", score=0.4, margin=0.1, uncertain=False, log_path=log)
    record_outcome("debug the traceback", outcome="positive", log_path=log)

    record_decision("remember my style", decided_class="user", score=0.4, margin=0.1, uncertain=False, log_path=log)
    record_outcome("remember my style", outcome="negative", log_path=log)

    agg = aggregate(log_path=log)
    assert agg["technical"].count == 2
    assert agg["technical"].outcomes.get("positive") == 2
    assert agg["user"].count == 1
    assert agg["user"].outcomes.get("negative") == 1
    assert positive_rate(agg["technical"]) == pytest.approx(1.0)
    assert positive_rate(agg["user"]) == pytest.approx(0.0)


def test_log_latest_outcome_wins(tmp_path: Path) -> None:
    log = tmp_path / "routing.jsonl"
    record_decision("task A", decided_class="technical", score=0.4, margin=0.1, uncertain=False, log_path=log)
    record_outcome("task A", outcome="negative", log_path=log)
    record_outcome("task A", outcome="positive", log_path=log)  # later outcome overrides
    agg = aggregate(log_path=log)
    assert agg["technical"].outcomes.get("positive") == 1
    # negative should not leak because we only keep the latest per task_hash
    assert "negative" not in agg["technical"].outcomes


def test_task_hash_is_stable() -> None:
    a = task_hash("Fix The Crash")
    b = task_hash("  fix the crash  ")
    assert a == b


def test_aggregate_missing_file_is_empty(tmp_path: Path) -> None:
    assert aggregate(log_path=tmp_path / "does-not-exist.jsonl") == {}
