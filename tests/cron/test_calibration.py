"""Tests for cron.calibration — outcome-inference drift detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from cron.calibration import (
    CalibrationResult,
    ConfusionMatrix,
    LabeledTurn,
    heuristic_judge,
    run_calibration_pass,
)


def _turn(user_msg: str, inferred: Optional[str], judge: Optional[str] = None) -> LabeledTurn:
    return LabeledTurn(
        session_id="s",
        user_msg=user_msg,
        assistant_msg=None,
        inferred=inferred,
        judge=judge,
    )


def test_confusion_matrix_perfect_agreement() -> None:
    cm = ConfusionMatrix()
    cm.record("positive", "positive")
    cm.record("negative", "negative")
    cm.record(None, None)
    assert cm.agreement == pytest.approx(1.0)
    assert cm.precision("positive") == pytest.approx(1.0)
    assert cm.recall("negative") == pytest.approx(1.0)


def test_confusion_matrix_partial() -> None:
    cm = ConfusionMatrix()
    cm.record("positive", "positive")
    cm.record("positive", "negative")
    cm.record("negative", "negative")
    cm.record("negative", "positive")
    assert cm.agreement == pytest.approx(0.5)
    assert cm.precision("positive") == pytest.approx(0.5)
    assert cm.recall("positive") == pytest.approx(0.5)


def test_run_pass_writes_log_row(tmp_path: Path) -> None:
    log = tmp_path / "calib.jsonl"
    samples = [
        _turn("thanks that works", inferred="positive", judge="positive"),
        _turn("great", inferred="positive", judge="positive"),
        _turn("no wrong", inferred="negative", judge="negative"),
        _turn("hmm", inferred=None, judge=None),
        _turn("ok", inferred=None, judge=None),
    ]
    result = run_calibration_pass(samples, log_path=log)
    assert result.sample_size == 5
    assert result.agreement == pytest.approx(1.0)
    assert result.alert is None
    rows = log.read_text().strip().split("\n")
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["agreement"] == pytest.approx(1.0)


def test_run_pass_alerts_on_low_agreement(tmp_path: Path) -> None:
    log = tmp_path / "calib.jsonl"
    # Inferrer says "positive" for everything, judge disagrees on 4/5.
    samples = [
        _turn("hello", inferred="positive", judge="negative"),
        _turn("hi", inferred="positive", judge="negative"),
        _turn("yes", inferred="positive", judge="negative"),
        _turn("ok", inferred="positive", judge="negative"),
        _turn("thanks", inferred="positive", judge="positive"),
    ]
    result = run_calibration_pass(samples, log_path=log)
    assert result.agreement == pytest.approx(0.2)
    assert result.alert is not None
    assert "agreement" in result.alert


def test_run_pass_alerts_on_low_class_precision(tmp_path: Path) -> None:
    log = tmp_path / "calib.jsonl"
    # 10 positive predictions, only 4 are actually positive → precision 0.4
    samples = []
    for _ in range(4):
        samples.append(_turn("thanks", inferred="positive", judge="positive"))
    for _ in range(6):
        samples.append(_turn("no thanks", inferred="positive", judge="negative"))
    # Pad neutrals so overall agreement is >= min_agreement
    for _ in range(14):
        samples.append(_turn("hmm", inferred=None, judge=None))
    result = run_calibration_pass(samples, log_path=log, min_agreement=0.5, min_class_precision=0.6)
    assert result.precision["positive"] == pytest.approx(0.4)
    assert result.alert is not None
    assert "positive" in result.alert


def test_run_pass_no_alert_on_trivial_sample(tmp_path: Path) -> None:
    log = tmp_path / "calib.jsonl"
    samples = [
        _turn("x", inferred="positive", judge="negative"),
        _turn("y", inferred="positive", judge="negative"),
    ]
    result = run_calibration_pass(samples, log_path=log)
    # Below minimum n=5 threshold; no alert regardless of disagreement.
    assert result.alert is None


def test_judge_applied_when_missing(tmp_path: Path) -> None:
    log = tmp_path / "calib.jsonl"
    samples = [
        _turn("thanks", inferred="positive", judge=None),
        _turn("broken", inferred="negative", judge=None),
        _turn("hmm", inferred=None, judge=None),
        _turn("perfect", inferred="positive", judge=None),
        _turn("wrong", inferred="negative", judge=None),
    ]
    result = run_calibration_pass(samples, judge=heuristic_judge, log_path=log)
    # Heuristic judge should label all four clearly-flagged ones matching the inferrer.
    assert result.agreement >= 0.8
