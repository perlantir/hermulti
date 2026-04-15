"""Outcome-inference calibration pass.

Periodically samples recent sessions with an *inferred* outcome label, asks a
judge (rule-based or LLM) to produce a ground-truth label for the same turn,
and compares the two. Emits a confusion matrix + agreement metrics to the
calibration log and raises a high-severity alert via ``reflection_log`` when
inferred-vs-true agreement drifts below configured thresholds.

This is the guardrail that protects Phase 1: if the heuristic inference in
``agent.outcome_signals`` stops tracking what users actually meant, trust
deltas computed downstream poison the learning loop. We'd rather learn that
the heuristics drifted than watch context-quality silently decay.

The calibration log lives at ``~/.hermes/calibration_log.jsonl``. Each row is
one pass: ``{timestamp, sample_size, agreement, precision_per_class,
recall_per_class, alert}``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from agent.outcome_signals import infer_outcome_from_turn

logger = logging.getLogger(__name__)

# Default thresholds — can be overridden via env vars. Below these, we emit
# an alert row and flag that the heuristics need retuning.
DEFAULT_MIN_AGREEMENT = 0.70
DEFAULT_MIN_CLASS_PRECISION = 0.60

_CALIBRATION_LOG_PATH = Path(
    os.environ.get(
        "HERMES_CALIBRATION_LOG",
        str(Path.home() / ".hermes" / "calibration_log.jsonl"),
    )
)


@dataclass
class LabeledTurn:
    """One calibration sample: a turn + its inferred label + judge label."""

    session_id: str
    user_msg: Optional[str]
    assistant_msg: Optional[str]
    inferred: Optional[str]
    judge: Optional[str]


@dataclass
class ConfusionMatrix:
    # true × predicted, with the third class ``None`` rolled into 'neutral'.
    labels: Sequence[str] = field(default_factory=lambda: ("positive", "neutral", "negative"))
    # matrix[true_idx][pred_idx]
    matrix: List[List[int]] = field(default_factory=lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def record(self, true_label: Optional[str], predicted_label: Optional[str]) -> None:
        ti = self._idx(true_label)
        pi = self._idx(predicted_label)
        self.matrix[ti][pi] += 1

    def _idx(self, label: Optional[str]) -> int:
        if label == "positive":
            return 0
        if label == "negative":
            return 2
        return 1  # neutral / None

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.matrix)

    @property
    def agreement(self) -> float:
        total = self.total
        if total == 0:
            return 1.0
        diag = sum(self.matrix[i][i] for i in range(3))
        return diag / total

    def precision(self, label: str) -> float:
        i = self._idx(label)
        col_total = sum(self.matrix[r][i] for r in range(3))
        if col_total == 0:
            return 1.0  # no predictions of this class → vacuously precise
        return self.matrix[i][i] / col_total

    def recall(self, label: str) -> float:
        i = self._idx(label)
        row_total = sum(self.matrix[i])
        if row_total == 0:
            return 1.0
        return self.matrix[i][i] / row_total


# ------------------------------------------------------------------
#  Judges


Judge = Callable[[LabeledTurn], Optional[str]]


def heuristic_judge(turn: LabeledTurn) -> Optional[str]:
    """Rule-based judge. Mirrors the inferrer but reads a broader context.

    Used as a test double and a safe fallback when no LLM is available; the
    intent of calibration is comparing two *different* labeling strategies,
    so in production this should be swapped for ``llm_judge``.
    """
    # A slightly broader phrase set than the production inferrer — differs
    # intentionally so calibration produces a non-trivial signal.
    text = (turn.user_msg or "").lower()
    if any(m in text for m in ("thanks", "perfect", "great", "exactly", "works")):
        return "positive"
    if any(m in text for m in ("wrong", "no,", "undo", "revert", "broken", "error")):
        return "negative"
    return None


# ------------------------------------------------------------------
#  Calibration pass


@dataclass
class CalibrationResult:
    sample_size: int
    agreement: float
    precision: dict
    recall: dict
    alert: Optional[str]
    timestamp: str


def run_calibration_pass(
    samples: Iterable[LabeledTurn],
    judge: Judge = heuristic_judge,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    min_class_precision: float = DEFAULT_MIN_CLASS_PRECISION,
    log_path: Optional[Path] = None,
) -> CalibrationResult:
    """Run one calibration pass and persist the result row.

    ``samples`` should already have ``inferred`` set (from the production
    inferrer). This function applies ``judge`` to produce the ground-truth
    label, computes the confusion matrix, and persists the metrics row.
    """
    cm = ConfusionMatrix()
    n = 0
    for turn in samples:
        if turn.judge is None:
            turn.judge = judge(turn)
        cm.record(true_label=turn.judge, predicted_label=turn.inferred)
        n += 1

    agreement = cm.agreement
    precision = {label: cm.precision(label) for label in cm.labels}
    recall = {label: cm.recall(label) for label in cm.labels}

    alert: Optional[str] = None
    if n >= 5:  # don't alert on trivial samples
        if agreement < min_agreement:
            alert = f"agreement {agreement:.2%} below threshold {min_agreement:.0%}"
        else:
            bad_class = next(
                (lbl for lbl, p in precision.items() if p < min_class_precision),
                None,
            )
            if bad_class is not None:
                alert = (
                    f"precision for class '{bad_class}' = {precision[bad_class]:.2%} "
                    f"below threshold {min_class_precision:.0%}"
                )

    result = CalibrationResult(
        sample_size=n,
        agreement=agreement,
        precision=precision,
        recall=recall,
        alert=alert,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    path = log_path or _CALIBRATION_LOG_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")
    except OSError as err:
        logger.warning("[calibration] failed to write log: %s", err)

    if alert:
        logger.warning("[calibration] ALERT — %s (n=%d)", alert, n)

    return result


def sample_recent_turns(db, limit: int = 100) -> List[LabeledTurn]:
    """Draw recent sessions with an outcome and infer the label on the last turn.

    ``db`` is a SessionDB-compatible object. The caller is responsible for
    passing a limit that matches the calibration cadence — the default is
    tuned for a weekly pass on a moderately busy agent.
    """
    samples: List[LabeledTurn] = []
    try:
        rows = db._execute_read(
            lambda c: c.execute(
                "SELECT id FROM sessions WHERE outcome IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        )
    except Exception as err:  # db missing / schema old
        logger.warning("[calibration] could not list sessions: %s", err)
        return samples

    for row in rows:
        session_id = row[0] if not hasattr(row, "keys") else row["id"]
        try:
            msg_rows = db._execute_read(
                lambda c, sid=session_id: c.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT 4",
                    (sid,),
                ).fetchall()
            )
        except Exception:
            continue
        user_msg = None
        assistant_msg = None
        for mrow in msg_rows:
            role = mrow[0] if not hasattr(mrow, "keys") else mrow["role"]
            content = mrow[1] if not hasattr(mrow, "keys") else mrow["content"]
            if role == "user" and user_msg is None:
                user_msg = content
            elif role == "assistant" and assistant_msg is None:
                assistant_msg = content
            if user_msg and assistant_msg:
                break
        inferred = infer_outcome_from_turn(user_msg, assistant_msg)
        samples.append(
            LabeledTurn(
                session_id=str(session_id),
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                inferred=inferred,
                judge=None,
            )
        )
    return samples
