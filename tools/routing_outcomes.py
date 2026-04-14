"""Routing-outcomes feedback edge.

Records each router decision + the downstream outcome so we can tell,
after the fact, whether the ``technical`` or ``user`` class actually
produced better completions than a plain ``ambiguous`` fallback. Nightly
aggregation over this log is the signal for tuning ``router_classifier``
seed sentences and the ``margin`` threshold.

Stored as JSONL at ``~/.hermes/routing_outcomes.jsonl`` — one row per
routing decision, optionally updated in place when the outcome lands
(the linkage is ``task_hash`` which the caller recomputes). For now we
append a second row with ``event="outcome"`` rather than mutating the
original; aggregation code picks the latest matching row by hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _default_log_path() -> Path:
    override = os.environ.get("HERMES_ROUTING_OUTCOMES_LOG")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "routing_outcomes.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_hash(task_description: str) -> str:
    return hashlib.sha256((task_description or "").strip().lower().encode()).hexdigest()[:16]


@dataclass
class RoutingRow:
    event: str  # "decision" | "outcome"
    task_hash: str
    timestamp: str
    decided_class: Optional[str] = None
    score: Optional[float] = None
    margin: Optional[float] = None
    uncertain: Optional[bool] = None
    outcome: Optional[str] = None
    tokens_used: Optional[int] = None


def record_decision(
    task_description: str,
    decided_class: str,
    score: float,
    margin: float,
    uncertain: bool,
    log_path: Optional[Path] = None,
) -> str:
    th = task_hash(task_description)
    row = RoutingRow(
        event="decision",
        task_hash=th,
        timestamp=_now_iso(),
        decided_class=decided_class,
        score=score,
        margin=margin,
        uncertain=uncertain,
    )
    _append(row, log_path or _default_log_path())
    return th


def record_outcome(
    task_description: str,
    outcome: Optional[str],
    tokens_used: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> None:
    th = task_hash(task_description)
    row = RoutingRow(
        event="outcome",
        task_hash=th,
        timestamp=_now_iso(),
        outcome=outcome,
        tokens_used=tokens_used,
    )
    _append(row, log_path or _default_log_path())


def _append(row: RoutingRow, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(asdict(row)) + "\n")
    except OSError as err:
        logger.warning("[routing-outcomes] failed to append: %s", err)


def _iter_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    try:
        with path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError as err:
        logger.warning("[routing-outcomes] failed to read: %s", err)


@dataclass
class ClassAggregate:
    count: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)  # "positive"/"negative"/"unknown" → count


def aggregate(log_path: Optional[Path] = None) -> Dict[str, ClassAggregate]:
    """Reduce the log to per-class outcome counts.

    For each task_hash we use the latest decision row's class and the latest
    outcome row's outcome. If no outcome row exists for a decision, that
    task contributes to count but not to outcome totals.
    """
    path = log_path or _default_log_path()
    latest_decision: Dict[str, dict] = {}
    latest_outcome: Dict[str, dict] = {}
    for row in _iter_rows(path):
        th = row.get("task_hash")
        if not th:
            continue
        if row.get("event") == "decision":
            latest_decision[th] = row
        elif row.get("event") == "outcome":
            latest_outcome[th] = row

    out: Dict[str, ClassAggregate] = {}
    for th, dec in latest_decision.items():
        cls = dec.get("decided_class") or "ambiguous"
        agg = out.setdefault(cls, ClassAggregate())
        agg.count += 1
        outcome_row = latest_outcome.get(th)
        if outcome_row:
            label = outcome_row.get("outcome") or "unknown"
            agg.outcomes[label] = agg.outcomes.get(label, 0) + 1
    return out


def positive_rate(agg: ClassAggregate) -> Optional[float]:
    total = sum(agg.outcomes.values())
    if total == 0:
        return None
    return agg.outcomes.get("positive", 0) / total
