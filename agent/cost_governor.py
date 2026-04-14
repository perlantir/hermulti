"""
Per-project daily LLM cost governor.

Tracks spend in a small JSON file (default: ``~/.hermes/cost_state.json``)
and enforces a configurable daily budget. ``check_budget`` raises
``BudgetExceeded`` once the project has consumed its daily allowance;
``record_spend`` must be called after each LLM response is received so
the counter reflects actual usage.

Budget configuration comes from env vars (``HERMES_DAILY_BUDGET_USD``
is the global default; per-project overrides live in the state file under
``budgets[project_id]``). The governor is intentionally process-local —
multi-process deployments should back this with Redis, but that's out of
scope for the primitive. File writes are atomic (write-temp + rename) to
survive concurrent sync callers within one process.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised by ``check_budget`` when a project is over its daily cap."""

    def __init__(self, project_id: str, spent_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"project {project_id} has spent ${spent_usd:.4f} today, "
            f"exceeding cap ${cap_usd:.4f}"
        )
        self.project_id = project_id
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


@dataclass
class BudgetStatus:
    project_id: str
    spent_today_usd: float
    cap_usd: Optional[float]
    allowed: bool
    period_start: str  # ISO date (YYYY-MM-DD) in UTC


@dataclass
class _State:
    # day_utc -> { project_id -> spent_usd }
    spend: dict[str, dict[str, float]] = field(default_factory=dict)
    # project_id -> cap_usd
    budgets: dict[str, float] = field(default_factory=dict)


def _default_state_path() -> Path:
    override = os.environ.get("HERMES_COST_STATE_PATH")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "cost_state.json"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _env_default_cap() -> Optional[float]:
    raw = os.environ.get("HERMES_DAILY_BUDGET_USD")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


class CostGovernor:
    """Thread-safe, file-backed cost governor."""

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self._path = state_path or _default_state_path()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    #  State I/O

    def _load(self) -> _State:
        if not self._path.exists():
            return _State()
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return _State()
        return _State(
            spend=dict(raw.get("spend", {})),
            budgets=dict(raw.get("budgets", {})),
        )

    def _save(self, state: _State) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp + rename.
        fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), prefix=".cost_state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"spend": state.spend, "budgets": state.budgets}, handle)
            os.replace(tmp_path, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _prune(self, state: _State) -> None:
        """Drop spend entries for days other than today (bounded size)."""
        today = _today_utc()
        state.spend = {today: state.spend.get(today, {})}

    # ------------------------------------------------------------------
    #  Public API

    def set_budget(self, project_id: str, cap_usd: Optional[float]) -> None:
        with self._lock:
            state = self._load()
            if cap_usd is None:
                state.budgets.pop(project_id, None)
            else:
                if cap_usd <= 0:
                    raise ValueError("cap_usd must be positive")
                state.budgets[project_id] = float(cap_usd)
            self._save(state)

    def status(self, project_id: str) -> BudgetStatus:
        with self._lock:
            state = self._load()
            today = _today_utc()
            spent = float(state.spend.get(today, {}).get(project_id, 0.0))
            cap = state.budgets.get(project_id)
            if cap is None:
                cap = _env_default_cap()
            allowed = cap is None or spent < cap
            return BudgetStatus(
                project_id=project_id,
                spent_today_usd=spent,
                cap_usd=cap,
                allowed=allowed,
                period_start=today,
            )

    def check_budget(self, project_id: str) -> None:
        """Raise BudgetExceeded if the project is over cap. Otherwise return."""
        s = self.status(project_id)
        if not s.allowed:
            raise BudgetExceeded(project_id, s.spent_today_usd, s.cap_usd or 0.0)

    def record_spend(self, project_id: str, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        with self._lock:
            state = self._load()
            self._prune(state)
            today = _today_utc()
            day = state.spend.setdefault(today, {})
            day[project_id] = float(day.get(project_id, 0.0)) + float(cost_usd)
            self._save(state)


# Module-level singleton for easy wiring.
_GOVERNOR: Optional[CostGovernor] = None
_GOVERNOR_LOCK = threading.Lock()


def get_governor() -> CostGovernor:
    global _GOVERNOR
    with _GOVERNOR_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = CostGovernor()
        return _GOVERNOR


def reset_governor_for_tests(state_path: Optional[Path] = None) -> CostGovernor:
    """Force a fresh governor. Tests only."""
    global _GOVERNOR
    with _GOVERNOR_LOCK:
        _GOVERNOR = CostGovernor(state_path=state_path)
        return _GOVERNOR


# Rough pricing table (USD per 1K tokens). Intentionally coarse — the goal is
# budget *gating*, not accounting. Callers that care about precise cost should
# compute their own and pass cost_usd directly to record_spend.
_PRICING_PER_1K = {
    "claude-opus": (0.015, 0.075),
    "claude-sonnet": (0.003, 0.015),
    "claude-haiku": (0.001, 0.005),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5": (0.0005, 0.0015),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    key = next((k for k in _PRICING_PER_1K if k in (model or "").lower()), None)
    if key is None:
        # Default to sonnet-class pricing as a conservative estimate.
        in_rate, out_rate = _PRICING_PER_1K["claude-sonnet"]
    else:
        in_rate, out_rate = _PRICING_PER_1K[key]
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


def current_project_id() -> Optional[str]:
    """Best-effort project-id lookup for call-sites that don't pass one through.

    Reads HERMES_PROJECT_ID from the environment. Returning None disables
    governance for that call — governance is advisory for anonymous calls.
    """
    pid = os.environ.get("HERMES_PROJECT_ID")
    if pid and pid.strip():
        return pid.strip()
    return None
