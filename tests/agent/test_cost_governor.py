"""Tests for agent.cost_governor — budget gating primitive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.cost_governor import (
    BudgetExceeded,
    CostGovernor,
    estimate_cost_usd,
)


@pytest.fixture
def gov(tmp_path: Path) -> CostGovernor:
    return CostGovernor(state_path=tmp_path / "cost.json")


def test_no_budget_set_allows_all(gov: CostGovernor) -> None:
    gov.record_spend("proj-a", 99.0)
    gov.check_budget("proj-a")  # does not raise
    s = gov.status("proj-a")
    assert s.allowed is True
    assert s.cap_usd is None
    assert s.spent_today_usd == pytest.approx(99.0)


def test_budget_enforced_per_project(gov: CostGovernor) -> None:
    gov.set_budget("proj-a", cap_usd=1.0)
    gov.record_spend("proj-a", 0.5)
    gov.check_budget("proj-a")  # 0.5 < 1.0, ok

    gov.record_spend("proj-a", 0.6)
    with pytest.raises(BudgetExceeded) as exc:
        gov.check_budget("proj-a")
    assert exc.value.project_id == "proj-a"
    assert exc.value.spent_usd == pytest.approx(1.1)
    assert exc.value.cap_usd == pytest.approx(1.0)

    # Other project unaffected.
    gov.check_budget("proj-b")


def test_set_budget_rejects_non_positive(gov: CostGovernor) -> None:
    with pytest.raises(ValueError):
        gov.set_budget("proj-a", cap_usd=0.0)


def test_state_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    g1 = CostGovernor(state_path=path)
    g1.set_budget("proj-a", 2.0)
    g1.record_spend("proj-a", 1.5)

    g2 = CostGovernor(state_path=path)
    s = g2.status("proj-a")
    assert s.spent_today_usd == pytest.approx(1.5)
    assert s.cap_usd == pytest.approx(2.0)


def test_state_file_has_0o600_perms(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    g = CostGovernor(state_path=path)
    g.record_spend("proj-a", 0.01)
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600 got {oct(mode)}"


def test_corrupt_state_file_resets_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    path.write_text("{not valid json")
    g = CostGovernor(state_path=path)
    # Should not raise; treated as empty state.
    s = g.status("proj-a")
    assert s.spent_today_usd == 0.0


def test_prune_keeps_only_today(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seed file with yesterday's entries.
    path.write_text(json.dumps({
        "spend": {"1999-01-01": {"proj-a": 999.0}},
        "budgets": {},
    }))
    g = CostGovernor(state_path=path)
    g.record_spend("proj-a", 0.1)  # triggers prune
    raw = json.loads(path.read_text())
    assert "1999-01-01" not in raw["spend"]
    assert "proj-a" in next(iter(raw["spend"].values()))


def test_record_spend_ignores_zero_or_negative(gov: CostGovernor) -> None:
    gov.record_spend("proj-a", 0.0)
    gov.record_spend("proj-a", -5.0)
    assert gov.status("proj-a").spent_today_usd == 0.0


def test_env_default_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_DAILY_BUDGET_USD", "0.10")
    g = CostGovernor(state_path=tmp_path / "cost.json")
    g.record_spend("proj-a", 0.15)
    with pytest.raises(BudgetExceeded):
        g.check_budget("proj-a")


def test_estimate_cost_sonnet() -> None:
    c = estimate_cost_usd("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    # 1000/1000 * 0.003 + 500/1000 * 0.015 = 0.003 + 0.0075 = 0.0105
    assert c == pytest.approx(0.0105, rel=1e-4)


def test_estimate_cost_unknown_model_defaults_to_sonnet() -> None:
    a = estimate_cost_usd("mystery-model-xyz", 1000, 1000)
    b = estimate_cost_usd("claude-sonnet", 1000, 1000)
    assert a == pytest.approx(b)
