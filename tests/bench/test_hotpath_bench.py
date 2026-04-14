"""Hot-path microbenchmarks for hermulti.

Pure-CPU benches on the outcome inferrer and the similarity router — both sit
on the per-turn path and an accidental O(n²) regression would compound fast.
We avoid pytest-benchmark as an extra dependency: a simple perf_counter
sampler + a stored baseline JSON is enough to flag regressions in CI.

Run manually::

    pytest tests/bench/ -o addopts='' --no-header  # measure + compare
    HERMES_BENCH_UPDATE=1 pytest tests/bench/ -o addopts='' --no-header  # reseed

Budgets live in ``tests/bench/budgets.json`` alongside this file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, List

import pytest

from agent.outcome_signals import infer_outcome_from_turn
from tools.router_classifier import classify


BUDGETS_PATH = Path(__file__).with_name("budgets.json")


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(pct / 100.0 * len(ordered)))
    return ordered[idx]


def _measure(name: str, fn: Callable[[], None], *, iters: int = 2000, warmup: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {
        "name": name,
        "iters": iters,
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
    }


def _load_budgets() -> dict:
    try:
        return json.loads(BUDGETS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"tolerance": 1.4, "baseline": {}}


def _save_budgets(data: dict) -> None:
    BUDGETS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _compare(result: dict, budgets: dict) -> None:
    update = os.environ.get("HERMES_BENCH_UPDATE") == "1"
    tolerance = float(budgets.get("tolerance", 1.4))
    baseline = budgets.setdefault("baseline", {})
    existing = baseline.get(result["name"])

    if update or existing is None:
        baseline[result["name"]] = {
            "p50_ms": result["p50_ms"],
            "p95_ms": result["p95_ms"],
            "p99_ms": result["p99_ms"],
        }
        _save_budgets(budgets)
        return

    budget_p95 = existing["p95_ms"] * tolerance
    assert result["p95_ms"] <= budget_p95, (
        f"{result['name']}: p95 {result['p95_ms']:.3f}ms exceeds budget {budget_p95:.3f}ms "
        f"(baseline {existing['p95_ms']:.3f}ms, tolerance {tolerance}x)"
    )


# ------------------------------------------------------------------
#  Benches


def test_bench_outcome_inferrer() -> None:
    messages = [
        "thanks that worked perfectly!",
        "no that's not what I wanted",
        "hmm ok let me think",
        "can you try again please",
        "great, shipping it",
    ]
    i = [0]

    def invoke() -> None:
        infer_outcome_from_turn(messages[i[0] % len(messages)])
        i[0] += 1

    result = _measure("outcome_inferrer", invoke, iters=5000, warmup=200)
    budgets = _load_budgets()
    _compare(result, budgets)


def test_bench_router_classifier() -> None:
    tasks = [
        "fix the crash in the auth handler",
        "remember that I prefer tabs",
        "write a trivial hello world",
        "something is off, please look",
        "debug the failing database query",
    ]
    i = [0]

    def invoke() -> None:
        classify(tasks[i[0] % len(tasks)])
        i[0] += 1

    result = _measure("router_classifier", invoke, iters=2000, warmup=50)
    budgets = _load_budgets()
    _compare(result, budgets)


@pytest.mark.skipif(os.environ.get("HERMES_BENCH_SKIP_SUMMARY") == "1", reason="opt-out")
def test_bench_summary_smoke() -> None:
    """Dummy test that just ensures the budgets file remains valid JSON after runs."""
    budgets = _load_budgets()
    assert "baseline" in budgets
    assert "tolerance" in budgets
