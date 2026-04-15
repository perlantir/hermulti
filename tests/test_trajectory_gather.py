"""Tests for TrajectoryCompressor.compress_many_async().

Validates that the batch helper uses asyncio.gather with a Semaphore(10)
so a batch of 10 slow items completes in ~1/10 of the sequential time,
and that results are returned in the same order as the inputs.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from trajectory_compressor import TrajectoryCompressor


ITEM_DELAY_S = 0.3
BATCH_SIZE = 10


def _make_compressor() -> TrajectoryCompressor:
    """Build a TrajectoryCompressor skipping real __init__.

    We stub ``process_entry_async`` directly on the instance, so most of
    the constructor's work (tokenizer, API client) is irrelevant.
    """
    comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
    comp.config = MagicMock()
    return comp


@pytest.mark.asyncio
async def test_batch_uses_gather_and_preserves_order():
    comp = _make_compressor()

    async def slow_process(entry):
        await asyncio.sleep(ITEM_DELAY_S)
        # Echo the idx back so we can verify order preservation.
        return ({"idx": entry["idx"]}, entry["idx"])

    comp.process_entry_async = slow_process

    entries = [{"idx": i} for i in range(BATCH_SIZE)]

    t0 = time.perf_counter()
    results = await comp.compress_many_async(entries)
    elapsed = time.perf_counter() - t0

    # Semaphore(10) with 10 items → all run concurrently → ~ITEM_DELAY_S.
    # Sequential would be 10 * ITEM_DELAY_S = 3.0s.  Assert <= 40% of
    # sequential (generous slack for CI jitter).
    sequential = BATCH_SIZE * ITEM_DELAY_S
    assert elapsed < sequential * 0.4, (
        f"expected parallel execution (<{sequential * 0.4:.2f}s), got {elapsed:.2f}s"
    )

    # Order preserved.
    assert [r[0]["idx"] for r in results] == list(range(BATCH_SIZE))
    assert [r[1] for r in results] == list(range(BATCH_SIZE))


@pytest.mark.asyncio
async def test_batch_respects_semaphore_cap():
    """If we hand compress_many_async 20 items, no more than 10 should be
    in flight at any moment — the 20 items should take ~2x ITEM_DELAY_S."""
    comp = _make_compressor()

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow_process(entry):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            await asyncio.sleep(ITEM_DELAY_S)
            return ({"idx": entry["idx"]}, entry["idx"])
        finally:
            async with lock:
                in_flight -= 1

    comp.process_entry_async = slow_process
    entries = [{"idx": i} for i in range(20)]

    t0 = time.perf_counter()
    results = await comp.compress_many_async(entries)
    elapsed = time.perf_counter() - t0

    assert peak <= 10, f"semaphore cap exceeded: peak={peak}"
    # Two batches of 10 → ~2*ITEM_DELAY_S.  Allow generous slack.
    assert elapsed < 4 * ITEM_DELAY_S
    assert [r[0]["idx"] for r in results] == list(range(20))


@pytest.mark.asyncio
async def test_batch_empty_input():
    comp = _make_compressor()
    comp.process_entry_async = MagicMock()  # Must not be called.
    assert await comp.compress_many_async([]) == []
    comp.process_entry_async.assert_not_called()
