"""`hermes wal status` — inspect HIPP0 provider WAL and dead-letter queues.

Walks every registered agent under HERMES_HOME and reports:
  - WAL depth (pending.jsonl entries)
  - dead-letter depth (dead_letter.jsonl entries)
  - oldest entry age across both

No retry / reset actions — inspection only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _iter_agent_dirs(root: Path) -> List[Path]:
    agents_root = root / "agents"
    if not agents_root.is_dir():
        return []
    return [p for p in sorted(agents_root.iterdir()) if p.is_dir()]


def _count_and_oldest(path: Path) -> Tuple[int, Optional[float]]:
    """Return (entry_count, oldest_timestamp) for a JSONL file. Missing -> (0, None)."""
    if not path.is_file():
        return 0, None
    count = 0
    oldest: Optional[float] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = record.get("timestamp") or record.get("dead_letter_timestamp")
            if isinstance(ts, (int, float)):
                if oldest is None or ts < oldest:
                    oldest = ts
    except OSError:
        return 0, None
    return count, oldest


def _fmt_age(ts: Optional[float], now: float) -> str:
    if ts is None:
        return "-"
    secs = max(0, int(now - ts))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


def wal_status(hermes_home: Optional[Path] = None) -> int:
    """Print WAL + dead-letter depth for every agent. Returns exit code."""
    root = hermes_home or _hermes_home()
    agent_dirs = _iter_agent_dirs(root)
    if not agent_dirs:
        print(f"No agents found under {root}/agents")
        return 0

    now = time.time()
    rows: List[Tuple[str, int, int, Optional[float]]] = []
    for agent_dir in agent_dirs:
        wal_n, wal_oldest = _count_and_oldest(agent_dir / "pending.jsonl")
        dl_n, dl_oldest = _count_and_oldest(agent_dir / "dead_letter.jsonl")
        candidates = [t for t in (wal_oldest, dl_oldest) if t is not None]
        oldest = min(candidates) if candidates else None
        rows.append((agent_dir.name, wal_n, dl_n, oldest))

    name_w = max(6, max(len(r[0]) for r in rows))
    print(f"{'agent':<{name_w}}  {'wal':>5}  {'dead':>5}  {'oldest':>7}")
    print("-" * (name_w + 24))
    for name, wal_n, dl_n, oldest in rows:
        print(f"{name:<{name_w}}  {wal_n:>5}  {dl_n:>5}  {_fmt_age(oldest, now):>7}")
    return 0
