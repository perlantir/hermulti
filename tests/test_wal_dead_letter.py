"""Unit tests for WAL dead-lettering of 4xx replay failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from agent.hipp0_memory_provider import Hipp0MemoryProvider


class _FakeResp:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode() if text else b""

    def json(self) -> Dict[str, Any]:
        return json.loads(self.text) if self.text else {}


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in used by _drain_wal()."""

    def __init__(self, responses: Dict[str, _FakeResp]) -> None:
        self._responses = responses
        self.calls: list = []

    async def post(self, path, *, json=None, params=None, headers=None):  # noqa: A002
        self.calls.append((path, json))
        return self._responses.get(path, _FakeResp(200, "{}"))

    async def aclose(self) -> None:
        pass


def _make_provider(tmp_path: Path, client: _FakeClient) -> Hipp0MemoryProvider:
    return Hipp0MemoryProvider(
        base_url="http://127.0.0.1:9",
        api_key="test",
        project_id="p",
        agent_name="a",
        agent_id="id",
        pending_wal_path=tmp_path / "pending.jsonl",
        memory_md_path=tmp_path / "MEMORY.md",
        client=client,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_drain_moves_4xx_entry_to_dead_letter(tmp_path: Path) -> None:
    client = _FakeClient({"/api/capture": _FakeResp(400, '{"error":"bad"}')})
    provider = _make_provider(tmp_path, client)

    # Pre-populate WAL with a capture entry that will 4xx on replay.
    provider._wal_append(
        {
            "kind": "capture",
            "path": "/api/capture",
            "body": {"conversation": "x"},
            "params": None,
            "headers": None,
            "timestamp": 123.0,
            "error": "prior outage",
        }
    )
    assert provider.wal_size() == 1
    assert provider.dead_letter_size() == 0

    await provider._drain_wal()

    # WAL emptied; dead-letter has the entry with enrichment fields.
    assert provider.wal_size() == 0
    assert provider.dead_letter_size() == 1
    dl_lines = (tmp_path / "dead_letter.jsonl").read_text().splitlines()
    dl_entry = json.loads(dl_lines[0])
    assert dl_entry["status_code"] == 400
    assert dl_entry["error_body"] == '{"error":"bad"}'
    assert dl_entry["kind"] == "capture"
    assert "dead_letter_timestamp" in dl_entry


@pytest.mark.asyncio
async def test_dead_letter_entries_are_not_retried(tmp_path: Path) -> None:
    # Second drain call on a 4xx-emptied WAL should not re-POST anything.
    client = _FakeClient({"/api/capture": _FakeResp(400, "bad")})
    provider = _make_provider(tmp_path, client)
    provider._wal_append(
        {
            "kind": "capture",
            "path": "/api/capture",
            "body": {"conversation": "x"},
            "timestamp": 1.0,
        }
    )
    await provider._drain_wal()
    assert len(client.calls) == 1

    # Second drain — WAL is empty, dead-letter retained but ignored.
    await provider._drain_wal()
    assert len(client.calls) == 1  # no additional POST
    assert provider.dead_letter_size() == 1


@pytest.mark.asyncio
async def test_drain_success_does_not_dead_letter(tmp_path: Path) -> None:
    client = _FakeClient({"/api/capture": _FakeResp(200, "{}")})
    provider = _make_provider(tmp_path, client)
    provider._wal_append(
        {
            "kind": "capture",
            "path": "/api/capture",
            "body": {"conversation": "ok"},
            "timestamp": 1.0,
        }
    )
    await provider._drain_wal()
    assert provider.wal_size() == 0
    assert provider.dead_letter_size() == 0


def test_wal_status_reports_depths(tmp_path: Path, capsys, monkeypatch) -> None:
    from hermes_cli.wal import wal_status

    agents_root = tmp_path / "agents"
    agent_dir = agents_root / "alice"
    agent_dir.mkdir(parents=True)
    (agent_dir / "pending.jsonl").write_text(
        json.dumps({"kind": "capture", "timestamp": 1.0}) + "\n"
    )
    (agent_dir / "dead_letter.jsonl").write_text(
        json.dumps({"kind": "capture", "dead_letter_timestamp": 2.0}) + "\n"
        + json.dumps({"kind": "compile", "dead_letter_timestamp": 3.0}) + "\n"
    )

    rc = wal_status(hermes_home=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "1" in out  # wal depth
    assert "2" in out  # dead-letter depth
