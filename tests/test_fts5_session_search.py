"""Tests for the v9 expanded FTS5 index + top-10 recent cache.

Verifies:
- messages_fts carries session_id / role / timestamp as UNINDEXED columns
- triggers keep those columns in sync on INSERT and UPDATE
- search_messages returns results after INSERT and reflects UPDATEd content
- list_sessions_rich() top-10 response is served from the 5-min TTL cache
"""

from __future__ import annotations

import time

import pytest

from hermes_state import SCHEMA_VERSION, SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "fts5.db")
    yield d
    d.close()


def test_schema_bumped_to_v9(db):
    cursor = db._conn.execute("SELECT version FROM schema_version")
    assert cursor.fetchone()[0] == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 9


def test_fts_table_has_metadata_columns(db):
    # Probe with the expanded column set — must not raise.
    cursor = db._conn.execute(
        "SELECT content, session_id, role, timestamp FROM messages_fts LIMIT 0"
    )
    names = {c[0] for c in cursor.description}
    assert {"content", "session_id", "role", "timestamp"} <= names


def test_insert_is_indexed(db):
    db.create_session("s1", "cli")
    db.append_message("s1", "user", "deploy the kubernetes cluster today")
    results = db.search_messages("kubernetes")
    assert len(results) == 1
    assert results[0]["session_id"] == "s1"


def test_fts_metadata_matches_messages(db):
    db.create_session("s2", "cli")
    db.append_message("s2", "assistant", "hello from docker world")
    row = db._conn.execute(
        "SELECT session_id, role FROM messages_fts WHERE messages_fts MATCH ?",
        ("docker",),
    ).fetchone()
    assert row is not None
    assert row["session_id"] == "s2"
    assert row["role"] == "assistant"


def test_update_reflected_in_fts(db):
    db.create_session("s3", "cli")
    db.append_message("s3", "user", "initial payload mentioning redis")
    msg_id = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = 's3'"
    ).fetchone()[0]

    # Searching the old term finds it.
    assert db.search_messages("redis")

    # Update content via a raw UPDATE (fires the UPDATE trigger).
    def _do(conn):
        conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("rewritten payload mentioning postgres", msg_id),
        )
    db._execute_write(_do)

    # Old term gone; new term found.
    assert db.search_messages("redis") == []
    results = db.search_messages("postgres")
    assert len(results) == 1
    assert results[0]["session_id"] == "s3"


def test_delete_trigger_removes_from_fts(db):
    db.create_session("s4", "cli")
    db.append_message("s4", "user", "ephemeral nginx content")
    assert db.search_messages("nginx")

    def _do(conn):
        conn.execute("DELETE FROM messages WHERE session_id = 's4'")
    db._execute_write(_do)

    assert db.search_messages("nginx") == []


def test_recent_cache_hits_within_ttl(db):
    db.create_session("s5", "cli")
    db.append_message("s5", "user", "first question")

    first = db.list_sessions_rich(limit=10)
    assert first and first[0]["id"] == "s5"

    # Write a new session WITHOUT busting the cache — top-10 is still the
    # stale snapshot because the TTL hasn't expired.
    db.create_session("s6", "cli")
    db.append_message("s6", "user", "second question")

    second = db.list_sessions_rich(limit=10)
    ids = [s["id"] for s in second]
    assert "s6" not in ids, "cache must be hit within TTL"
    assert ids == [s["id"] for s in first]


def test_recent_cache_expires(db):
    db.create_session("s7", "cli")
    db.append_message("s7", "user", "seed")

    _ = db.list_sessions_rich(limit=10)

    # Force-expire the cache by rewinding expiry.
    db._RECENT_CACHE_TTL_S = 0.0
    # Prior entries' expires_at is already in the past after zeroing,
    # but to be safe also clear the dict.
    db._recent_cache.clear()

    db.create_session("s8", "cli")
    db.append_message("s8", "user", "fresh")

    fresh = db.list_sessions_rich(limit=10)
    assert "s8" in [s["id"] for s in fresh]


def test_recent_cache_bypassed_for_large_limit(db):
    db.create_session("s9", "cli")
    db.append_message("s9", "user", "msg")

    # limit > 10 must not hit the cache path; verify by checking cache dict
    # remains empty after the call.
    db._recent_cache.clear()
    _ = db.list_sessions_rich(limit=50)
    assert db._recent_cache == {}

    _ = db.list_sessions_rich(limit=10)
    assert db._recent_cache != {}
