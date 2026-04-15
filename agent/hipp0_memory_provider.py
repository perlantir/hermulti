"""HIPP0-backed memory provider for persistent Hermes agents.

This is the HTTP client that sits between a named persistent agent
(``hermes_cli.agent_registry``) and a running HIPP0 instance. It talks
the contract locked down in the ``feat/persistent-agents-hipp0`` task
brief — see ``HIPP0_REQUESTS.md`` at the repo root for any divergences
this file has had to flag while the sibling repo evolves.

The class implements the sync :class:`agent.memory_provider.MemoryProvider`
abstract base so Hermes runtime plumbing can hold it, but its *primary*
API is the set of ``async`` HTTP methods consumed directly by
:class:`tools.persistent_delegate_tool.PersistentDelegateTool`:

    - :meth:`start_session` / :meth:`end_session`
    - :meth:`capture` — fire-and-forget conversation ingestion
    - :meth:`compile` — task-aware context retrieval
    - :meth:`record_outcome` — reinforcement signal
    - :meth:`upsert_user_fact` — cross-agent user facts

Write-Ahead Log
---------------
Every async call is wrapped in a WAL. A failed request (connection
error or 5xx) is appended as one JSON line to the agent's
``pending.jsonl``. On the next successful call we drain that WAL
oldest-first, dropping lines as they replay. This is the mechanism
that prevents memory loss when HIPP0 is transiently down.

Degraded mode
-------------
If :meth:`compile` fails after retries we fall back to reading
``MEMORY.md`` off disk and return a :class:`CompiledContext` with
``degraded=True``. Callers (PersistentDelegateTool, gateway) should
surface the flag in the agent's first line so users know why recall
looks thin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# Outcome literal — matches HIPP0's /api/outcomes contract.
OutcomeLiteral = Literal["positive", "negative", "neutral"]

# A conservative per-request timeout. HIPP0 is expected to be local or
# in-region; long tails usually mean a dead peer we should fall back
# from rather than wait on.
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0

# How many times we retry an individual HTTP call (connection or 5xx)
# before writing the request to the WAL.
_RETRY_ATTEMPTS = 3
_RETRY_INITIAL_DELAY = 0.4  # seconds; doubles each retry

# Circuit breaker tuning for compile(). Three unavailable events inside
# a 60s sliding window trips the breaker OPEN for 2 minutes; the next
# call after cooldown is a HALF_OPEN probe. A success on probe closes
# the breaker. A failure on probe re-opens it for another 2 minutes.
_CB_FAIL_THRESHOLD = 3
_CB_WINDOW_SECONDS = 60.0
_CB_OPEN_SECONDS = 120.0

# Prepend a stale-memory marker to the rendered compile block when the
# last successful compile is older than this OR the breaker is OPEN.
_STALE_MEMORY_THRESHOLD_SECONDS = 30 * 60


class _CompileCircuitBreaker:
    """Minimal circuit breaker for Hipp0MemoryProvider.compile().

    State transitions:
      CLOSED --(3 timeouts in 60s)--> OPEN
      OPEN   --(2m elapsed)--------->  HALF_OPEN  (on next call)
      HALF_OPEN --(success)---------> CLOSED
      HALF_OPEN --(failure)---------> OPEN (new 2m cooldown)
    """

    def __init__(
        self,
        *,
        fail_threshold: int = _CB_FAIL_THRESHOLD,
        window_seconds: float = _CB_WINDOW_SECONDS,
        open_seconds: float = _CB_OPEN_SECONDS,
        clock: Optional[Any] = None,
    ) -> None:
        self._fail_threshold = fail_threshold
        self._window = window_seconds
        self._open_for = open_seconds
        self._clock = clock or time.monotonic
        self._failures: List[float] = []
        self._state: str = "CLOSED"
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        # Lazy transition OPEN -> HALF_OPEN when cooldown elapsed.
        if self._state == "OPEN" and self._opened_at is not None:
            if self._clock() - self._opened_at >= self._open_for:
                self._state = "HALF_OPEN"
        return self._state

    def allow(self) -> bool:
        """Return True if a call should proceed, False if short-circuited."""
        return self.state != "OPEN"

    def record_success(self) -> None:
        self._failures.clear()
        self._state = "CLOSED"
        self._opened_at = None

    def record_failure(self) -> None:
        now = self._clock()
        if self._state == "HALF_OPEN":
            # Probe failed: re-open for a fresh cooldown.
            self._state = "OPEN"
            self._opened_at = now
            self._failures = [now]
            return
        # Trim outside-window failures and append the new one.
        cutoff = now - self._window
        self._failures = [t for t in self._failures if t >= cutoff]
        self._failures.append(now)
        if len(self._failures) >= self._fail_threshold:
            self._state = "OPEN"
            self._opened_at = now


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompiledContext:
    """Result of a ``POST /api/compile`` call.

    ``degraded=True`` means HIPP0 was unreachable and the payload was
    reconstructed from the agent's local ``MEMORY.md`` snapshot.
    """

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    cache_hit: bool = False
    role_signal: Optional[Dict[str, Any]] = None
    contrastive_pairs: Optional[List[Dict[str, Any]]] = None
    degraded: bool = False
    degraded_reason: Optional[str] = None
    # Extended fields for audit trail
    decisions_considered: int = 0
    decisions_included: int = 0
    user_facts: List[Dict[str, Any]] = field(default_factory=list)
    compilation_time_ms: int = 0
    token_count: int = 0
    raw_response: Optional[Dict[str, Any]] = None
    # Minutes since the provider's last successful compile(). Set when
    # the breaker is OPEN or recall is stale (>30m). None = fresh.
    stale_minutes: Optional[int] = None

    def as_prompt_block(self) -> str:
        """Render the compiled context as a plain-text prompt block.

        Used by PersistentDelegateTool when constructing the delegate's
        system prompt. Keeps a deterministic shape so snapshot tests are
        stable.
        """
        if self.degraded:
            header = "## Compiled context (DEGRADED — HIPP0 unreachable)"
            if self.degraded_reason:
                header += f"\n_Reason: {self.degraded_reason}_"
        else:
            header = "## Compiled context"

        lines: List[str] = []
        if self.stale_minutes is not None:
            lines.append(
                f"[STALE MEMORY: last successful compile {self.stale_minutes}m ago]"
            )
        lines.extend([header, ""])
        if self.decisions:
            for d in self.decisions:
                text = d.get("text", "")
                score = d.get("score")
                did = d.get("id", "?")
                lines.append(f"- [{did}] {text}" + (f" (score={score})" if score is not None else ""))
        elif not self.user_facts:
            lines.append("_(no decisions returned)_")
            return "\n".join(lines)

        # Render user_facts — preferences, habits, vibes, identity about the
        # user. These are project-scoped in HIPP0 and must reach the model;
        # without this block the agent cannot recall cross-agent preferences
        # even though /api/compile returns them in the JSON payload.
        if self.user_facts:
            rendered: List[str] = []
            for f in self.user_facts:
                # Strict schema: require "key". Log-and-drop malformed
                # entries so the legacy `fact_key` fallback can't mask a
                # broken HIPP0 contract.
                key = f.get("key")
                if not isinstance(key, str) or not key:
                    logger.warning(
                        "HIPP0 user_fact missing 'key'; dropping entry: %r", f
                    )
                    continue
                value = f.get("value", "")
                rendered.append(f"- **{key}**: {value}")
            if rendered:
                lines.append("")
                lines.append(f"## User Facts ({len(rendered)})")
                lines.extend(rendered)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Hipp0Error(RuntimeError):
    """Base class for HIPP0 HTTP failures."""


class Hipp0HTTPError(Hipp0Error):
    """Raised when HIPP0 returned a non-success status the caller can see."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HIPP0 {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class Hipp0UnavailableError(Hipp0Error):
    """Raised when HIPP0 is unreachable after retries (conn error or 5xx)."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class Hipp0MemoryProvider(MemoryProvider):
    """HTTP memory provider backed by a HIPP0 instance.

    Instances are bound to a specific agent (``agent_name`` / ``agent_id``)
    and a specific HIPP0 project (``project_id``). One instance owns one
    ``httpx.AsyncClient`` with connection pooling; reuse it across calls.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        project_id: str,
        agent_name: str,
        agent_id: str,
        *,
        pending_wal_path: Optional[Path] = None,
        memory_md_path: Optional[Path] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.project_id = project_id
        self.agent_name = agent_name
        self.agent_id = agent_id

        self._pending_wal_path = Path(pending_wal_path) if pending_wal_path else None
        self._memory_md_path = Path(memory_md_path) if memory_md_path else None

        self._session_id: Optional[str] = None

        self._compile_breaker = _CompileCircuitBreaker()
        # Wall-clock timestamp of the last successful compile(). Used by
        # CompiledContext.as_prompt_block() to render a stale-memory
        # marker when recall may be out of date.
        self._last_compile_success_ts: Optional[float] = None

        # Serialize WAL file I/O so concurrent _wal_append / _drain_wal
        # calls cannot interleave read→write and lose records. Created
        # lazily on first use to bind to the correct event loop.
        self._wal_lock: Optional[asyncio.Lock] = None

        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                timeout,
                connect=_DEFAULT_CONNECT_TIMEOUT_SECONDS,
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"hermes-agent/hipp0-provider ({agent_name})",
            },
        )
        self._owns_client = client is None

    # ------------------------------------------------------------------ core

    @property
    def name(self) -> str:  # pragma: no cover — trivial
        return "hipp0"

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def is_available(self) -> bool:  # pragma: no cover — config-only check
        return bool(self.base_url and self.api_key and self.project_id)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Sync stub to satisfy the :class:`MemoryProvider` ABC.

        The Hipp0 provider is driven primarily by the async API that
        :class:`tools.persistent_delegate_tool.PersistentDelegateTool`
        consumes directly. When Hermes's sync memory manager hosts the
        provider as a normal plugin, the session id has already been
        negotiated by the delegate tool — we just cache it so any
        subsequent capture/compile calls use the correct id.
        """
        if session_id:
            self._session_id = session_id

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # This provider's value is in compile/capture orchestration
        # driven by PersistentDelegateTool, not in per-turn tool calls.
        return []

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ----------------------------------------------------------------- HTTP

    async def start_session(
        self,
        *,
        platform: str,
        user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
    ) -> str:
        """Ask HIPP0 for a server-generated session id.

        HIPP0 owns all session ids — we never client-generate them. The
        returned id is cached on the provider and reused for subsequent
        capture/compile calls until :meth:`end_session` or a new
        :meth:`start_session`.

        The ``user_id`` parameter is the external-system user identifier
        (e.g. a Telegram user id) and is serialized as ``external_user_id``
        on the wire — that's the key HIPP0's ``/api/hermes/session/start``
        handler reads. See HIPP0_REQUESTS.md §1.
        """
        payload = {
            "project_id": self.project_id,
            "agent_name": self.agent_name,
            "platform": platform,
            "external_user_id": user_id,
            "external_chat_id": external_chat_id,
        }
        data = await self._post_json("/api/hermes/session/start", payload)
        session_id = data.get("session_id")
        if not session_id:
            raise Hipp0Error(f"session/start missing session_id: {data!r}")
        self._session_id = session_id
        return session_id

    async def end_session(self) -> List[str]:
        """End the current HIPP0 session and return any summary snippet ids."""
        if not self._session_id:
            return []
        payload = {"session_id": self._session_id}
        try:
            data = await self._post_json("/api/hermes/session/end", payload)
        finally:
            self._session_id = None
        return list(data.get("summary_snippet_ids") or [])

    async def capture(
        self,
        conversation_text: str,
        *,
        source: str = "hermes",
        source_event_id: Optional[str] = None,
        source_channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fire-and-forget conversation ingestion.

        HIPP0 returns ``202`` with ``capture_id`` + ``status`` immediately;
        snippet extraction is async. Poll ``GET /api/capture/:id`` later
        if you need the extracted decision ids.
        """
        if len(conversation_text) > 500_000:
            raise ValueError("conversation_text exceeds 500_000 chars")
        payload = {
            "agent_name": self.agent_name,
            "project_id": self.project_id,
            "conversation": conversation_text,
            "session_id": self._session_id,
            "source": source,
            "source_event_id": source_event_id,
            "source_channel": source_channel,
        }
        return await self._post_json("/api/capture", payload, wal_kind="capture")

    async def compile(
        self,
        task_description: str,
        *,
        max_tokens: int = 4000,
        fast_mode: bool = True,
        task_session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        session_lookback_days: Optional[int] = None,
        include_superseded: bool = False,
        include_role_signal: bool = True,
    ) -> CompiledContext:
        """Task-aware context retrieval with degraded-mode fallback."""
        if len(task_description) > 100_000:
            raise ValueError("task_description exceeds 100_000 chars")

        body: Dict[str, Any] = {
            "agent_name": self.agent_name,
            "project_id": self.project_id,
            "task_description": task_description,
            "max_tokens": max_tokens,
            "include_superseded": include_superseded,
            "include_role_signal": include_role_signal,
        }
        if task_session_id is not None:
            body["task_session_id"] = task_session_id
        if namespace is not None:
            body["namespace"] = namespace
        if session_lookback_days is not None:
            body["session_lookback_days"] = session_lookback_days

        # Fast mode: skip shared-pattern expansion and explanations,
        # tighten the score threshold. Full mode is reserved for
        # session-start compiles.
        if fast_mode:
            params = {
                "format": "json",
                "depth": "default",
                "threshold": "0.6",
                "include_patterns": "false",
                "explain": "false",
            }
        else:
            params = {
                "format": "json",
                "depth": "full",
                "threshold": "0.5",
                "include_patterns": "true",
                "explain": "false",
            }

        # Circuit breaker: short-circuit to degraded-mode while OPEN so we
        # don't pile up doomed requests against a dead HIPP0.
        if not self._compile_breaker.allow():
            return self._degraded_compile(
                f"circuit breaker OPEN (cooldown {int(_CB_OPEN_SECONDS)}s)"
            )

        try:
            data = await self._post_json(
                "/api/compile",
                body,
                params=params,
                wal_kind="compile",
                allow_wal=False,  # compile is read; no point queueing
            )
        except Hipp0UnavailableError as e:
            self._compile_breaker.record_failure()
            return self._degraded_compile(str(e))
        except Hipp0HTTPError as e:
            # 4xx is a hard contract bug — surface it. 5xx fell through
            # to Hipp0UnavailableError via retry.
            if 500 <= e.status_code < 600:
                self._compile_breaker.record_failure()
                return self._degraded_compile(str(e))
            raise

        self._compile_breaker.record_success()
        self._last_compile_success_ts = time.time()

        return CompiledContext(
            decisions=list(data.get("decisions") or []),
            total_tokens=int(data.get("total_tokens") or 0),
            cache_hit=bool(data.get("cache_hit")),
            role_signal=data.get("role_signal"),
            contrastive_pairs=data.get("contrastive_pairs"),
            degraded=False,
            decisions_considered=int(data.get("decisions_considered") or 0),
            decisions_included=int(data.get("decisions_included") or 0),
            user_facts=list(data.get("user_facts") or []),
            compilation_time_ms=int(data.get("compilation_time_ms") or 0),
            token_count=int(data.get("token_count") or 0),
            raw_response=data,
        )

    async def record_outcome(
        self,
        snippet_ids: List[str],
        outcome: OutcomeLiteral,
        *,
        signal_source: str,
        note: Optional[str] = None,
    ) -> None:
        """Record reinforcement signal for a set of snippets.

        Hermes must call this at end of turn. ``signal_source`` is
        free-form on the server side (e.g. ``telegram_reaction``,
        ``repeat_question``, ``manual``), but the provider still
        surfaces it as a kwarg so callers can be explicit.

        Posts to the H6 ``POST /api/hermes/outcomes`` endpoint, not the
        older compile-request-based ``/api/outcomes`` path. See
        ``HIPP0_REQUESTS.md §6`` for the contract split. The new
        endpoint is keyed by opaque ``session_id`` only — no
        ``agent_name`` on the wire.
        """
        if not snippet_ids:
            return
        payload: Dict[str, Any] = {
            "project_id": self.project_id,
            "session_id": self._session_id,
            "snippet_ids": list(snippet_ids),
            "outcome": outcome,
            "signal_source": signal_source,
        }
        if note is not None:
            payload["note"] = note
        await self._post_json("/api/hermes/outcomes", payload, wal_kind="outcome")

    async def record_decision(
        self,
        title: str,
        rationale: str,
        tags: Optional[List[str]] = None,
        confidence: str = "medium",
        agent_name: Optional[str] = None,
    ) -> bool:
        """Record a decision signal to hipp0. Non-fatal on failure."""
        if not self.project_id:
            return False
        try:
            # hipp0 requires `description` (not `content`) and `project_id`
            # on the unscoped /api/decisions route. Omitting either yields a
            # 400 VALIDATION_ERROR.
            payload: Dict[str, Any] = {
                "project_id": self.project_id,
                "title": title,
                "description": rationale,
                "made_by": agent_name or "hermes",
                "tags": tags or [],
                "confidence": confidence,
                "source": "auto_capture",
            }
            data = await self._post_json("/api/decisions", payload)
            return bool(data) or True
        except Exception as exc:
            logger.debug("[hipp0] record_decision failed: %s", exc)
            return False

    async def upsert_user_fact(
        self,
        user_id: str,
        facts: List[Dict[str, Any]],
        *,
        etag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upsert facts about a user with optimistic concurrency control.

        ``facts`` is a list of ``{"key", "value", "additive"}`` dicts.
        Pass the ``etag`` returned by the last successful upsert as the
        ``If-Match`` header to avoid clobbering concurrent edits.

        ``user_id`` is the external-system user identifier and is sent as
        ``external_user_id`` — that's the key HIPP0's handler reads.
        """
        payload = {
            "project_id": self.project_id,
            "external_user_id": user_id,
            "facts": list(facts),
        }
        headers = {"If-Match": etag} if etag else None
        return await self._post_json(
            "/api/hermes/user-facts",
            payload,
            extra_headers=headers,
        )

    async def register(self, soul: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Register this agent with HIPP0 and return ``{agent_id, created}``.

        AgentRegistry.register_agent() writes the profile locally; callers
        that want a HIPP0 agent id should invoke this afterwards and then
        persist ``agent_id`` into the local config via
        :func:`hermes_cli.agent_registry.update_agent_config`.
        """
        payload = {
            "project_id": self.project_id,
            "agent_name": self.agent_name,
            "soul": soul,
            "config": config,
        }
        data = await self._post_json("/api/hermes/register", payload)
        returned_id = data.get("agent_id")
        if returned_id:
            self.agent_id = returned_id
        return data

    # --------------------------------------------------------- HTTP internals

    async def _post_json(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        params: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        wal_kind: Optional[str] = None,
        allow_wal: bool = True,
    ) -> Dict[str, Any]:
        """POST JSON with retries, WAL on outage, and WAL drain on success."""
        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt < _RETRY_ATTEMPTS:
            try:
                resp = await self._client.post(
                    path,
                    json=body,
                    params=params,
                    headers=extra_headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                await asyncio.sleep(_RETRY_INITIAL_DELAY * (2**attempt))
                attempt += 1
                continue

            if 500 <= resp.status_code < 600:
                last_exc = Hipp0HTTPError(resp.status_code, resp.text)
                await asyncio.sleep(_RETRY_INITIAL_DELAY * (2**attempt))
                attempt += 1
                continue

            if resp.status_code >= 400:
                # 4xx: hard client-side contract error. Don't WAL, don't retry.
                raise Hipp0HTTPError(resp.status_code, resp.text)

            # Success — drain any queued WAL entries in the background
            # (best-effort). If draining fails the WAL stays put.
            if wal_kind is not None:
                await self._drain_wal()
            try:
                return resp.json() if resp.content else {}
            except json.JSONDecodeError:
                return {}

        # Exhausted retries.
        if wal_kind is not None and allow_wal:
            self._wal_append(
                {
                    "kind": wal_kind,
                    "path": path,
                    "body": body,
                    "params": params,
                    "headers": extra_headers,
                    "timestamp": time.time(),
                    "error": str(last_exc),
                }
            )
            logger.warning(
                "HIPP0 %s unreachable after %d retries; queued to WAL: %s",
                path,
                _RETRY_ATTEMPTS,
                last_exc,
            )
        raise Hipp0UnavailableError(
            f"HIPP0 {path} unreachable after {_RETRY_ATTEMPTS} retries: {last_exc}"
        )

    # ----------------------------------------------------------------- WAL

    def _get_wal_lock(self) -> asyncio.Lock:
        if self._wal_lock is None:
            self._wal_lock = asyncio.Lock()
        return self._wal_lock

    @staticmethod
    def _write_secure(path: Path, content: str) -> None:
        """Atomically write *content* to *path* with 0o600 permissions.

        Writes to a sibling tmp file, chmods before rename so the mode
        is applied before the file is visible at the final name.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # os.open + write to set mode atomically (avoids umask-dependent
        # initial perms that Path.write_text would create).
        import os as _os
        fd = _os.open(tmp, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o600)
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        _os.replace(tmp, path)

    @staticmethod
    def _append_secure(path: Path, line: str) -> None:
        """Append *line* to *path*, creating it with 0o600 if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        import os as _os
        # O_APPEND is atomic on POSIX for writes < PIPE_BUF; JSON lines
        # here are always under that. Create with 0o600 if not present.
        existed = path.exists()
        fd = _os.open(path, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o600)
        try:
            with _os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)
        finally:
            if not existed:
                try:
                    _os.chmod(path, 0o600)
                except OSError:
                    pass

    def _wal_append(self, record: Dict[str, Any]) -> None:
        if not self._pending_wal_path:
            logger.error(
                "HIPP0 WAL miss: no pending.jsonl path configured; dropping %s",
                record.get("kind"),
            )
            return
        self._append_secure(self._pending_wal_path, json.dumps(record) + "\n")

    async def _drain_wal(self) -> None:
        """Replay WAL entries oldest-first. Drops on success, keeps on failure.

        Serialized under _wal_lock so a concurrent _wal_append cannot be
        lost between the read and the rewrite, and so two concurrent
        drains cannot double-post records.
        """
        if not self._pending_wal_path or not self._pending_wal_path.exists():
            return
        async with self._get_wal_lock():
            if not self._pending_wal_path.exists():
                return
            try:
                lines = self._pending_wal_path.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                logger.warning("HIPP0 WAL: could not read %s: %s", self._pending_wal_path, e)
                return

            remaining: List[str] = []
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("HIPP0 WAL: dropping malformed line %d", i)
                    continue
                try:
                    resp = await self._client.post(
                        record["path"],
                        json=record.get("body") or {},
                        params=record.get("params"),
                        headers=record.get("headers"),
                    )
                except (httpx.TransportError, httpx.TimeoutException):
                    # Keep this line and all subsequent lines in order.
                    remaining.append(line)
                    remaining.extend(lines[i + 1 :])
                    break
                if resp.status_code >= 500:
                    remaining.append(line)
                    remaining.extend(lines[i + 1 :])
                    break
                # 4xx: bad contract — move to dead_letter.jsonl for operator
                # inspection rather than silently dropping. 2xx: drop normally.
                if resp.status_code >= 400:
                    logger.warning(
                        "HIPP0 WAL: dead-lettering 4xx entry %s (%d)",
                        record.get("kind"),
                        resp.status_code,
                    )
                    self._dead_letter_append(record, resp.status_code, resp.text)
                continue

            if remaining:
                self._write_secure(
                    self._pending_wal_path, "\n".join(remaining) + "\n"
                )
            else:
                try:
                    self._pending_wal_path.unlink()
                except OSError:
                    pass

    def _dead_letter_path(self) -> Optional[Path]:
        if not self._pending_wal_path:
            return None
        return self._pending_wal_path.with_name("dead_letter.jsonl")

    def _dead_letter_append(
        self, record: Dict[str, Any], status_code: int, error_body: str
    ) -> None:
        dl_path = self._dead_letter_path()
        if dl_path is None:
            return
        entry = {
            **record,
            "dead_letter_timestamp": time.time(),
            "status_code": status_code,
            "error_body": error_body[:2000],
        }
        self._append_secure(dl_path, json.dumps(entry) + "\n")

    def dead_letter_size(self) -> int:
        """Return the number of dead-lettered entries (observability helper)."""
        dl_path = self._dead_letter_path()
        if not dl_path or not dl_path.exists():
            return 0
        try:
            return sum(
                1 for line in dl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            return 0

    def wal_size(self) -> int:
        """Return the number of queued WAL entries (test + observability helper)."""
        if not self._pending_wal_path or not self._pending_wal_path.exists():
            return 0
        try:
            return sum(
                1 for line in self._pending_wal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            return 0

    # -------------------------------------------------------- degraded mode

    def _degraded_compile(self, reason: str) -> CompiledContext:
        """Build a CompiledContext from the local MEMORY.md cache."""
        decisions: List[Dict[str, Any]] = []
        total_tokens = 0
        if self._memory_md_path and self._memory_md_path.is_file():
            try:
                text = self._memory_md_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "HIPP0 degraded: failed to read %s: %s",
                    self._memory_md_path,
                    e,
                )
                text = ""
            if text.strip():
                decisions.append(
                    {
                        "id": "local-memory",
                        "text": text,
                        "score": None,
                        "source": "MEMORY.md",
                    }
                )
                # Rough token estimate: 4 chars per token.
                from agent.model_metadata import estimate_tokens_rough
                total_tokens = max(1, estimate_tokens_rough(text))
        logger.warning("HIPP0 compile degraded: %s", reason)
        return CompiledContext(
            decisions=decisions,
            total_tokens=total_tokens,
            cache_hit=False,
            degraded=True,
            degraded_reason=reason,
            stale_minutes=self._compute_stale_minutes(force=True),
        )

    def _compute_stale_minutes(self, *, force: bool = False) -> Optional[int]:
        """Return minutes since last successful compile, or None if fresh.

        When ``force`` is True (degraded path, or breaker open) we always
        emit a staleness number — 999 if nothing has ever succeeded —
        so callers can render the stale-memory marker. Otherwise we only
        return a value when the breaker is OPEN or the gap exceeds
        ``_STALE_MEMORY_THRESHOLD_SECONDS``.
        """
        now = time.time()
        last = self._last_compile_success_ts
        if last is None:
            return 999 if force else None
        gap = now - last
        if force or self._compile_breaker.state == "OPEN" or gap >= _STALE_MEMORY_THRESHOLD_SECONDS:
            return max(0, int(gap // 60))
        return None
