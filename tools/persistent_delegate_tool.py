"""Persistent-delegate tool: invoke a named agent as a subagent.

This is the H3 piece of the ``feat/persistent-agents-hipp0`` stack.
It's the orchestration glue between:

  - :mod:`hermes_cli.agent_registry` — loads the named agent's
    SOUL.md + config.yaml off disk.
  - :class:`agent.hipp0_memory_provider.Hipp0MemoryProvider` — runs
    the HIPP0 session lifecycle (start → compile → capture → end),
    WAL + degraded mode included.
  - A Hermes ``AIAgent`` subagent (or any injectable runner) that
    actually drives the LLM conversation under the delegate's SOUL.

Flow for a single :meth:`PersistentDelegateTool.invoke` call::

    1. AgentRegistry.get_agent(name)        -> AgentProfile
    2. Hipp0MemoryProvider(agent + hipp0)   -> bound provider
    3. provider.start_session(...)          -> server UUID
    4. provider.compile(task, fast=True)    -> CompiledContext
    5. build system prompt = SOUL + compiled context + degraded warn
    6. runner(task, system_prompt)          -> delegate's final reply
    7. provider.capture(transcript)         -> 202 (async snippet extract)
    8. return {response, session_id, captured, degraded, ...}

:func:`persistent_delegate_task_handler` is the thin sync wrapper
registered with ``tools.registry`` so the model can fire off
``persistent_delegate_task`` from a parent agent. It runs the async
flow on a fresh event loop (or inside the parent's loop when one is
active) and forwards the result through the standard tool-reply
protocol.

The runner is injected so tests can exercise the full orchestration
without spinning up a real LLM — see
``tests/tools/test_persistent_delegate_tool.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from agent.hipp0_memory_provider import (
    CompiledContext,
    Hipp0MemoryProvider,
    Hipp0UnavailableError,
)
from hermes_cli.agent_registry import (
    AgentNotFoundError,
    AgentProfile,
    get_agent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


# A runner takes (task_description, system_prompt, context) and returns
# the delegate's final message + a raw transcript (string form suitable
# for HIPP0 capture). Kept narrow so tests can inject a trivial stub
# and production code can wrap AIAgent.run_conversation.
DelegateRunner = Callable[
    [str, str, "DelegateRunContext"],
    Awaitable["DelegateRunResult"],
]


@dataclass
class DelegateRunContext:
    """Everything the runner needs beyond ``task`` and ``system_prompt``."""

    agent: AgentProfile
    session_id: str
    compiled: CompiledContext
    provider: Hipp0MemoryProvider
    parent_agent: Optional[Any] = None  # parent AIAgent, when available


@dataclass
class DelegateRunResult:
    """Result returned by a :data:`DelegateRunner`."""

    final_message: str
    transcript: str  # full text to hand to HIPP0 capture


@dataclass
class PersistentDelegateResult:
    """Public shape of an :meth:`PersistentDelegateTool.invoke` call."""

    agent_name: str
    session_id: str
    response: str
    captured: Dict[str, Any]
    compiled_degraded: bool
    compiled_token_count: int
    degraded_reason: Optional[str] = None

    def to_tool_payload(self) -> Dict[str, Any]:
        """Serializable dict for the outer `handle_function_call` result."""
        return {
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "response": self.response,
            "captured": self.captured,
            "compiled_degraded": self.compiled_degraded,
            "compiled_token_count": self.compiled_token_count,
            "degraded_reason": self.degraded_reason,
        }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PersistentDelegateError(RuntimeError):
    """Raised when a persistent delegate cannot complete its task."""


class PersistentDelegateConfigError(PersistentDelegateError):
    """Raised when HIPP0 env / agent config is insufficient to run a delegate."""


# ---------------------------------------------------------------------------
# Compile-result TTL cache — absorbs N-subagent fan-out on the same task.
# ---------------------------------------------------------------------------


_COMPILE_CACHE_TTL_SECONDS = 300.0  # 5 minutes
_compile_cache: Dict[str, tuple] = {}  # key -> (expires_at, CompiledContext)
_compile_cache_lock = asyncio.Lock()


def _compile_cache_key(
    project_id: str,
    task: str,
    *,
    fast_mode: bool,
    namespace: Optional[str],
) -> str:
    h = hashlib.sha256(task.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{project_id}|{h}|{int(fast_mode)}|{namespace or '-'}"


async def _compile_cache_get(key: str) -> Optional["CompiledContext"]:
    async with _compile_cache_lock:
        entry = _compile_cache.get(key)
        if entry is None:
            return None
        expires_at, compiled = entry
        if expires_at < time.monotonic():
            _compile_cache.pop(key, None)
            return None
        return compiled


async def _compile_cache_put(key: str, compiled: "CompiledContext") -> None:
    async with _compile_cache_lock:
        _compile_cache[key] = (
            time.monotonic() + _COMPILE_CACHE_TTL_SECONDS,
            compiled,
        )


def _compile_cache_clear() -> None:
    """Test hook: drop all entries."""
    _compile_cache.clear()


# ---------------------------------------------------------------------------
# Task classifier — pick the cheapest compile mode for the task.
# ---------------------------------------------------------------------------


_SELF_CONTAINED_PATTERNS = (
    re.compile(r"\bfrom scratch\b", re.I),
    re.compile(r"\bhello[\s-]world\b", re.I),
    re.compile(r"\bwrite\s+a\s+(?:simple|small|trivial|basic)\b", re.I),
    re.compile(r"\bpure\s+function\b", re.I),
)

_TECHNICAL_KEYWORDS = (
    "bug", "error", "fix", "crash", "stack trace", "traceback",
    "exception", "how to", "debug",
)

_USER_KEYWORDS = (
    "preference", "style", "like", "remember", "my ",
    "i prefer", "i like", "remind me",
)


def classify_task(task_description: str) -> Dict[str, Any]:
    """Classify a task into a compile-mode hint.

    Returns a dict with one of:

    * ``{"skip_compile": True}`` — self-contained tasks (e.g. "write a
      hello world from scratch") don't need cross-session memory.
    * ``{"namespace": "technical", "fast_mode": False}`` — debugging /
      how-to tasks; use full compile scoped to technical namespace.
    * ``{"namespace": "user", "fast_mode": True}`` — preference /
      style / identity tasks; scope to user namespace.
    * ``{"namespace": None, "fast_mode": True}`` — default: full
      compile in fast mode, no namespace filter.

    Uses the similarity-based ``router_classifier`` when available and
    falls back to the keyword heuristic for short/empty inputs or when
    the import is missing (import-cycle safety during test collection).

    Also logs the routing decision to ``routing_outcomes`` when a similarity
    decision was produced, so the feedback edge can learn over time.

    Pure from the caller's perspective; the side-effect is append-only
    logging to ``~/.hermes/routing_outcomes.jsonl``.
    """
    t = (task_description or "").lower().strip()
    if not t:
        return {"namespace": None, "fast_mode": True}

    try:
        from tools.router_classifier import classify as _similarity_classify
        from tools.router_classifier import decision_to_classify_task_hint
        from tools.routing_outcomes import record_decision

        dec = _similarity_classify(task_description)
        hint = decision_to_classify_task_hint(dec)
        # Fire-and-forget log. Any failure must not break the routing path.
        try:
            record_decision(
                task_description,
                decided_class=dec.cls,
                score=dec.score,
                margin=dec.margin,
                uncertain=dec.uncertain,
            )
        except Exception:
            pass
        return hint
    except Exception:
        # Fallback to the legacy keyword classifier below.
        pass

    # Self-contained heuristic: short tasks with "from scratch" /
    # "hello world" markers and no proper nouns (uppercase words
    # mid-sentence) are unlikely to benefit from memory.
    for pat in _SELF_CONTAINED_PATTERNS:
        if pat.search(task_description):
            # Reject if the task mentions proper nouns mid-sentence,
            # which usually means a project-specific reference.
            tokens = task_description.split()
            has_proper_noun = any(
                i > 0 and tok[:1].isupper() and tok[1:2].islower()
                for i, tok in enumerate(tokens)
            )
            if not has_proper_noun:
                return {"skip_compile": True}
            break

    if any(k in t for k in _TECHNICAL_KEYWORDS):
        return {"namespace": "technical", "fast_mode": False}

    if any(k in t for k in _USER_KEYWORDS):
        return {"namespace": "user", "fast_mode": True}

    return {"namespace": None, "fast_mode": True}


def _tokenize(text: str) -> set:
    """Lowercase, split on non-word chars, drop short tokens/stopwords."""
    _STOP = {
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
        "with", "is", "are", "was", "were", "be", "this", "that",
        "it", "as", "at", "by", "from", "if", "you", "i", "we",
    }
    return {
        w for w in re.findall(r"[a-z0-9]{3,}", text.lower())
        if w not in _STOP
    }


def _slice_compiled_per_task(
    broad: "CompiledContext",
    tasks: List[str],
) -> List["CompiledContext"]:
    """Score each broad decision against each task and split per subagent.

    Each decision goes to the subagent whose task shares the most
    tokens with the decision text (ties broken by order). If no
    subagent matches, the decision is dropped for that batch.
    ``user_facts`` go to every subagent (project-wide preferences).
    """
    task_tokens = [_tokenize(t) for t in tasks]
    n = len(tasks)
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

    for d in broad.decisions or []:
        dtoks = _tokenize(str(d.get("text", "")))
        if not dtoks:
            continue
        scores = [len(dtoks & tt) for tt in task_tokens]
        best = max(scores) if scores else 0
        if best == 0:
            continue  # irrelevant to every subagent — drop
        buckets[scores.index(best)].append(d)

    return [
        CompiledContext(
            decisions=buckets[i],
            total_tokens=sum(
                int(d.get("tokens") or 0) for d in buckets[i]
            ) or broad.total_tokens // max(n, 1),
            cache_hit=broad.cache_hit,
            role_signal=broad.role_signal,
            contrastive_pairs=broad.contrastive_pairs,
            degraded=broad.degraded,
            degraded_reason=broad.degraded_reason,
            decisions_considered=broad.decisions_considered,
            decisions_included=len(buckets[i]),
            user_facts=list(broad.user_facts or []),
            compilation_time_ms=broad.compilation_time_ms,
            token_count=broad.token_count,
            raw_response={"sliced_from_batch": True},
        )
        for i in range(n)
    ]


_REDUNDANCY_THRESHOLD = 0.8  # 80% of decisions already present → skip


def _drop_redundant_compiled(
    compiled: "CompiledContext",
    recent_messages: List[Dict[str, Any]],
    *,
    max_chars: int = 16_000,
) -> "CompiledContext":
    """Drop compiled decisions already carried by the conversation.

    For each decision, compute token-overlap ratio against the
    concatenated text of the last few messages. If >= 80% of tokens
    appear in the conversation, drop it. If >= 80% of ALL decisions
    are redundant, zero out the decisions list entirely (avoids a
    nearly-empty compile block whose header adds noise).
    """
    if not compiled.decisions:
        return compiled

    # Join the tail of recent messages into one searchable blob.
    buf: List[str] = []
    remaining = max_chars
    for m in reversed(recent_messages):
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, str) or not content:
            continue
        if len(content) > remaining:
            buf.append(content[-remaining:])
            break
        buf.append(content)
        remaining -= len(content)
        if remaining <= 0:
            break
    convo_tokens = _tokenize(" ".join(buf))
    if not convo_tokens:
        return compiled

    kept: List[Dict[str, Any]] = []
    redundant = 0
    for d in compiled.decisions:
        dtoks = _tokenize(str(d.get("text", "")))
        if not dtoks:
            kept.append(d)
            continue
        overlap = len(dtoks & convo_tokens) / len(dtoks)
        if overlap >= _REDUNDANCY_THRESHOLD:
            redundant += 1
        else:
            kept.append(d)

    total = len(compiled.decisions)
    # If overwhelmingly redundant, drop everything.
    if redundant / total >= _REDUNDANCY_THRESHOLD:
        kept = []

    if len(kept) == total:
        return compiled  # nothing dropped; keep identity

    return CompiledContext(
        decisions=kept,
        total_tokens=compiled.total_tokens,
        cache_hit=compiled.cache_hit,
        role_signal=compiled.role_signal,
        contrastive_pairs=compiled.contrastive_pairs,
        degraded=compiled.degraded,
        degraded_reason=compiled.degraded_reason,
        decisions_considered=compiled.decisions_considered,
        decisions_included=len(kept),
        user_facts=list(compiled.user_facts or []),
        compilation_time_ms=compiled.compilation_time_ms,
        token_count=compiled.token_count,
        raw_response={
            "skipped_redundant": total - len(kept),
            "original": compiled.raw_response,
        },
    )


# ---------------------------------------------------------------------------
# Core tool
# ---------------------------------------------------------------------------


class PersistentDelegateTool:
    """Orchestrates a persistent named-agent delegation.

    Parameters
    ----------
    base_url, api_key:
        HIPP0 endpoint and bearer token. Defaults pull from
        ``HIPP0_BASE_URL`` / ``HIPP0_API_KEY`` env vars.
    runner:
        Injectable coroutine that actually runs the delegate
        conversation. The default runner wraps Hermes's ``AIAgent``
        via :func:`_default_aiagent_runner`; tests pass a stub.
    provider_factory:
        Injectable factory that builds the :class:`Hipp0MemoryProvider`.
        Default wires the real provider. Tests inject a factory that
        uses a shared httpx.AsyncClient pointed at the mock HIPP0.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        runner: Optional[DelegateRunner] = None,
        provider_factory: Optional[
            Callable[[AgentProfile], Hipp0MemoryProvider]
        ] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "HIPP0_BASE_URL", "http://localhost:3000"
        )
        self.api_key = api_key or os.environ.get("HIPP0_API_KEY", "")
        self._runner = runner or _default_aiagent_runner
        self._provider_factory = provider_factory

    async def invoke(
        self,
        agent_name: str,
        task: str,
        *,
        platform: str = "cli",
        user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
        parent_agent: Optional[Any] = None,
        end_session: bool = False,
        precompiled: Optional[CompiledContext] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> PersistentDelegateResult:
        """Run a persistent delegate end-to-end.

        Parameters
        ----------
        agent_name:
            Name registered under ``<hermes_root>/agents/<name>/``.
        task:
            The user-visible task description.
        platform:
            Platform the delegation is happening on (``"cli"``,
            ``"telegram"``, …). Forwarded to HIPP0 session/start.
        user_id, external_chat_id:
            Optional platform identifiers. Passed to HIPP0 so sessions
            can be scoped per-user.
        parent_agent:
            Parent ``AIAgent`` instance, if one is driving the
            delegation. Exposed to the runner via ``DelegateRunContext``
            so implementations can inherit credentials, display, etc.
        end_session:
            When True, call ``session/end`` after a successful capture.
            Telegram keeps the session open across messages; one-shot
            CLI invocations should pass True so HIPP0 runs the
            rolling-summary flush.
        """
        try:
            profile = get_agent(agent_name)
        except AgentNotFoundError as e:
            raise PersistentDelegateError(
                f"Persistent delegate {agent_name!r} not registered: {e}"
            ) from e

        if not profile.config.project_id:
            raise PersistentDelegateConfigError(
                f"Agent {agent_name!r} has no project_id. Register with HIPP0 "
                f"via Hipp0MemoryProvider.register() and persist the result "
                f"via hermes_cli.agent_registry.update_agent_config."
            )
        if not self.api_key:
            raise PersistentDelegateConfigError(
                "HIPP0_API_KEY env var is not set. Cannot invoke a "
                "persistent delegate without HIPP0 credentials."
            )

        provider = self._make_provider(profile)
        try:
            session_id = await provider.start_session(
                platform=platform,
                user_id=user_id,
                external_chat_id=external_chat_id,
            )
            hint = classify_task(task)
            if precompiled is not None:
                # Parent supplied a pre-sliced CompiledContext (fan-out
                # path in invoke_batch). Skip the per-subagent compile
                # round-trip entirely.
                compiled = precompiled
            elif hint.get("skip_compile"):
                # Self-contained task: synthesize an empty CompiledContext
                # rather than round-tripping to HIPP0. Saves one network
                # call; the degraded flag stays False because this was an
                # intentional skip, not a failure.
                compiled = CompiledContext(
                    decisions=[],
                    total_tokens=0,
                    cache_hit=False,
                    degraded=False,
                    raw_response={"skipped": "self_contained_task"},
                )
            else:
                fast = bool(hint.get("fast_mode", True))
                ns = hint.get("namespace")
                cache_key = _compile_cache_key(
                    str(profile.config.project_id),
                    task,
                    fast_mode=fast,
                    namespace=ns,
                )
                compiled = await _compile_cache_get(cache_key)
                if compiled is None:
                    compiled = await provider.compile(
                        task_description=task,
                        fast_mode=fast,
                        namespace=ns,
                    )
                    if not compiled.degraded:
                        # Don't cache degraded results — they're local
                        # fallbacks, and caching would pin us in the
                        # degraded state past the 5m window.
                        await _compile_cache_put(cache_key, compiled)
            # If most compiled content is already present in the
            # parent's recent messages, skip re-injection to save
            # tokens + avoid nagging the model with duplicates.
            effective_recent = recent_messages
            if effective_recent is None and parent_agent is not None:
                effective_recent = getattr(
                    parent_agent, "_session_messages", None
                )
            if effective_recent:
                compiled = _drop_redundant_compiled(
                    compiled, effective_recent
                )

            system_prompt = self._build_system_prompt(
                profile, compiled, platform=platform,
            )

            run_ctx = DelegateRunContext(
                agent=profile,
                session_id=session_id,
                compiled=compiled,
                provider=provider,
                parent_agent=parent_agent,
            )
            run_result = await self._runner(task, system_prompt, run_ctx)
            if not isinstance(run_result, DelegateRunResult):
                raise PersistentDelegateError(
                    f"Delegate runner returned {type(run_result).__name__}, "
                    "expected DelegateRunResult"
                )

            try:
                captured = await provider.capture(
                    run_result.transcript,
                    source="hermes",
                    source_channel=external_chat_id,
                )
            except Hipp0UnavailableError as e:
                # Capture failed after retries — the WAL already has it.
                # Don't fail the delegation; log and proceed.
                logger.warning(
                    "persistent delegate %r: capture WALed (%s)",
                    agent_name,
                    e,
                )
                captured = {"status": "walled", "capture_id": None}

            if end_session:
                try:
                    await provider.end_session()
                except Hipp0UnavailableError as e:
                    logger.warning(
                        "persistent delegate %r: session/end failed: %s",
                        agent_name,
                        e,
                    )

            return PersistentDelegateResult(
                agent_name=agent_name,
                session_id=session_id,
                response=run_result.final_message,
                captured=captured,
                compiled_degraded=compiled.degraded,
                compiled_token_count=compiled.total_tokens,
                degraded_reason=compiled.degraded_reason,
            )
        finally:
            await provider.aclose()

    async def invoke_batch(
        self,
        tasks: List[Dict[str, Any]],
        *,
        platform: str = "cli",
        user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
        parent_agent: Optional[Any] = None,
        end_session: bool = False,
    ) -> List[PersistentDelegateResult]:
        """Fan out N subagents with a single shared compile.

        ``tasks`` is a list of ``{"agent_name": ..., "task": ...}``.
        The parent performs one broad compile (joined task descriptions)
        and slices the returned decisions / user_facts per subagent
        using simple token-overlap scoring; each subagent then runs
        ``invoke()`` with that per-agent slice as ``precompiled``.

        All subagents must share the same ``project_id`` — otherwise a
        cross-project compile would leak context. Mixed-project batches
        fall back to parallel per-task ``invoke()`` calls.
        """
        if not tasks:
            return []

        # Resolve profiles up-front so we can group by project.
        profiles: List[AgentProfile] = []
        for t in tasks:
            name = t.get("agent_name")
            if not name:
                raise PersistentDelegateError(
                    "invoke_batch: each task must have 'agent_name'"
                )
            try:
                profiles.append(get_agent(name))
            except AgentNotFoundError as e:
                raise PersistentDelegateError(
                    f"Persistent delegate {name!r} not registered: {e}"
                ) from e

        project_ids = {str(p.config.project_id) for p in profiles}
        same_project = len(project_ids) == 1 and "None" not in project_ids

        if not same_project or len(tasks) < 2:
            # Nothing to share — fan out the unchanged per-task path.
            return await asyncio.gather(*[
                self.invoke(
                    agent_name=t["agent_name"],
                    task=t["task"],
                    platform=platform,
                    user_id=user_id,
                    external_chat_id=external_chat_id,
                    parent_agent=parent_agent,
                    end_session=end_session,
                )
                for t in tasks
            ])

        # ── One broad compile for the whole batch ──────────────────────
        # We borrow the first agent's provider to do the compile (same
        # project_id by construction). The per-subagent invokes still
        # need their own providers for session/capture — those are
        # cheap compared to compile.
        broad_task = "\n".join(t["task"] for t in tasks)
        pilot_provider = self._make_provider(profiles[0])
        try:
            await pilot_provider.start_session(
                platform=platform,
                user_id=user_id,
                external_chat_id=external_chat_id,
            )
            broad = await pilot_provider.compile(
                task_description=broad_task,
                fast_mode=True,
            )
        finally:
            await pilot_provider.aclose()

        # Slice per subagent.
        slices = _slice_compiled_per_task(broad, [t["task"] for t in tasks])

        return await asyncio.gather(*[
            self.invoke(
                agent_name=t["agent_name"],
                task=t["task"],
                platform=platform,
                user_id=user_id,
                external_chat_id=external_chat_id,
                parent_agent=parent_agent,
                end_session=end_session,
                precompiled=s,
            )
            for t, s in zip(tasks, slices)
        ])

    # ------------------------------------------------------------------ helpers

    def _make_provider(self, profile: AgentProfile) -> Hipp0MemoryProvider:
        if self._provider_factory is not None:
            return self._provider_factory(profile)
        return Hipp0MemoryProvider(
            base_url=self.base_url,
            api_key=self.api_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

    @staticmethod
    def _build_system_prompt(
        profile: AgentProfile,
        compiled: CompiledContext,
        *,
        platform: Optional[str] = None,
    ) -> str:
        """Compose the delegate's system prompt via the slim builder.

        Delegates don't want the full Hermes operating manual — their
        persona (SOUL.md) is authoritative and HIPP0 owns their memory.
        Delegates always use :func:`agent.prompt_builder.build_slim_system_prompt`.
        Degraded-mode annotations come from :meth:`CompiledContext.as_prompt_block`.
        """
        from agent.prompt_builder import build_slim_system_prompt

        return build_slim_system_prompt(
            profile.soul,
            compiled_context_block=compiled.as_prompt_block(),
            platform_hint=platform,
        )


# ---------------------------------------------------------------------------
# Default runner — AIAgent-backed (lazy import so test runs don't need the
# full Hermes runtime to exercise this module).
# ---------------------------------------------------------------------------


async def _default_aiagent_runner(
    task: str,
    system_prompt: str,
    ctx: DelegateRunContext,
) -> DelegateRunResult:
    """Default runner: spawn a lightweight AIAgent and drive a single turn.

    Runs synchronously inside an executor because ``AIAgent.chat`` is
    blocking. The delegate inherits model / credentials from the parent
    when one is provided, otherwise falls back to the agent's own
    config.yaml.
    """
    parent = ctx.parent_agent

    def _run_sync() -> DelegateRunResult:
        from run_agent import AIAgent  # lazy import

        agent_cfg = ctx.agent.config
        model = (getattr(parent, "model", None) or agent_cfg.model)
        base_url = getattr(parent, "base_url", None)
        api_key = getattr(parent, "api_key", None)

        delegate = AIAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_iterations=10,
            quiet_mode=True,
            ephemeral_system_prompt=system_prompt,
            platform=getattr(parent, "platform", "cli"),
            skip_context_files=True,
            skip_memory=True,  # Hipp0 provider owns memory, not builtin
            log_prefix=f"[persistent:{ctx.agent.name}]",
        )
        delegate._delegate_depth = (getattr(parent, "_delegate_depth", 0) or 0) + 1

        final = delegate.chat(task)
        transcript_lines = [
            f"USER: {task}",
            f"ASSISTANT({ctx.agent.name}): {final}",
        ]
        return DelegateRunResult(
            final_message=final,
            transcript="\n".join(transcript_lines),
        )

    loop = asyncio.get_event_loop()
    # Hard 120-second timeout for delegated tasks to prevent runaway sub-agents
    delegate_timeout = int(os.environ.get("HERMES_DELEGATE_TIMEOUT", "120"))
    return await asyncio.wait_for(
        loop.run_in_executor(None, _run_sync),
        timeout=delegate_timeout,
    )


# ---------------------------------------------------------------------------
# Tool registration — sync bridge for tools/registry.py
# ---------------------------------------------------------------------------


PERSISTENT_DELEGATE_TASK_SCHEMA = {
    "name": "persistent_delegate_task",
    "description": (
        "Hand a task to a persistent named agent (e.g. 'alice', 'bob') whose "
        "memory lives in HIPP0. The named agent shares cross-session memory "
        "with every other persistent agent in the same project — anything "
        "Alice learns is retrievable by Bob.\n\n"
        "Use this INSTEAD of delegate_task when:\n"
        "- The task should be handled by a specific persona (sales, product, …).\n"
        "- You want the conversation to feed the shared HIPP0 memory graph.\n"
        "- You want the delegate to remember past user preferences.\n\n"
        "The tool runs the delegate end-to-end: HIPP0 session start, context "
        "compile, delegate conversation, transcript capture, optional "
        "session end. The returned payload includes the delegate's final "
        "message plus a 'captured.capture_id' you can poll with "
        "GET /api/capture/:id if you need extracted snippet ids."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the persistent agent (e.g. 'alice'). Must exist "
                    "under <hermes_root>/agents/."
                ),
            },
            "task": {
                "type": "string",
                "description": "Task description the delegate should act on.",
            },
            "platform": {
                "type": "string",
                "description": (
                    "Platform the delegation is happening on "
                    "('cli', 'telegram', 'discord', 'web'). Defaults to 'cli'."
                ),
            },
            "user_id": {
                "type": "string",
                "description": "Optional external user id (e.g. Telegram user id).",
            },
            "external_chat_id": {
                "type": "string",
                "description": "Optional external chat id (e.g. Telegram chat id).",
            },
            "end_session": {
                "type": "boolean",
                "description": (
                    "If true, call HIPP0 session/end after the delegate "
                    "completes. Use for one-shot CLI invocations; keep "
                    "false for long-lived Telegram conversations."
                ),
            },
        },
        "required": ["agent_name", "task"],
    },
}


def persistent_delegate_task_handler(
    args: Dict[str, Any],
    *,
    parent_agent: Optional[Any] = None,
    **_: Any,
) -> str:
    """Sync tool-registry handler.

    Runs :meth:`PersistentDelegateTool.invoke` on a fresh event loop.
    Returns a JSON string (the standard tool-reply shape) so the
    parent AIAgent can inject it into the conversation.
    """
    agent_name = args.get("agent_name")
    task = args.get("task")
    if not agent_name or not task:
        return json.dumps(
            {"error": "persistent_delegate_task requires agent_name and task"}
        )

    tool = PersistentDelegateTool()

    async def _run() -> PersistentDelegateResult:
        return await tool.invoke(
            agent_name=agent_name,
            task=task,
            platform=args.get("platform", "cli"),
            user_id=args.get("user_id"),
            external_chat_id=args.get("external_chat_id"),
            parent_agent=parent_agent,
            end_session=bool(args.get("end_session", False)),
        )

    try:
        result = asyncio.run(_run())
    except PersistentDelegateError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("persistent_delegate_task failed")
        return json.dumps({"error": f"internal error: {e}"})

    return json.dumps(result.to_tool_payload())


# Registry wiring — guarded so importing the module in tests that don't
# rely on the full tool registry (e.g. this phase's unit tests) doesn't
# force all of tools/registry.py's import graph to load.
def _register() -> None:  # pragma: no cover - import-time wiring
    try:
        from tools.registry import registry
    except Exception as e:
        logger.debug("persistent_delegate_tool: registry import skipped (%s)", e)
        return
    registry.register(
        name="persistent_delegate_task",
        toolset="delegation",
        schema=PERSISTENT_DELEGATE_TASK_SCHEMA,
        handler=lambda args, **kw: persistent_delegate_task_handler(
            args, parent_agent=kw.get("parent_agent")
        ),
        emoji="🧠",
    )


_register()
