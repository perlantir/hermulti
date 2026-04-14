"""Daily self-reflection engine for persistent agents.

Runs on a stagger-scheduled daily cron per agent. Gathers the last N
days of session outcomes + tool usage + memory, sends a structured
analysis prompt to Claude Haiku via the existing auxiliary_client
infrastructure, and applies confidence-gated improvements:

- Memory proposals tagged "[auto-reflection]" are appended/replaced in
  MEMORY.md (max :data:`MAX_MEMORY_PER_CYCLE` per run).
- Skill proposals are NEVER auto-applied — logged for user review.
- Cross-agent observations are captured to HIPP0 via the normal
  ``/api/capture`` pipeline with ``source="reflection"``.

All actions append-only logged to ``~/.hermes/agents/<name>/reflection_log.jsonl``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence thresholds and caps
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "memory_add": 0.5,
    "memory_replace": 0.7,
    "skill_create": 1.0,
    "skill_update": 1.0,
    "skill_search_hub": 0.3,
    "cross_agent": 0.6,
}

MAX_MEMORY_PER_CYCLE = 3
# Skills are now auto-creatable (capped at 1/cycle) but only after passing an
# evidence gate: the candidate must be backed by at least one NEGATIVE-outcome
# session that mentions a topic token from the proposed skill name within the
# lookback window.  Without a prior failure to anchor the skill to, we log the
# proposal as "skill_eval_gate_failed" and skip creation.
MAX_SKILLS_PER_CYCLE = 1
# Window (days) searched for prior-negative evidence when scoring a candidate.
SKILL_EVIDENCE_LOOKBACK_DAYS = 7

AUTO_REFLECTION_TAG = "[auto-reflection]"
MIN_OVERALL_CONFIDENCE = 0.5
MIN_SESSIONS_REQUIRED = 3
DEFAULT_LOOKBACK_DAYS = 7
# Sessions older than this with NULL outcome are backfilled via heuristic
# before the reflection prompt is built, so stale entries don't sit forever
# as "no outcome recorded".
AGED_NULL_OUTCOME_DAYS = 3
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReflectionInput:
    agent_name: str
    recent_sessions: List[Dict[str, Any]] = field(default_factory=list)
    positive_sessions: List[Dict[str, Any]] = field(default_factory=list)
    negative_sessions: List[Dict[str, Any]] = field(default_factory=list)
    current_skills: List[str] = field(default_factory=list)
    memory_snapshot: str = ""
    user_snapshot: str = ""
    tool_usage: Dict[str, int] = field(default_factory=dict)
    compiled_context: Optional[str] = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS


@dataclass
class ReflectionOutput:
    skill_proposals: List[Dict[str, Any]] = field(default_factory=list)
    memory_proposals: List[Dict[str, Any]] = field(default_factory=list)
    cross_agent_observations: List[Dict[str, Any]] = field(default_factory=list)
    skill_gap_queries: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _agent_dir(agent_name: str) -> Path:
    from hermes_cli.agent_registry import get_agent_dir
    return get_agent_dir(agent_name)


def _reflection_log_path(agent_name: str) -> Path:
    return _agent_dir(agent_name) / "reflection_log.jsonl"


def _append_log(agent_name: str, entry: Dict[str, Any]) -> None:
    path = _reflection_log_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("timestamp", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data gathering — no LLM call
# ---------------------------------------------------------------------------


def _state_db_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _backfill_aged_null_outcomes(
    con: sqlite3.Connection, pool: List[Dict[str, Any]]
) -> None:
    """Infer outcomes for aged NULL-outcome sessions and persist them.

    Sessions that ended more than ``AGED_NULL_OUTCOME_DAYS`` days ago without
    any reaction signal are unlikely to ever receive one. Run the same
    turn-boundary heuristic over the *last* user message in each such session
    and, if it yields a confident label, write it back so reflection can use
    it on future runs. Neutral/unknown cases are left NULL — the reflection
    prompt already buckets them as "no outcome recorded".
    """
    try:
        from agent.outcome_signals import infer_outcome_from_turn
    except Exception:
        return
    cutoff = time.time() - AGED_NULL_OUTCOME_DAYS * 86400
    for s in pool:
        if s.get("outcome"):
            continue
        ended = s.get("ended_at") or s.get("started_at") or 0
        if not ended or ended > cutoff:
            continue
        last_user = con.execute(
            """SELECT content FROM messages
               WHERE session_id = ? AND role = 'user'
               ORDER BY timestamp DESC LIMIT 1""",
            (s["id"],),
        ).fetchone()
        text = (last_user["content"] if last_user else "") or ""
        inferred = infer_outcome_from_turn(text, None, None)
        if inferred is None:
            continue
        try:
            con.execute(
                "UPDATE sessions SET outcome = ?, outcome_source = ? "
                "WHERE id = ? AND outcome IS NULL",
                (inferred, "reflection_backfill", s["id"]),
            )
            con.commit()
            s["outcome"] = inferred
            s["outcome_source"] = "reflection_backfill"
        except sqlite3.Error as exc:
            logger.debug("NULL-outcome backfill failed for %s: %s", s.get("id"), exc)


def _query_sessions(
    agent_name: str,
    lookback_days: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return sessions for *agent_name* in the lookback window, split by outcome.

    Sessions are filtered by the ``agent_name`` column (set when an
    AIAgent is constructed for a persistent agent). Rows with a NULL
    ``agent_name`` are legacy/cross-flow sessions written before this
    column existed — we include them as a best-effort fallback so old
    sessions still show up in every agent's reflection rather than
    vanishing entirely.
    """
    cutoff = time.time() - lookback_days * 86400
    db_path = _state_db_path()
    if not db_path.exists():
        return {"all": [], "positive": [], "negative": [], "neutral": []}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT id, source, agent_name, started_at, ended_at, message_count,
                      outcome, outcome_source, outcome_detail
               FROM sessions
               WHERE started_at >= ?
                 AND (agent_name = ? OR agent_name IS NULL)
               ORDER BY started_at DESC
               LIMIT 500""",
            (cutoff, agent_name),
        ).fetchall()
        pool = [dict(r) for r in rows]
        # Attach the first user message as a summary
        for s in pool:
            msg_row = con.execute(
                """SELECT content FROM messages
                   WHERE session_id = ? AND role = 'user'
                   ORDER BY timestamp ASC LIMIT 1""",
                (s["id"],),
            ).fetchone()
            s["first_user_message"] = (msg_row["content"] or "")[:200] if msg_row else ""
        _backfill_aged_null_outcomes(con, pool)
        return {
            "all": pool,
            "positive": [s for s in pool if s.get("outcome") == "positive"],
            "negative": [s for s in pool if s.get("outcome") == "negative"],
            "neutral": [s for s in pool if not s.get("outcome")],
        }
    finally:
        con.close()


def _query_tool_usage(agent_name: str, lookback_days: int) -> Dict[str, int]:
    cutoff = time.time() - lookback_days * 86400
    db_path = _state_db_path()
    if not db_path.exists():
        return {}
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """SELECT m.tool_name, COUNT(*) as n
               FROM messages m JOIN sessions s ON m.session_id = s.id
               WHERE s.started_at >= ?
                 AND (s.agent_name = ? OR s.agent_name IS NULL)
                 AND m.tool_name IS NOT NULL
               GROUP BY m.tool_name
               ORDER BY n DESC
               LIMIT 50""",
            (cutoff, agent_name),
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}
    finally:
        con.close()


def _read_text_file(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8")
        return data[:limit]
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.debug("read %s failed: %s", path, exc)
        return ""


def _list_skills(agent_name: str) -> List[str]:
    skills_dir = _agent_dir(agent_name) / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(
        p.name for p in skills_dir.iterdir()
        if p.is_dir() and (p / "SKILL.md").is_file()
    )


UNUSED_SKILL_AGE_DAYS = 30


def _propose_unused_skill_deprecation(
    agent_name: str, tool_usage: Dict[str, int]
) -> None:
    """Log deprecation proposals (never auto-delete) for skills unused >30d.

    A skill is considered unused when its ``SKILL.md`` mtime is older than
    ``UNUSED_SKILL_AGE_DAYS`` and no token from the skill name appears as a
    substring of any recently-used tool name.  Pure log entry — a human
    reviews the reflection log to prune.
    """
    skills_dir = _agent_dir(agent_name) / "skills"
    if not skills_dir.is_dir():
        return
    cutoff = time.time() - UNUSED_SKILL_AGE_DAYS * 86400
    lowered_tools = [t.lower() for t in tool_usage.keys()]
    for p in sorted(skills_dir.iterdir()):
        skill_md = p / "SKILL.md"
        if not (p.is_dir() and skill_md.is_file()):
            continue
        try:
            mtime = skill_md.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        name_tokens = [t for t in _WORD_RE.findall(p.name.lower()) if len(t) >= 3]
        used = any(
            any(tok in tool for tok in name_tokens)
            for tool in lowered_tools
        )
        if used:
            continue
        _append_log(agent_name, {
            "action": "skill_deprecation_proposal",
            "data": {
                "skill": p.name,
                "path": str(p),
                "mtime": mtime,
                "age_days": (time.time() - mtime) / 86400,
                "reason": "unused_30d",
            },
            "applied": False,
        })


async def _try_compile_context(agent_name: str) -> Optional[str]:
    """Best-effort call to HIPP0 compile for self-improvement context."""
    try:
        from hermes_cli.agent_registry import get_agent
        profile = get_agent(agent_name)
        project_id = getattr(profile.config, "project_id", None)
        if not project_id:
            return None
        base_url = os.environ.get("HIPP0_BASE_URL", "http://127.0.0.1:3100")
        api_key = os.environ.get("HIPP0_API_KEY", "")
        if not api_key:
            key_file = os.environ.get("HIPP0_API_KEY_FILE")
            if key_file and Path(key_file).is_file():
                api_key = Path(key_file).read_text(encoding="utf-8").strip()
        if not api_key:
            return None
        import httpx
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        ) as client:
            resp = await client.post(
                "/api/hermes/compile",
                json={
                    "agent_name": agent_name,
                    "project_id": str(project_id),
                    "task": "self-improvement analysis",
                    "max_tokens": 2000,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("compiled_context") or data.get("context") or None
    except Exception as exc:
        logger.debug("compile call failed: %s", exc)
        return None


async def gather_reflection_input(
    agent_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    include_compile: bool = True,
) -> ReflectionInput:
    """Assemble reflection inputs. Native coroutine so the compile-context
    fetch joins the caller's running loop instead of opening a fresh one
    (which would crash under gateway concurrency)."""
    sessions = _query_sessions(agent_name, lookback_days)
    tool_usage = _query_tool_usage(agent_name, lookback_days)
    skills = _list_skills(agent_name)
    memory = _read_text_file(_agent_dir(agent_name) / "MEMORY.md", limit=4000)
    user_md = _read_text_file(_agent_dir(agent_name) / "USER.md", limit=2000)
    compiled: Optional[str] = None
    if include_compile:
        try:
            compiled = await _try_compile_context(agent_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("compile fetch failed: %s", exc)
            compiled = None
    return ReflectionInput(
        agent_name=agent_name,
        recent_sessions=sessions["all"],
        positive_sessions=sessions["positive"],
        negative_sessions=sessions["negative"],
        current_skills=skills,
        memory_snapshot=memory,
        user_snapshot=user_md,
        tool_usage=tool_usage,
        compiled_context=compiled,
        lookback_days=lookback_days,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _fmt_session_line(s: Dict[str, Any]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("started_at") or 0))
    first = (s.get("first_user_message") or "").replace("\n", " ")
    detail = s.get("outcome_detail") or ""
    return f"- {ts} | {first[:180]} | detail={detail[:120]}"


def _build_reflection_prompt(rin: ReflectionInput) -> str:
    total = len(rin.recent_sessions)
    pos = len(rin.positive_sessions)
    neg = len(rin.negative_sessions)
    neutral = max(0, total - pos - neg)
    neg_lines = "\n".join(_fmt_session_line(s) for s in rin.negative_sessions[:10]) or "(none)"
    pos_lines = "\n".join(_fmt_session_line(s) for s in rin.positive_sessions[:10]) or "(none)"
    tool_lines = "\n".join(
        f"  {name}: {count}"
        for name, count in sorted(rin.tool_usage.items(), key=lambda x: -x[1])[:20]
    ) or "  (no tool usage recorded)"
    skills_list = ", ".join(rin.current_skills[:50]) or "(none)"
    return f"""You are analyzing the recent performance of agent "{rin.agent_name}" to identify concrete improvements.

RECENT SESSIONS (last {rin.lookback_days} days):
- Total: {total}
- Positive outcomes: {pos}
- Negative outcomes: {neg}
- No outcome recorded: {neutral}

NEGATIVE OUTCOME SESSIONS (what went wrong — analyze these carefully):
{neg_lines}

POSITIVE OUTCOME SESSIONS (what went right — learn from these):
{pos_lines}

TOOL USAGE (last {rin.lookback_days} days):
{tool_lines}

CURRENT SKILLS ({len(rin.current_skills)}):
{skills_list}

CURRENT MEMORY ({len(rin.memory_snapshot)} chars):
{rin.memory_snapshot[:2000]}

Respond with ONLY this JSON (no markdown fences, no preamble):
{{
  "skill_proposals": [
    {{"action": "create|update|search_hub", "name": "skill-name", "reason": "why this would help based on the session data", "priority": "high|medium|low", "content_hint": "brief description"}}
  ],
  "memory_proposals": [
    {{"action": "add|replace", "target": "memory|user", "content": "the exact text to add/replace", "reason": "why this should be remembered based on the evidence"}}
  ],
  "cross_agent_observations": [
    {{"observation": "what was learned", "relevant_agents": ["agent1", "agent2"], "reason": "why they should know this"}}
  ],
  "skill_gap_queries": ["search terms for skill hub"],
  "overall_confidence": 0.0
}}

RULES:
- Only propose improvements with CLEAR EVIDENCE from the session data above.
- Prefer skill proposals for REPEATED patterns (3+ similar sessions).
- Prefer memory entries for one-off learnings or behavioral notes.
- Cross-agent observations should be genuinely useful to the named agents based on their roles.
- Set confidence below 0.5 if the data is too sparse to draw conclusions.
- If nothing needs improvement, return empty arrays and confidence 0.0.
- Memory content must be concise (under 200 chars per entry).
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_reflection_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("`").rstrip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _call_haiku(prompt: str) -> Optional[str]:
    from agent.auxiliary_client import async_call_llm
    try:
        resp = await async_call_llm(
            task="reflection",
            provider="anthropic",
            model=HAIKU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        choice = resp.choices[0]
        return getattr(choice.message, "content", None) or ""
    except Exception as exc:
        logger.warning("Haiku reflection call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _apply_memory_add(agent_name: str, content: str) -> None:
    memory_path = _agent_dir(agent_name) / "MEMORY.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    tagged = f"- {AUTO_REFLECTION_TAG} {content.strip()}"
    existing = ""
    if memory_path.is_file():
        existing = memory_path.read_text(encoding="utf-8")
    if tagged in existing:
        return
    sep = "" if existing.endswith("\n") or not existing else "\n"
    with open(memory_path, "a", encoding="utf-8") as f:
        if not existing:
            f.write("# MEMORY\n\n")
        f.write(f"{sep}{tagged}\n")


def _apply_memory_replace(agent_name: str, old: str, new: str) -> Optional[str]:
    memory_path = _agent_dir(agent_name) / "MEMORY.md"
    if not memory_path.is_file():
        return None
    existing = memory_path.read_text(encoding="utf-8")
    if old not in existing:
        return None
    tagged_new = f"{AUTO_REFLECTION_TAG} {new.strip()}"
    memory_path.write_text(existing.replace(old, tagged_new, 1), encoding="utf-8")
    return old


# ---------------------------------------------------------------------------
# Skill eval gate + auto-apply
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[a-z0-9]+")


def _skill_topic_tokens(proposal: Dict[str, Any]) -> List[str]:
    """Extract lowercase alphanumeric tokens from the skill's name/hint.

    Short stopword-like tokens ( <3 chars ) are dropped so we match on
    signal-bearing words rather than "to" / "a" / "of".
    """
    parts: List[str] = []
    for key in ("name", "content_hint", "reason"):
        parts.append(str(proposal.get(key) or ""))
    text = " ".join(parts).lower().replace("-", " ").replace("_", " ")
    return [t for t in _WORD_RE.findall(text) if len(t) >= 3]


def _score_skill_candidate(
    agent_name: str,
    proposal: Dict[str, Any],
    rin: ReflectionInput,
) -> Dict[str, Any]:
    """Require at least one prior NEGATIVE session in the lookback window
    whose first-user-message contains a topic token from the proposal.
    """
    tokens = _skill_topic_tokens(proposal)
    if not tokens:
        return {"passed": False, "reason": "no_topic_tokens",
                "tokens": [], "matches": 0}
    matches = 0
    matched_sessions: List[str] = []
    for s in rin.negative_sessions:
        text = (s.get("first_user_message") or "").lower()
        if any(tok in text for tok in tokens):
            matches += 1
            matched_sessions.append(s.get("id") or "")
    passed = matches >= 1
    return {
        "passed": passed,
        "reason": "ok" if passed else "no_prior_negative",
        "tokens": tokens,
        "matches": matches,
        "matched_sessions": matched_sessions[:5],
        "lookback_days": rin.lookback_days,
    }


def _apply_skill_create(
    agent_name: str, proposal: Dict[str, Any]
) -> Optional[Path]:
    """Create a minimal SKILL.md scaffold. Returns path, or None if the
    skill already exists or the name is invalid.
    """
    raw_name = str(proposal.get("name") or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", raw_name).strip("-")
    if not name:
        return None
    skills_dir = _agent_dir(agent_name) / "skills"
    skill_dir = skills_dir / name
    skill_md = skill_dir / "SKILL.md"
    if skill_md.is_file():
        return None
    skill_dir.mkdir(parents=True, exist_ok=True)
    reason = proposal.get("reason") or ""
    hint = proposal.get("content_hint") or ""
    body = (
        f"---\nname: {name}\nsource: auto-reflection\n"
        f"created_at: {time.time()}\n---\n\n"
        f"# {name}\n\n{hint}\n\n## Why\n{reason}\n"
    )
    skill_md.write_text(body, encoding="utf-8")
    return skill_md


# ---------------------------------------------------------------------------
# skill_outcomes A/B baseline + auto-invoke
# ---------------------------------------------------------------------------


def _ensure_skill_outcomes_table(con: sqlite3.Connection) -> None:
    """Create skill_outcomes table on demand.

    Kept in reflection.py (rather than hermes_state migrations) because it's
    a reflection-private artifact — it would be dead weight in session DBs
    for installs that never run reflection.
    """
    con.execute(
        """CREATE TABLE IF NOT EXISTS skill_outcomes (
               skill_id TEXT NOT NULL,
               agent_name TEXT NOT NULL,
               session_id TEXT,
               outcome TEXT,
               kind TEXT NOT NULL,
               ts REAL NOT NULL
           )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_skill_outcomes_skill "
        "ON skill_outcomes(skill_id, ts)"
    )


def _record_skill_outcome(
    skill_id: str,
    agent_name: str,
    session_id: Optional[str],
    outcome: Optional[str],
    kind: str,
    ts: Optional[float] = None,
) -> None:
    """Append a row to skill_outcomes. Best-effort; swallows DB errors."""
    db_path = _state_db_path()
    if not db_path.parent.exists():
        return
    try:
        con = sqlite3.connect(str(db_path))
        try:
            _ensure_skill_outcomes_table(con)
            con.execute(
                "INSERT INTO skill_outcomes "
                "(skill_id, agent_name, session_id, outcome, kind, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (skill_id, agent_name, session_id, outcome, kind,
                 ts if ts is not None else time.time()),
            )
            con.commit()
        finally:
            con.close()
    except sqlite3.Error as exc:
        logger.debug("skill_outcomes write failed: %s", exc)


def _register_skill_autoinvoke(
    agent_name: str, proposal: Dict[str, Any], rin: ReflectionInput
) -> None:
    """Match up to 3 recent sessions to the skill and capture a 7d baseline.

    The scheduler process has no live agent session to inject into, so
    "auto-invocation" here means wiring the skill into the outcome ledger:
      1. Baseline: same-topic outcomes in the 7d BEFORE skill creation are
         written as ``kind='baseline'`` rows.
      2. Matches: up to 3 recent sessions whose first-user-message contains
         a topic token are written as ``kind='match'`` rows so the outcome
         pipeline can later write a ``kind='post'`` row for the delta.

    An A/B summary is appended to the reflection log immediately.
    """
    skill_id = re.sub(r"[^a-z0-9]+", "-",
                      str(proposal.get("name") or "").lower()).strip("-")
    if not skill_id:
        return
    tokens = _skill_topic_tokens(proposal)
    now = time.time()
    baseline_cutoff = now - 7 * 86400
    baseline_outcomes: List[str] = []
    db_path = _state_db_path()
    matched_ids: List[str] = []
    if db_path.exists() and tokens:
        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
            try:
                rows = con.execute(
                    """SELECT s.id, s.outcome, m.content
                       FROM sessions s
                       LEFT JOIN messages m ON m.session_id = s.id
                                           AND m.role = 'user'
                       WHERE s.started_at >= ?
                         AND (s.agent_name = ? OR s.agent_name IS NULL)
                       ORDER BY s.started_at DESC
                       LIMIT 500""",
                    (baseline_cutoff, agent_name),
                ).fetchall()
            finally:
                con.close()
            seen: set = set()
            for r in rows:
                sid = r["id"]
                if sid in seen:
                    continue
                text = (r["content"] or "").lower()
                if any(tok in text for tok in tokens):
                    seen.add(sid)
                    if r["outcome"]:
                        baseline_outcomes.append(r["outcome"])
        except sqlite3.Error as exc:
            logger.debug("skill baseline query failed: %s", exc)

    for oc in baseline_outcomes:
        _record_skill_outcome(skill_id, agent_name, None, oc, "baseline", now)

    for s in rin.recent_sessions[:50]:
        text = (s.get("first_user_message") or "").lower()
        if any(tok in text for tok in tokens):
            matched_ids.append(s.get("id") or "")
            _record_skill_outcome(
                skill_id, agent_name, s.get("id"),
                s.get("outcome"), "match", now,
            )
            if len(matched_ids) >= 3:
                break

    pos = sum(1 for o in baseline_outcomes if o == "positive")
    neg = sum(1 for o in baseline_outcomes if o == "negative")
    _append_log(agent_name, {
        "action": "skill_autoinvoke_registered",
        "data": {
            "skill_id": skill_id,
            "matched_sessions": matched_ids,
            "baseline": {
                "total": len(baseline_outcomes),
                "positive": pos, "negative": neg,
                "ratio": (pos / len(baseline_outcomes))
                         if baseline_outcomes else None,
            },
        },
        "applied": True,
    })


def record_skill_outcome_for_session(
    skill_id: str,
    agent_name: str,
    session_id: str,
    outcome: str,
) -> None:
    """Public hook: called from the outcome-recording pipeline on sessions
    that were previously registered as a match for ``skill_id``.

    Writes a ``kind='post'`` row so the A/B delta can be computed later.
    """
    _record_skill_outcome(skill_id, agent_name, session_id, outcome, "post")


async def _capture_cross_agent_observation(
    agent_name: str, obs: Dict[str, Any]
) -> bool:
    try:
        from hermes_cli.agent_registry import get_agent
        from tools.persistent_delegate_tool import PersistentDelegateTool
        profile = get_agent(agent_name)
        tool = PersistentDelegateTool()
        provider = tool._make_provider(profile)
        try:
            observation = obs.get("observation", "")
            relevant = ", ".join(obs.get("relevant_agents") or [])
            reason = obs.get("reason", "")
            text = (
                f"REFLECTION by {agent_name}: {observation}\n"
                f"Relevant agents: {relevant}\n"
                f"Reason: {reason}"
            )
            await provider.capture(text, source="reflection")
            return True
        finally:
            await provider.aclose()
    except Exception as exc:
        logger.warning("cross-agent capture failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_reflection(
    agent_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    dry_run: bool = False,
) -> ReflectionOutput:
    """Run a reflection cycle for *agent_name*.

    Returns a :class:`ReflectionOutput` even on failure (empty fields).
    """
    out = ReflectionOutput()
    rin = await gather_reflection_input(agent_name, lookback_days=lookback_days)

    if len(rin.recent_sessions) < MIN_SESSIONS_REQUIRED:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "insufficient_sessions",
            "data": {"session_count": len(rin.recent_sessions)},
            "applied": False,
        })
        return out

    prompt = _build_reflection_prompt(rin)
    raw = await _call_haiku(prompt)
    if not raw:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "llm_call_failed",
            "applied": False,
        })
        return out
    parsed = _parse_reflection_json(raw)
    if not parsed:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "parse_error",
            "data": {"raw": raw[:500]},
            "applied": False,
        })
        return out

    out.skill_proposals = parsed.get("skill_proposals") or []
    out.memory_proposals = parsed.get("memory_proposals") or []
    out.cross_agent_observations = parsed.get("cross_agent_observations") or []
    out.skill_gap_queries = parsed.get("skill_gap_queries") or []
    try:
        out.overall_confidence = float(parsed.get("overall_confidence") or 0.0)
    except (TypeError, ValueError):
        out.overall_confidence = 0.0

    _append_log(agent_name, {
        "action": "reflection_complete",
        "data": {
            "overall_confidence": out.overall_confidence,
            "memory_count": len(out.memory_proposals),
            "skill_count": len(out.skill_proposals),
            "cross_agent_count": len(out.cross_agent_observations),
            "dry_run": dry_run,
        },
        "applied": False,
    })

    if dry_run or out.overall_confidence < MIN_OVERALL_CONFIDENCE:
        if not dry_run:
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "below_overall_confidence",
                "data": {"confidence": out.overall_confidence},
                "applied": False,
            })
        return out

    # Apply memory proposals (capped)
    applied_memory = 0
    for mp in out.memory_proposals:
        if applied_memory >= MAX_MEMORY_PER_CYCLE:
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "memory_cap_reached",
                "data": mp,
                "applied": False,
            })
            continue
        action = (mp.get("action") or "add").lower()
        target = (mp.get("target") or "memory").lower()
        if target != "memory":
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "user_md_edits_disabled",
                "data": mp,
                "applied": False,
            })
            continue
        threshold = CONFIDENCE_THRESHOLDS.get(
            f"memory_{action}", CONFIDENCE_THRESHOLDS["memory_add"]
        )
        if out.overall_confidence < threshold:
            _append_log(agent_name, {
                "action": "skipped_below_threshold",
                "data": {"proposal": mp, "threshold": threshold,
                         "confidence": out.overall_confidence},
                "applied": False,
            })
            continue
        content = (mp.get("content") or "").strip()
        if not content:
            continue
        try:
            if action == "replace":
                old_target = (mp.get("target_text") or content)
                old_value = _apply_memory_replace(agent_name, old_target, content)
                _append_log(agent_name, {
                    "action": "memory_replace",
                    "data": {"content": content, "replaced": old_value,
                             "reason": mp.get("reason")},
                    "applied": old_value is not None,
                })
                if old_value is not None:
                    applied_memory += 1
            else:
                _apply_memory_add(agent_name, content)
                _append_log(agent_name, {
                    "action": "memory_add",
                    "data": {"content": f"{AUTO_REFLECTION_TAG} {content}",
                             "reason": mp.get("reason")},
                    "applied": True,
                })
                applied_memory += 1
        except Exception as exc:
            logger.warning("memory apply failed: %s", exc)
            _append_log(agent_name, {
                "action": "error",
                "data": {"proposal": mp, "error": str(exc)},
                "applied": False,
            })

    # Skill proposals — apply at most MAX_SKILLS_PER_CYCLE, gated by evidence eval.
    applied_skills = 0
    for sp in out.skill_proposals:
        action = (sp.get("action") or "create").lower()
        if action != "create" or applied_skills >= MAX_SKILLS_PER_CYCLE:
            _append_log(agent_name, {
                "action": "skill_proposal",
                "data": sp,
                "applied": False,
                "reason": "requires_user_review"
                          if action != "create" else "skill_cap_reached",
            })
            continue
        evidence = _score_skill_candidate(agent_name, sp, rin)
        if not evidence.get("passed"):
            _append_log(agent_name, {
                "action": "skill_eval_gate_failed",
                "data": {"proposal": sp, "evidence": evidence},
                "applied": False,
            })
            continue
        try:
            created_path = _apply_skill_create(agent_name, sp)
        except Exception as exc:
            logger.warning("skill create failed: %s", exc)
            _append_log(agent_name, {
                "action": "error",
                "data": {"proposal": sp, "error": str(exc)},
                "applied": False,
            })
            continue
        if not created_path:
            _append_log(agent_name, {
                "action": "skill_proposal",
                "data": sp,
                "applied": False,
                "reason": "skill_already_exists",
            })
            continue
        _append_log(agent_name, {
            "action": "skill_create",
            "data": {"proposal": sp, "path": str(created_path),
                     "evidence": evidence},
            "applied": True,
        })
        applied_skills += 1
        # Auto-invoke wiring: match recent sessions + capture 7d baseline so
        # subsequent record_outcome calls on matched sessions can be scored
        # as an A/B delta vs the pre-creation window.
        try:
            _register_skill_autoinvoke(agent_name, sp, rin)
        except Exception as exc:
            logger.debug("skill auto-invoke wiring failed: %s", exc)

    # Cross-agent observations
    if out.overall_confidence >= CONFIDENCE_THRESHOLDS["cross_agent"]:
        for obs in out.cross_agent_observations:
            ok = await _capture_cross_agent_observation(agent_name, obs)
            _append_log(agent_name, {
                "action": "cross_agent_capture",
                "data": obs,
                "applied": ok,
            })

    # Skill gap queries
    for q in out.skill_gap_queries:
        _append_log(agent_name, {
            "action": "skill_gap_search",
            "data": {"query": q},
            "applied": False,
        })

    # Propose deprecation for skills that haven't been touched / used in 30d.
    try:
        _propose_unused_skill_deprecation(agent_name, rin.tool_usage)
    except Exception as exc:
        logger.debug("unused skill proposal failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Cron registration
# ---------------------------------------------------------------------------


def stagger_schedule(agent_name: str) -> str:
    """Generate a staggered daily cron schedule based on agent name hash."""
    h = int(hashlib.md5(agent_name.encode()).hexdigest()[:8], 16)
    hour = (h % 6) + 2   # 2am - 7am
    minute = h % 60
    return f"{minute} {hour} * * *"


def _reflection_job_name(agent_name: str) -> str:
    return f"reflection:{agent_name}"


def ensure_reflection_cron_jobs(
    agents: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Register a daily reflection cron job for each agent.

    Idempotent: jobs with the reserved name prefix ``reflection:<agent>``
    are replaced if already present so the schedule stays in sync with
    :func:`stagger_schedule`.

    Returns a mapping of ``agent_name -> schedule``.
    """
    from cron.jobs import load_jobs, save_jobs, create_job, remove_job
    from hermes_cli.agent_registry import list_agents

    names = agents or list_agents()
    schedules: Dict[str, str] = {}
    existing_jobs = load_jobs()
    existing_by_name = {j.get("name"): j for j in existing_jobs}

    for agent in names:
        job_name = _reflection_job_name(agent)
        schedule = stagger_schedule(agent)
        schedules[agent] = schedule
        if job_name in existing_by_name:
            prior = existing_by_name[job_name]
            if prior.get("schedule_display") == schedule and prior.get("job_type") == "reflection":
                continue
            try:
                remove_job(prior["id"])
            except Exception:
                pass

        prompt = (
            f"[reflection job for {agent}] "
            "This job is handled by the reflection runner, not the agent."
        )
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=job_name,
            repeat=None,
            deliver="local",
        )
        # Tag the job so the scheduler can dispatch it to run_reflection.
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["job_type"] = "reflection"
                j["reflection_agent"] = agent
                break
        save_jobs(jobs)

    return schedules


def run_reflection_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entry point used by the cron scheduler."""
    agent = job.get("reflection_agent") or ""
    if not agent:
        return {"ok": False, "error": "missing reflection_agent"}
    try:
        result = asyncio.run(run_reflection(agent, dry_run=False))
        return {
            "ok": True,
            "confidence": result.overall_confidence,
            "memory_proposals": len(result.memory_proposals),
            "skill_proposals": len(result.skill_proposals),
            "cross_agent_observations": len(result.cross_agent_observations),
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("reflection job failed: %s", exc)
        return {"ok": False, "error": str(exc)}
