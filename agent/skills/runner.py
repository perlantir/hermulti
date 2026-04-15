"""
SkillRunner: executes a matched skill by calling an LLM with the skill body
as instruction, the event as context, and parsing structured JSON actions.
Each action is dispatched to a corresponding tool method on the hipp0 provider.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agent.skills.loader import Skill
from agent.skills.matcher import SkillEvent

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal async LLM interface the runner needs."""
    async def call(self, system: str, user: str, *, max_tokens: int = 1500, temperature: float = 0.2) -> str: ...


class Hipp0ProviderProto(Protocol):
    """Subset of Hipp0MemoryProvider methods the runner dispatches to."""
    async def record_decision(self, *, title: str, rationale: str, tags: list[str] | None = None,
                              confidence: str = "medium", agent_name: str | None = None) -> bool: ...
    async def record_outcome(self, *args: Any, **kwargs: Any) -> bool: ...


@dataclass
class SkillResult:
    skill_name: str
    actions_attempted: int = 0
    actions_succeeded: int = 0
    actions_failed: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    raw_output: str = ""           # the raw LLM output (for debugging)
    actions: list[dict[str, Any]] = field(default_factory=list)  # parsed actions


_SYSTEM_TEMPLATE = """You are a skill executor. You will be given:
1. The skill instructions (markdown).
2. The triggering event (type and payload).

Read the skill instructions and decide what actions to take.
Output JSON ONLY in this exact shape:

{{
  "actions": [
    {{"type": "<action_name>", "args": {{...}} }},
    ...
  ]
}}

Supported action types:
  - record_decision: args={{"title", "rationale", "tags": [...], "confidence": "high|medium|low"}}
  - record_outcome:  args={{"session_id", "outcome": "positive|negative|neutral", "signal_source", "snippet_ids": [...] }}
  - log:             args={{"message"}}
  - noop:            args={{"reason"}}

If the skill instructions say to do nothing for this event, return {{"actions": []}}.
If the event is irrelevant, return {{"actions": [{{"type": "noop", "args": {{"reason": "..."}} }}]}}.

Be conservative. Only emit record_decision when the event clearly contains an explicit decision."""


class SkillRunner:
    """Executes a single skill against a single event."""

    def __init__(
        self,
        llm_client: LLMClient | None,
        hipp0_provider: Hipp0ProviderProto | None,
        *,
        agent_name: str = "hermes",
    ):
        self._llm = llm_client
        self._provider = hipp0_provider
        self._agent_name = agent_name

    async def run(self, skill: Skill, event: SkillEvent) -> SkillResult:
        result = SkillResult(skill_name=skill.name)

        if self._llm is None:
            result.error = "no LLM client configured"
            return result

        system = _SYSTEM_TEMPLATE
        user = self._build_user_prompt(skill, event)

        try:
            raw = await self._llm.call(system=system, user=user, max_tokens=1500, temperature=0.2)
        except Exception as exc:
            result.error = f"LLM call failed: {exc}"
            logger.debug("[skill:%s] LLM error: %s", skill.name, exc)
            return result

        result.raw_output = raw
        actions = self._parse_actions(raw)
        result.actions = actions
        result.actions_attempted = len(actions)

        for action in actions:
            ok = await self._dispatch_action(action, skill, event)
            if ok:
                result.actions_succeeded += 1
            else:
                result.actions_failed += 1

        return result

    @staticmethod
    def _build_user_prompt(skill: Skill, event: SkillEvent) -> str:
        return (
            f"# Skill: {skill.name}\n\n"
            f"## Skill instructions\n{skill.body}\n\n"
            f"## Event\n"
            f"type: {event.type.value}\n"
            f"text: {event.text[:4000]}\n"
            f"metadata: {json.dumps(event.metadata, default=str)[:1000]}\n\n"
            "Return JSON only."
        )

    @staticmethod
    def _parse_actions(raw: str) -> list[dict[str, Any]]:
        """Extract the actions array from the LLM response."""
        if not raw:
            return []
        # Find the first {...} block (LLMs sometimes wrap in prose or code fences)
        m = re.search(r'\{[\s\S]*\}', raw)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except (ValueError, json.JSONDecodeError):
            return []
        actions = parsed.get('actions') if isinstance(parsed, dict) else None
        if not isinstance(actions, list):
            return []
        # Sanity-check each action
        clean: list[dict[str, Any]] = []
        for a in actions:
            if isinstance(a, dict) and isinstance(a.get('type'), str):
                clean.append({'type': a['type'], 'args': a.get('args') or {}})
        return clean

    async def _dispatch_action(self, action: dict[str, Any], skill: Skill, event: SkillEvent) -> bool:
        atype = action.get('type', '')
        args = action.get('args', {}) or {}

        if atype == 'log':
            logger.info("[skill:%s] %s", skill.name, args.get('message', ''))
            return True
        if atype == 'noop':
            logger.debug("[skill:%s] noop: %s", skill.name, args.get('reason', ''))
            return True

        if self._provider is None:
            return False

        if atype == 'record_decision':
            try:
                return bool(await self._provider.record_decision(
                    title=str(args.get('title', ''))[:200],
                    rationale=str(args.get('rationale', ''))[:2000],
                    tags=list(args.get('tags') or [])[:10],
                    confidence=str(args.get('confidence', 'medium')),
                    agent_name=self._agent_name,
                ))
            except Exception as exc:
                logger.debug("[skill:%s] record_decision failed: %s", skill.name, exc)
                return False

        if atype == 'record_outcome':
            try:
                return bool(await self._provider.record_outcome(
                    session_id=args.get('session_id'),
                    outcome=args.get('outcome', 'neutral'),
                    signal_source=args.get('signal_source', f'skill:{skill.name}'),
                    snippet_ids=list(args.get('snippet_ids') or []),
                ))
            except Exception as exc:
                logger.debug("[skill:%s] record_outcome failed: %s", skill.name, exc)
                return False

        logger.debug("[skill:%s] Unknown action type: %s", skill.name, atype)
        return False
