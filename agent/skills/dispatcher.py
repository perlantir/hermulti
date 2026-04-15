"""
SkillDispatcher: orchestrates skill execution.

Responsibilities:
  1. Load skills via SkillLoader (once at construction).
  2. On each event, ask TriggerMatcher for matched skills.
  3. Execute matched skills via SkillRunner with priority ordering:
     - brain-ops READ phase fires before other skills on PRE_TASK events
     - brain-ops WRITE phase fires after other skills on POST_DECISION/POST_OUTCOME
     - signal-detector always runs in parallel (fire-and-forget) on INBOUND/OUTBOUND messages
     - Other matched skills run sequentially after the READ phase

Disable via HIPP0_SKILL_DISPATCHER=off (default: 'on' if an LLM is configured).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from agent.skills.loader import Skill, SkillSet, load_skills
from agent.skills.matcher import EventType, SkillEvent, TriggerMatcher
from agent.skills.runner import Hipp0ProviderProto, LLMClient, SkillResult, SkillRunner

logger = logging.getLogger(__name__)


@dataclass
class DispatchSummary:
    event_type: str
    matched_skills: list[str] = field(default_factory=list)
    results: list[SkillResult] = field(default_factory=list)
    parallel_tasks: int = 0       # signal-detector fire-and-forget tasks created


class SkillDispatcher:
    """Orchestrates skill execution for an agent's lifecycle events."""

    def __init__(
        self,
        skills_dir: str | None = None,
        llm_client: LLMClient | None = None,
        hipp0_provider: Hipp0ProviderProto | None = None,
        *,
        agent_name: str = "hermes",
        skill_set: Optional[SkillSet] = None,
    ):
        self._skill_set = skill_set if skill_set is not None else load_skills(skills_dir)
        self._matcher = TriggerMatcher(self._skill_set)
        self._runner = SkillRunner(llm_client, hipp0_provider, agent_name=agent_name)
        self._enabled = self._compute_enabled(llm_client)
        self._background_tasks: set[asyncio.Task[SkillResult]] = set()
        if self._enabled:
            logger.info(
                "[skill-dispatcher] Enabled. %d skills loaded from %s",
                len(self._skill_set.skills), self._skill_set.skills_dir,
            )
        else:
            logger.debug("[skill-dispatcher] Disabled (no LLM client or HIPP0_SKILL_DISPATCHER=off)")

    @staticmethod
    def _compute_enabled(llm_client: LLMClient | None) -> bool:
        env = os.environ.get('HIPP0_SKILL_DISPATCHER', 'auto').lower()
        if env in ('off', 'false', '0'):
            return False
        if env in ('on', 'true', '1'):
            return True
        # 'auto': enabled iff an LLM client is wired
        return llm_client is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def skills(self) -> list[Skill]:
        return list(self._skill_set.skills)

    async def dispatch(self, event: SkillEvent) -> DispatchSummary:
        """Dispatch an event to matched skills and return a summary.

        Order of execution:
          1. PRE_TASK phase: run brain-ops READ first (sequential, awaited)
          2. signal-detector (always fire-and-forget on INBOUND/OUTBOUND)
          3. Other matched skills (sequential, awaited if mutating)
          4. POST_DECISION/POST_OUTCOME: brain-ops WRITE last
        """
        summary = DispatchSummary(event_type=event.type.value)

        if not self._enabled:
            return summary

        try:
            matched = self._matcher.match(event)
        except Exception as exc:
            logger.warning("[skill-dispatcher] match failed: %s", exc)
            return summary

        if not matched:
            return summary

        summary.matched_skills = [s.name for s in matched]

        # Partition the matched set
        signal_detector = next((s for s in matched if s.name == 'signal-detector'), None)
        brain_ops = next((s for s in matched if s.name == 'brain-ops'), None)
        others = [s for s in matched if s.name not in ('signal-detector', 'brain-ops')]

        # 1. brain-ops READ first on PRE_TASK
        if brain_ops and event.type == EventType.PRE_TASK:
            try:
                summary.results.append(await self._runner.run(brain_ops, event))
            except Exception as exc:
                logger.debug("[skill-dispatcher] brain-ops READ error: %s", exc)

        # 2. signal-detector: always parallel/fire-and-forget on inbound/outbound messages
        if signal_detector and event.type in (EventType.INBOUND_MESSAGE, EventType.OUTBOUND_MESSAGE):
            task = asyncio.create_task(self._safe_run(signal_detector, event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            summary.parallel_tasks += 1

        # 3. Other matched skills, sequential
        for skill in others:
            try:
                summary.results.append(await self._runner.run(skill, event))
            except Exception as exc:
                logger.debug("[skill-dispatcher] %s error: %s", skill.name, exc)

        # 4. brain-ops WRITE last on POST_DECISION/POST_OUTCOME
        if brain_ops and event.type in (EventType.POST_DECISION, EventType.POST_OUTCOME):
            try:
                summary.results.append(await self._runner.run(brain_ops, event))
            except Exception as exc:
                logger.debug("[skill-dispatcher] brain-ops WRITE error: %s", exc)

        return summary

    async def _safe_run(self, skill: Skill, event: SkillEvent) -> SkillResult:
        """Wrapper that swallows exceptions for fire-and-forget tasks."""
        try:
            return await self._runner.run(skill, event)
        except Exception as exc:
            logger.debug("[skill-dispatcher:bg] %s failed: %s", skill.name, exc)
            return SkillResult(skill_name=skill.name, error=str(exc))

    async def close(self) -> None:
        """Wait for any outstanding background tasks. Safe to call multiple times."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()
