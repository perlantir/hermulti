"""
TriggerMatcher: maps SkillEvents to skills whose triggers match.

Strategy:
  1. Each skill's textual triggers are pre-compiled into regex patterns +
     event-type tags via `_compile_trigger`.
  2. On dispatch, the matcher walks all skills and returns those whose
     compiled triggers match the event.

Future-proof: an optional LLM classifier can be plugged in for ambiguous
events (gated by HIPP0_SKILL_LLM_MATCH=on); regex matching always runs first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from agent.skills.loader import Skill, SkillSet


class EventType(str, Enum):
    INBOUND_MESSAGE = "inbound_message"      # user message arrives
    OUTBOUND_MESSAGE = "outbound_message"    # assistant response produced
    PRE_TASK = "pre_task"                    # before task work starts
    POST_DECISION = "post_decision"          # after a decision is recorded
    POST_OUTCOME = "post_outcome"            # after an outcome is recorded
    NEW_ENTITY = "new_entity"                # entity mentioned for first time
    INGEST_DOCUMENT = "ingest_document"      # PDF/transcript handed to agent
    HEALTH_CHECK = "health_check"            # explicit maintenance request


@dataclass
class SkillEvent:
    type: EventType
    text: str = ""                             # free-form text payload
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class CompiledTrigger:
    """A pre-compiled trigger ready for matching."""
    skill_name: str
    raw: str
    regex: Optional[re.Pattern[str]] = None    # set when text-based pattern present
    event_types: list[EventType] = field(default_factory=list)
    always_on: bool = False                    # always returns True for any inbound event


# Hand-mapping of common trigger phrases to event types.
# Triggers come from human-written SKILL.md files - we map known phrases to
# concrete event types. Anything that doesn't match a known phrase becomes a
# regex over the event's text payload.
_PHRASE_TO_EVENTS: list[tuple[re.Pattern[str], list[EventType]]] = [
    (re.compile(r"every inbound message", re.I), [EventType.INBOUND_MESSAGE]),
    (re.compile(r"\boutbound\b|\bafter the assistant responds\b", re.I), [EventType.OUTBOUND_MESSAGE]),
    (re.compile(r"\bbefore any task\b|\bstarting a (?:new )?task\b|\bpre[- ]task\b", re.I), [EventType.PRE_TASK]),
    (re.compile(r"\bafter (?:a|any) decision\b|\bdecision recorded\b|\bpost[- ]decision\b", re.I), [EventType.POST_DECISION]),
    (re.compile(r"\bafter (?:a|any) outcome\b|\bpost[- ]outcome\b|\bouttcome (?:known|recorded|signal)\b", re.I), [EventType.POST_OUTCOME]),
    (re.compile(r"task complete", re.I), [EventType.POST_OUTCOME]),
    (re.compile(r"new entity mentioned|entity mention", re.I), [EventType.NEW_ENTITY, EventType.INBOUND_MESSAGE]),
    (re.compile(r"\bingest(?:ing)?\b.*\b(pdf|transcript|document)\b|user provides a document", re.I), [EventType.INGEST_DOCUMENT]),
    (re.compile(r"health check|clean up memory|run health|stale", re.I), [EventType.HEALTH_CHECK]),
    (re.compile(r"creating/?merging/?exploring a knowledge branch|merge branch|create a branch", re.I), [EventType.PRE_TASK]),
]


def _compile_trigger(skill_name: str, raw: str) -> CompiledTrigger:
    """Compile a raw trigger string into matchable form."""
    raw_stripped = raw.strip()
    always_on = '(always-on)' in raw_stripped.lower()

    event_types: list[EventType] = []
    for pat, evts in _PHRASE_TO_EVENTS:
        if pat.search(raw_stripped):
            for e in evts:
                if e not in event_types:
                    event_types.append(e)

    # Extract a quoted text fragment as a literal-substring regex (e.g. "we decided to")
    quoted = re.findall(r'"([^"]+)"', raw_stripped)
    bracketed = re.findall(r'\[([^\]]+)\]', raw_stripped)
    text_fragments = [q for q in quoted if q]
    text_fragments += [b for b in bracketed if b and not any(c in b for c in '[](){}')]

    regex: Optional[re.Pattern[str]] = None
    if text_fragments:
        # Combine fragments into one regex, escaping each
        parts = [re.escape(frag) for frag in text_fragments]
        regex = re.compile(r'(' + r'|'.join(parts) + r')', re.I)
    elif not event_types:
        # Free-form trigger: build a loose keyword regex from significant words
        # Strip parenthesised hints and extract words >= 4 chars
        cleaned = re.sub(r'\([^)]*\)', '', raw_stripped)
        words = [w for w in re.findall(r'[A-Za-z][A-Za-z0-9_-]{3,}', cleaned)]
        if words:
            kw = r'\b(' + r'|'.join(re.escape(w) for w in words[:6]) + r')\b'
            regex = re.compile(kw, re.I)

    return CompiledTrigger(
        skill_name=skill_name,
        raw=raw_stripped,
        regex=regex,
        event_types=event_types,
        always_on=always_on,
    )


class TriggerMatcher:
    def __init__(self, skill_set: SkillSet, llm_classifier: Optional[Callable[[SkillEvent, list[Skill]], list[str]]] = None):
        self._skill_set = skill_set
        self._llm_classifier = llm_classifier
        self._compiled: dict[str, list[CompiledTrigger]] = {}
        for skill in skill_set.skills:
            self._compiled[skill.name] = [_compile_trigger(skill.name, t) for t in skill.triggers]

    def match(self, event: SkillEvent) -> list[Skill]:
        """Return the list of skills whose triggers match the event."""
        matched: list[Skill] = []
        for skill in self._skill_set.skills:
            for trig in self._compiled[skill.name]:
                if self._trigger_matches(trig, event):
                    matched.append(skill)
                    break
        # Optional LLM classifier for events with no regex match (or to disambiguate)
        if self._llm_classifier and not matched:
            try:
                names = self._llm_classifier(event, self._skill_set.skills)
                for name in names:
                    sk = self._skill_set.get(name)
                    if sk and sk not in matched:
                        matched.append(sk)
            except Exception:
                pass
        return matched

    @staticmethod
    def _trigger_matches(trig: CompiledTrigger, event: SkillEvent) -> bool:
        # Always-on triggers fire on inbound/outbound message events
        if trig.always_on and event.type in (EventType.INBOUND_MESSAGE, EventType.OUTBOUND_MESSAGE):
            return True
        # Event-type matches (e.g. PRE_TASK trigger fires on a PRE_TASK event)
        if trig.event_types and event.type in trig.event_types:
            return True
        # Text regex match against event payload
        if trig.regex and event.text and trig.regex.search(event.text):
            return True
        return False
