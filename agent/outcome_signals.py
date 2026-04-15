"""Heuristic turn-boundary outcome inference.

A tiny, pure helper used to close the outcome-signal loop on a per-turn basis.
The caller (``run_agent.py`` turn loop, reflection backfill) decides what to do
with the inferred label — this module only classifies.

Return values match the vocabulary accepted by
``hermes_state.SessionDB.record_outcome`` / the hipp0 provider:
``"positive"``, ``"negative"``, or ``None`` for "no confident signal".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DecisionSignal:
    title: str
    rationale: str
    tags: list[str] = field(default_factory=list)
    confidence: str = "medium"  # high | medium | low


# Patterns that indicate a decision statement
_DECISION_PATTERNS = [
    r"(?:I'?ll|I will|we'?ll|we will|going to|decided to|choosing to)\s+(.{10,120})",
    r"(?:decided|decision|choosing|going with|opted for|selected)\s*[:\-]?\s*(.{10,120})",
    r"(?:rejected|ruled out|not going with|avoiding)\s+(.{2,120})\s+because\s+(.{10,200})",
    r"(?:the (?:best|right|correct) approach is|we should use)\s+(.{10,120})",
    r"(?:we|I)\s+(?:definitely|absolutely|must|clearly|always)\s+(?:must\s+)?(?:use|need|require|apply|implement)\s+(.{5,120})",
]

_CONFIDENCE_HIGH = re.compile(r"\b(definitely|clearly|absolutely|must|always)\b", re.I)
_CONFIDENCE_LOW = re.compile(r"\b(might|could|perhaps|maybe|probably|consider)\b", re.I)


def extract_decision_signals(turn_text: str, agent_name: str = "hermes") -> list[DecisionSignal]:
    """
    Scan assistant turn text for decision statements.
    Returns up to 5 signals per turn to avoid noise.
    """
    signals: list[DecisionSignal] = []

    for pattern in _DECISION_PATTERNS:
        for match in re.finditer(pattern, turn_text, re.IGNORECASE):
            full_match = match.group(0).strip()
            # Skip very short or very long matches
            if len(full_match) < 15 or len(full_match) > 300:
                continue

            # Infer confidence from language
            if _CONFIDENCE_HIGH.search(full_match):
                confidence = "high"
            elif _CONFIDENCE_LOW.search(full_match):
                confidence = "low"
            else:
                confidence = "medium"

            # Extract rough tags from capitalized nouns in the match
            tags = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', full_match)
            tags = [t.lower().replace(' ', '-') for t in tags if len(t) > 2][:5]

            signals.append(DecisionSignal(
                title=full_match[:120],
                rationale=full_match,
                tags=tags,
                confidence=confidence,
            ))

        if len(signals) >= 5:
            break

    return signals[:5]


_POSITIVE_MARKERS = (
    "thanks",
    "thank you",
    "perfect",
    "great",
    "exactly",
    "awesome",
    "nice work",
    "works",
)

_NEGATIVE_MARKERS = (
    "no,",
    "no.",
    "wrong",
    "that's not",
    "thats not",
    "undo",
    "revert",
    "not what",
    "incorrect",
)


def infer_outcome_from_turn(
    user_msg: Optional[str],
    assistant_msg: Optional[str] = None,
    prior_context: Optional[Any] = None,
) -> Optional[str]:
    """Infer a coarse outcome label from a single turn's user message.

    The signal comes from the *user's* message — it is feedback on whatever
    the assistant did previously. ``assistant_msg`` and ``prior_context`` are
    accepted for future enrichment but currently unused.

    Returns ``"positive"``, ``"negative"``, or ``None`` when no confident
    signal is detected. Negative markers take precedence over positive so a
    mixed message ("thanks but that's wrong") is flagged negative.
    """
    if not user_msg:
        return None
    text = user_msg.lower()
    if any(marker in text for marker in _NEGATIVE_MARKERS):
        return "negative"
    if any(marker in text for marker in _POSITIVE_MARKERS):
        return "positive"
    return None
