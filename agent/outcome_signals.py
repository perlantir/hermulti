"""Heuristic turn-boundary outcome inference.

A tiny, pure helper used to close the outcome-signal loop on a per-turn basis.
The caller (``run_agent.py`` turn loop, reflection backfill) decides what to do
with the inferred label — this module only classifies.

Return values match the vocabulary accepted by
``hermes_state.SessionDB.record_outcome`` / the hipp0 provider:
``"positive"``, ``"negative"``, or ``None`` for "no confident signal".
"""

from __future__ import annotations

from typing import Any, Optional


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
