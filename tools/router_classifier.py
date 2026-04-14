"""Similarity-based task classifier for the persistent_delegate router.

The v1 router used a keyword switch — misroutes anything phrased obliquely
(e.g. "the thing is weird" has no keyword hit). This module upgrades the
decision to a lexical cosine over labeled seed sentences per class, which
behaves like a small embedding classifier without pulling sentence-transformers
into the hot path.

Each class (``technical``, ``user``, ``self_contained``, ``ambiguous``) owns a
seed list; at classify time the task is tokenised and its TF-IDF-lite vector is
cosine-compared against each class centroid. The top class wins, subject to
a margin threshold — below the margin, the task routes to ``ambiguous`` and
flags ``routing_uncertain=True`` so the caller can opt for the safer full
compile.

Design choices:

* Zero external ML deps — works in the Termux/low-resource install path.
* Seeds live next to the module for easy hand-tuning; future work can load
  them from a YAML file once Phase 13's routing_outcomes feedback edge is
  producing enough data to justify periodic re-seeding.
* Deterministic — same task text always produces the same decision, so the
  routing_outcomes log is cleanly attributable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ------------------------------------------------------------------
#  Seed sentences. Hand-tuned; ~12 per class. Tune via routing_outcomes.


_SEEDS: Dict[str, List[str]] = {
    "technical": [
        "fix the bug in the auth handler",
        "debug the stack trace we just saw",
        "why does this keep crashing",
        "the database connection errored out again",
        "tests are failing after the last refactor",
        "how do I make this query run faster",
        "production is returning 500s",
        "the response time is regressing",
        "track down the memory leak",
        "this deadlock is intermittent",
        "something broke after the migration",
        "it is weird today, the queue is stuck",
    ],
    "user": [
        "remember that I prefer tabs over spaces",
        "my style is concise variable names",
        "I like short functions",
        "remind me about the deadline tomorrow",
        "I prefer dark mode in all our dashboards",
        "I always want commits signed",
        "my usual workflow starts with a branch",
        "note that I live in Europe timezone",
        "I like to see tests run before merge",
        "remember my phone number is sensitive, do not log it",
        "prefer MLA citation style",
        "my editor is neovim",
    ],
    "self_contained": [
        "write a hello world program",
        "print fizzbuzz from scratch",
        "implement a trivial factorial function",
        "give me a tiny demo of websockets",
        "write a simple example of a python decorator",
        "pure function that reverses a string",
        "minimal example of async await",
        "show the syntax of a Go goroutine",
        "a basic shell script that prints hostname",
        "write a toy HTTP server in Rust",
        "a small script that counts lines",
        "demo of a pure lambda in lisp",
    ],
    "ambiguous": [
        "can you take a look",
        "something is off",
        "check this for me",
        "thoughts",
        "any ideas",
        "what would you do here",
        "review please",
        "quick question",
        "whats wrong",
        "not sure about this",
        "see attached",
        "help with this",
    ],
}


# ------------------------------------------------------------------
#  Tokenisation & vectorisation


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "are", "was", "were", "be", "this", "that", "these", "those",
    "do", "does", "did", "it", "its", "about", "from", "at", "by", "as",
    "can", "you",
})


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) > 1 and t.lower() not in _STOPWORDS]


def _vectorize(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    # TF-IDF-lite: tf is raw count; idf is precomputed per-class training pass.
    counts = Counter(tokens)
    return {t: c * idf.get(t, 1.0) for t, c in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _build_idf(seeds: Dict[str, List[str]]) -> Dict[str, float]:
    # Document frequency across all seeds; idf = log(N / (1 + df)).
    docs = [_tokens(s) for cls_seeds in seeds.values() for s in cls_seeds]
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        for tok in set(doc):
            df[tok] += 1
    return {tok: math.log(n_docs / (1 + count)) + 1.0 for tok, count in df.items()}


def _build_centroids(seeds: Dict[str, List[str]], idf: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    centroids: Dict[str, Dict[str, float]] = {}
    for cls, sentences in seeds.items():
        agg: Dict[str, float] = {}
        for sentence in sentences:
            vec = _vectorize(_tokens(sentence), idf)
            for tok, w in vec.items():
                agg[tok] = agg.get(tok, 0.0) + w
        # Normalize by number of seeds so centroids with more examples don't dominate.
        if sentences:
            for tok in agg:
                agg[tok] /= len(sentences)
        centroids[cls] = agg
    return centroids


_IDF = _build_idf(_SEEDS)
_CENTROIDS = _build_centroids(_SEEDS, _IDF)


# ------------------------------------------------------------------
#  Public API


@dataclass
class RouterDecision:
    cls: str
    score: float
    margin: float  # score - runner-up score
    scores: Dict[str, float] = field(default_factory=dict)
    uncertain: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "class": self.cls,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "uncertain": self.uncertain,
        }


DEFAULT_MARGIN = 0.05
DEFAULT_MIN_SCORE = 0.10


def classify(task_description: str, *, margin: float = DEFAULT_MARGIN, min_score: float = DEFAULT_MIN_SCORE) -> RouterDecision:
    """Score the task against each class centroid; return the top class.

    ``margin`` — top class must beat runner-up by at least this much; otherwise
    the call is flagged uncertain and the caller should treat it as
    ``ambiguous``.
    ``min_score`` — below this absolute score, the class is also considered
    unreliable (short tasks hit few seed tokens).
    """
    text = (task_description or "").strip()
    if not text:
        return RouterDecision(cls="ambiguous", score=0.0, margin=0.0, uncertain=True)

    vec = _vectorize(_tokens(text), _IDF)
    scores: Dict[str, float] = {
        cls: _cosine(vec, centroid) for cls, centroid in _CENTROIDS.items()
    }
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cls, top_score = ordered[0]
    runner_score = ordered[1][1] if len(ordered) > 1 else 0.0
    actual_margin = top_score - runner_score
    uncertain = top_score < min_score or actual_margin < margin
    return RouterDecision(
        cls=top_cls if not uncertain else "ambiguous",
        score=top_score,
        margin=actual_margin,
        scores=scores,
        uncertain=uncertain,
    )


def decision_to_classify_task_hint(dec: RouterDecision) -> Dict[str, object]:
    """Translate a RouterDecision into the legacy classify_task() hint shape.

    Used by persistent_delegate_tool to swap in the new classifier without
    changing downstream call sites. Callers that want the full decision
    (for routing_outcomes logging) should consume ``classify`` directly.
    """
    if dec.cls == "self_contained":
        return {"skip_compile": True, "routing_uncertain": dec.uncertain}
    if dec.cls == "technical":
        return {"namespace": "technical", "fast_mode": False, "routing_uncertain": dec.uncertain}
    if dec.cls == "user":
        return {"namespace": "user", "fast_mode": True, "routing_uncertain": dec.uncertain}
    # ambiguous → default to full fast compile, no namespace; flag uncertainty
    # so the caller can widen the compile if desired.
    return {"namespace": None, "fast_mode": True, "routing_uncertain": True}
