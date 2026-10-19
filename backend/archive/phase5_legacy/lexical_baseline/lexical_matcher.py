"""
Lexical matcher — Ahadi (EDM 2022) exact-match baseline.

Decides "does this course cover this skill?" by **surface-form word overlap**,
not embedding similarity. A skill is COVERED if all of its content words
(after lowercasing, punctuation stripping, and stopword removal) appear in
some course unit (a CLO or a lesson). No embeddings, no ESCO.

Drop-in replacement for clo_skill_alignment/matcher.py:match — same arguments
and same return shape — so the runner/coverage/classify logic is identical and
only the matching mechanism differs.
"""

from __future__ import annotations

import re
import time
from typing import Sequence


# Minimal English stopword list — removed from both sides before overlap so
# that function words don't create spurious matches. Deliberately small and
# domain-neutral (no skill/verb words like "design" or "analyze" are removed).
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "be", "this", "that", "these",
    "those", "it", "its", "their", "your", "our", "using", "use", "used",
    "into", "via", "per", "etc", "including", "include", "such",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_OVERLAP = 1.0   # 1.0 = full containment (true exact-match baseline)


def _tokens(text: str) -> set[str]:
    """Lowercase → strip punctuation → tokenize → drop stopwords."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _build_course_units(
    lessons: Sequence[dict],
    learning_outcomes: Sequence[str],
) -> tuple[list[str], list[set[str]], list[dict]]:
    """Flatten lessons + outcomes into parallel (text, token-set, source) lists."""
    texts:  list[str]      = []
    toksets: list[set[str]] = []
    sources: list[dict]    = []

    for l in lessons:
        text = f"{l.get('topic', '')}. {l.get('covers', '')}".strip(". ")
        texts.append(text)
        toksets.append(_tokens(text))
        sources.append({"kind": "lesson", "n": l.get("n"), "topic": l.get("topic")})

    for i, o in enumerate(learning_outcomes, 1):
        texts.append(o)
        toksets.append(_tokens(o))
        sources.append({"kind": "outcome", "n": i})

    return texts, toksets, sources


def match(
    lessons: Sequence[dict],
    learning_outcomes: Sequence[str],
    job_skills: Sequence[str],
    threshold: float = _DEFAULT_OVERLAP,
    job_freq: dict[str, int] | None = None,
) -> dict:
    """Lexical word-overlap matching. Same return shape as the semantic matcher.

    Args:
        lessons:           [{n, topic, covers}] from pdf_parser
        learning_outcomes: [CLO strings]
        job_skills:        raw job-skill strings (NOT ESCO-normalized)
        threshold:         min fraction of a skill's content tokens that must
                           appear in a single course unit (1.0 = full
                           containment = true exact match)
        job_freq:          {skill: frequency} for demand weighting

    Returns:
        { matched, gaps, scores, metadata } — identical schema to matcher.match
    """
    t0 = time.time()
    job_freq = job_freq or {}

    texts, toksets, sources = _build_course_units(lessons, learning_outcomes)

    matched: list[dict] = []
    gaps:    list[dict] = []

    for skill in job_skills:
        sk_tokens = _tokens(skill)
        freq = int(job_freq.get(skill, 0))

        if not sk_tokens:
            # skill reduced to nothing but stopwords → cannot match
            gaps.append({"skill": skill, "score": 0.0, "job_freq": freq})
            continue

        best_ratio = 0.0
        best_idx = -1
        for i, unit in enumerate(toksets):
            overlap = len(sk_tokens & unit) / len(sk_tokens)
            if overlap > best_ratio:
                best_ratio = overlap
                best_idx = i

        record = {"skill": skill, "score": round(best_ratio, 3), "job_freq": freq}
        if best_ratio >= threshold and best_idx >= 0:
            record["evidence"] = texts[best_idx]
            record["source"]   = sources[best_idx]
            matched.append(record)
        else:
            gaps.append(record)

    matched.sort(key=lambda r: r["score"],    reverse=True)
    gaps.sort(   key=lambda r: r["job_freq"], reverse=True)

    coverage_pct = round(100 * len(matched) / max(len(job_skills), 1), 1)
    demand_weighted_score = sum(r["job_freq"] for r in matched)

    return {
        "matched": matched,
        "gaps":    gaps,
        "scores": {
            "coverage_pct":          coverage_pct,
            "demand_weighted_score": demand_weighted_score,
        },
        "metadata": {
            "lessons":         len(lessons),
            "outcomes":        len(learning_outcomes),
            "skills_compared": len(job_skills),
            "threshold_used":  threshold,
            "model":           "lexical-word-overlap",
            "esco":            False,
            "processed_ms":    int((time.time() - t0) * 1000),
        },
    }
