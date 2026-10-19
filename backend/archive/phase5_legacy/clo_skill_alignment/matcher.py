"""
Match course curriculum to job skills via cosine similarity + max pool.

Algorithm (Path A):
  1. Build a list of "course strings" — one per lesson + one per learning outcome
  2. Embed course strings + job skills with the same SBERT model
  3. Compute cosine_sim(course, skills) — shape (K, N)
  4. For each skill, take MAX across all course strings (best evidence)
  5. Threshold at 0.60 → matched / gaps

Each lesson is rendered as "topic. covers" so the embedding captures
both the topic name and what's actually taught.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from embedder import Embedder


_DEFAULT_THRESHOLD = 0.60


def _build_course_strings(
    lessons: Sequence[dict],
    learning_outcomes: Sequence[str],
) -> tuple[list[str], list[dict]]:
    """Flatten lessons + outcomes into embed-ready strings.

    Returns:
        (strings, sources)  — same length, parallel lists.
        sources[i] tags string[i] with its origin: {"kind": "lesson"|"outcome", ...}
    """
    strings:  list[str]  = []
    sources:  list[dict] = []

    for l in lessons:
        text = f"{l['topic']}. {l['covers']}".strip(". ")
        strings.append(text)
        sources.append({"kind": "lesson", "n": l["n"], "topic": l["topic"]})

    for i, o in enumerate(learning_outcomes, 1):
        strings.append(o)
        sources.append({"kind": "outcome", "n": i})

    return strings, sources


def match(
    lessons: Sequence[dict],
    learning_outcomes: Sequence[str],
    job_skills: Sequence[str],
    threshold: float = _DEFAULT_THRESHOLD,
    job_freq: dict[str, int] | None = None,
    course_units: Sequence[str] | None = None,
) -> dict:
    """Run Path A matching end-to-end.

    Args:
        lessons:           list of {n, topic, covers} dicts from pdf_parser
        learning_outcomes: list of CLO strings from pdf_parser
        job_skills:        top-N job skills from Phase 3
        threshold:         cosine score above which a skill is "matched"
        job_freq:          optional {skill: frequency} for demand weighting
        course_units:      optional explicit course-side strings (e.g. ESCO
                           labels of CLO skills). When given, these replace the
                           CLO/lesson text as the course representation, so the
                           match is ESCO-label ↔ ESCO-label (symmetric).

    Returns:
        {
            "matched":   [{ skill, score, evidence, source, job_freq }],
            "gaps":      [{ skill, score, job_freq }],
            "scores":    { coverage_pct, demand_weighted_score },
            "metadata":  { lessons, outcomes, skills_compared, threshold, model, processed_ms }
        }
    """
    t0 = time.time()
    embedder = Embedder()

    if course_units:
        course_strings = list(course_units)
        sources = [{"kind": "skill", "n": i + 1} for i in range(len(course_strings))]
    else:
        course_strings, sources = _build_course_strings(lessons, learning_outcomes)
    course_vecs = embedder.embed(course_strings, task="course")
    skill_vecs  = embedder.embed(list(job_skills), task="skill")

    # both L2-normalized → cosine = dot product.
    # errstate guards against spurious FPE warnings raised by some BLAS
    # backends (notably macOS Accelerate) during float32 matmul; the
    # vectors are unit-norm so the result is always well-defined.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = course_vecs @ skill_vecs.T      # shape (K, N)
    max_scores = sim.max(axis=0)              # (N,)
    best_idx   = sim.argmax(axis=0)           # (N,)

    job_freq = job_freq or {}

    matched: list[dict] = []
    gaps:    list[dict] = []
    for i, skill in enumerate(job_skills):
        score    = float(max_scores[i])
        idx      = int(best_idx[i])
        freq     = int(job_freq.get(skill, 0))
        record   = {"skill": skill, "score": round(score, 3), "job_freq": freq}
        if score >= threshold:
            record["evidence"] = course_strings[idx]
            record["source"]   = sources[idx]
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
            "lessons":          len(lessons),
            "outcomes":         len(learning_outcomes),
            "skills_compared":  len(job_skills),
            "threshold_used":   threshold,
            "model":            embedder.model_name,
            "processed_ms":     int((time.time() - t0) * 1000),
        },
    }


# ---------- CLI end-to-end test ----------

if __name__ == "__main__":
    import sys

    from pdf_parser import parse_pdf

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else \
        "../../school_program/automa_theory_course.pdf"

    fake_skills = [
        # automata-theory skills SHOULD score high (>0.60)
        "regular expressions",
        "context-free grammars",
        "finite automata",
        "Turing machines",
        "formal language theory",
        "computational complexity",
        # generic dev skills SHOULD score low (<0.60) on this course
        "Python",
        "SQL",
        "REST APIs",
        "Git",
        "React",
        "machine learning",
        "Docker",
    ]
    fake_freq = {
        "Python": 412, "SQL": 287, "REST APIs": 145, "Git": 98, "React": 87,
        "machine learning": 156, "Docker": 73,
        "regular expressions": 34, "context-free grammars": 8,
        "finite automata": 5, "Turing machines": 3,
        "formal language theory": 2, "computational complexity": 11,
    }

    print("=" * 70)
    print("Step 1: parse PDF (LLM-structured)")
    print("=" * 70)
    parsed = parse_pdf(pdf_path)
    print(f"course:   {parsed['course']}")
    print(f"lessons:  {len(parsed['lessons'])}")
    print(f"outcomes: {len(parsed['learning_outcomes'])}")
    print()

    print("=" * 70)
    print("Step 2: embed + match")
    print("=" * 70)
    result = match(
        parsed["lessons"],
        parsed["learning_outcomes"],
        fake_skills,
        job_freq=fake_freq,
    )

    print()
    print(f"coverage:        {result['scores']['coverage_pct']}%")
    print(f"demand-weighted: {result['scores']['demand_weighted_score']}")
    print(f"latency:         {result['metadata']['processed_ms']} ms")
    print()

    print("✅ MATCHED")
    print("-" * 70)
    for r in result["matched"]:
        src = r["source"]
        tag = f"lesson {src['n']} ({src['topic']})" if src["kind"] == "lesson" else f"outcome {src['n']}"
        print(f"  {r['score']:.3f}  {r['skill']:<28}  📊 {r['job_freq']:>3} jobs   ← {tag}")
        print(f"           evidence: {r['evidence'][:90]}{'...' if len(r['evidence']) > 90 else ''}")
    print()

    print("❌ GAPS  (highest demand first)")
    print("-" * 70)
    for r in result["gaps"]:
        print(f"  {r['score']:.3f}  {r['skill']:<28}  📊 {r['job_freq']:>3} jobs")
