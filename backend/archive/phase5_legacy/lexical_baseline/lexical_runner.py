"""
Lexical baseline runner — Ahadi exact-match end-to-end.

Uses the SAME coverage formula and MAINTAIN/REVISE/UPDATE thresholds as
clo_skill_alignment/runner.py (copied here so the baseline is standalone and
identical except for the matching mechanism). Skills are used RAW (no ESCO).

CLI:
    python lexical_runner.py                     # built-in Automata sample
    python lexical_runner.py --job-freq f.json   # real {skill: freq}
    python lexical_runner.py --pdf course.pdf     # parse a real syllabus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lexical_matcher import match as run_lexical_match


# Same thresholds as the semantic pipeline (held constant for the ablation)
TAU_HIGH = 0.66   # >= → MAINTAIN
TAU_LOW  = 0.33   # >= and < TAU_HIGH → REVISE ; < TAU_LOW → UPDATE
OVERLAP_THRESHOLD = 1.0   # full word containment = exact match
K_GAPS = 5


def classify(coverage: float) -> str:
    if coverage >= TAU_HIGH:
        return "MAINTAIN"
    if coverage >= TAU_LOW:
        return "REVISE"
    return "UPDATE"


def frequency_weighted_coverage(matched: list[dict],
                                job_freq: dict[str, int]) -> float:
    total_freq = sum(job_freq.values()) or 1
    matched_freq = sum(r.get("job_freq", 0) for r in matched)
    return round(matched_freq / total_freq, 3)


def analyze_course(course_data: dict[str, Any],
                   job_freq: dict[str, int],
                   overlap_threshold: float = OVERLAP_THRESHOLD,
                   k_gaps: int = K_GAPS) -> dict[str, Any]:
    """Run the lexical baseline on a parsed course + raw job-skill frequencies."""
    course = course_data.get("course", {}) or {}
    lessons = course_data.get("lessons", []) or []
    learning_outcomes = course_data.get("learning_outcomes", []) or []

    if not job_freq:
        raise ValueError("job_freq is empty")

    result = run_lexical_match(
        lessons=lessons,
        learning_outcomes=learning_outcomes,
        job_skills=list(job_freq.keys()),
        threshold=overlap_threshold,
        job_freq=job_freq,
    )

    coverage_weighted = frequency_weighted_coverage(result["matched"], job_freq)
    label = classify(coverage_weighted)

    return {
        "course": {
            "code":      course.get("code"),
            "title":     course.get("title"),
            "programme": course.get("programme"),
        },
        "recommendation": {
            "label":             label,
            "coverage_weighted": coverage_weighted,
            "coverage_pct":      result["scores"]["coverage_pct"],
            "matched_count":     len(result["matched"]),
            "skills_compared":   result["metadata"]["skills_compared"],
        },
        "matched": result["matched"],
        "gaps":    result["gaps"][:k_gaps],
        "method":  "lexical",
        "metadata": result["metadata"],
    }


def analyze_one(course_data: dict[str, Any],
                job_freq: dict[str, int],
                overlap_threshold: float = OVERLAP_THRESHOLD,
                k_gaps: int = K_GAPS) -> dict[str, Any]:
    """Per-course lexical report shaped for curriculum_runner.aggregate_curriculum.

    Mirrors clo_skill_alignment/curriculum_runner.py:analyze_one so the shared,
    matcher-agnostic aggregate_curriculum() works unchanged for the lexical
    method (programme view).
    """
    course = course_data.get("course", {}) or {}
    lessons = course_data.get("lessons", []) or []
    los = course_data.get("learning_outcomes", []) or []

    result = run_lexical_match(
        lessons=lessons,
        learning_outcomes=los,
        job_skills=list(job_freq.keys()),
        threshold=overlap_threshold,
        job_freq=job_freq,
    )
    coverage_weighted = frequency_weighted_coverage(result["matched"], job_freq)
    return {
        "course": course,
        "label": classify(coverage_weighted),
        "coverage_weighted": coverage_weighted,
        "matched": result["matched"],
        "matched_skills_set": {m["skill"] for m in result["matched"]},
        "gaps_top_k": result["gaps"][:k_gaps],
    }


# ---------- built-in sample (Automata Theory) for an instant, Ollama-free demo ----------

SAMPLE_COURSE = {
    "course": {"code": "GICI3AT", "title": "Automata Theory",
               "programme": "Computer Sciences"},
    "learning_outcomes": [
        "Analyze formal languages and computational models, including finite "
        "automata, pushdown automata, and Turing machines, in terms of their "
        "structure, behavior, and computational power.",
        "Design computational models and formal representations to solve "
        "language recognition and computation problems using automata theory "
        "techniques.",
        "Evaluate problem-solving approaches in automata theory with respect to "
        "logical rigor, correctness, and professional responsibility.",
    ],
    "lessons": [
        {"n": 1, "topic": "Deterministic Finite Automata (DFA)",
         "covers": "Design DFA models to solve regular language recognition problems."},
        {"n": 2, "topic": "Regular Expressions",
         "covers": "Analyze regular expressions and equivalent finite automata."},
        {"n": 3, "topic": "Context-Free Grammars (CFG)",
         "covers": "Design context-free grammars for specified languages."},
        {"n": 4, "topic": "Pushdown Automata and Turing Machines",
         "covers": "Compare computational power of pushdown automata and Turing machines."},
    ],
}

# Mix of skills that SHOULD lexically match (share words with the CLOs) and
# generic ones that should NOT — to show the baseline discriminating.
SAMPLE_JOB_FREQ = {
    "finite automata": 40,
    "regular expressions": 34,
    "context-free grammars": 8,
    "turing machines": 3,
    "formal languages": 12,
    "problem solving": 117,
    "communication skills": 252,
    "customer service": 162,
    "microsoft excel": 95,
    "teamwork": 182,
}


def _print_report(report: dict) -> None:
    rec = report["recommendation"]
    emoji = {"MAINTAIN": "🟢", "REVISE": "🟡", "UPDATE": "🔴"}[rec["label"]]
    print("=" * 70)
    print(f"COURSE:  {report['course']['code']} — {report['course']['title']}")
    print(f"METHOD:  lexical (exact word-overlap, no ESCO)")
    print("=" * 70)
    print(f"  {emoji} {rec['label']}   coverage {rec['coverage_weighted']*100:.1f}%  "
          f"(weighted) · {rec['matched_count']}/{rec['skills_compared']} matched")
    print()
    print("✅ MATCHED")
    print("-" * 70)
    for r in report["matched"]:
        src = r["source"]
        tag = (f"lesson {src['n']} ({src['topic'][:30]})"
               if src["kind"] == "lesson" else f"outcome {src['n']}")
        print(f"  {r['score']:.2f}  {r['skill']:<26} 📊 {r['job_freq']:>4}  ← {tag}")
    if not report["matched"]:
        print("  (none)")
    print()
    print(f"🔴 TOP {len(report['gaps'])} GAPS")
    print("-" * 70)
    for r in report["gaps"]:
        print(f"  {r['score']:.2f}  {r['skill']:<26} 📊 {r['job_freq']:>4}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", help="parse a real syllabus PDF (needs Ollama)")
    p.add_argument("--job-freq", help="path to a {skill: freq} JSON (raw, no ESCO)")
    args = p.parse_args()

    if args.pdf:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clo_skill_alignment"))
        from pdf_parser import parse_pdf
        course_data = parse_pdf(args.pdf)
    else:
        course_data = SAMPLE_COURSE

    if args.job_freq:
        job_freq = {k: int(v) for k, v in json.loads(Path(args.job_freq).read_text()).items()}
    else:
        job_freq = SAMPLE_JOB_FREQ

    report = analyze_course(course_data, job_freq)
    _print_report(report)
