"""
End-to-end orchestrator for Phase 5 curriculum-job alignment.

Combines:
  pdf_parser.parse_pdf           (Stage 1 + 2)
  clo_skill_extractor (optional, currently bypassed — match on CLO text directly)
  matcher.match                  (cosine similarity matching)
  threshold classification       (MAINTAIN / REVISE / UPDATE)

CLI usage:
    python runner.py                       # default: Automata Theory PDF
    python runner.py /path/to/course.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matcher import match as run_matcher
from pdf_parser import parse_pdf


# default thresholds — tunable
TAU_HIGH = 0.66   # >= this → MAINTAIN
TAU_LOW  = 0.33   # >= this and < TAU_HIGH → REVISE
                  # < TAU_LOW → UPDATE
MATCH_THRESHOLD = 0.60   # per-skill cosine threshold for "matched"
K_GAPS = 5               # how many top-frequency gaps to report


def classify(coverage: float) -> str:
    """Map coverage score to action label."""
    if coverage >= TAU_HIGH:
        return "MAINTAIN"
    if coverage >= TAU_LOW:
        return "REVISE"
    return "UPDATE"


def frequency_weighted_coverage(matched: list[dict],
                                job_freq: dict[str, int]) -> float:
    """Compute Σ f(s) for matched skills / Σ f(s) over all skills."""
    total_freq = sum(job_freq.values()) or 1
    matched_freq = sum(r.get("job_freq", 0) for r in matched)
    return round(matched_freq / total_freq, 3)


def analyze_course(course_data: dict[str, Any],
                   job_freq: dict[str, int],
                   match_threshold: float = MATCH_THRESHOLD,
                   k_gaps: int = K_GAPS,
                   course_units: list[str] | None = None) -> dict[str, Any]:
    """Run the matching pipeline on a parsed course + aggregated job skills.

    Args:
        course_data: output of pdf_parser.parse_pdf()
        job_freq:    {skill: frequency} (output of aggregate_skills.py)
        match_threshold: cosine threshold for per-skill matching
        k_gaps:      number of top gaps to report
        course_units: optional ESCO labels for the course side (symmetric ESCO)

    Returns:
        Full recommendation report.
    """
    course = course_data.get("course", {}) or {}
    lessons = course_data.get("lessons", []) or []
    learning_outcomes = course_data.get("learning_outcomes", []) or []

    if not job_freq:
        raise ValueError("job_freq is empty — run aggregate_skills.py first")

    skills_list = list(job_freq.keys())

    # call existing matcher (it already produces matched / gaps / scores)
    matched_result = run_matcher(
        lessons=lessons,
        learning_outcomes=learning_outcomes,
        job_skills=skills_list,
        threshold=match_threshold,
        job_freq=job_freq,
        course_units=course_units,
    )

    # frequency-weighted coverage (more meaningful than count-based)
    coverage_weighted = frequency_weighted_coverage(matched_result["matched"],
                                                     job_freq)
    label = classify(coverage_weighted)

    # top-K gaps already sorted by job_freq desc in matcher; just trim
    top_gaps = matched_result["gaps"][:k_gaps]

    return {
        "course": {
            "code":      course.get("code"),
            "title":     course.get("title"),
            "credits":   course.get("credits"),
            "programme": course.get("programme"),
        },
        "recommendation": {
            "label":              label,
            "coverage_weighted":  coverage_weighted,
            "coverage_pct":       matched_result["scores"]["coverage_pct"],
            "matched_count":      len(matched_result["matched"]),
            "skills_compared":    matched_result["metadata"]["skills_compared"],
        },
        "matched":  matched_result["matched"],
        "gaps":     top_gaps,
        "thresholds": {
            "match_threshold": match_threshold,
            "tau_high":        TAU_HIGH,
            "tau_low":         TAU_LOW,
        },
        "metadata": matched_result["metadata"],
    }


def load_job_freq(path: str | Path) -> dict[str, int]:
    """Load a job_freq.json file (output of aggregate_skills.py)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- CLI ----------

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else (
        "./"
        "school_program/automa_theory_course.pdf"
    )
    job_freq_path = Path(__file__).parent / "job_freq.json"

    if not job_freq_path.exists():
        print(f"⚠️ {job_freq_path} not found.")
        print("Run aggregate_skills.py first.")
        sys.exit(1)

    print(f"[1/3] Parsing PDF: {pdf_path}")
    parsed = parse_pdf(pdf_path)
    print(f"      → {len(parsed['lessons'])} lessons, "
          f"{len(parsed['learning_outcomes'])} CLOs")
    print()

    print(f"[2/3] Loading job skills from {job_freq_path.name}")
    job_freq = load_job_freq(job_freq_path)
    print(f"      → {len(job_freq)} skills with frequencies")
    print()

    print(f"[3/3] Running matcher + classification...")
    report = analyze_course(parsed, job_freq)

    print()
    print("=" * 70)
    course = report["course"]
    print(f"COURSE:    {course['code']} — {course['title']}")
    print(f"PROGRAMME: {course['programme']}")
    print("=" * 70)
    rec = report["recommendation"]
    print()
    if rec["label"] == "MAINTAIN":
        emoji = "🟢"
    elif rec["label"] == "REVISE":
        emoji = "🟡"
    else:
        emoji = "🔴"
    print(f"  RECOMMENDATION:   {emoji} {rec['label']}")
    print(f"  Coverage score:   {rec['coverage_weighted'] * 100:.1f}%  "
          f"(frequency-weighted)")
    print(f"  Matched count:    {rec['matched_count']} of "
          f"{rec['skills_compared']} skills")
    print(f"  Latency:          {report['metadata']['processed_ms']} ms")
    print()

    print("✅ MATCHED SKILLS")
    print("-" * 70)
    if not report["matched"]:
        print("  (none above threshold)")
    for r in report["matched"]:
        src = r["source"]
        if src["kind"] == "lesson":
            tag = f"lesson {src['n']} ({src['topic'][:40]})"
        else:
            tag = f"outcome {src['n']}"
        print(f"  {r['score']:.3f}  {r['skill']:<30}  "
              f"📊 {r['job_freq']:>4}   ← {tag}")
    print()

    print(f"🔴 TOP {len(report['gaps'])} GAPS (high market demand, not taught)")
    print("-" * 70)
    for r in report["gaps"]:
        print(f"  {r['score']:.3f}  {r['skill']:<30}  📊 {r['job_freq']:>4}")
