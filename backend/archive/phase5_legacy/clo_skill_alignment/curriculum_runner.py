"""
Phase 5 — Curriculum-level (programme-wide) recommendation runner.

Aggregates the per-course matching produced by `runner.py` into a single
programme-level label and gap list, following:

    matched(C) = ⋃_{c ∈ C} matched(c)
    coverage(C) = Σ f(s) for s ∈ matched(C) / Σ f(s) over all skills
    label = MAINTAIN / REVISE / UPDATE on the curriculum coverage

For this v1 we accept:
  - 1 real PDF (Automata Theory)
  - n MOCK course objects (constructed inline) to simulate a full
    Computer Sciences programme

This is enough to demo the programme-level view to the supervisor / the
audience without having every ITC PDF in hand.

CLI usage:
    python curriculum_runner.py                # default CS programme demo
    python curriculum_runner.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matcher import match as run_matcher
from pdf_parser import parse_pdf
from runner import (
    K_GAPS,
    MATCH_THRESHOLD,
    TAU_HIGH,
    TAU_LOW,
    classify,
    frequency_weighted_coverage,
    load_job_freq,
)


# ---------- MOCK COURSE DATA ----------
# Each mock course mimics the shape of pdf_parser.parse_pdf() output:
#   { "course": {...}, "lessons": [...], "learning_outcomes": [...] }

MOCK_COURSES: list[dict[str, Any]] = [
    {
        "course": {
            "code": "GICI3PROG",
            "title": "Programming for Data Science",
            "credits": 3,
            "programme": "Engineering Degree in Computer Sciences",
        },
        "lessons": [
            {"n": 1, "topic": "Python fundamentals",
             "covers": "Write idiomatic Python programs and use modules."},
            {"n": 2, "topic": "Data analysis with pandas",
             "covers": "Use pandas DataFrames for data analysis and aggregation."},
            {"n": 3, "topic": "REST APIs in Flask",
             "covers": "Design REST API endpoints in Python using Flask."},
            {"n": 4, "topic": "Version control with Git",
             "covers": "Use Git for source control, branches, and pull requests."},
            {"n": 5, "topic": "Containerization with Docker",
             "covers": "Build, run and orchestrate Docker containers for apps."},
        ],
        "learning_outcomes": [
            "Design Python programs that read, transform, and visualize data.",
            "Build a REST API in Flask that exposes data-analysis results.",
            "Use Git and Docker to package and deploy applications.",
        ],
    },
    {
        "course": {
            "code": "GICI3DB",
            "title": "Database Design and Administration",
            "credits": 3,
            "programme": "Engineering Degree in Computer Sciences",
        },
        "lessons": [
            {"n": 1, "topic": "Relational design and normalization",
             "covers": "Design normalized relational schemas (1NF–3NF)."},
            {"n": 2, "topic": "SQL querying",
             "covers": "Write SQL SELECT, JOIN, GROUP BY, window functions."},
            {"n": 3, "topic": "Indexing and performance",
             "covers": "Optimize queries with indexes and execution plans."},
            {"n": 4, "topic": "Data analysis with SQL",
             "covers": "Perform data analysis directly in SQL."},
        ],
        "learning_outcomes": [
            "Design relational schemas and write complex SQL queries.",
            "Perform data analysis on tabular data using SQL.",
        ],
    },
    {
        "course": {
            "code": "GICI3ML",
            "title": "Introduction to Machine Learning",
            "credits": 3,
            "programme": "Engineering Degree in Computer Sciences",
        },
        "lessons": [
            {"n": 1, "topic": "Supervised learning fundamentals",
             "covers": "Classification, regression, train/test split."},
            {"n": 2, "topic": "Model evaluation",
             "covers": "Cross-validation, metrics, statistics."},
            {"n": 3, "topic": "TensorFlow modeling",
             "covers": "Build and train neural networks in TensorFlow."},
            {"n": 4, "topic": "Natural language processing",
             "covers": "Apply NLP techniques to text classification."},
        ],
        "learning_outcomes": [
            "Apply machine-learning techniques to classification and regression.",
            "Build and evaluate neural models using TensorFlow.",
            "Apply natural-language-processing methods to real text data.",
        ],
    },
    {
        "course": {
            "code": "GICI3ENG",
            "title": "English for IT and Technical Communication",
            "credits": 2,
            "programme": "Engineering Degree in Computer Sciences",
        },
        "lessons": [
            {"n": 1, "topic": "Technical English writing",
             "covers": "Write clear technical reports in English."},
            {"n": 2, "topic": "Stakeholder communication",
             "covers": "Communicate technical results to non-technical stakeholders."},
            {"n": 3, "topic": "Presentation skills",
             "covers": "Deliver effective presentations using slide decks."},
        ],
        "learning_outcomes": [
            "Communicate technical content clearly in English, written and spoken.",
            "Write project reports with proper structure and citations.",
            "Collaborate effectively in teams and with stakeholders.",
        ],
    },
    {
        "course": {
            "code": "GICI3PM",
            "title": "Project Management for Engineers",
            "credits": 2,
            "programme": "Engineering Degree in Computer Sciences",
        },
        "lessons": [
            {"n": 1, "topic": "Project planning and milestones",
             "covers": "Plan projects, define milestones and dependencies."},
            {"n": 2, "topic": "Team leadership and management",
             "covers": "Lead engineering teams, manage tasks and people."},
            {"n": 3, "topic": "Risk and stakeholder management",
             "covers": "Identify risks and align stakeholders."},
        ],
        "learning_outcomes": [
            "Apply project-management techniques to plan and deliver engineering projects.",
            "Lead a team, manage stakeholders, and communicate progress.",
            "Solve problems collaboratively under deadline pressure.",
        ],
    },
]


# ---------- per-course matching (reuse runner logic) ----------

def analyze_one(course_data: dict[str, Any],
                job_freq: dict[str, int],
                match_threshold: float = MATCH_THRESHOLD,
                k_gaps: int = K_GAPS,
                course_units: list[str] | None = None) -> dict[str, Any]:
    """Run per-course matching for a single parsed course."""
    course = course_data.get("course", {}) or {}
    lessons = course_data.get("lessons", []) or []
    los = course_data.get("learning_outcomes", []) or []

    matched_result = run_matcher(
        lessons=lessons,
        learning_outcomes=los,
        job_skills=list(job_freq.keys()),
        threshold=match_threshold,
        job_freq=job_freq,
        course_units=course_units,
    )
    coverage_weighted = frequency_weighted_coverage(matched_result["matched"],
                                                     job_freq)
    return {
        "course": course,
        "label": classify(coverage_weighted),
        "coverage_weighted": coverage_weighted,
        "matched": matched_result["matched"],
        "matched_skills_set": {m["skill"] for m in matched_result["matched"]},
        "gaps_top_k": matched_result["gaps"][:k_gaps],
    }


# ---------- programme-level aggregation ----------

def aggregate_curriculum(per_course_reports: list[dict],
                         job_freq: dict[str, int],
                         k_gaps: int = K_GAPS) -> dict[str, Any]:
    """Union matched skills across courses, recompute coverage."""
    if not per_course_reports:
        raise ValueError("Empty per_course_reports")

    # union of matched skill names across all courses
    matched_union: set[str] = set()
    for cr in per_course_reports:
        matched_union.update(cr["matched_skills_set"])

    # frequency-weighted coverage at programme level
    total_freq = sum(job_freq.values()) or 1
    matched_freq = sum(job_freq[s] for s in matched_union if s in job_freq)
    coverage = round(matched_freq / total_freq, 3)
    label = classify(coverage)

    # programme-level gaps = top-K skills no course matched
    gap_pairs = sorted(
        ((s, f) for s, f in job_freq.items() if s not in matched_union),
        key=lambda x: x[1],
        reverse=True,
    )[:k_gaps]

    # per-course summary table
    course_table = []
    for cr in per_course_reports:
        course_table.append({
            "code": cr["course"].get("code"),
            "title": cr["course"].get("title"),
            "label": cr["label"],
            "coverage_weighted": cr["coverage_weighted"],
            "matched_count": len(cr["matched_skills_set"]),
        })

    return {
        "programme": {
            "name": per_course_reports[0]["course"].get("programme",
                                                        "Unknown programme"),
            "label": label,
            "coverage_weighted": coverage,
            "matched_count": len(matched_union),
            "skills_compared": len(job_freq),
            "course_count": len(per_course_reports),
            "matched_freq_sum": matched_freq,
            "total_freq_sum": total_freq,
        },
        "matched_union": sorted(matched_union),
        "gaps": [{"skill": s, "job_freq": f} for s, f in gap_pairs],
        "courses": course_table,
        "thresholds": {
            "match": MATCH_THRESHOLD,
            "tau_high": TAU_HIGH,
            "tau_low": TAU_LOW,
        },
    }


# ---------- CLI ----------

def _emoji(label: str) -> str:
    return {"MAINTAIN": "🟢", "REVISE": "🟡", "UPDATE": "🔴"}.get(label, "?")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--real-pdf",
                   default="./"
                           "school_program/automa_theory_course.pdf",
                   help="Real PDF to include alongside the mock courses.")
    p.add_argument("--skill-source", choices=("dummy", "real"), default="dummy",
                   help="dummy = curated test skills, real = job_freq.json")
    p.add_argument("--match-threshold", type=float, default=0.50,
                   help="Per-skill cosine threshold (default 0.50 for demo).")
    p.add_argument("--json", default=None,
                   help="If given, write the full report JSON to this path.")
    args = p.parse_args()

    here = Path(__file__).parent
    if args.skill_source == "dummy":
        with open(here / "dummy_job_skills.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        job_freq = {k: int(v) for k, v in data["skills"].items()}
    else:
        job_freq = load_job_freq(here / "job_freq.json")

    print(f"[curriculum_runner] skill source: {args.skill_source} · "
          f"{len(job_freq)} skills loaded")

    # Build the course list: real Automata + mocks
    all_courses: list[dict[str, Any]] = []
    print(f"[curriculum_runner] parsing real PDF: {args.real_pdf}")
    real = parse_pdf(args.real_pdf)
    all_courses.append(real)
    print(f"  → {real['course'].get('code')} "
          f"({len(real['lessons'])} lessons, "
          f"{len(real['learning_outcomes'])} CLOs)")

    print(f"[curriculum_runner] loading {len(MOCK_COURSES)} mock courses")
    all_courses.extend(MOCK_COURSES)

    # Per-course analysis
    print(f"\n[curriculum_runner] running per-course matching "
          f"(threshold={args.match_threshold})...")
    per_course = []
    for c in all_courses:
        code = c["course"].get("code")
        title = c["course"].get("title")
        print(f"  · {code:<12} {title}")
        per_course.append(analyze_one(c, job_freq,
                                       match_threshold=args.match_threshold))

    # Curriculum-level aggregation
    print(f"\n[curriculum_runner] aggregating programme-level coverage...")
    report = aggregate_curriculum(per_course, job_freq)

    # Pretty print
    print()
    print("=" * 70)
    prog = report["programme"]
    print(f"PROGRAMME: {prog['name']}")
    print("=" * 70)
    print()
    print(f"  {_emoji(prog['label'])} {prog['label']}     "
          f"coverage {prog['coverage_weighted'] * 100:.1f}%")
    print(f"  Courses analysed:    {prog['course_count']}")
    print(f"  Matched (union):     {prog['matched_count']} unique skills")
    print(f"  Skills compared:     {prog['skills_compared']}")
    print(f"  Matched demand sum:  {prog['matched_freq_sum']:,} jobs")
    print(f"  Total demand sum:    {prog['total_freq_sum']:,} jobs")
    print()

    print("─" * 70)
    print(f"{'Course':<28} {'Label':<10} {'Coverage':>10} {'Matched':>10}")
    print("─" * 70)
    for c in report["courses"]:
        print(f"{(c['code'] or ''):<8} "
              f"{(c['title'] or '')[:18]:<18}  "
              f"{_emoji(c['label'])} {c['label']:<8} "
              f"{c['coverage_weighted'] * 100:>7.1f}%   "
              f"{c['matched_count']:>4}")
    print("─" * 70)
    print()

    print(f"🔴 TOP {len(report['gaps'])} PROGRAMME GAPS "
          f"(no course teaches these):")
    print("─" * 70)
    for g in report["gaps"]:
        print(f"  • {g['skill']:<35}  📊 {g['job_freq']:>4} jobs")
    print()

    print(f"✅ UNION OF MATCHED SKILLS ({len(report['matched_union'])}):")
    print("─" * 70)
    for s in report["matched_union"]:
        print(f"  • {s}")
    print()

    if args.json:
        # serialize per-course matched-sets to lists for JSON
        for cr in per_course:
            cr["matched_skills_set"] = sorted(cr["matched_skills_set"])
        out = {
            "report": report,
            "per_course_detail": per_course,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"📄 Full report written to {args.json}")


if __name__ == "__main__":
    main()
