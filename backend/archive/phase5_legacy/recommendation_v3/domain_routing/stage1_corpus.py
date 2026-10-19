"""Stage 1 — the taxonomy + the corpus slice it is built from.

CATEGORY_DOMAINS is the ONE manual input of the whole domain pipeline: which
career domains live inside which job_category bucket of the LinkedIn corpus.
No seed titles, no regex — a domain is defined by its NAME alone; stages 2–4
derive everything else from data.

Adding a future ITC faculty = add its job category + domain names here and run
`python stage4_profiles.py <category>`.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent                                   # recommendation_v3/
_CORPUS = _V3.parent / "linkedin_job_posting" / "job_postings_categorized.jsonl"

# Shared data-file location (written by stage 4, read by stage 5 + build_inventory).
PROFILES_FILE = _V3 / "domain_profiles.json"

# job_category bucket → the career domains inside it. A domain's cohort is
# bootstrapped ONLY from postings of its own category slice, so taxonomies never
# bleed into each other.
CATEGORY_DOMAINS: dict[str, list[str]] = {
    "it": [
        "artificial_intelligence",
        "data_science",
        "software_engineering",
        "cybersecurity",
        "cloud_devops",
        "networking",
        "data_engineering",
        "data_analytics",
        "qa_testing",
        "business_analysis",
        "web_frontend",
        "mobile_development",
        "embedded_systems",
        "database_administration",
        "it_operations",
        "it_support",
        "project_product_management",
        "blockchain",
        "ui_ux_design",
        "game_development",
    ],
    # ── Future ITC faculties — uncomment / extend when their courses arrive ──
    # "engineering": [
    #     "civil_engineering", "mechanical_engineering", "electrical_engineering",
    #     "chemical_engineering", "industrial_engineering", "environmental_engineering",
    # ],
    # "construction": [
    #     "structural_engineering", "architecture", "construction_management",
    #     "surveying", "urban_planning",
    # ],
    # "health": [
    #     "nursing", "pharmacy", "medical_laboratory", "public_health",
    # ],
    # "finance": [
    #     "accounting", "banking", "auditing", "financial_analysis",
    # ],
}

# Skills excluded from a profile: generic soft skills + credentials. They appear
# in EVERY domain, so keeping them would blur the profiles into each other.
_EXCLUDE = {
    "communication", "communication skills", "teamwork", "collaboration",
    "leadership", "problem solving", "problemsolving", "time management",
    "critical thinking", "attention to detail", "interpersonal skills",
    "organizational skills", "adaptability", "creativity", "presentation",
    "project management", "analytical skills", "written communication",
    "verbal communication", "research", "documentation", "planning",
    "bachelor's degree", "bachelors degree", "master's degree", "phd",
    "computer science", "degree", "english", "microsoft office", "ms office",
}


def load_corpus(categories: set[str]) -> dict[str, tuple[list, Counter]]:
    """One corpus pass → {category: (posts, title_count)} for the given
    job_category slices. posts = [(title_lower, skills_frozenset)]."""
    if not _CORPUS.exists():
        raise FileNotFoundError(f"corpus not found at {_CORPUS}")
    out: dict[str, tuple[list, Counter]] = {c: ([], Counter()) for c in categories}
    scanned = 0
    with open(_CORPUS, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cat = rec.get("job_category")
            if cat not in out:
                continue
            title = (rec.get("job_title", "") or "").strip().lower()
            title = re.sub(r"\s+", " ", title)
            if not title:
                continue
            skills = frozenset(
                s.strip().lower() for s in rec.get("skills", []) if s.strip()
            ) - _EXCLUDE
            posts, title_count = out[cat]
            posts.append((title, skills))
            title_count[title] += 1
    for c in categories:
        posts, tc = out[c]
        print(f"scanned {scanned:,} postings · {len(posts):,} in '{c}' · "
              f"{len(tc):,} unique titles", file=sys.stderr)
    return out
