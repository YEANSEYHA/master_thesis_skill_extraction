#!/usr/bin/env python3
"""IT-posting filter for the Phase-3 stage2 corpus — conditional, two-stage.

Rule (in order):
  1. CATEGORY first — if the LLM-extracted `job_category` is IT-related, keep.
  2. TITLE fallback — else, if the `job_title` is a genuine tech role, keep.
     (title-only excludes 'business development' / 'IT sales' false hits)

This is the reliable IT filter: `job_category` alone recovers ~240 of the
Cambodian corpus (vs 91 by title alone); the title stage adds the few IT jobs
whose category was mislabeled (e.g. "DevOps Engineer" tagged `engineering`).

    python it_filter.py                         # counts + stage breakdown
    python it_filter.py --write stage2_it.json  # also write the IT-only subset
    python it_filter.py --input <stage2_result.json>
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # phase3 root (script now lives one level down)
DEFAULT_INPUT = HERE / "outputs" / "stage2" / "api_full" / "stage2_result.json"

# Stage 1 — IT-related job_category labels (the LLM's own category field).
CATEGORY_IT = re.compile(
    r"\b(it|ict|information technology|software|web|programming|developer|"
    r"network|database|system (admin|development)|cyber|data (science|analytics|"
    r"engineering)|artificial intelligence|machine learning|computer|"
    r"quality assurance|devops|telecommunication|digital)\b")

# Stage 2 — genuine tech titles (fallback when category misses). Deliberately
# excludes 'business development', 'IT sales', 'learning and development'.
TITLE_IT = re.compile(
    r"\b(software|developer|programmer|full.?stack|front.?end|back.?end|"
    r"web design|\.net|java(script)?|python|php|golang|node\.?js|react|"
    r"it officer|it support|it manager|it staff|it admin|it programmer|"
    r"system admin|sysadmin|network (admin|engineer|specialist)|"
    r"database admin|dba|data analyst|data engineer|data scientist|devops|"
    r"cyber ?security|qa engineer|software tester|mobile developer|"
    r"android developer|ios developer|ui/ux|ux/ui|ai engineer|"
    r"machine learning engineer|blockchain)\b", re.I)


def _category_text(rec: dict) -> str:
    c = rec.get("output", {}).get("job_category")
    if isinstance(c, list):
        return " , ".join(str(x) for x in c).lower()
    return str(c or "").lower()


def _title_text(rec: dict) -> str:
    return str(rec.get("output", {}).get("job_title") or "")


def classify_it(rec: dict) -> str | None:
    """'category' | 'title' | None — which stage kept the posting (None = drop).
    Category is checked FIRST; title is only consulted when category misses."""
    if CATEGORY_IT.search(_category_text(rec)):
        return "category"
    if TITLE_IT.search(_title_text(rec)):
        return "title"
    return None


def is_it_posting(rec: dict) -> bool:
    return classify_it(rec) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter the stage2 corpus to IT postings (category-first, title-fallback).")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--write", default="", help="write the IT-only subset to this path")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    by_stage = Counter()
    it_only = []
    for rec in data:
        stage = classify_it(rec)
        if stage:
            by_stage[stage] += 1
            it_only.append(rec)

    total = len(data)
    kept = len(it_only)
    print(f"corpus                 : {total} postings")
    print(f"  stage 1 category=IT  : {by_stage['category']}")
    print(f"  stage 2 title=IT     : {by_stage['title']}   (category missed, title caught)")
    print(f"  IT total (kept)      : {kept}  ({kept/total*100:.1f}%)")
    print(f"  dropped (non-IT)     : {total - kept}")

    if args.write:
        out = Path(args.write)
        if not out.is_absolute():
            out = HERE / out
        out.write_text(json.dumps(it_only, ensure_ascii=False, indent=2))
        print(f"\nwrote IT-only subset -> {out}  ({kept} postings)")


if __name__ == "__main__":
    main()
