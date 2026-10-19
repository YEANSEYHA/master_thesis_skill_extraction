"""
Domain router — classify an uploaded course into the job-market category(ies)
it belongs to, so the job corpus can be filtered to the relevant labour-market
segment before computing curriculum-job coverage.

This is the "domain-conditioned matching" step: a Computer-Science course
should be scored against IT/software postings, not against the whole general
job corpus (which is soft-skill dominated). See curricula_recommendation.md §0.7.

Uses qwen3 (Ollama), mirroring pdf_parser.py. Pure classification — the caller
supplies the list of available `job_category` labels (derived from Phase-3
stage-2 outputs). On any Ollama error it returns [] so the UI can fall back to
manual category selection.
"""

from __future__ import annotations

import os

import ollama
from pydantic import BaseModel, Field


_LLM_MODEL = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")


class _CategoryChoice(BaseModel):
    categories: list[str] = Field(
        ...,
        description="subset of the provided job categories that match the course's field",
    )


_SYSTEM_PROMPT = """\
You map a university course to the job-market categories it prepares students for.

RULES:
  1. Choose ONLY from the provided list of available categories. Never invent new ones.
  2. Judge the course's field from the COURSE TITLE and topics FIRST — not from the
     degree/programme name. Degree names are often misleading: a programme called
     "Engineering Degree in Computer Sciences" is a COMPUTER SCIENCE course, not a
     physical-engineering one.
  3. Field → category mapping:
     - Computer Science / software / programming / IT / informatics / data /
       networks / cybersecurity / web / databases  →  "it" (and only very
       closely-related IT categories).
     - "engineering" in this job list means PHYSICAL engineering jobs (mechanical,
       civil, electrical, construction, audio/AV, maintenance). Do NOT pick
       "engineering" for a Computer Science / software course, even if the degree
       is named an "Engineering Degree".
     - Finance/accounting course → "finance", "banking", "accounting"; etc.
  4. Prefer the BROAD, HIGH-COUNT canonical category (e.g. "it") over niche or
     compound aliases (e.g. "information technology solution / computer science").
     The number in parentheses after each category is its posting count.
  5. Be PRECISE: return only categories whose field genuinely matches (usually 1-2).
     Exclude tangential ones. A Computer Science course should NOT return "sales",
     "hospitality", or "engineering".
  6. Copy the category strings EXACTLY as given (without the count).
  7. If none clearly apply, return an empty list."""

_USER_PROMPT = """\
COURSE  (judge the field from the Title/topics, not the degree name)
  Programme: {programme}
  Title:     {title}
  Synopsis:  {synopsis}

AVAILABLE JOB CATEGORIES — "label (posting count)"; choose labels only, copy exactly:
{categories}

Return the 1-2 categories whose FIELD best matches this course."""


def classify_course(course: dict,
                    available_categories: list) -> list[str]:
    """Return the subset of `available_categories` matching the course's domain.

    Args:
        course: parsed course dict, expects keys among {programme, title,
                synopsis}. Accepts either the flat course dict or the full
                parse with a nested "course" key.
        available_categories: candidate job categories — either a list of label
                strings, or a list of {"category", "count"} dicts (counts are
                shown to the model so it prefers the canonical high-count label).

    Returns:
        list of categories (exact strings from the input). [] on error or no match.
    """
    if not available_categories:
        return []

    # Accept list[str] or list[{category, count}]; build (label, count) pairs.
    pairs: list[tuple[str, int | None]] = []
    for x in available_categories:
        if isinstance(x, dict):
            pairs.append((x.get("category", ""), x.get("count")))
        else:
            pairs.append((str(x), None))
    labels = [lbl for lbl, _ in pairs if lbl]

    c = course.get("course", course) if isinstance(course, dict) else {}
    programme = (c.get("programme") or "").strip() or "(unspecified)"
    title = (c.get("title") or "").strip() or "(unspecified)"
    synopsis = (course.get("synopsis") or c.get("synopsis") or "").strip() or "(none)"

    cat_block = "\n".join(
        f"  - {lbl}" + (f" ({cnt})" if cnt is not None else "")
        for lbl, cnt in pairs if lbl
    )

    try:
        response = ollama.chat(
            model=_LLM_MODEL,
            think=False,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT.format(
                    programme=programme, title=title,
                    synopsis=synopsis[:600], categories=cat_block)},
            ],
            format=_CategoryChoice.model_json_schema(),
            options={"temperature": 0},
        )
        chosen = _CategoryChoice.model_validate_json(response.message.content).categories
    except Exception as e:
        print(f"[domain_router] classify failed ({e}) — returning []")
        return []

    # Keep only exact members of the provided list (case-insensitive match,
    # but return the canonical label as supplied).
    avail = {lbl.lower(): lbl for lbl in labels}
    out, seen = [], set()
    for ch in chosen:
        key = (ch or "").strip().lower()
        if key in avail and key not in seen:
            out.append(avail[key])
            seen.add(key)
    return out


if __name__ == "__main__":
    cats = ["it", "sales", "finance", "marketing", "computer - general",
            "information technology", "engineering", "education", "hospitality/hotels"]
    course = {"course": {"programme": "Computer Sciences", "title": "Automata Theory"}}
    print("course:", course["course"])
    print("→ matched categories:", classify_course(course, cats))
