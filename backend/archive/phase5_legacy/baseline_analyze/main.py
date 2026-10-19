

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from helpers import read_jsonl

# ── paths ───────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PHASE5 = _HERE.parent
_LINKEDIN = _PHASE5 / "linkedin_job_posting" / "job_postings_categorized.jsonl"
_ESCO_OUT = _HERE / "linked_esco_normalize.jsonl"


# ── Module 1: load the LinkedIn job dataset ──────────────────────────────────────

def load_linkedin_jobs(path: Path = _LINKEDIN, limit: int | None = None,
                       category: str | None = None) -> list[dict]:
    """Load LinkedIn job postings from the categorized JSONL corpus.

    Each line is a posting:
        {"job_title", "skills": [...], "job_category", "location",
         "company_name", "full_url"}

    Args:
        path:     JSONL file (default: categorized LinkedIn corpus).
        limit:    stop after N postings (None = all ~1.35M).
        category: keep only postings in this job_category (e.g. "it").

    Returns:
        list of posting dicts.
    """
    jobs: list[dict] = []
    for rec in read_jsonl(path):
        if category and rec.get("job_category") != category:
            continue
        jobs.append(rec)
        if limit and len(jobs) >= limit:
            break
    return jobs


# ── Aggregate job skills → demand frequency table ────────────────────────────────

def aggregate_job_skills(category: str = "it") -> dict[str, int]:
    """Collapse all postings of one category into a {skill: frequency} table.

    Frequency = number of postings that list the skill (counted once per
    posting, so a skill repeated in one posting still counts +1). This is the
    labour-market DEMAND signal — the weight used by the coverage score.

    Args:
        category: job_category to keep (e.g. "it"). Filters by type only.

    Returns:
        {skill: how many postings of this category asked for it}.
    """
    jobs = load_linkedin_jobs(category=category)

    # pass 1 — per-posting demand count on the lowercased raw label
    raw: dict[str, int] = {}
    for j in jobs:
        for s in {x.strip().lower() for x in j.get("skills", []) if x.strip()}:
            raw[s] = raw.get(s, 0) + 1

    # pass 2 — collapse spacing/punctuation variants ("problem solving",
    # "problem-solving", "problemsolving" → one entry). The merge key strips
    # everything but [a-z0-9]; the most-demanded spelling is kept as the
    # display label and the variants' frequencies are summed.
    groups: dict[str, list[tuple[str, int]]] = {}
    for label, f in raw.items():
        key = re.sub(r"[^a-z0-9]", "", label)
        groups.setdefault(key, []).append((label, f))

    freq: dict[str, int] = {}
    for variants in groups.values():
        display = max(variants, key=lambda lf: lf[1])[0]   # most-frequent spelling
        freq[display] = sum(f for _, f in variants)
    return freq




# ── Semantic embedding (SBERT all-mpnet-base-v2) ─────────────────────────────────

def embed_skills(skills: list[str]) -> np.ndarray:
    """Embed a list of skill strings → (N, D) L2-normalized matrix.

    Uses the shared SBERT Embedder (all-mpnet-base-v2) — a proper
    sentence-similarity model with a calibrated cosine space (unlike the
    JobBERT→ESCO mapping we disabled). Because rows are L2-normalized, the
    cosine similarity between any two skills is just their dot product.

    First draft: plain embedding, no caching/batching tuning yet.
    """
    return _baseline_embedder()().embed(skills, task="skill")


_Embedder = None
_EMBEDDER_LOCK = threading.Lock()


def _baseline_embedder():
    """Load THIS package's embedder.py under a unique module name, so it isn't
    shadowed by clo_skill_alignment's identically-named `embedder` (which gets
    cached first inside the FastAPI process).

    Lock-guarded: under FastAPI's threadpool two concurrent requests could both
    pass the `is None` check and exec_module twice, producing two *separate*
    Embedder classes (each with its own singleton lock). The two models then
    load concurrently and race on MPS → "Cannot copy out of meta tensor".
    """
    global _Embedder
    with _EMBEDDER_LOCK:
        if _Embedder is None:
            import importlib.util as _ilu
            path = Path(__file__).resolve().parent / "embedder.py"
            spec = _ilu.spec_from_file_location("baseline_embedder", path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _Embedder = mod.Embedder
    return _Embedder


# ── Coverage score: course skills vs labour-market demand ────────────────────────

# action-label cutoffs on the demand-weighted coverage (paper defaults)
_TAU_HIGH = 0.66   # ≥ this → MAINTAIN
_TAU_LOW  = 0.33   # <  this → UPDATE  (between the two → REVISE)


def _label(coverage: float) -> str:
    """Map a [0,1] coverage fraction to a MAINTAIN / REVISE / UPDATE action."""
    if coverage >= _TAU_HIGH:
        return "MAINTAIN"
    if coverage >= _TAU_LOW:
        return "REVISE"
    return "UPDATE"


def compute_coverage(course_skills: list[str], job_freq: dict[str, int],
                     threshold: float = 0.60, top_n: int | None = None,
                     min_freq: int = 1) -> dict:
    """Compute how much of labour-market DEMAND a course's skills cover.

    Direction: a labour-market skill is "covered" if SOME course skill is
    semantically close to it (cosine ≥ threshold). This is the AMC-style
    coverage of the job market by the curriculum — symmetric to the
    clo_skill_alignment matcher, but run on the AMC baseline's skill sets.

    Steps:
      1. (optional) keep only the top-N most-demanded job skills (embedding the
         full IT vocabulary — thousands of distinct strings — is slow).
      2. embed course skills + job skills with the same SBERT model.
      3. cosine matrix (course × job), L2-normalized → dot product.
      4. max-pool down the course axis → best course evidence per job skill.
      5. threshold → matched job skills vs gaps.
      6. coverage = unweighted (count) AND demand-weighted (Σ freq).
      7. demand-weighted coverage → MAINTAIN / REVISE / UPDATE.

    Args:
        course_skills: curriculum-side skill phrases (e.g. get_course_skills()).
        job_freq:      {skill: # postings that asked for it} from
                       aggregate_job_skills().
        threshold:     cosine score above which a job skill counts as covered.
        top_n:         cap the demand table to its top-N skills by frequency.

    Returns:
        {
          "matched": [{skill, score, job_freq, evidence}],   # demand met by course
          "gaps":    [{skill, score, job_freq}],             # demand NOT met
          "scores":  {coverage_pct, demand_weighted_pct, demand_total, demand_met},
          "label":   "MAINTAIN" | "REVISE" | "UPDATE",
          "metadata":{course_skills, job_skills_compared, threshold},
        }
    """
    # 1) trim the demand table to the most-demanded skills (keeps embedding cheap)
    #    min_freq drops the long noise tail (freq-1 one-off strings); top_n then
    #    caps to the N most-demanded of what remains.
    items = sorted(job_freq.items(), key=lambda kv: kv[1], reverse=True)
    if min_freq > 1:
        items = [(s, f) for s, f in items if f >= min_freq]
    if top_n:
        items = items[:top_n]
    job_skills = [s for s, _ in items]
    print("Job Skills",job_skills)
    print("\n\n\n\n\n\n\n")
    print("Course Skill",course_skills)
    if not job_skills or not course_skills:
        return {"matched": [], "gaps": [], "label": "UPDATE",
                "scores": {"coverage_pct": 0.0, "demand_weighted_pct": 0.0,
                           "demand_total": 0, "demand_met": 0},
                "metadata": {"course_skills": len(course_skills),
                             "job_skills_compared": 0, "threshold": threshold}}

    # 2-3) embed both sides and build the cosine matrix (course × job)
    course_vecs = embed_skills(course_skills)
    job_vecs    = embed_skills(job_skills)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = course_vecs @ job_vecs.T        # (C, J)

    # 4) best course evidence per job skill
    max_scores = sim.max(axis=0)              # (J,)
    best_idx   = sim.argmax(axis=0)           # (J,)

    # 5-6) split into matched / gaps, accumulate demand weight
    matched: list[dict] = []
    gaps:    list[dict] = []
    demand_total = 0
    demand_met   = 0
    for j, skill in enumerate(job_skills):
        score = float(max_scores[j])
        freq  = int(items[j][1])
        demand_total += freq
        rec = {"skill": skill, "score": round(score, 3), "job_freq": freq}
        if score >= threshold:
            rec["evidence"] = course_skills[int(best_idx[j])]
            matched.append(rec)
            demand_met += freq
        else:
            gaps.append(rec)

    matched.sort(key=lambda r: r["job_freq"], reverse=True)
    gaps.sort(   key=lambda r: r["job_freq"], reverse=True)

    coverage_pct        = round(100 * len(matched) / len(job_skills), 1)
    demand_weighted     = demand_met / demand_total if demand_total else 0.0
    demand_weighted_pct = round(100 * demand_weighted, 1)

    return {
        "matched": matched,
        "gaps":    gaps,
        "scores": {
            "coverage_pct":        coverage_pct,        # unweighted (% of skills)
            "demand_weighted_pct": demand_weighted_pct, # weighted (% of postings' demand)
            "demand_total":        demand_total,
            "demand_met":          demand_met,
        },
        "label":    _label(demand_weighted),
        "metadata": {
            "course_skills":       len(course_skills),
            "job_skills_compared": len(job_skills),
            "threshold":           threshold,
        },
    }


if __name__ == "__main__":
    pass
