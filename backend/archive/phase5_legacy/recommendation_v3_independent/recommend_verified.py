"""Recommendation V3 (independent) — RETRIEVE → VERIFY analyze.

Two-stage curriculum↔market matching that fixes the embedding-only failures
(blob dilution + range-compression false positives):

    course skills
        │  group (consolidate_v3, no market vocab)
        ▼
    STAGE A — RETRIEVE (embedding, high recall)
        │  for each market skill, shortlist top-N competencies by max-pooled
        │  cosine ≥ floor (MiniLM). Proposes candidates; does not decide.
        ▼
    STAGE B — VERIFY (qwen3:14b, high precision)
        │  per (market skill, candidate) ask "does this competency teach it?"
        │  yes/no + reason. Covered = any yes; attribute to best-scoring yes.
        ▼
    demand-weighted coverage → gaps by severity → MAINTAIN/REVISE/UPDATE

Reuses recommendation_v3 helpers (_core, _infer_tools, _severity, _gap_label,
_course_reps) so the return shape matches recommend_independent() — the existing
frontend renders it unchanged.

Run:  python recommend_verified.py
"""

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent / "recommendation_v3"
for _p in (str(_HERE), str(_V3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import consolidate_v3                              # noqa: E402  grouping (no market vocab)
import recommend as _rec                           # noqa: E402  _core/_infer_tools/_course_reps
import coverage as _cov                            # noqa: E402  _severity/_gap_label/_recommend_text
import esco_llm as _esco_llm                       # noqa: E402  C1: Qwen3 top-5 → LLM pick (V6)

# ESCO mapping mode for the V6 (group=False) path — flip this one name to switch:
#   _esco_llm.normalize            top-5 → qwen3:14b pick/reject (no threshold)
#   _esco_llm.normalize_threshold  playground-style: top-1 cosine + 0.80 keep-raw
_ESCO_NORMALIZE = _esco_llm.normalize_threshold
import esco_match as _match                        # noqa: E402  retrieve → verify → score core
import cosine_match as _cosine                     # noqa: E402  embedding-only ablation (no LLM)

# Full-result cache — same document + settings ⇒ instant replay (dev convenience).
_ANALYZE_CACHE = _HERE / ".analyze_cache.json"


def _cache_read() -> dict:
    try:
        return json.loads(_ANALYZE_CACHE.read_text())
    except Exception:
        return {}


def analyze_verified(course_skills: list[str], domains: list[str], top_k: int = 10,
                     markdown: str = "", course_code: str = "", course_title: str = "",
                     force: bool = False, group: bool = True,
                     matcher: str = "verify") -> dict:
    """Two-stage retrieve→verify analyze. Same return shape as recommend_independent.

    group=False (V6): skip competency grouping — each raw course skill is its own
    matching unit for retrieve→verify.
    matcher="cosine": embedding-only ablation — covered iff best cosine ≥ τ,
    no LLM verification (cosine_match.py).
    """
    key = hashlib.sha1(json.dumps(
        ["v3",  # bump to invalidate cache: floor 0.55→0.70 + threshold-ESCO mode
         sorted(course_skills), sorted(domains), top_k, group, matcher,
         # market-vocab version: a rebuilt canonical_vocab.json (new market core)
         # must not replay results computed against the old market
         int(_rec._VOCAB.stat().st_mtime),
         hashlib.sha1(markdown.encode()).hexdigest()]).encode()).hexdigest()
    if not force:
        hit = _cache_read().get(key)
        if hit:
            return {**hit, "cached": True}

    skills = [s for s in course_skills if s and s.strip()]
    have = {s.lower() for s in skills}
    inferred_tools = [t for t in _rec._infer_tools(markdown) if t not in have]
    skills = skills + inferred_tools

    esco_rows = None
    if group:
        con = consolidate_v3.consolidate_independent(
            skills, course_code=course_code, course_title=course_title, force=force)
        competencies = con["competencies"]
        cached = con.get("cached", False)
    else:
        # C1 — ESCO-anchor each raw skill (Qwen3 top-5 → qwen3:14b pick/reject),
        # then merge same-label skills into units (esco_match.build_course_units).
        esco_rows = _ESCO_NORMALIZE(skills)
        competencies = _match.build_course_units(esco_rows)
        cached = False
    reps = _rec._course_reps(competencies)

    core, n_concepts = _rec._core(domains, top_k)
    if not core or not reps:
        return {"domains": domains, "alignment_score": 0.0, "coverage_pct": 0.0,
                "recommendation": "UPDATE", "n_core": 0, "n_covered": 0,
                "covered": [], "gaps": [], "near_taught": [], "specialized": reps,
                "extra_market": [], "competencies": competencies,
                "market_size": n_concepts, "top_k": top_k, "esco": esco_rows,
                "method": ("raw_cosine_argmax" if matcher == "cosine"
                           else "independent_retrieve_verify" if group else "raw_retrieve_verify")}

    core_names = [c for c, _, _ in core]
    # Symmetric ESCO (V6): anchor the MARKET side too, so retrieval + verification
    # compare ESCO label ↔ ESCO label (kept-raw names used where the LLM rejected).
    # Demand/denominator stays per original market row (no merging yet — Step 3).
    market_esco = None
    if group:
        match_names = core_names
        verify_names = core_names
    else:
        market_esco = _ESCO_NORMALIZE(core_names)
        match_names = [r["esco"] for r in market_esco]
        # Broad ESCO anchors retrieve well but must not be the VERIFY target
        # (cissp→"cyber security" would be trivially covered): verify against
        # the raw market skill unless the mapping was exact.
        verify_names = [r["esco"] if r.get("match") == "exact" else r["orig"]
                        for r in market_esco]

    # STEP 1-4: retrieve → decide (LLM verify OR cosine-argmax) → demand-weighted coverage
    engine = _cosine.match if matcher == "cosine" else _match.match
    m = engine(core, match_names, reps, verify_names)
    covered, gaps, off_core = m["covered"], m["gaps"], m["off_core"]
    completeness = m["coverage"]

    all_demand = m["all_demand"]
    for g in gaps:
        share = (g["demand"] / all_demand) if all_demand else 0.0
        g["share"] = round(share, 4)
        g["severity"] = _cov._severity(share)
    n_crit = sum(1 for g in gaps if g["severity"] == "critical")
    n_med = sum(1 for g in gaps if g["severity"] == "medium")
    recommendation = _cov._gap_label(n_crit)

    strengths = [c["skill"] for c in covered]
    critical_gaps = [g for g in gaps if g["severity"] == "critical"]
    suggestions = [g["skill"] for g in gaps
                   if g["severity"] in ("critical", "medium")
                   and g["skill"] not in _rec._NO_SUGGEST][:6]

    result = {
        "domains": domains,
        "alignment_score": round(100 * completeness, 1),
        "coverage": round(completeness, 4),
        "coverage_pct": round(100 * completeness, 1),
        "recommendation": recommendation,
        "recommendation_text": _cov._recommend_text(domains, strengths, critical_gaps,
                                                    recommendation),
        "n_core": len(core),
        "n_covered": len(covered),
        "n_critical": n_crit,
        "n_medium": n_med,
        "covered": covered,
        "gaps": gaps,
        "near_taught": [],
        "specialized": off_core,
        "extra_market": [],
        "competencies": competencies,
        "suggestions": suggestions,
        "inferred_tools": inferred_tools,
        "market_size": n_concepts,
        "top_k": top_k,
        "cached": cached,
        "esco": esco_rows,
        "market_esco": market_esco,
        "method": ("raw_cosine_argmax" if matcher == "cosine"
                   else "independent_retrieve_verify" if group else "raw_retrieve_verify"),
    }
    cache = _cache_read()
    cache[key] = result
    _ANALYZE_CACHE.write_text(json.dumps(cache))
    return result


if __name__ == "__main__":
    import json
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
          "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
          "Gradient Boosting", "Data Exploration and Analysis (EDA)",
          "AI Application Development", "Algorithm Evaluation", "Data Preparation",
          "Algorithm Selection for Datasets", "AI Sub-Fields"]
    r = analyze_verified(ai, ["artificial_intelligence"], top_k=10,
                         course_code="AIECS031",
                         course_title="Introduction to Artificial Intelligence")
    print(f"\ncoverage {r['alignment_score']}%  ({r['n_covered']}/{r['n_core']})  "
          f"→ {r['recommendation']}")
    print("\nCOVERED:")
    for c in r["covered"]:
        print(f"  ✓ {c['skill']:20s} 📊{c['demand']:<4} ← {c['course_competency']}"
              f"   ({c['reason']})")
    print("\nGAPS:")
    for g in r["gaps"]:
        print(f"  ✗ {g['skill']:20s} 📊{g['demand']:<4} [{g['severity']}]")
