"""Recommendation V3 — Stage 5+6: exact-match coverage + gap-count recommendation.

The orchestrator that ties the canonical pipeline together:

    course skills (+ syllabus markdown → inferred tools)
        │
        ▼  Stage 4  consolidate() → competencies {market, specialized} + evidence
        │
        ▼  Stage 5  domain core = top-K canonical concepts by importance
        │           coverage = Σ demand(core ∩ MARKET competencies) / Σ demand(core)
        │           EXACT canonical-label match — no cosine threshold (both sides
        │           speak the same canonical vocabulary). SPECIALIZED excluded.
        │
        ▼  Stage 6  gap severity (demand share) → recommendation by CRITICAL COUNT
        ▼
    {coverage_pct, recommendation, covered[+evidence], gaps, specialized, …}

Why exact match (not cosine): after stage 4, a course competency is a canonical
concept and the core is the top-K of the SAME canonical vocabulary, so coverage is
a set intersection on labels — deterministic, no τ to tune. Cosine lives only in
routing (stage 2) and the stage-3 de-dup veto.

Self-contained: reuses consolidate.py (stage 4) + coverage.py's pure helpers
(severity, label, tool inference) + canonical_vocab.json (stage 3). No reach
outside recommendation_v3/.
"""

import json
import sys
from pathlib import Path

import numpy as np

# make sibling imports work when this file is loaded by path (FastAPI)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consolidate                                   # noqa: E402  (stage 4)
import coverage                                      # noqa: E402  (reuse pure helpers)

_HERE = Path(__file__).resolve().parent
_VOCAB = _HERE / "canonical_vocab.json"
_CAT_VOCAB = _HERE / "category_vocab.json"   # 17 raw job_category buckets (LinkedIn 1.35M)

_TOP_K = coverage._TOP_K                              # 30 core concepts
_severity = coverage._severity
_gap_label = coverage._gap_label
_recommend_text = coverage._recommend_text
_infer_tools = coverage.infer_tools
_NO_SUGGEST = coverage._NO_SUGGEST

_GAP_VETO_TAU = 0.62     # a core concept this close to a taught skill is taught under
                         # a different label, not a real gap (display + critical count only)


def _veto_taught_gaps(gaps: list[dict], taught: list[str],
                      tau: float) -> tuple[list[dict], list[dict]]:
    """Remove core concepts from the gap list when some taught skill/competency is
    semantically near them (cosine ≥ tau) — they're taught under a DIFFERENT label,
    not missing (e.g. a cybersecurity course shouldn't "miss" cybersecurity /
    incident response). Coverage is unchanged (exact-match); this only filters the
    gap DISPLAY + the critical-gap count. Returns (true_gaps, near_taught)."""
    taught = [t for t in dict.fromkeys(taught) if t and t.strip()]
    if not gaps or not taught:
        return gaps, []
    G = consolidate._encode([g["skill"] for g in gaps])
    T = consolidate._encode(taught)
    sims = consolidate._sims(G, T.T)                 # (n_gaps, n_taught)
    true_gaps, near = [], []
    for i, g in enumerate(gaps):
        j = int(sims[i].argmax())
        sc = float(sims[i][j])
        if sc >= tau:
            near.append({**g, "near": taught[j], "score": round(sc, 3)})
        else:
            true_gaps.append(g)
    return true_gaps, near


def all_categories() -> list[dict]:
    """The unique job_category buckets of the raw LinkedIn corpus, with posting
    counts — feeds the UI's Compare-against chip list. Built by
    build_category_vocab.py; [] if the vocab file hasn't been built yet."""
    try:
        cat = json.loads(_CAT_VOCAB.read_text(encoding="utf-8"))
        return [{"name": c, "n_postings": v.get("n_postings")}
                for c, v in cat.items()]
    except Exception:
        return []


def _core(domains: list[str], top_k: int) -> tuple[list[tuple[str, int, float]], int]:
    """Merge the routed domains' canonical vocabs → top-K core concepts by importance.

    Returns ([(concept, demand, importance)], n_concepts_total). Demand and
    importance are SUMMED across domains a concept appears in (one merged market).

    A name not in canonical_vocab.json is looked up in category_vocab.json (the 17
    raw job_category buckets of the LinkedIn corpus, built by
    build_category_vocab.py) with importance = raw demand — so the UI can compare
    against a whole category, not just a curated career domain.
    """
    cv = json.loads(_VOCAB.read_text(encoding="utf-8"))
    try:
        cat_v = json.loads(_CAT_VOCAB.read_text(encoding="utf-8"))
    except Exception:
        cat_v = {}
    imp: dict[str, float] = {}
    dem: dict[str, int] = {}
    for d in domains:
        dom = cv.get(d)
        if dom:
            for t, s in dom.get("importance", {}).items():
                imp[t] = imp.get(t, 0.0) + float(s)
            for t, f in dom.get("demand", {}).items():
                dem[t] = dem.get(t, 0) + int(f)
        else:  # raw category bucket: no curated importance — rank by demand
            for t, f in (cat_v.get(d) or {}).get("demand", {}).items():
                imp[t] = imp.get(t, 0.0) + float(f)
                dem[t] = dem.get(t, 0) + int(f)
    ranked = sorted(imp.items(), key=lambda kv: -kv[1])[:top_k]
    core = [(t, dem.get(t, 0), imp[t]) for t, _ in ranked]
    return core, len(imp)


def recommend(course_skills: list[str], domains: list[str],
              top_k: int = _TOP_K, markdown: str = "",
              course_code: str = "", force: bool = False,
              gap_veto_tau: float = _GAP_VETO_TAU) -> dict:
    """Score a course against its routed domain(s)' canonical demand core.

    1. fold in tools named only in the syllabus (python, pandas…) so they aren't
       false gaps, 2. consolidate raw skills → competencies (stage 4),
    3. exact-match the MARKET competencies against the top-K core, 4. demand-
    weighted coverage, 5. gap severity → recommendation by critical count.
    """
    skills = [s for s in course_skills if s and s.strip()]

    # #1 fold in tools the extractor missed (named only in References/Tools)
    have = {s.lower() for s in skills}
    inferred_tools = [t for t in _infer_tools(markdown) if t not in have]
    skills = skills + inferred_tools

    # #2 stage 4 — consolidate into market + specialized competencies (cached)
    con = consolidate.consolidate(skills, domains, course_code=course_code, force=force)
    competencies = con["competencies"]
    ev_map = {c["competency"].lower(): c["evidence"] for c in competencies}
    market_labels = {c["competency"].lower() for c in competencies
                     if c["source"] == "market"}

    # #3 stage 5 — domain core + EXACT-match coverage (demand-weighted)
    core, n_concepts = _core(domains, top_k)
    if not core or not skills:
        return {"domains": domains, "alignment_score": 0.0, "coverage_pct": 0.0,
                "recommendation": "UPDATE", "n_core": 0, "n_covered": 0,
                "covered": [], "gaps": [], "near_taught": [], "specialized": [],
                "competencies": competencies, "market_size": n_concepts, "top_k": top_k}

    covered, gaps = [], []
    cov_demand = all_demand = 0
    core_names = {c.lower() for c, _, _ in core}
    for concept, demand, importance in core:
        all_demand += demand
        if concept.lower() in market_labels:
            cov_demand += demand
            covered.append({"skill": concept, "demand": demand,
                            "importance": round(importance, 1),
                            "evidence": ev_map.get(concept.lower(), [])})
        else:
            gaps.append({"skill": concept, "demand": demand,
                         "importance": round(importance, 1)})
    completeness = (cov_demand / all_demand) if all_demand else 0.0

    # semantic gap veto — coverage above is unchanged (exact-match, deterministic);
    # this only drops core concepts that ARE taught under a different label from the
    # gap display + the critical-gap count (root-cause-1 fix).
    taught_repr = list(skills) + [c["competency"] for c in competencies]
    gaps, near_taught = _veto_taught_gaps(gaps, taught_repr, gap_veto_tau)

    # #4 stage 6 — gap severity (demand share) → recommendation by critical count
    for g in gaps:
        share = (g["demand"] / all_demand) if all_demand else 0.0
        g["share"] = round(share, 4)
        g["severity"] = _severity(share)
    n_crit = sum(1 for g in gaps if g["severity"] == "critical")
    n_med = sum(1 for g in gaps if g["severity"] == "medium")
    recommendation = _gap_label(n_crit)

    # #5 explainability sets (Ahadi 3-set, canonical)
    strengths = [c["skill"] for c in covered]
    critical_gaps = [g for g in gaps if g["severity"] == "critical"]
    suggestions = [g["skill"] for g in gaps
                   if g["severity"] in ("critical", "medium")
                   and g["skill"] not in _NO_SUGGEST][:6]
    specialized = [{"competency": c["competency"], "evidence": c["evidence"]}
                   for c in competencies if c["source"] == "specialized"]
    # taught market skills BELOW the top-K core (relevant, just not top demand)
    extra_market = [{"competency": c["competency"], "evidence": c["evidence"]}
                    for c in competencies if c["source"] == "market"
                    and c["competency"].lower() not in core_names]
    # backward-compat field for the current UI (rewritten in stage 7)
    course_mismatch = [{"skill": x["competency"], "score": 0.0,
                        "evidence": x["evidence"]}
                       for x in specialized + extra_market]

    return {
        "domains": domains,
        "alignment_score": round(100 * completeness, 1),
        "coverage": round(completeness, 4),
        "coverage_pct": round(100 * completeness, 1),
        "recommendation": recommendation,
        "recommendation_text": _recommend_text(domains, strengths, critical_gaps,
                                               recommendation),
        "n_core": len(core),
        "n_covered": len(covered),
        "n_critical": n_crit,
        "n_medium": n_med,
        "covered": covered,                          # aligned (with evidence)
        "gaps": gaps,                                # market mismatch (Sought − Taught)
        "near_taught": near_taught,                  # core vetoed from gaps (taught, other label)
        "specialized": specialized,                  # off-market topics (display only)
        "extra_market": extra_market,                # taught market, below core
        "course_mismatch": course_mismatch,          # back-compat (specialized + extra)
        "competencies": competencies,                # full stage-4 output
        "suggestions": suggestions,
        "inferred_tools": inferred_tools,
        "market_size": n_concepts,
        "top_k": top_k,
        "cached": con.get("cached", False),
    }


def _course_reps(competencies: list[dict]) -> list[dict]:
    """Course-side competency text used for independent semantic comparison."""
    reps = []
    for c in competencies:
        ev = [e for e in c.get("evidence", []) if e]
        label = str(c.get("competency", "")).strip()
        if not label:
            continue
        text = f"{label}: {', '.join(ev)}" if ev else label
        reps.append({"label": label, "text": text, "evidence": ev})
    return reps


def recommend_independent(course_skills: list[str], domains: list[str],
                          top_k: int = _TOP_K, markdown: str = "",
                          course_code: str = "", course_title: str = "",
                          force: bool = False,
                          match_threshold: float = coverage._MATCH_TAU) -> dict:
    """Score independent course competencies against the routed market core.

    Difference from `recommend()`:
    - course competencies are grouped without market vocabulary;
    - market data is used only after grouping, as the comparison target;
    - coverage uses cosine between course competency representations and market
      core skills, rather than exact canonical-label matching.
    """
    skills = [s for s in course_skills if s and s.strip()]

    have = {s.lower() for s in skills}
    inferred_tools = [t for t in _infer_tools(markdown) if t not in have]
    skills = skills + inferred_tools

    con = consolidate.consolidate_independent(
        skills, course_code=course_code, course_title=course_title, force=force
    )
    competencies = con["competencies"]
    reps = _course_reps(competencies)

    core, n_concepts = _core(domains, top_k)
    if not core or not reps:
        return {"domains": domains, "alignment_score": 0.0, "coverage_pct": 0.0,
                "recommendation": "UPDATE", "n_core": 0, "n_covered": 0,
                "covered": [], "gaps": [], "near_taught": [], "specialized": reps,
                "extra_market": [], "competencies": competencies,
                "market_size": n_concepts, "top_k": top_k,
                "method": "independent_course_grouping"}

    core_names = [c for c, _, _ in core]
    C = np.ascontiguousarray(consolidate._encode(core_names))
    R = np.ascontiguousarray(consolidate._encode([r["text"] for r in reps]))
    sim = consolidate._sims(C, R.T)
    best = sim.max(axis=1)
    arg = sim.argmax(axis=1)

    covered, gaps = [], []
    cov_demand = all_demand = 0
    for i, (concept, demand, importance) in enumerate(core):
        all_demand += demand
        j = int(arg[i])
        score = float(best[i])
        if score >= match_threshold:
            cov_demand += demand
            covered.append({
                "skill": concept,
                "demand": demand,
                "importance": round(importance, 1),
                "score": round(score, 3),
                "course_competency": reps[j]["label"],
                "evidence": reps[j]["evidence"],
            })
        else:
            gaps.append({
                "skill": concept,
                "demand": demand,
                "importance": round(importance, 1),
                "score": round(score, 3),
                "nearest_course_competency": reps[j]["label"],
            })
    completeness = (cov_demand / all_demand) if all_demand else 0.0

    # Course competencies that do not cover any scored top-K market concept.
    best_per_rep = sim.max(axis=0) if sim.size else np.zeros(len(reps))
    nearest_core = sim.argmax(axis=0) if sim.size else np.zeros(len(reps), dtype=int)
    off_core = []
    for j, rep in enumerate(reps):
        score = float(best_per_rep[j])
        if score < match_threshold:
            off_core.append({
                "competency": rep["label"],
                "evidence": rep["evidence"],
                "score": round(score, 3),
                "nearest_market_skill": core_names[int(nearest_core[j])] if core_names else None,
            })

    for g in gaps:
        share = (g["demand"] / all_demand) if all_demand else 0.0
        g["share"] = round(share, 4)
        g["severity"] = _severity(share)
    n_crit = sum(1 for g in gaps if g["severity"] == "critical")
    n_med = sum(1 for g in gaps if g["severity"] == "medium")
    recommendation = _gap_label(n_crit)

    strengths = [c["skill"] for c in covered]
    critical_gaps = [g for g in gaps if g["severity"] == "critical"]
    suggestions = [g["skill"] for g in gaps
                   if g["severity"] in ("critical", "medium")
                   and g["skill"] not in _NO_SUGGEST][:6]

    return {
        "domains": domains,
        "alignment_score": round(100 * completeness, 1),
        "coverage": round(completeness, 4),
        "coverage_pct": round(100 * completeness, 1),
        "recommendation": recommendation,
        "recommendation_text": _recommend_text(domains, strengths, critical_gaps,
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
        "course_mismatch": [{"skill": x["competency"], "score": x["score"],
                             "evidence": x["evidence"]}
                            for x in off_core],
        "competencies": competencies,
        "suggestions": suggestions,
        "inferred_tools": inferred_tools,
        "market_size": n_concepts,
        "match_threshold": match_threshold,
        "top_k": top_k,
        "cached": con.get("cached", False),
        "method": "independent_course_grouping",
    }


if __name__ == "__main__":
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
          "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
          "Gradient Boosting", "Data Exploration and Analysis (EDA)",
          "AI Application Development", "Algorithm Evaluation", "Data Preparation",
          "Algorithm Selection for Datasets", "AI Sub-Fields"]
    r = recommend(ai, ["artificial_intelligence", "data_science"],
                  course_code="AIECS031", force=True)
    print(f"domains={r['domains']}  market_size={r['market_size']}")
    print(f"coverage {r['coverage_pct']}%  → {r['recommendation']}  "
          f"({r['n_covered']}/{r['n_core']} core covered, "
          f"{r['n_critical']} critical gaps)\n")
    print("ALIGNED (covered core, with evidence):")
    for c in r["covered"]:
        print(f"  ✓ {c['skill']:<22} demand={c['demand']:<6} ← {', '.join(c['evidence'][:4])}"
              f"{'…' if len(c['evidence']) > 4 else ''}")
    print("\nMARKET GAPS (top 8 by demand):")
    for g in sorted(r["gaps"], key=lambda d: -d["demand"])[:8]:
        print(f"  ✗ {g['skill']:<22} demand={g['demand']:<6} [{g['severity']}]")
    print("\nSPECIALIZED (off-market, display only):")
    for s in r["specialized"]:
        print(f"  • {s['competency']}  ← {', '.join(s['evidence'][:4])}")
    print(f"\nrecommendation_text: {r['recommendation_text']}")
