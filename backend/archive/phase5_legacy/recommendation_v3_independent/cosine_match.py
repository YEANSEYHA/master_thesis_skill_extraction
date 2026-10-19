"""V6 ablation — embedding-only matching (NO LLM verify).

Same Qwen3 cosine retrieval as esco_match (max-pool over label+evidence),
but the covered/gap decision is just the single highest score vs a threshold:

    covered  iff  best cosine ≥ TAU

Drop-in replacement for esco_match.match() — identical return shape — so
recommend_verified can switch between the two with one parameter.
"""

import esco_match as _m

TAU = 0.70   # covered/gap threshold (Qwen3 cosine scale)


def match(core: list[tuple], match_names: list[str], units: list[dict],
          verify_names: list[str] | None = None, tau: float = TAU) -> dict:
    """core = [(raw_name, demand, importance)] · match_names = ESCO (or raw)
    names aligned to core · units = course reps [{label, evidence}].
    verify_names is accepted for signature parity with esco_match (unused —
    there is no LLM verification step here)."""
    # floor=0 keeps the nearest unit for EVERY skill (gaps still show it)
    shortlist = _m.retrieve(match_names, units, top_n=1, floor=0.0)

    covered, gaps, used = [], [], set()
    cov_demand = all_demand = 0
    for i, (concept, demand, importance) in enumerate(core):
        target = match_names[i]
        all_demand += demand
        j, score = shortlist[i][0] if shortlist[i] else (None, 0.0)

        if j is not None and score >= tau:
            used.add(j)
            cov_demand += demand
            covered.append({"skill": concept, "demand": demand,
                            "esco": target if target != concept else None,
                            "importance": round(importance, 1),
                            "score": round(score, 3),
                            "course_competency": units[j]["label"],
                            "evidence": units[j]["evidence"],
                            "reason": f"cosine {score:.2f} ≥ τ={tau}"})
        else:
            gaps.append({"skill": concept, "demand": demand,
                         "esco": target if target != concept else None,
                         "importance": round(importance, 1),
                         "score": round(score, 3),
                         "nearest_course_competency":
                             units[j]["label"] if j is not None else None})

    off_core = [{"competency": u["label"], "evidence": u["evidence"]}
                for j, u in enumerate(units) if j not in used]
    coverage = (cov_demand / all_demand) if all_demand else 0.0

    return {"covered": covered, "gaps": gaps, "off_core": off_core,
            "coverage": coverage, "cov_demand": cov_demand, "all_demand": all_demand}
