"""Recommendation V3 — ESCO-tab coverage: raw↔raw vs ESCO↔ESCO, both matched
by COSINE ≥ τ (semantic), demand-weighted.

Course skills vs the routed domain(s)' top-K market-demand skills (raw skill→freq
from domain_inventory.json). Two modes:

    raw   : compare the raw strings on both sides.
    esco  : map BOTH sides to ESCO preferred labels first (symmetric ESCO), then
            compare. Variants collapse onto a shared ESCO vocabulary before the
            cosine match, so e.g. "supervised learning" and "machine learning"
            both become "machine learning".

A market core skill is COVERED if some course skill matches it at cosine ≥ τ.
    coverage = Σ demand(covered core) / Σ demand(top-K core).

Self-contained: reuses route._embedder() + esco.normalize(); reads
domain_inventory.json. No reach outside recommendation_v3/.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_routing import stage5_route as route   # noqa: E402  (MiniLM embedder)
import esco    # noqa: E402  (ESCO normalize)

_HERE = Path(__file__).resolve().parent
_INV = _HERE / "domain_inventory.json"

_DEFAULT_TAU = 0.70        # cosine match threshold (a course skill "covers" a market skill)
_ESCO_TAU = 0.70          # ESCO-normalization threshold (below → keep raw, no forced label)


def _market_core(domains: list[str], top_k: int) -> tuple[list[tuple[str, int]], int]:
    """Merge the routed domains' raw skill→freq tables → top-K by summed demand.

    Returns ([(raw_skill, demand)], n_skills_total)."""
    inv = json.loads(_INV.read_text(encoding="utf-8"))
    dem: dict[str, int] = {}
    for d in domains:
        for s, f in ((inv.get(d) or {}).get("skills", {})).items():
            dem[s] = dem.get(s, 0) + int(f)
    ranked = sorted(dem.items(), key=lambda kv: -kv[1])[:top_k]
    return ranked, len(dem)


def _esco_labels(skills: list[str]) -> dict[str, str]:
    """{raw_skill: ESCO preferred label or raw skill} for the ESCO mode."""
    return {r["orig"]: r["esco"] for r in esco.normalize(skills, threshold=_ESCO_TAU)}


def _encode(texts: list[str]) -> np.ndarray:
    emb = route._embedder()
    return np.asarray(
        emb.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def analyze(course_skills: list[str], domains: list[str], top_k: int = 15,
            mode: str = "esco", threshold: float = _DEFAULT_TAU) -> dict:
    """Demand-weighted coverage of the routed market core by the course skills.

    mode="raw": compare raw strings. mode="esco": symmetric ESCO (both sides →
    ESCO labels) then compare. Match = cosine ≥ threshold."""
    course = [s for s in course_skills if s and s.strip()]
    core, market_total = _market_core(domains, top_k)   # [(raw_skill, demand)]

    base = {"domains": domains, "mode": mode, "top_k": top_k,
            "threshold": round(threshold, 2), "market_size": market_total}
    if not course or not core:
        return {**base, "alignment_score": 0.0, "coverage_pct": 0.0,
                "covered": [], "gaps": [], "off_core": []}

    market_raw = [s for s, _ in core]
    market_dem = [d for _, d in core]

    # representation per side (raw string, or ESCO label in esco mode)
    if mode == "esco":
        c_map = _esco_labels(course)
        m_map = _esco_labels(market_raw)
        course_repr = [c_map[s] for s in course]
        market_repr = [m_map[s] for s in market_raw]
    else:
        course_repr = course
        market_repr = market_raw

    C = np.ascontiguousarray(_encode(course_repr).astype(np.float64))
    M = np.ascontiguousarray(_encode(market_repr).astype(np.float64))
    with np.errstate(all="ignore"):
        sims = M @ C.T                                   # (k_core, m_course)

    covered, gaps = [], []
    cov_demand = all_demand = 0
    course_hit = [False] * len(course)                   # course skill matched ≥1 core skill
    for j, (mskill, demand) in enumerate(core):
        all_demand += demand
        row = sims[j]
        hits = [i for i in range(len(course)) if row[i] >= threshold]
        if hits:
            cov_demand += demand
            for i in hits:
                course_hit[i] = True
            ev = sorted(hits, key=lambda i: -row[i])
            entry = {"skill": mskill, "demand": demand,
                     "score": round(float(row[max(hits, key=lambda i: row[i])]), 3),
                     "evidence": [course[i] for i in ev]}
            if mode == "esco":
                entry["esco"] = market_repr[j]
            covered.append(entry)
        else:
            gaps.append({"skill": mskill, "demand": demand})

    completeness = (cov_demand / all_demand) if all_demand else 0.0

    # course skills that matched NO core market skill (taught, outside top-K demand)
    off_core = []
    for i, s in enumerate(course):
        if not course_hit[i]:
            best = float(sims[:, i].max()) if sims.size else 0.0
            entry = {"skill": s, "score": round(best, 3)}
            if mode == "esco":
                entry["esco"] = course_repr[i]
            off_core.append(entry)

    return {**base,
            "alignment_score": round(100 * completeness, 1),
            "coverage_pct": round(100 * completeness, 1),
            "covered": covered,           # market core covered (with course-skill evidence)
            "gaps": gaps,                 # market core NOT covered
            "off_core": off_core}         # course skills matching no core demand


if __name__ == "__main__":
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
          "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
          "Gradient Boosting", "Data Exploration and Analysis (EDA)",
          "AI Application Development", "Algorithm Evaluation", "Data Preparation",
          "Algorithm Selection for Datasets", "AI Sub-Fields"]
    for mode in ("raw", "esco"):
        r = analyze(ai, ["artificial_intelligence"], top_k=15, mode=mode)
        print(f"\n=== mode={mode}  coverage={r['coverage_pct']}%  "
              f"({len(r['covered'])}/{r['top_k']} core covered) ===")
        for c in r["covered"]:
            print(f"  ✓ {c['skill']:<24} demand={c['demand']:<6} {c['score']}  "
                  f"← {', '.join(c['evidence'][:3])}")
        print("  gaps:", ", ".join(g["skill"] for g in r["gaps"]))
