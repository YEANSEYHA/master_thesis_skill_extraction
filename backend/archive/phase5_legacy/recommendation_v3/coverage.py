"""Recommendation V3 — Stage 6+7: domain-scoped coverage → recommendation.

Given a course's skills and the career-domain(s) it routed to:
  1. pull those domains' demand inventories and MERGE them into one market,
  2. take the top-K most in-demand skills (the domain "core"),
  3. for each core skill, check if a taught skill covers it (embedding cosine ≥ τ),
  4. completeness = Σ demand(covered core) / Σ demand(all core)  [demand-weighted],
  5. label MAINTAIN / REVISE / UPDATE, and list the demand-sorted gaps.

This is the fair metric: the denominator is the COURSE'S OWN domain core, not all
143K IT postings — so a focused course is scored against the right market.

Self-contained: reuses route.py's embedder (no duplicate model load).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# make `import route` work even when this file is loaded by path (FastAPI)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_routing import stage5_route as route  # noqa: E402  (MiniLM embedder + domain set)

_HERE = Path(__file__).resolve().parent
_INV = _HERE / "domain_inventory.json"

_MATCH_TAU = 0.60      # cosine ≥ → a core skill is "covered" by a taught skill
_TOP_K = 30            # # of core (most in-demand) domain skills scored against
_CRIT_SHARE = 0.05     # an uncovered core skill with ≥5% of core demand → CRITICAL
_MED_SHARE = 0.02      # 2–5% → MEDIUM   (< 2% → LOW)


def _inventory() -> dict:
    if not _INV.exists():
        raise FileNotFoundError(f"{_INV} not found — run build_inventory.py first.")
    return json.loads(_INV.read_text(encoding="utf-8"))


def _severity(share: float) -> str:
    return ("critical" if share >= _CRIT_SHARE
            else "medium" if share >= _MED_SHARE else "low")


def _gap_label(n_critical: int) -> str:
    """Recommendation derived from the GAP LIST (the explanation layer), NOT the
    raw alignment %: the verdict follows directly from how many HIGH-DEMAND skills
    are missing. This is what a curriculum committee actually reasons about, and it
    discriminates where the score does not — a covered high-demand skill (python,
    ML) is NOT a gap, so a well-aligned course is not punished by the senior-stack
    denominator. Mapping (0–1 → MAINTAIN, 2–3 → REVISE, >3 → UPDATE) is forgiving of
    a single missing common skill (e.g. sql) but escalates as real gaps accumulate."""
    if n_critical > 3:
        return "UPDATE"
    if n_critical >= 2:
        return "REVISE"
    return "MAINTAIN"


def _recommend_text(domains: list[str], strengths: list[str],
                    critical: list[dict], recommendation: str = "") -> str:
    """Descriptive summary of the alignment — what the course covers and which
    high-demand skills it misses. Deliberately NO categorical verdict
    (MAINTAIN/REVISE/UPDATE); the alignment % is the signal we surface."""
    dom = " / ".join(d.replace("_", " ") for d in domains)
    cov = ", ".join(strengths[:3]) if strengths else "core concepts"
    if not critical:
        return (f"The course covers the high-demand {dom} skills ({cov}) — no critical "
                f"gaps against the current market.")
    gaps = ", ".join(g["skill"] for g in critical[:4])
    return (f"The course covers {cov}, but {dom} job postings frequently require "
            f"{gaps}.")


# ── #1: tool inference — recover implied tools the extractor missed ────────────
# Languages/libraries a course actually uses are often named only in the
# References/Tools section (w3schools.com/python, Pandas notebooks), never in a
# CLO/lesson — so skill extraction misses them and they show up as FALSE gaps
# (python). Deterministic whitelist scan (no LLM): add them as taught skills so
# e.g. "K-Means + EDA in Python" stops counting `python` as missing.
_L, _R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
_TOOL_PATTERNS = {
    "python":       rf"{_L}python{_R}",
    "pandas":       rf"{_L}pandas{_R}",
    "numpy":        rf"{_L}numpy{_R}",
    "scikit-learn": rf"{_L}scikit-?learn{_R}|{_L}sklearn{_R}",
    "jupyter":      rf"{_L}jupyter{_R}|{_L}colab{_R}",
    "matplotlib":   rf"{_L}matplotlib{_R}",
    "tensorflow":   rf"{_L}tensorflow{_R}",
    "pytorch":      rf"{_L}pytorch{_R}",
    "sql":          rf"{_L}sql{_R}",
    "tableau":      rf"{_L}tableau{_R}",
    "r":            rf"{_L}rstudio{_R}|{_L}r\s+programming{_R}",
}
_EXAMPLE_LINE = re.compile(r"\be\.?\s?g\b|\bfor example\b|\bexample[s]?:", re.I)


def infer_tools(markdown: str) -> list[str]:
    """Whitelisted languages/libraries named in the syllabus markdown (high
    precision, no LLM). Skips template-example lines ('E.g: …, Python, …')."""
    if not markdown:
        return []
    found = []
    for name, pat in _TOOL_PATTERNS.items():
        for mo in re.finditer(pat, markdown, re.I):
            ls = markdown.rfind("\n", 0, mo.start()) + 1
            le = markdown.find("\n", mo.end())
            line = markdown[ls:le if le != -1 else None]
            if _EXAMPLE_LINE.search(line):
                continue                          # boilerplate, keep looking
            found.append(name)
            break
    return found


# ── #2: skills we never SUGGEST adding (still shown as gaps for transparency) ──
# General-purpose / systems languages and non-teachable generics — recommending
# "add Java/C++ to Introduction to AI" is the kind of thing a committee questions.
_NO_SUGGEST = {
    "java", "c++", "c#", "scala", "go", "rust",
    "engineering", "troubleshooting", "agile", "software development",
    "software engineering", "reporting", "excel",
}


def analyze(course_skills: list[str], domains: list[str],
            top_k: int = _TOP_K, match_threshold: float = _MATCH_TAU,
            markdown: str = "") -> dict:
    """Coverage of a course against its routed domain(s)' demand core.

    markdown: optional syllabus markdown — whitelisted tools named in it
    (python, pandas, …) are added as taught skills so they stop being false gaps.
    """
    inv = _inventory()
    course_skills = [s for s in course_skills if s and s.strip()]

    # #1 fold in tools the extractor missed (named only in References/Tools)
    have = {s.lower() for s in course_skills}
    inferred_tools = [t for t in infer_tools(markdown) if t not in have]
    course_skills = course_skills + inferred_tools

    # 1) merge routed domains → one market {skill: demand}
    market: Counter = Counter()
    for d in domains:
        for s, f in inv.get(d, {}).get("skills", {}).items():
            market[s] += int(f)

    if not market or not course_skills:
        return {"domains": domains, "coverage": 0.0, "coverage_pct": 0.0,
                "recommendation": "UPDATE", "n_core": 0, "n_covered": 0,
                "covered": [], "gaps": [], "market_size": len(market)}

    # 2) top-K core skills by demand
    core = market.most_common(top_k)               # [(skill, demand), …]
    core_names = [s for s, _ in core]

    # 3) embed core + taught skills, cosine each core → nearest taught skill
    emb = route._embedder()
    C = np.asarray(emb.encode(core_names, normalize_embeddings=True), dtype=np.float64)
    T = np.asarray(emb.encode(course_skills, normalize_embeddings=True), dtype=np.float64)
    sim = np.ascontiguousarray(C) @ np.ascontiguousarray(T).T   # (K, n_taught)
    best = sim.max(axis=1)
    arg = sim.argmax(axis=1)

    # course-side mismatch (Taught − Sought): taught skills that don't match ANY
    # top-K core skill ≥ τ — i.e. things the course teaches that this market core
    # doesn't name (specialised/foundational topics, or genuinely off-market).
    best_per_taught = sim.max(axis=0) if sim.size else np.zeros(len(course_skills))
    course_mismatch = sorted(
        ({"skill": course_skills[j], "score": round(float(best_per_taught[j]), 3)}
         for j in range(len(course_skills)) if best_per_taught[j] < match_threshold),
        key=lambda d: -d["score"])

    # 4) covered vs gap, demand-weighted
    covered, gaps = [], []
    cov_demand = all_demand = 0
    for i, (skill, demand) in enumerate(core):
        all_demand += demand
        if best[i] >= match_threshold:
            cov_demand += demand
            covered.append({"skill": skill, "demand": demand,
                            "score": round(float(best[i]), 3),
                            "evidence": course_skills[int(arg[i])]})
        else:
            gaps.append({"skill": skill, "demand": demand,
                         "score": round(float(best[i]), 3)})
    completeness = (cov_demand / all_demand) if all_demand else 0.0

    # 5) gap SEVERITY (by demand share of the core) → drives the recommendation
    for g in gaps:
        share = (g["demand"] / all_demand) if all_demand else 0.0
        g["share"] = round(share, 4)
        g["severity"] = _severity(share)
    n_crit = sum(1 for g in gaps if g["severity"] == "critical")
    n_med = sum(1 for g in gaps if g["severity"] == "medium")
    recommendation = _gap_label(n_crit)

    # 6) actionable layer: strengths, demand-ranked additions, plain-language verdict
    strengths = [c["skill"] for c in covered]
    critical_gaps = [g for g in gaps if g["severity"] == "critical"]
    # #2 suggest only field-appropriate additions (drop general-purpose langs)
    suggestions = [g["skill"] for g in gaps
                   if g["severity"] in ("critical", "medium")
                   and g["skill"] not in _NO_SUGGEST][:6]

    return {
        "domains": domains,
        # quantitative research metric (kept internal; shown small in the UI)
        "alignment_score": round(100 * completeness, 1),
        "coverage": round(completeness, 4),
        "coverage_pct": round(100 * completeness, 1),
        # actionable verdict — derived from the gap list, not the raw %
        "recommendation": recommendation,
        "recommendation_text": _recommend_text(domains, strengths, critical_gaps,
                                                recommendation),
        "n_core": len(core),
        "n_covered": len(covered),
        "n_critical": n_crit,
        "n_medium": n_med,
        "covered": covered,
        "gaps": gaps,                              # market mismatch (Sought − Taught)
        "course_mismatch": course_mismatch,        # course mismatch (Taught − Sought)
        "suggestions": suggestions,
        "inferred_tools": inferred_tools,          # tools folded in from the syllabus
        "market_size": len(market),
        "match_threshold": match_threshold,
        "top_k": top_k,
    }


if __name__ == "__main__":
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
          "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
          "Gradient Boosting", "Data Exploration and Analysis (EDA)",
          "AI Application Development", "Algorithm Evaluation", "Data Preparation",
          "Algorithm Selection for Datasets", "AI Sub-Fields"]
    r = analyze(ai, ["artificial_intelligence", "data_science"])
    print(f"domains={r['domains']}  market_size={r['market_size']}")
    print(f"coverage {r['coverage_pct']}%  → {r['recommendation']}  "
          f"({r['n_covered']}/{r['n_core']} core covered)\n")
    print("COVERED core:")
    for c in r["covered"]:
        print(f"  ✓ {c['skill']:<24} demand={c['demand']:<6} ← {c['evidence']} ({c['score']})")
    print("\nTOP GAPS:")
    for g in r["gaps"][:8]:
        print(f"  ✗ {g['skill']:<24} demand={g['demand']}")
