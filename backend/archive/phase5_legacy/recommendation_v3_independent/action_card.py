"""V7 — label-free ACTION CARD analyze (playground/coverage_score design).

Wraps the frozen V6 engine (recommend_verified.analyze_verified, group=False)
for the market→course direction, and ADDS the missing course→market direction
(relevance), then shapes both into the card payload:

    Taught & in demand   X of N taught units (verified pairs)     ← relevance
    Field core covered   Y of K                                   ← V6 coverage
    ACTIONS
      ADD    — most-demanded field skills not taught (from V6 gaps)
      RENAME — taught, demanded under different market wording
      REVIEW — taught, no verified demand (foundations are NOT trim candidates)
    (supplementary: importance-weighted coverage %)

Labels + headline-% are retired by design (2026-07-12): no threshold survived
granularity changes, so the recommendation = the evidence-ranked action list.
V6 code is reused untouched; everything new lives in this file.

Run:  python action_card.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent / "recommendation_v3"
for _p in (str(_HERE), str(_V3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recommend as _rec                    # noqa: E402  _core / _course_reps
import esco_match as _match                 # noqa: E402  embed / verify / FLOOR / TOP_N
import recommend_verified as _rv            # noqa: E402  the frozen V6 engine

VOCAB_N = 200        # relevance is measured against the top-N market vocab —
                     # a skill demanded once is noise, not demand
_CACHE_FILE = _HERE / ".action_card_cache.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _cache_read() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except Exception:
        return {}


def _retrieve_for_units(units: list[dict], vocab_names: list[str],
                        top_n: int = _match.TOP_N,
                        floor: float = _match.FLOOR) -> list[list[tuple]]:
    """Reverse of esco_match.retrieve — per course UNIT: top-N market vocab
    indices by max-pooled cosine ≥ floor."""
    parts, owner = [], []
    for j, u in enumerate(units):
        for p in [u["label"], *u["evidence"]]:
            parts.append(p)
            owner.append(j)

    sim_parts = _match.embed(parts) @ _match.embed(vocab_names).T  # parts × vocab
    sim = np.full((len(units), len(vocab_names)), -1.0)
    for k, j in enumerate(owner):
        np.maximum(sim[j], sim_parts[k], out=sim[j])               # max-pool → unit

    return [[(int(i), float(sim[j][i]))
             for i in np.argsort(-sim[j]) if sim[j][i] >= floor][:top_n]
            for j in range(len(units))]


def _relevance(units: list[dict], vocab: list[tuple]) -> tuple[list, list]:
    """Course→market: is each taught unit demanded anywhere in the vocab?
    vocab = [(name, demand, importance)] from _rec._core. Returns
    (in_demand, review) — review = taught with no verified demand."""
    names = [v[0] for v in vocab]
    shortlist = _retrieve_for_units(units, names)
    in_demand, review = [], []
    for j, u in enumerate(units):
        hits = []
        for i, cos in shortlist[j]:
            ok, why = _match.verify(names[i], u["label"], u["evidence"])
            if ok:
                hits.append((i, cos, why))
        if hits:
            i, cos, why = max(hits, key=lambda h: h[1])            # best-cosine yes
            in_demand.append({"name": u["label"], "evidence": u["evidence"],
                              "sought_as": names[i], "demand": vocab[i][1],
                              "cos": round(cos, 3), "reason": why})
        else:
            review.append({"name": u["label"], "evidence": u["evidence"]})
    return in_demand, review


def analyze_action_card(course_skills: list[str], domains: list[str],
                        top_k: int = 30, vocab_n: int = VOCAB_N,
                        markdown: str = "", course_code: str = "",
                        course_title: str = "", force: bool = False) -> dict:
    """V7 analyze: V6 coverage (unchanged engine) + relevance → card payload."""
    key = hashlib.sha1(json.dumps(
        ["v7.1", sorted(course_skills), sorted(domains), top_k, vocab_n,
         int(_rec._VOCAB.stat().st_mtime),
         hashlib.sha1(markdown.encode()).hexdigest()]).encode()).hexdigest()
    if not force:
        hit = _cache_read().get(key)
        if hit:
            return {**hit, "cached": True}

    # Market→course: the frozen V6 retrieve→verify engine, untouched.
    base = _rv.analyze_verified(
        course_skills, domains, top_k=top_k, markdown=markdown,
        course_code=course_code, course_title=course_title,
        force=force, group=False, matcher="verify")

    # Course→market: relevance against the wider vocab (not just the top-K core).
    reps = _rec._course_reps(base.get("competencies") or [])
    vocab, _ = _rec._core(domains, vocab_n)
    if reps and vocab:
        in_demand, review = _relevance(reps, vocab)
    else:
        in_demand, review = [], [{"name": r["label"], "evidence": r["evidence"]}
                                 for r in reps]

    # RENAME — taught and sought, but under different market wording.
    renames = [r for r in in_demand
               if _norm(r["name"]) != _norm(r["sought_as"]) and r["cos"] < 0.98]

    result = {
        "domains": base.get("domains", domains),
        "top_k": base.get("top_k", top_k),
        "vocab_n": len(vocab),
        "n_taught": len(reps),
        "n_in_demand": len(in_demand),
        "taught_in_demand": in_demand,
        "n_core": base.get("n_core", 0),
        "core_covered": base.get("covered", []),
        "add": base.get("gaps", []),
        "renames": renames,
        "review": review,
        "coverage_pct": base.get("coverage_pct",
                                 base.get("alignment_score", 0.0)),
        "esco": base.get("esco"),
        "inferred_tools": base.get("inferred_tools", []),
        "method": "action_card_v7",
    }
    cache = _cache_read()
    cache[key] = result
    _CACHE_FILE.write_text(json.dumps(cache))
    return result


if __name__ == "__main__":
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees",
          "Cross-Validation", "Logistic Regression", "K-Nearest Neighbor",
          "Gradient Descent", "Gradient Boosting",
          "Data Exploration and Analysis (EDA)", "Data Preparation"]
    r = analyze_action_card(ai, ["artificial_intelligence"], top_k=15,
                            course_code="AIECS031",
                            course_title="Introduction to Artificial Intelligence")
    print(f"Taught & in demand  {r['n_in_demand']} of {r['n_taught']}")
    for x in r["taught_in_demand"]:
        print(f"  ✔ {x['name']:<32} sought as '{x['sought_as']}' (cos {x['cos']:.2f})")
    print(f"\nField core covered  {len(r['core_covered'])} of {r['n_core']}")
    for x in r["core_covered"]:
        print(f"  ✔ {x['skill']:<32} <- {x['course_competency']} ({x['score']:.2f})")
    print("\nADD:")
    for x in r["add"][:5]:
        print(f"  ✘ {x['skill']:<32} demand {x['demand']}  "
              f"nearest: {x.get('nearest_course_competency')}")
    print("\nRENAME:")
    for x in r["renames"]:
        print(f"  ↺ '{x['name']}' -> market says '{x['sought_as']}' ({x['cos']:.2f})")
    print("\nREVIEW:")
    for x in r["review"]:
        print(f"  • {x['name']}")
    print(f"\n(supplementary coverage {r['coverage_pct']:.1f}%)")
