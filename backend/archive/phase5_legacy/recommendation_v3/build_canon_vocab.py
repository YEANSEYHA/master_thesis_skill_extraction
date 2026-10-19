"""Recommendation V3 — Stage 3: build the CANONICAL MARKET VOCABULARY per domain.

Pipeline (PIPELINE.md stage 3):

    domain_inventory.json
        │  importance(skill) = demand × distinctiveness (smoothed IDF over domains)
        ▼
    top-N important skills
        │  semantic de-duplication of near-synonyms
        │  (qwen3 groups synonyms; the canonical representative is the
        │   highest-IMPORTANCE market term — reproducible + reviewable)
        ▼
    canonical vocabulary (~50 concepts)   [+ "other" handled at stage 4]
        → cache: canonical_vocab.json
          { domain: {n_postings, vocab[], importance{}, demand{}, merges{}} }

Why SMOOTHED IDF (not log(N/df)): with only 6 broad IT sub-domains, the core
technical skills (python, sql, machine learning, aws) appear in ALL of them, so a
pure IDF would zero them out. Smoothed IDF = log((N+1)/(df+1)) + 1 keeps pan-domain
skills (idf→1.0) while still lifting domain-distinctive ones (pytorch/deep-learning
idf→1.85). The effect is intentionally mild → the vocabulary is demand-led, which
is correct for a "market-demand vocabulary".

De-dup = LLM PROPOSES, EMBEDDING VETOES (hybrid). A cosine threshold ALONE can't
find synonyms — on the AI top-60, `data science ↔ data engineering` (0.740, DISTINCT
roles) outranks `data analysis ↔ data analytics` (0.731, true synonyms), so no
single threshold separates them. But qwen3 ALONE over-merges loose buckets (it once
swallowed aws/azure/spark/kafka into "tableau"). So: qwen3 proposes synonym groups
(temp 0), then each proposed pair must also clear an embedding cosine FLOOR
(VETO_FLOOR) to the group representative — embeddings can't decide synonymy but they
reliably reject absurd long-distance merges (tableau↔aws ≈ 0.22 ≪ 0.60). Measured
separation: true synonyms ≥ 0.60, catastrophic merges < 0.31. The representative is
the highest-IMPORTANCE market term (reproducible); the merge map is written for review.

When two terms merge, their demand/importance is COMBINED onto the representative
(same concept → its market demand consolidates) — which also fixes the split
denominator on the market side.

Self-contained: reads only domain_inventory.json; qwen3 via Ollama (offline build,
not the FastAPI request path). On Ollama failure it degrades to dedup="none".

Run:  python build_canon_vocab.py [domain ...]     (default: all domains)
      python build_canon_vocab.py --top-n 60 --dedup none
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_INV = _HERE / "domain_inventory.json"
_OUT = _HERE / "canonical_vocab.json"

_TOP_N = 60          # important skills pulled before de-dup (→ ~50 after merges)
_LLM = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")
_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")
_VETO_FLOOR = 0.60   # an LLM-proposed merge must clear this cosine to the rep, else
                     # it is vetoed (true synonyms ≥0.60; tableau↔aws ≈0.22 — see doc)

# Expert blocklist — concepts that must NEVER be absorbed into another (the third
# de-dup layer). These are CLOSE-but-DISTINCT pairs the embedding veto can't catch:
# `data science↔data engineering` (0.740) sits ABOVE the true synonym
# `data analysis↔data analytics` (0.731), so no cosine floor separates them, and
# qwen3 is inconsistent across domains (kept ML/DL apart in AI, merged them in DS).
# Member-side protection (these never merge AWAY); high-demand reps still absorb
# real synonyms. Reviewable domain-expert constraint.
_PROTECT = {
    "machine learning",                              # ≠ deep learning (the #1 AI skill —
                                                     # must stay its own concept, never fold
                                                     # into deep learning even when DL ranks higher)
    "deep learning", "artificial intelligence",      # ≠ machine learning
    "data engineering",                              # ≠ data science
    "cisa", "cism",                                  # distinct certifications
    "application security", "security operations",   # ≠ network security
    "networking", "network management",              # ≠ network design
    # related-but-DISTINCT security concepts (cosine 0.63–0.70 — the veto can't
    # separate these from true synonyms; only an expert blocklist can). Each is a
    # different role / learning outcome / market demand than its near-neighbour.
    "security architecture", "security controls", "security policies",  # ≠ network security
    "cloud computing",                               # ≠ cloud security
    "threat intelligence",                           # ≠ threat modeling
}


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL, device=_device())


# ── importance = demand × distinctiveness (smoothed IDF over domains) ──────────
def _smoothed_idf(inv: dict, doms: list[str]) -> dict[str, float]:
    """idf(s) = log((N+1)/(df+1)) + 1, df = #domains whose inventory lists s.

    Smoothed so a pan-domain skill (df=N) keeps idf=1.0 (not 0) — see module doc.
    """
    n = len(doms)
    df: Counter = Counter()
    for d in doms:
        for s in inv[d].get("skills", {}):
            df[s] += 1
    return {s: math.log((n + 1) / (c + 1)) + 1.0 for s, c in df.items()}


def _ranked(domain: str, inv: dict, idf: dict[str, float]) -> list[tuple[str, int, float]]:
    """[(skill, demand, importance)] for one domain, importance-sorted desc."""
    skills = inv[domain].get("skills", {})
    out = [(s, int(f), int(f) * idf.get(s, 1.0)) for s, f in skills.items()]
    out.sort(key=lambda t: -t[2])
    return out


# ── semantic de-dup of near-synonyms (qwen3 groups; code picks representative) ──
_DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        }
    },
    "required": ["groups"],
}

_DEDUP_SYS = """\
You are given a list of labour-market SKILL TERMS. Group together ONLY terms that are
INTERCHANGEABLE — swapping one for the other in a job posting would NOT change what
the job asks for. This is SYNONYM detection (the SAME skill, different wording), NOT
grouping of related or similar skills.

THE TEST — apply to every candidate pair:
  "Would replacing term A with term B change the meaning of a job posting?"
     NO  → same skill        → MERGE
     YES → different skill   → DO NOT MERGE  (even if closely related / similar)

MERGE (interchangeable — same skill: spelling/format variants, abbreviations, acronyms):
  "cyber security" + "cybersecurity"
  "machine learning" + "ml"
  "data analysis" + "data analytics"
  "certified information systems security professional (cissp)" + "cissp"

DO NOT MERGE (related but DISTINCT — different role / learning outcome / market demand):
  "network security"  vs  "security architecture"   (securing a network ≠ designing it)
  "network security"  vs  "security controls"  vs  "security policies"
  "cloud security"    vs  "cloud computing"          (securing the cloud ≠ building on it)
  "threat modeling"   vs  "threat intelligence"      (a design activity ≠ an intel feed)
  "data science"      vs  "data engineering"         (different roles)
  "machine learning"  vs  "deep learning"            (subset, not the same)
  "python"  vs  "r"                                  (different languages)
  a tool vs the concept it serves ("tableau" vs "data visualization")

Only return groups with 2+ members. Singletons are implied — do not list them.
When unsure, DO NOT merge. Return JSON: {"groups": [["term","term",...], ...]}."""


def _llm_dedup(terms: list[str]) -> list[list[str]]:
    """qwen3 → list of synonym groups (each a list of input terms). [] on failure."""
    user = "Terms:\n" + "\n".join(f"- {t}" for t in terms)
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            # num_predict caps runaway generations: grammar-constrained JSON at
            # temp 0 can repetition-loop without ever closing the object (seen
            # 15k+ tokens on one domain). A truncated reply fails json.loads →
            # the except below degrades that domain to dedup=none, build moves on.
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 2000},
            format=_DEDUP_SCHEMA,
            messages=[{"role": "system", "content": _DEDUP_SYS},
                      {"role": "user", "content": user}],
        )
        groups = json.loads(resp.message.content).get("groups", [])
    except Exception as e:
        print(f"[v3:dedup] LLM failed ({e}) — no de-dup applied", file=sys.stderr, flush=True)
        return []
    # keep only valid terms, drop singletons, assign each term to its FIRST group
    valid = set(terms)
    seen: set[str] = set()
    out: list[list[str]] = []
    for g in groups:
        members = [t for t in g if t in valid and t not in seen]
        if len(members) >= 2:
            out.append(members)
            seen.update(members)
    return out


def _veto(groups: list[list[str]], importance: dict[str, float],
          floor: float) -> list[tuple[str, list[str]]]:
    """LLM-proposed groups → [(rep, kept_members)]. rep = highest importance; a
    member is kept only if cosine(member, rep) ≥ floor (embedding veto of absurd
    long-distance merges). Groups left with no surviving member are dropped."""
    terms = sorted({t for g in groups for t in g})
    if not terms:
        return []
    emb = _embedder()
    V = np.asarray(emb.encode(terms, normalize_embeddings=True), dtype=np.float64)
    row = {t: i for i, t in enumerate(terms)}
    out: list[tuple[str, list[str]]] = []
    for g in groups:
        rep = max(g, key=lambda t: importance[t])
        kept, vetoed = [], []
        for o in g:
            if o == rep:
                continue
            if o in _PROTECT:                          # expert blocklist (layer 3)
                vetoed.append((o, "protected"))
                continue
            cos = float(V[row[o]] @ V[row[rep]])
            (kept if cos >= floor else vetoed).append((o, round(cos, 3)))
        if vetoed:
            print(f"[v3:veto] {rep}  ⟵̸  "
                  + ", ".join(f"{o}({c})" for o, c in vetoed), file=sys.stderr, flush=True)
        if kept:
            out.append((rep, [o for o, _ in kept]))
    return out


def build_canon_vocab(domain: str, top_n: int = _TOP_N,
                      dedup: str = "llm", veto_floor: float = _VETO_FLOOR,
                      inv: dict | None = None, idf: dict | None = None) -> dict:
    """Build one domain's canonical vocabulary payload (see module doc)."""
    inv = inv if inv is not None else json.loads(_INV.read_text(encoding="utf-8"))
    doms = [k for k in inv if k != "_meta"]
    if domain not in doms:
        raise KeyError(f"{domain!r} not in inventory (have: {doms})")
    idf = idf if idf is not None else _smoothed_idf(inv, doms)

    ranked = _ranked(domain, inv, idf)[:top_n]
    terms = [s for s, _, _ in ranked]
    demand = {s: d for s, d, _ in ranked}
    importance = {s: round(i, 1) for s, _, i in ranked}

    # LLM proposes synonym groups → embedding vetoes absurd merges → [(rep, members)]
    proposed = _llm_dedup(terms) if dedup == "llm" else []
    vetted = _veto(proposed, importance, veto_floor) if proposed else []

    # representative = highest-importance member; combine demand + importance onto it
    merges: dict[str, list[str]] = {}
    drop: set[str] = set()
    for rep, others in vetted:
        merges[rep] = others
        for o in others:
            demand[rep] += demand[o]
            importance[rep] = round(importance[rep] + importance[o], 1)
            drop.add(o)

    vocab = [t for t in terms if t not in drop]
    vocab.sort(key=lambda t: -importance[t])      # re-rank after merging

    return {
        "n_postings": inv[domain].get("n_postings", 0),
        "vocab": vocab,                            # canonical concepts (importance-desc)
        "importance": {t: importance[t] for t in vocab},
        "demand": {t: demand[t] for t in vocab},
        "merges": merges,                          # representative → [absorbed synonyms]
    }


def build_all(domains: list[str] | None = None, top_n: int = _TOP_N,
              dedup: str = "llm", veto_floor: float = _VETO_FLOOR) -> dict:
    inv = json.loads(_INV.read_text(encoding="utf-8"))
    doms = domains or [k for k in inv if k != "_meta"]
    idf = _smoothed_idf(inv, [k for k in inv if k != "_meta"])
    out = {"_meta": {"top_n": top_n, "dedup": dedup, "idf": "smoothed",
                     "veto_floor": veto_floor, "embedder": _MODEL,
                     "llm": _LLM if dedup == "llm" else None}}
    for d in doms:
        out[d] = build_canon_vocab(d, top_n=top_n, dedup=dedup,
                                   veto_floor=veto_floor, inv=inv, idf=idf)
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3 — build canonical market vocab")
    ap.add_argument("domains", nargs="*", help="domains to build (default: all)")
    ap.add_argument("--top-n", type=int, default=_TOP_N)
    ap.add_argument("--dedup", choices=["llm", "none"], default="llm")
    ap.add_argument("--veto-floor", type=float, default=_VETO_FLOOR)
    a = ap.parse_args()

    out = build_all(a.domains or None, top_n=a.top_n, dedup=a.dedup,
                    veto_floor=a.veto_floor)
    print(f"→ wrote {_OUT.name}  (top_n={a.top_n}, dedup={a.dedup})\n")
    for d in [k for k in out if k != "_meta"]:
        p = out[d]
        print(f"■ {d}  — {p['n_postings']:,} postings · {len(p['vocab'])} canonical concepts")
        print("   " + ", ".join(p["vocab"][:12]) + " …")
        if p["merges"]:
            for rep, members in p["merges"].items():
                print(f"     merged: {rep}  ⟵  {', '.join(members)}")
        print()
