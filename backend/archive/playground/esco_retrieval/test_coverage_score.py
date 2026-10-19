"""Playground — Step 1 only: cosine retrieve top-5 distinct ESCO labels per skill."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

job_markets=[
  "cybersecurity",
  "vulnerability management",
  "cissp",
  "information security",
  "network security",
  "incident response",
  "threat modeling",
  "cloud security",
  "siem",
  "application security"
]


course_skills=[
  "Cybercrime Analysis",
  "Secure Coding",
  "Secure Software Development",
  "Digital Forensics",
  "Ethical Software Development",
  "Input Validation",
  "Secure Data Storage",
  "Access Control",
  "SQL Injection",
  "Cross-Site Scripting (XSS)",
  "Digital Evidence Analysis",
  "Secure Web Application Development",
  "Network Security Controls",
  "Digital Forensics Tools",
  "Incident Evidence Analysis",
  "Ethical Response Planning",
  "sql"
]

# ESCO CSV + Qwen3 embedding bank shared with recommendation_v3 (410MB, prebuilt).
_V3 = Path(__file__).resolve().parents[2] / "phase5_legacy" / "recommendation_v3"
_CSV = _V3 / "esco_skills_en.csv"
_EMB_CACHE = _V3 / ".esco_emb_qwen3.npz"

_EMBED_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
_TOP_K = 5


def _load_surfaces():
    """CSV → (surfaces, preferred): every preferredLabel + altLabel is a
    searchable surface; each surface points back at its preferred label."""
    surfaces, preferred = [], []
    with open(_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = (row.get("preferredLabel") or "").strip()
            if not p:
                continue
            surfaces.append(p)
            preferred.append(p)
            for a in (row.get("altLabels") or "").split("\n"):
                a = a.strip()
                if a:
                    surfaces.append(a)
                    preferred.append(p)
    return surfaces, preferred


def _embed(texts: list[str], batch: int = 256):
    """L2-normalized Qwen3 embeddings via Ollama."""
    import ollama
    vecs = []
    for i in range(0, len(texts), batch):
        r = ollama.embed(model=_EMBED_MODEL, input=list(texts[i:i + batch]))
        vecs.extend(r.embeddings)
    v = np.asarray(vecs, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


@lru_cache(maxsize=1)
def _bank():
    """(preferred_per_surface, V) — Qwen3-embedded ESCO surface bank."""
    surfaces, preferred = _load_surfaces()
    if _EMB_CACHE.exists():
        z = np.load(_EMB_CACHE)
        if z["V"].shape[0] == len(surfaces):
            return preferred, z["V"]
    V = _embed(surfaces)                       # one-time ~19 min; then cached
    np.savez(_EMB_CACHE, V=V)
    return preferred, V


# ── Step 1 · retrieve top-k candidate labels per skill ──────────────────────────

def candidates(skills: list[str], top_k: int = _TOP_K):
    """Per skill: top-k DISTINCT preferred labels by max cosine over surfaces.

    Returns, aligned to `skills`: [[(esco_label, cosine), …top_k], …]."""
    preferred, V = _bank()
    Q = _embed(skills).astype(np.float64)
    sims = Q @ np.ascontiguousarray(V.astype(np.float64)).T   # skills × surfaces
    out = []
    for i in range(len(skills)):
        best: dict[str, float] = {}
        for j in np.argsort(-sims[i])[:400]:   # scan until k distinct labels
            lbl = preferred[int(j)]
            if lbl not in best:
                best[lbl] = float(sims[i][int(j)])
            if len(best) >= top_k:
                break
        out.append(sorted(best.items(), key=lambda kv: -kv[1]))
    return out


# ── Step 2 · rerank: LLM picks the right candidate (or rejects all) ─────────────

import json

_PICK_MODEL = "qwen3:14b"

_PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": ["integer", "null"]},
        "match": {"type": "string", "enum": ["exact", "broader", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["choice", "match", "reason"],
}

_PICK_SYS = (
    "You map a course/job SKILL onto the official ESCO skill taxonomy. Given a "
    "skill and numbered candidate ESCO labels, pick the ONE candidate that denotes "
    "the same skill (match=exact), or the standard broader category it belongs to "
    "(match=broader) when no exact candidate exists. If every candidate is wrong, "
    "answer choice=null, match=none. NEVER pick a lexical look-alike from another "
    "domain (e.g. 'Logistic Regression' is NOT 'logistics'; 'Gradient Descent' is "
    "NOT sports 'periodisation'; 'Pandas' the Python library is NOT about games or "
    "animals). Keep reason to one short clause."
)


def _pick(skill: str, cands: list[tuple[str, float]]) -> dict:
    """LLM verdict: which candidate (1-based) is the right ESCO label, if any."""
    import ollama
    lines = "\n".join(f"{i + 1}. {lbl}" for i, (lbl, _) in enumerate(cands))
    user = (f"Skill: {skill}\nCandidate ESCO labels:\n{lines}\n"
            f"Which candidate (1-{len(cands)}) is the correct label, or none?")
    r = ollama.chat(model=_PICK_MODEL, think=False,
                    options={"temperature": 0, "num_ctx": 2048},
                    format=_PICK_SCHEMA,
                    messages=[{"role": "system", "content": _PICK_SYS},
                              {"role": "user", "content": user}])
    return json.loads(r.message.content)


def rerank(skills: list[str], top_k: int = _TOP_K) -> list[dict]:
    """Step 1 + Step 2: retrieve top-k candidates, LLM picks one / rejects all.

    Returns, aligned to `skills`:
    [{orig, esco, score, rank, match, reason, candidates}] — `esco` keeps the
    raw name when the LLM rejects every candidate (match=none)."""
    out = []
    for skill, cands in zip(skills, candidates(skills, top_k=top_k)):
        d = _pick(skill, cands) if cands else {"choice": None, "match": "none",
                                               "reason": "no candidates"}
        ci = d.get("choice")
        ok = (isinstance(ci, int) and 1 <= ci <= len(cands)
              and d.get("match") in ("exact", "broader"))
        out.append({
            "orig": skill,
            "esco": cands[ci - 1][0] if ok else skill,
            "score": round(cands[ci - 1][1], 3) if ok else 0.0,
            "rank": ci if ok else None,
            "match": d.get("match", "none") if ok else "none",
            "reason": str(d.get("reason", "")).strip(),
            "candidates": [{"label": lbl, "score": round(sc, 3)} for lbl, sc in cands],
        })
    return out


# ── Step 3 · MERGE: course skills sharing an ESCO label → one unit ──────────────

def build_course_units(esco_rows: list[dict]) -> list[dict]:
    """Merge skills that share an ESCO label into one unit; raw names = evidence."""
    by_label: dict[str, list[str]] = {}
    for r in esco_rows:
        by_label.setdefault(r["esco"], []).append(r["orig"])
    return [{"label": lbl,
             "evidence": [o for o in origs if o.lower() != lbl.lower()]}
            for lbl, origs in by_label.items()]


# ── Step 4 · RETRIEVE: shortlist course units per market skill ──────────────────

_FLOOR = 0.55       # retrieve floor (Qwen3 cosine scale)
_TOP_N = 2          # candidate units per market skill handed to verify


def retrieve(market_names: list[str], units: list[dict],
             top_n: int = _TOP_N, floor: float = _FLOOR):
    """Per market skill: top-N unit indices by max-pooled cosine ≥ floor.

    Cosine is computed against the unit label AND every evidence string, then
    max-pooled to the unit — so a specific raw topic can carry the match even
    when the merged label is broad."""
    parts, owner = [], []
    for j, u in enumerate(units):
        for p in [u["label"], *u["evidence"]]:
            parts.append(p)
            owner.append(j)

    sim_parts = _embed(market_names).astype(np.float64) @ \
        _embed(parts).astype(np.float64).T                     # market × parts
    sim = np.full((len(market_names), len(units)), -1.0)
    for k, j in enumerate(owner):
        np.maximum(sim[:, j], sim_parts[:, k], out=sim[:, j])  # max-pool → unit

    return [[(int(j), float(sim[i][j]))
             for j in np.argsort(-sim[i]) if sim[i][j] >= floor][:top_n]
            for i in range(len(market_names))]


# ── Step 5 · VERIFY: does the course unit actually teach the market skill? ──────

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"teaches": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["teaches", "reason"],
}
_VERIFY_SYS = ("You judge whether a university course COMPETENCY actually teaches a "
               "labour-market SKILL. Say teaches=true only if the competency "
               "genuinely covers that skill — not merely related. Keep reason to "
               "one short clause.")


def verify(skill: str, label: str, evidence: list[str]):
    import ollama
    user = (f"Market skill: {skill}\n"
            f"Course competency: {label}\n"
            f"Topics taught under it: {', '.join(evidence) or label}\n"
            f"Does this competency teach '{skill}'?")
    try:
        r = ollama.chat(model=_PICK_MODEL, think=False,
                        options={"temperature": 0, "num_ctx": 4096},
                        format=_VERIFY_SCHEMA,
                        messages=[{"role": "system", "content": _VERIFY_SYS},
                                  {"role": "user", "content": user}])
        d = json.loads(r.message.content)
        return bool(d.get("teaches")), str(d.get("reason", "")).strip()
    except Exception as e:
        return False, f"verify error: {e}"


# ── Step 6 · SCORE: coverage over the market skills ─────────────────────────────

def coverage_score(market_rows: list[dict], units: list[dict]) -> dict:
    """market_rows = rerank(job_markets) · units = build_course_units(rerank(course)).

    Retrieval uses the ESCO anchor; verification uses the RAW name whenever the
    anchor is broader (a broad label like 'cyber security' would make specific
    skills like cissp trivially 'covered'). Demand is uniform (1 per skill) —
    plug real demand counts in when wiring to the inventory."""
    match_names = [r["esco"] for r in market_rows]
    verify_names = [r["esco"] if r["match"] == "exact" else r["orig"]
                    for r in market_rows]
    shortlist = retrieve(match_names, units)

    covered, gaps, used = [], [], set()
    for i, row in enumerate(market_rows):
        hits = []
        for j, score in shortlist[i]:
            ok, reason = verify(verify_names[i], units[j]["label"], units[j]["evidence"])
            if ok:
                hits.append((j, score, reason))

        if hits:
            j, score, reason = max(hits, key=lambda h: h[1])   # best-cosine yes
            used.add(j)
            covered.append({"skill": row["orig"], "esco": row["esco"],
                            "score": round(score, 3),
                            "course_unit": units[j]["label"], "reason": reason})
        else:
            near = shortlist[i][0] if shortlist[i] else None
            gaps.append({"skill": row["orig"], "esco": row["esco"],
                         "score": round(near[1], 3) if near else 0.0,
                         "nearest_unit": units[near[0]]["label"] if near else None})

    off_core = [u["label"] for j, u in enumerate(units) if j not in used]
    return {"covered": covered, "gaps": gaps, "off_core": off_core,
            "coverage": len(covered) / len(market_rows) if market_rows else 0.0}

# V6 coverage algorithm — embeddings shortlist, the LLM decides:
#
#   ANCHOR   1. candidates()         each skill → top-5 ESCO labels (cosine recall)
#            2. rerank()             LLM picks one: exact / broader / keep raw
#   PREP     3. build_course_units() same-ESCO course skills → 1 unit {label, evidence}
#   MATCH    4. retrieve()           per market skill: top-2 units (max-pool ≥ 0.55)
#            5. verify()             LLM: "does it teach X?" (X = raw name if broader)
#   SCORE    6. coverage_score()     covered / gaps / off_core → coverage %
#
# Cosine only ranks candidates — every accept/reject decision is an LLM verdict.


if __name__ == "__main__":
    print("── market side: rerank ─────────────────────────────────────────")
    market = rerank(job_markets)
    for r in market:
        tag = f"[{r['match']}]" if r["match"] != "none" else "[KEPT RAW]"
        print(f"  {r['orig']:<28} -> {r['esco']:<42} {tag}")

    print("\n── course side: rerank + merge ─────────────────────────────────")
    course = rerank(course_skills)
    units = build_course_units(course)
    print(f"  {len(course_skills)} course skills -> {len(units)} units")
    for u in units:
        ev = f"  <- {', '.join(u['evidence'])}" if u["evidence"] else ""
        print(f"  • {u['label']}{ev}")

    print("\n── retrieve + verify + score ───────────────────────────────────")
    res = coverage_score(market, units)
    print(f"\nCOVERAGE: {res['coverage']:.1%}  "
          f"({len(res['covered'])}/{len(market)} market skills covered)\n")
    for c in res["covered"]:
        print(f"  ✔ {c['skill']:<26} <- {c['course_unit']:<40} {c['score']:.2f}  {c['reason']}")
    for g in res["gaps"]:
        print(f"  ✘ {g['skill']:<26} gap (nearest: {g['nearest_unit']}, {g['score']:.2f})")
    print("\n  off-core (taught, not in this market top-10):")
    for lbl in res["off_core"]:
        print(f"    • {lbl}")
