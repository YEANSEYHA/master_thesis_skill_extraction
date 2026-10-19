"""V6 matching core — both sides ESCO-anchored, then:

    retrieve (Qwen3 cosine, max-pool)  →  verify (qwen3:14b yes/no)
    →  credit best yes  →  demand-weighted coverage
"""

import json

import numpy as np

EMBED_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
VERIFY_MODEL = "qwen3:14b"
FLOOR = 0.70        # retrieve floor (Qwen3 cosine scale); was 0.55 — see recall note
TOP_N = 2           # candidates per market skill handed to verify

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"teaches": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["teaches", "reason"],
}
VERIFY_SYS = ("You judge whether a university course COMPETENCY actually teaches a "
              "labour-market SKILL. Say teaches=true only if the competency "
              "genuinely covers that skill — not merely related. Keep reason to "
              "one short clause.")


def embed(texts: list[str]):
    import ollama
    v = np.asarray(ollama.embed(model=EMBED_MODEL, input=list(texts)).embeddings,
                   dtype=np.float64)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def build_course_units(esco_rows: list[dict]) -> list[dict]:
    """Merge skills that share an ESCO label into one unit; raw names = evidence."""
    by_label: dict[str, list[str]] = {}
    for r in esco_rows:
        by_label.setdefault(r["esco"], []).append(r["orig"])
    return [{"competency": lbl,
             "evidence": [o for o in origs if o.lower() != lbl.lower()]}
            for lbl, origs in by_label.items()]


# ── STEP 1 · RETRIEVE ───────────────────────────────────────────────────────────
def retrieve(market_names: list[str], units: list[dict],
             top_n: int = TOP_N, floor: float = FLOOR):
    """Per market skill: top-N unit indices by max-pooled cosine ≥ floor."""
    parts, owner = [], []
    for j, u in enumerate(units):
        for p in [u["label"], *u["evidence"]]:
            parts.append(p)
            owner.append(j)

    sim_parts = embed(market_names) @ embed(parts).T          # market × parts
    sim = np.full((len(market_names), len(units)), -1.0)
    for k, j in enumerate(owner):
        np.maximum(sim[:, j], sim_parts[:, k], out=sim[:, j])  # max-pool → unit

    return [[(int(j), float(sim[i][j]))
             for j in np.argsort(-sim[i]) if sim[i][j] >= floor][:top_n]
            for i in range(len(market_names))]


# ── STEP 2 · VERIFY ─────────────────────────────────────────────────────────────
def verify(skill: str, label: str, evidence: list[str]):
    import ollama
    user = (f"Market skill: {skill}\n"
            f"Course competency: {label}\n"
            f"Topics taught under it: {', '.join(evidence) or label}\n"
            f"Does this competency teach '{skill}'?")
    try:
        r = ollama.chat(model=VERIFY_MODEL, think=False,
                        options={"temperature": 0, "num_ctx": 4096},
                        format=VERIFY_SCHEMA,
                        messages=[{"role": "system", "content": VERIFY_SYS},
                                  {"role": "user", "content": user}])
        d = json.loads(r.message.content)
        return bool(d.get("teaches")), str(d.get("reason", "")).strip()
    except Exception as e:
        return False, f"verify error: {e}"


# ── STEP 3 + 4 · DECIDE, CREDIT, SCORE ──────────────────────────────────────────
def match(core: list[tuple], match_names: list[str], units: list[dict],
          verify_names: list[str] | None = None) -> dict:
    """core = [(raw_name, demand, importance)] · match_names = ESCO (or raw) names
    aligned to core · units = course reps [{label, evidence}].

    verify_names: what STEP 3 asks about, aligned to core. Broad ESCO anchors are
    good for RETRIEVAL but must not replace the VERIFICATION target (a broad label
    erases specificity: cissp→"cyber security" would be trivially "covered").
    Defaults to match_names.

    Returns {covered, gaps, off_core, coverage, cov_demand, all_demand}."""
    verify_names = verify_names or match_names
    shortlist = retrieve(match_names, units)

    covered, gaps, used = [], [], set()
    cov_demand = all_demand = 0
    for i, (concept, demand, importance) in enumerate(core):
        target = match_names[i]
        all_demand += demand

        hits = []
        for j, score in shortlist[i]:
            ok, reason = verify(verify_names[i], units[j]["label"], units[j]["evidence"])
            if ok:
                hits.append((j, score, reason))

        if hits:
            j, score, reason = max(hits, key=lambda h: h[1])   # best-cosine yes
            used.add(j)
            cov_demand += demand
            covered.append({"skill": concept, "demand": demand,
                            "esco": target if target != concept else None,
                            "importance": round(importance, 1),
                            "score": round(score, 3),
                            "course_competency": units[j]["label"],
                            "evidence": units[j]["evidence"], "reason": reason})
        else:
            near = shortlist[i][0] if shortlist[i] else None
            gaps.append({"skill": concept, "demand": demand,
                         "esco": target if target != concept else None,
                         "importance": round(importance, 1),
                         "score": round(near[1], 3) if near else 0.0,
                         "nearest_course_competency":
                             units[near[0]]["label"] if near else None})

    off_core = [{"competency": u["label"], "evidence": u["evidence"]}
                for j, u in enumerate(units) if j not in used]
    coverage = (cov_demand / all_demand) if all_demand else 0.0

    return {"covered": covered, "gaps": gaps, "off_core": off_core,
            "coverage": coverage, "cov_demand": cov_demand, "all_demand": all_demand}
