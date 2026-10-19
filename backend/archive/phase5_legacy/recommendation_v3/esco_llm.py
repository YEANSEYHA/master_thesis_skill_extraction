"""Recommendation V3 — ESCO anchoring v2: Qwen3 retrieve top-5 → LLM pick/reject.

    skill ──Qwen3-Embedding──► top-5 distinct ESCO candidate labels
          ──qwen3:14b────────► pick the correct one / reject all

Fixes the two embedding-only failure modes seen in the 3-model comparison:
  - lexical traps ("Logistic Regression" → "logistics") — the LLM rejects them;
  - Qwen3 range compression (garbage at 0.75-0.79) — no threshold needed at all,
    the LLM verdict replaces the τ gate.

Same surface bank parse as esco.py (preferredLabel + altLabels). Qwen3 surface
embeddings cached to .esco_emb_qwen3.npz (~410MB, one-time build ~19 min via
Ollama). LLM verdicts cached per skill to .esco_llm_cache.json so re-uploads
cost zero LLM calls.

Return shape is a superset of esco.normalize():
    [{orig, esco, score, mapped, match: exact|broader|none, reason, candidates}]
"""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CSV = _HERE / "esco_skills_en.csv"
_EMB_CACHE = _HERE / ".esco_emb_qwen3.npz"
_LLM_CACHE = _HERE / ".esco_llm_cache.json"

_EMBED_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
_PICK_MODEL = "qwen3:14b"
_TOP_K = 5


def _load_surfaces():
    """Same parse as esco.py — keep the two banks index-compatible."""
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


def _candidates(skills: list[str], top_k: int = _TOP_K):
    """Per skill: top-k DISTINCT preferred labels by max cosine over surfaces."""
    preferred, V = _bank()
    Q = _embed(skills).astype(np.float64)
    sims = Q @ np.ascontiguousarray(V.astype(np.float64)).T
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


def normalize(skills: list[str], top_k: int = _TOP_K, use_cache: bool = True):
    """Map each skill to an LLM-verified ESCO label, or keep raw on reject.

    Returns [{orig, esco, score, mapped, match, reason, candidates}] — `esco`
    is the label used for matching (= orig when mapped is False)."""
    skills = [s for s in skills if s and s.strip()]
    if not skills:
        return []

    cache: dict = {}
    if use_cache and _LLM_CACHE.exists():
        try:
            cache = json.loads(_LLM_CACHE.read_text())
        except Exception:
            cache = {}

    todo = [s for s in skills if s.lower() not in cache]
    if todo:
        for s, cands in zip(todo, _candidates(todo, top_k=top_k)):
            try:
                d = _pick(s, cands) if cands else {"choice": None, "match": "none",
                                                   "reason": "no candidates"}
            except Exception as e:
                d = {"choice": None, "match": "none", "reason": f"pick error: {e}"}
            ci = d.get("choice")
            ok = (isinstance(ci, int) and 1 <= ci <= len(cands)
                  and d.get("match") in ("exact", "broader"))
            cache[s.lower()] = {
                "orig": s,
                "esco": cands[ci - 1][0] if ok else s,
                "score": round(cands[ci - 1][1] if ok
                               else (cands[0][1] if cands else 0.0), 3),
                "mapped": bool(ok),
                "match": d.get("match", "none") if ok else "none",
                "reason": str(d.get("reason", "")).strip(),
                "candidates": [{"label": lbl, "score": round(sc, 3)}
                               for lbl, sc in cands],
            }
        if use_cache:
            _LLM_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    return [dict(cache[s.lower()], orig=s) for s in skills]


def normalize_threshold(skills: list[str], floor: float = 0.80,
                        top_k: int = _TOP_K):
    """Playground-style ESCO mapping: top-1 cosine + keep-raw floor, NO LLM pick.

    Same row schema as normalize() so build_course_units / verify_names logic
    work unchanged. Cheaper (zero LLM calls) but reintroduces the two failure
    modes the LLM pick was built for: lexical traps and range-compression
    drift near the floor (e.g. tool→concept snaps at 0.80-0.82)."""
    skills = [s for s in skills if s and s.strip()]
    if not skills:
        return []
    rows = []
    for s, cands in zip(skills, _candidates(skills, top_k=top_k)):
        label, score = cands[0] if cands else (s, 0.0)
        mapped = bool(cands) and score >= floor
        esco = label if mapped else s
        match = ("exact" if mapped and label.lower() == s.lower()
                 else "broader" if mapped else "none")
        rows.append({
            "orig": s,
            "esco": esco,
            "score": round(score, 3),
            "mapped": mapped,
            "match": match,
            "reason": (f"top-1 cosine {score:.2f} >= {floor}" if mapped
                       else f"top-1 cosine {score:.2f} < {floor} — kept raw"),
            "candidates": [{"label": lbl, "score": round(sc, 3)}
                           for lbl, sc in cands],
        })
    return rows


if __name__ == "__main__":
    tests = ["Supervised Learning", "K-Means", "Gaussian Mixture",
             "Logistic Regression", "Gradient Descent", "Gradient Boosting",
             "Data Exploration and Analysis (EDA)", "Data Preparation",
             "Python", "Pandas", "Jupyter"]
    for r in normalize(tests):
        tag = f"[{r['match']}]" if r["mapped"] else "[KEPT RAW]"
        print(f"{r['orig']:<38} -> {r['esco']:<42} {r['score']:.2f} {tag:<10} {r['reason']}")
