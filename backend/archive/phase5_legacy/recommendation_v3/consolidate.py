"""Recommendation V3 — Stage 4: COMPETENCY CONSOLIDATION.

Group a course's raw extracted skills into competencies, classifying each INTO the
canonical market vocabulary (stage 3) where it fits, else naming a SPECIALIZED
(curriculum-specific) competency. Evidence (the original skills) is kept under each.

    raw course skills  +  routed domain(s)
        │  load canonical vocab for those domains (union, importance-ranked)
        ▼
    ONE qwen3 call (temp 0): classify each skill into a canonical concept
        ("market") or a short curriculum-specific label ("specialized")
        ▼
    validate (labels ∈ vocab for market; evidence ∈ input; nothing dropped)
        ▼
    competencies[]  { competency, source: market|specialized, evidence:[raw…] }
        → cache: course_code → .competency_cache.json   (run once per syllabus)

Design (PIPELINE.md stage 4): the LLM is used ONLY as a classifier INTO the fixed
market vocabulary — it does not invent the vocabulary. Market labels are validated
against the vocab (snap to canonical casing; demote stragglers to specialized).
SPECIALIZED is the readable escape hatch for off-market / foundational content
(e.g. "Automata Theory") — kept for display, excluded from market scoring downstream.

Self-contained: reads only canonical_vocab.json; qwen3 via Ollama. On LLM failure
returns a single specialized "uncategorized" group (safe degradation).

Run:  python consolidate.py
"""

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VOCAB = _HERE / "canonical_vocab.json"
_CACHE = _HERE / ".competency_cache.json"

sys.path.insert(0, str(_HERE))
import clean  # noqa: E402  (stage ① CLEAN, sibling module)

_LLM = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")
_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")  # same as the vocab builder
_SNAP_FLOOR = 0.70     # P2: snap an LLM "market" label to the nearest canonical concept
_ATTACH_FLOOR = 0.62   # P1: attach a dropped skill to an existing competency, else own group


# ── embedder (shared model, lazy) — used only to repair LLM output, not to group ──
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


def _encode(texts: list[str]):
    """L2-normalized embeddings (float64) so a dot product is cosine similarity."""
    import numpy as np
    return np.asarray(_embedder().encode(texts, normalize_embeddings=True), dtype=np.float64)


def _sims(mat, vec):
    """Cosine similarities (mat is row-normalized, vec normalized). Wrapped in
    errstate to silence the spurious Apple-Accelerate matmul FP flags (the values
    are finite — verified — the warnings are backend noise, not real errors)."""
    import numpy as np
    with np.errstate(all="ignore"):
        return mat @ vec


# ── canonical vocab for the routed domain(s) ──────────────────────────────────
def _vocab_for(domains: list[str]) -> list[str]:
    """Union of the routed domains' canonical vocabs, ranked by max importance."""
    cv = json.loads(_VOCAB.read_text(encoding="utf-8"))
    imp: dict[str, float] = {}
    for d in domains:
        dom = cv.get(d)
        if not dom:
            continue
        for term, score in dom.get("importance", {}).items():
            imp[term] = max(imp.get(term, 0.0), float(score))
    return [t for t, _ in sorted(imp.items(), key=lambda kv: -kv[1])]


# ── LLM consolidation ─────────────────────────────────────────────────────────
_SCHEMA = {
    "type": "object",
    "properties": {
        "competencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "competency": {"type": "string"},
                    "source": {"type": "string", "enum": ["market", "specialized"]},
                    "evidence": {"type": "array", "items": {"type": "string"},
                                 "minItems": 1},
                },
                "required": ["competency", "source", "evidence"],
            },
        }
    },
    "required": ["competencies"],
}

_SYS = """\
You group a university course's extracted SKILLS into COMPETENCIES. Each competency
has a label, a source, and its evidence (the original skills it groups).

RULES
- The "competency" label is a real SUBJECT or SKILL concept (e.g. "machine learning",
  "Complex Numbers", "Calculus", "Numerical Methods"). It is NEVER one of the words
  "market", "specialized", "uncategorized", "other" or "misc" — those are not labels.
  "source" tells where the label came from; "competency" is the concept name itself.
- PREFER a label from the CANONICAL VOCABULARY the user provides. When a group of
  skills clearly corresponds to a canonical concept, use that EXACT label (verbatim)
  and set source = "market".
    e.g. K-Means, Decision Trees, Logistic Regression, KNN → "machine learning" (market)
- If a group of skills fits NO canonical concept, create a SHORT, meaningful
  curriculum-specific label (the subject name) and set source = "specialized". Still
  split off-market content into MULTIPLE concepts — do not dump everything into one.
    e.g. Pushdown Automata, Turing Machines → "Automata Theory" (specialized)
         Differentiation, Integration, Maclaurin Series → "Calculus" (specialized)
         Complex Numbers, De Moivre's Theorem → "Complex Numbers" (specialized)
- Every input skill must appear as evidence under EXACTLY ONE competency.
- Evidence must be the ORIGINAL skill strings, verbatim. Do not invent skills.
- Group by concept, not by wording — several techniques can share one competency.

Return JSON: {"competencies":[{"competency","source","evidence":[...]}, ...]}."""


def _llm_consolidate(skills: list[str], vocab: list[str]) -> list[dict]:
    user = ("Canonical vocabulary (use these EXACT labels for market competencies):\n"
            + ", ".join(vocab) + "\n\nCourse skills to group:\n"
            + "\n".join(f"- {s}" for s in skills))

    # ── debug: what goes INTO the LLM (curriculum skills + market vocab) ──
    print("\n[v3:consolidate] ── INPUT ────────────────────────────────",
          file=sys.stderr)
    print(f"[v3:consolidate] curriculum skills ({len(skills)}): "
          + ", ".join(skills), file=sys.stderr)
    print(f"[v3:consolidate] market vocab ({len(vocab)}): "
          + ", ".join(vocab[:20]) + (" …" if len(vocab) > 20 else ""),
          file=sys.stderr, flush=True)
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_SCHEMA,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": user}],
        )
        groups = json.loads(resp.message.content).get("competencies", [])
        # ── debug: what comes OUT of the LLM (raw groups, pre-validation) ──
        print("[v3:consolidate] ── OUTPUT (raw LLM groups) ──────────────",
              file=sys.stderr)
        for g in groups:
            print(f"[v3:consolidate]   {g.get('competency', '?')}  "
                  f"[{g.get('source', '?')}]  ← "
                  + ", ".join(g.get("evidence", [])), file=sys.stderr)
        print("[v3:consolidate] ─────────────────────────────────────────",
              file=sys.stderr, flush=True)
        return groups
    except Exception as e:
        print(f"[v3:consolidate] LLM failed ({e})", file=sys.stderr, flush=True)
        return []


_INDEPENDENT_SCHEMA = {
    "type": "object",
    "properties": {
        "competencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "competency": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"},
                                 "minItems": 1},
                },
                "required": ["competency", "evidence"],
            },
        }
    },
    "required": ["competencies"],
}

_INDEPENDENT_SYS = """\
You group a university course's extracted SKILLS into COURSE COMPETENCIES.

IMPORTANT
- Use ONLY the course skills provided by the user.
- Do NOT use job-posting data, market demand, canonical market vocabulary, or
  industry popularity to choose labels.
- Group by the concept taught in the course, not by whether the concept is
  demanded in the labour market.

RULES
- The "competency" label is a short subject/skill concept taught by the course
  (e.g. "Supervised Learning", "Model Evaluation", "Database Design").
- Every input skill must appear as evidence under EXACTLY ONE competency.
- Evidence must be the ORIGINAL skill strings, verbatim. Do not invent skills.
- Split large subjects into meaningful sub-competencies when the skills support
  it; do not put everything under one broad label.
- Keep closely related techniques together when they are naturally taught as the
  same competency.

Return JSON: {"competencies":[{"competency","evidence":[...]}, ...]}."""


def _llm_independent_consolidate(skills: list[str]) -> list[dict]:
    """qwen3 course-only grouping. No market vocabulary is provided."""
    user = "Course skills to group:\n" + "\n".join(f"- {s}" for s in skills)
    print("\n[v3:independent] ── INPUT ───────────────────────────────",
          file=sys.stderr)
    print(f"[v3:independent] curriculum skills ({len(skills)}): "
          + ", ".join(skills), file=sys.stderr, flush=True)
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_INDEPENDENT_SCHEMA,
            messages=[{"role": "system", "content": _INDEPENDENT_SYS},
                      {"role": "user", "content": user}],
        )
        groups = json.loads(resp.message.content).get("competencies", [])
        print("[v3:independent] ── OUTPUT (raw LLM groups) ───────",
              file=sys.stderr)
        for g in groups:
            print(f"[v3:independent]   {g.get('competency', '?')}  ← "
                  + ", ".join(g.get("evidence", [])), file=sys.stderr)
        print("[v3:independent] ───────────────────────────────────",
              file=sys.stderr, flush=True)
        return groups
    except Exception as e:
        print(f"[v3:independent] LLM failed ({e})", file=sys.stderr, flush=True)
        return []


# source-words the LLM sometimes leaks into the `competency` field — not real labels
_DEGENERATE = {"", "market", "specialized", "specialised", "uncategorized",
               "uncategorised", "course", "other", "misc", "none", "n/a"}


# ── validation: snap market labels, keep evidence honest, drop nothing ─────────
def _snap_market(label: str, vocab: list[str], canon: dict[str, str],
                 vemb_box: dict) -> tuple[str, bool]:
    """P2 — resolve an LLM "market" label to a canonical concept.

    Exact (case-insensitive) match wins. Otherwise snap to the NEAREST canonical
    concept by embedding when cosine ≥ _SNAP_FLOOR (so "java programming"→"java",
    "OOP"→… only if genuinely close). Below the floor → not market (demote).
    `vemb_box` memoizes the vocab embeddings across calls within one _validate.
    """
    hit = canon.get(label.lower())
    if hit:
        return hit, True                                  # exact → canonical casing
    if not vocab:
        return label, False
    import numpy as np
    if "V" not in vemb_box:
        vemb_box["V"] = _encode(vocab)                    # encode the vocab once
    sims = _sims(vemb_box["V"], _encode([label])[0])
    i = int(np.argmax(sims))
    if sims[i] >= _SNAP_FLOOR:
        return vocab[i], True                             # snapped to a canonical concept
    return label, False                                   # genuinely off-vocab → specialized


def _reassign_leftover(leftover: list[str], comps: dict[tuple, dict],
                       source: str = "specialized") -> None:
    """P1 — place skills the LLM dropped (no junk 'uncategorized' bucket).

    Attach each to the nearest EXISTING competency when cosine ≥ _ATTACH_FLOOR;
    otherwise give it its OWN specialized group named after the skill itself.
    """
    import numpy as np
    items = list(comps.values())                          # snapshot (existing groups)
    L = _encode([c["competency"] for c in items])
    Q = _encode(leftover)
    sims = _sims(Q, L.T)                                  # (n_leftover, n_existing)
    for i, skill in enumerate(leftover):
        j = int(np.argmax(sims[i]))
        if float(sims[i][j]) >= _ATTACH_FLOOR:
            items[j]["evidence"].append(skill)            # fold into nearest competency
        else:
            key = (source, skill.lower())                 # honest, self-named group
            comps.setdefault(key, {"competency": skill, "source": source,
                                   "evidence": []})["evidence"].append(skill)


def _validate(raw: list[dict], skills: list[str], vocab: list[str],
              course_title: str = "") -> list[dict]:
    canon = {v.lower(): v for v in vocab}                 # lower → canonical casing
    by_norm = {s.lower().strip(): s for s in skills}      # input skills (orig casing)
    assigned: set[str] = set()
    comps: dict[tuple, dict] = {}                         # (source,label) → competency
    vemb_box: dict = {}                                   # lazy vocab-embedding memo

    for c in raw:
        label = str(c.get("competency", "")).strip()
        source = c.get("source", "specialized")
        if not label:
            continue
        if source == "market":                            # P2: exact, else embed-snap
            label, is_market = _snap_market(label, vocab, canon, vemb_box)
            source = "market" if is_market else "specialized"
        ev = []
        for e in c.get("evidence", []):
            o = by_norm.get(str(e).lower().strip())
            if o and o not in assigned:
                ev.append(o)
                assigned.add(o)
        if not ev:
            continue
        if label.lower() in _DEGENERATE:                  # source word leaked as label →
            label = ev[0]                                 # name after evidence; keep distinct
        key = (source, label.lower())
        if key in comps:
            comps[key]["evidence"].extend(ev)             # merge duplicate labels
        else:
            comps[key] = {"competency": label, "source": source, "evidence": ev}

    # P1 — nothing dropped: reassign unassigned skills (LLM omissions)
    leftover = [s for s in skills if s not in assigned]
    if leftover:
        if comps:
            _reassign_leftover(leftover, comps)           # embed-attach or self-name
        else:                                             # total LLM failure → safe fallback
            label = course_title.strip() or "uncategorized"  # name the blob after the course
            comps[("specialized", label.lower())] = {
                "competency": label, "source": "specialized",
                "evidence": leftover}

    out = list(comps.values())
    out.sort(key=lambda c: (c["source"] != "market", -len(c["evidence"])))
    return out


def _validate_independent(raw: list[dict], skills: list[str],
                          course_title: str = "") -> list[dict]:
    by_norm = {s.lower().strip(): s for s in skills}
    assigned: set[str] = set()
    comps: dict[tuple, dict] = {}

    for c in raw:
        label = str(c.get("competency", "")).strip()
        if not label:
            continue
        ev = []
        for e in c.get("evidence", []):
            o = by_norm.get(str(e).lower().strip())
            if o and o not in assigned:
                ev.append(o)
                assigned.add(o)
        if not ev:
            continue
        if label.lower() in _DEGENERATE:
            label = ev[0]
        key = ("course", label.lower())
        if key in comps:
            comps[key]["evidence"].extend(ev)
        else:
            comps[key] = {"competency": label, "source": "course", "evidence": ev}

    leftover = [s for s in skills if s not in assigned]
    if leftover:
        if comps:
            _reassign_leftover(leftover, comps, source="course")
        else:
            label = course_title.strip() or "uncategorized"
            comps[("course", label.lower())] = {
                "competency": label, "source": "course", "evidence": leftover}

    out = list(comps.values())
    out.sort(key=lambda c: -len(c["evidence"]))
    return out


# ── cache ──────────────────────────────────────────────────────────────────────
def _key(skills: list[str], domains: list[str], title: str = "") -> str:
    payload = json.dumps([sorted(s.lower().strip() for s in skills),
                          sorted(domains), title.strip().lower()], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def consolidate(course_skills: list[str], domains: list[str],
                course_code: str = "", course_title: str = "",
                force: bool = False) -> dict:
    """Classify a course's raw skills into market + specialized competencies.

    Returns {competencies[], market[], specialized[], domains, course_code,
    n_input, cached}. Cached per (skills, domains, title) hash — runs the LLM once
    per syllabus. `course_code` is stored for readability only; `course_title`
    names the fallback blob if the LLM grouping fails (else "uncategorized").
    """
    skills = clean.clean_skills(course_skills)       # stage ① CLEAN
    domains = [d for d in domains if d]
    key = _key(skills, domains, course_title)

    cache = _load_cache()
    if not force and key in cache:
        return {**cache[key], "cached": True}

    vocab = _vocab_for(domains)
    raw = _llm_consolidate(skills, vocab) if (skills and vocab) else []
    comps = _validate(raw, skills, vocab, course_title)
    if not comps and skills:                              # LLM down → safe fallback
        label = course_title.strip() or "uncategorized"
        comps = [{"competency": label, "source": "specialized",
                  "evidence": skills}]

    result = {
        "competencies": comps,
        "market": [c["competency"] for c in comps if c["source"] == "market"],
        "specialized": [c["competency"] for c in comps if c["source"] == "specialized"],
        "domains": domains,
        "course_code": course_code,
        "n_input": len(skills),
    }
    cache[key] = result
    _CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "cached": False}


def consolidate_independent(course_skills: list[str], course_code: str = "",
                            course_title: str = "", force: bool = False) -> dict:
    """Group a course's raw skills independently from job-posting vocabulary.

    Returns course-side competencies only. Labour-market data is intentionally not
    used in this step; it is reserved for the later comparison/scoring step.
    """
    skills = clean.clean_skills(course_skills)       # stage ① CLEAN
    key = _key(skills, ["__independent_course_grouping__"], course_title)

    cache = _load_cache()
    if not force and key in cache:
        return {**cache[key], "cached": True}

    raw = _llm_independent_consolidate(skills) if skills else []
    comps = _validate_independent(raw, skills, course_title)
    if not comps and skills:
        label = course_title.strip() or "uncategorized"
        comps = [{"competency": label, "source": "course", "evidence": skills}]

    result = {
        "competencies": comps,
        "course": [c["competency"] for c in comps],
        "market": [],
        "specialized": [],
        "domains": [],
        "course_code": course_code,
        "n_input": len(skills),
        "method": "independent_course_grouping",
    }
    cache[key] = result
    _CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "cached": False}


if __name__ == "__main__":
    ai = ["Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
          "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
          "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
          "Gradient Boosting", "Data Exploration and Analysis (EDA)",
          "AI Application Development", "Algorithm Evaluation", "Data Preparation",
          "Algorithm Selection for Datasets", "AI Sub-Fields"]
    r = consolidate(ai, ["artificial_intelligence", "data_science"],
                    course_code="AIECS031", force=True)
    print(f"domains={r['domains']}  input={r['n_input']}  cached={r['cached']}")
    print(f"market={r['market']}\nspecialized={r['specialized']}\n")
    for c in r["competencies"]:
        tag = "■" if c["source"] == "market" else "○"
        print(f"{tag} {c['competency']}  ({c['source']}, {len(c['evidence'])})")
        for e in c["evidence"]:
            print(f"     ▸ {e}")
