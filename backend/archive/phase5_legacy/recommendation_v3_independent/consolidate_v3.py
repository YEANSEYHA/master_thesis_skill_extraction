"""Recommendation V3 (independent) — COURSE COMPETENCY GROUPING.

Group a course's raw extracted skills into competencies using ONLY the course's
own skills — no labour-market vocabulary is injected (that is reserved for the
later comparison step, to avoid describing the curriculum in the market's words).

    raw course skills
        │  stage ① CLEAN (encoding/whitespace/casing/dedup)
        ▼
    ONE qwen3 call (temp 0): group skills into course competencies
        ▼
    validate (evidence ∈ input; one group per skill; nothing dropped)
        ▼
    competencies[]  { competency, source: course, evidence:[raw…] }
        → cache: (skills, title) → .competency_cache.json

Self-contained: qwen3 via Ollama for grouping; MiniLM only to repair LLM
omissions (attach dropped skills), never to group. On LLM failure the input is
returned as a single course-named group (safe degradation).

Run:  python consolidate_v3.py
"""

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".competency_cache.json"

sys.path.insert(0, str(_HERE))
import clean  # noqa: E402  (stage ① CLEAN, sibling module)

_LLM = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")
_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")
_ATTACH_FLOOR = 0.62   # attach a dropped skill to an existing competency, else own group


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
    """Cosine similarities (mat is row-normalized, vec normalized). errstate
    silences the spurious Apple-Accelerate matmul FP flags (values are finite)."""
    import numpy as np
    with np.errstate(all="ignore"):
        return mat @ vec


# ── LLM grouping (no market vocabulary) ─────────────────────────────────────────
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

This must work for ANY discipline — computing, engineering (civil, electrical,
chemical, mechanical), the sciences, mathematics, health, business, law, the
arts. The examples below show the LOGIC, not the subject area. Use the course's
own discipline; never force a course into computing terms.

IMPORTANT
- Use ONLY the course skills provided by the user.
- Do NOT use job-posting data, market demand, or industry popularity to choose
  labels. Group by the concept TAUGHT in the course, not by whether the concept
  is demanded in the labour market.

RULES
- The "competency" label is a short subject/skill concept taught by the course.
- Every input skill must appear as evidence under EXACTLY ONE competency.
- Evidence must be the ORIGINAL skill strings, verbatim. Do not invent skills.
- Split a large subject into meaningful sub-competencies WHEN the skills support
  it; keep closely related techniques together. Do not over-split a narrow course.

EXAMPLE (computing):
  Linear Regression, Decision Trees, KNN        -> "Supervised Learning"
  K-Means, Gaussian Mixture                     -> "Unsupervised Learning"
EXAMPLE (mathematics — same logic, different field):
  Differentiation, Maclaurin Series             -> "Calculus"
  De Moivre's Theorem, Loci of Complex Numbers  -> "Complex Numbers"
EXAMPLE (civil engineering):
  Bending Moment, Shear Force                   -> "Structural Analysis"

Return JSON: {"competencies":[{"competency","evidence":[...]}, ...]}."""


def _llm_independent_consolidate(skills: list[str]) -> list[dict]:
    """qwen3 course-only grouping. No market vocabulary is provided."""
    user = "Course skills to group:\n" + "\n".join(f"- {s}" for s in skills)
    print("\n========== PROMPT → LLM ==========", file=sys.stderr)
    print("[SYSTEM]\n" + _INDEPENDENT_SYS, file=sys.stderr)
    print("\n[USER]\n" + user, file=sys.stderr, flush=True)
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_INDEPENDENT_SCHEMA,
            messages=[{"role": "system", "content": _INDEPENDENT_SYS},
                      {"role": "user", "content": user}],
        )
        content = resp.message.content
        print("\n========== RAW RESULT ← LLM ==========", file=sys.stderr)
        print(content, file=sys.stderr, flush=True)
        return json.loads(content).get("competencies", [])
    except Exception as e:
        print(f"[v3:independent] LLM failed ({e})", file=sys.stderr, flush=True)
        return []


# source-words the LLM sometimes leaks into the `competency` field — not real labels
_DEGENERATE = {"", "market", "specialized", "specialised", "uncategorized",
               "uncategorised", "course", "other", "misc", "none", "n/a"}


def _reassign_leftover(leftover: list[str], comps: dict[tuple, dict],
                       source: str = "course") -> None:
    """Place skills the LLM dropped (no junk 'uncategorized' bucket).

    Attach each to the nearest EXISTING competency when cosine ≥ _ATTACH_FLOOR;
    otherwise give it its OWN group named after the skill itself.
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
def _key(skills: list[str], title: str = "") -> str:
    payload = json.dumps([sorted(s.lower().strip() for s in skills),
                          title.strip().lower()], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def consolidate_independent(course_skills: list[str], course_code: str = "",
                            course_title: str = "", force: bool = False) -> dict:
    """Group a course's raw skills independently from job-posting vocabulary.

    Returns course-side competencies only. Labour-market data is intentionally not
    used in this step; it is reserved for the later comparison/scoring step.
    """
    skills = clean.clean_skills(course_skills)       # stage ① CLEAN
    key = _key(skills, course_title)

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
    consolidate_independent(ai, course_code="AIECS031", force=True)
