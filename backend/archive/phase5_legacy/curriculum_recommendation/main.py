"""Curriculum Recommendation pipeline — single self-contained module.

See propose_pipeline.md. All stages live here:

  1. LLM Skill Extraction          — get_market_skills() (LinkedIn) + get_course_skills() (PDF)
  2. Semantic Skill Matching       — match_skills()
  3. Importance-Weighted Coverage  — compute_coverage()   Importance = Demand × Specificity
  4. Recommendation label          — _label()             MAINTAIN / REVISE / UPDATE
  5. Explainability                — _explain()            strengths / critical gaps / additions

Entry points:
  recommend(pdf_source, …)        full pipeline from a syllabus PDF (Steps 1→5)
  recommend_from_skills(skills, …) Steps 2→5 when course skills already extracted
                                   (the UI persists Step-1 skills, so it can skip re-parsing)

Heavy deps (ollama, pymupdf4llm, sentence-transformers) load lazily on first use.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PHASE5 = _HERE.parent
_LINKEDIN_INV = _PHASE5 / "clo_skill_alignment" / "linkedin_inventory.json"
_PARSE_PDF = _PHASE5 / "baseline_analyze" / "parse_pdf" / "main.py"
_EMBEDDER = _PHASE5 / "baseline_analyze" / "embedder.py"
_CATEGORIZED = _PHASE5 / "linkedin_job_posting" / "job_postings_categorized.jsonl"

# Field = a sub-domain of the labour market, defined by job-TITLE patterns.
# Aggregating the skills of postings whose title matches gives a demand table
# scoped to that field (e.g. AI), instead of the whole coarse `it` category.
# Title patterns that select field postings at ingest. Kept tight on purpose:
# bare "research scientist"/"applied scientist" titles also match bio/physics/
# chemistry labs, leaking molecular-biology & physics demand into the AI core —
# so they are intentionally EXCLUDED. AI research postings still match via the
# specific role keywords below (ml/ai/data scientist, etc.).
_FIELD_PATTERNS = {
    "ai": r"\b(machine learning|deep learning|artificial intelligence|"
          r"data scientist|data science|ml engineer|ai engineer|ml/ai|ai/ml|"
          r"nlp|computer vision|mlops|ml ops)\b",
}

# ── tunables ───────────────────────────────────────────────────────────────────
_DEFAULT_TOP_N = 2000     # market vocab depth (domain-scoped)
_DEFAULT_TOP_K = 50       # # of CORE (most in-demand) field skills the audit scores against
_DEFAULT_TAU = 0.70       # cosine match threshold
TAU_HIGH = 0.70           # completeness ≥ → MAINTAIN
TAU_LOW = 0.40            # completeness ≥ → REVISE   (< → UPDATE)


# ════════════════════════════════════════════════════════════════════════════
#  lazy loaders
# ════════════════════════════════════════════════════════════════════════════

_inv_cache: dict | None = None
_pdf_to_skills_fn = None
_Embedder = None


def _load_inventory() -> dict:
    """Lazy-load the trimmed LinkedIn inventory (~200KB, cached)."""
    global _inv_cache
    if _inv_cache is None:
        if not _LINKEDIN_INV.exists():
            raise FileNotFoundError(
                f"LinkedIn inventory not found at {_LINKEDIN_INV} — run "
                "build_linkedin_inventory.py + trim_inventory.py first."
            )
        _inv_cache = json.loads(_LINKEDIN_INV.read_text(encoding="utf-8"))
    return _inv_cache


def _parse_pdf_mod():
    """Load the existing parse_pdf module by path (pulls ollama + pymupdf4llm
    only when first called). Exposes pdf_to_skills() and the schema-locked
    qwen3 helper _chat() that we reuse for the field classifier."""
    global _pdf_to_skills_fn
    if _pdf_to_skills_fn is None:
        spec = _ilu.spec_from_file_location("curric_parse_pdf", _PARSE_PDF)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pdf_to_skills_fn = mod
    return _pdf_to_skills_fn


def _embedder():
    """Load the SBERT embedder by path under a unique module name (so it does
    not collide with the other phase-5 packages' `embedder` modules)."""
    global _Embedder
    if _Embedder is None:
        spec = _ilu.spec_from_file_location("curric_embedder", _EMBEDDER)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _Embedder = mod.Embedder
    return _Embedder()


# ════════════════════════════════════════════════════════════════════════════
#  Stage 1 — LLM Skill Extraction
# ════════════════════════════════════════════════════════════════════════════

def available_categories() -> list[str]:
    """LinkedIn job categories present in the inventory."""
    return list(_load_inventory()["categories"].keys())


def get_market_skills(category: str = "it", top_n: int | None = None,
                      min_freq: int = 1) -> list[dict]:
    """Input 1: labour-market skills for one job category (from LinkedIn).

    Returns [{skill, demand, specificity}, …] sorted by demand desc.
      demand      = # postings in the category asking for the skill
      specificity = cross-category IDF (distinctiveness; 1.0 if absent)
    """
    inv = _load_inventory()
    cats = inv["categories"]
    if category not in cats:
        raise ValueError(f"unknown category {category!r}; choose from {list(cats)}")
    idf = inv.get("idf", {})
    rows = [
        {"skill": s, "demand": int(f), "specificity": float(idf.get(s, 1.0))}
        for s, f in cats[category].items() if int(f) >= min_freq
    ]
    rows.sort(key=lambda r: r["demand"], reverse=True)
    return rows[:top_n] if top_n else rows


def get_course_skills(pdf_source) -> dict:
    """Input 2: course skills from a syllabus PDF (path or upload bytes).

    Returns {course:{code,title,…}, skills:[{name,source,evidence}, …]}.
    """
    return _parse_pdf_mod().pdf_to_skills(pdf_source)


def course_skill_names(course: dict) -> list[str]:
    """Flatten get_course_skills() output → skill-name strings."""
    return [s["name"] for s in course.get("skills", [])]


# ── Field-scoped market (e.g. AI) — the FAIR comparison target ─────────────────
#
# The `it` category is too broad (java, aws, kubernetes, project management…). To
# audit an AI curriculum we want only AI-field demand: aggregate the skills of
# postings whose TITLE is an AI/ML/Data role. Built once → cached JSON.

_field_cache: dict[str, dict] = {}

# Skills excluded from a field's market vocab (denominator AND missing-skill
# suggestions). Two groups:
#   • soft skills — real demand but not what defines an AI curriculum's core;
#     including them both pads the denominator and yields useless "add
#     communication" advice.
#   • off-domain — leak in via broad research-role titles (bio/physics labs).
#     Fix A removes most at ingest; this is the belt-and-suspenders skill-level
#     guard (and also cleans the already-built cache without a rebuild).
_SOFT_SKILLS = {
    "communication", "communication skills", "collaboration", "teamwork",
    "leadership", "project management", "problem solving", "problemsolving",
    "research", "presentation", "presentation skills", "interpersonal skills",
    "time management", "critical thinking", "adaptability", "creativity",
    "attention to detail", "organization", "work ethic", "multitasking",
}
_OFF_DOMAIN_SKILLS = {
    "molecular biology", "physics", "chemistry", "biology", "genetics",
    "biochemistry", "cell biology", "neuroscience", "clinical research",
    "pharmacology", "microbiology", "organic chemistry", "biotechnology",
}
_FIELD_EXCLUDE = _SOFT_SKILLS | _OFF_DOMAIN_SKILLS


def _role_regex(role_patterns: list[str]):
    """Compile a word-boundary, case-insensitive title regex from role keywords."""
    import re
    alts = "|".join(re.escape(p) for p in role_patterns if p)
    return re.compile(rf"\b({alts})\b", re.I)


def _resolve_patterns(field: str, role_patterns: list[str] | None) -> list[str]:
    """role_patterns wins (from classify_field); else fall back to the legacy
    hardcoded _FIELD_PATTERNS regex split into its alternations."""
    import re
    if role_patterns:
        return [p.strip().lower() for p in role_patterns if p.strip()]
    if field in _FIELD_PATTERNS:
        inner = re.sub(r"^\\b\(|\)\\b$", "", _FIELD_PATTERNS[field])
        return [p for p in inner.split("|") if p]
    raise ValueError(
        f"no role_patterns for field {field!r} and no legacy pattern — "
        "pass role_patterns (e.g. from classify_field)."
    )


def build_field_inventory(field: str, role_patterns: list[str] | None = None,
                          min_freq: int = 3) -> dict:
    """Stream the categorized LinkedIn corpus, keep postings whose TITLE matches
    the field's role patterns, aggregate their skills → {skill: demand}. Cached
    to <field>_field_inventory.json next to this module.

    role_patterns: lowercase job-title keywords (from classify_field). If omitted,
    falls back to the legacy hardcoded _FIELD_PATTERNS for that field.
    """
    from collections import Counter

    if not _CATEGORIZED.exists():
        raise FileNotFoundError(f"corpus not found at {_CATEGORIZED}")
    patterns = _resolve_patterns(field, role_patterns)
    pat = _role_regex(patterns)

    counts: Counter = Counter()
    n_post = 0
    with open(_CATEGORIZED, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if pat.search(rec.get("job_title", "") or ""):
                n_post += 1
                for s in {x.strip().lower() for x in rec.get("skills", []) if x.strip()}:
                    counts[s] += 1

    skills = {s: c for s, c in counts.items() if c >= min_freq}
    out = {"field": field, "n_postings": n_post, "skills": skills,
           "meta": {"source": _CATEGORIZED.name, "min_freq": min_freq,
                    "role_patterns": patterns}}
    (_HERE / f"{field}_field_inventory.json").write_text(
        json.dumps(out), encoding="utf-8")
    return out


def get_field_skills(field: str = "ai", role_patterns: list[str] | None = None,
                     top_n: int | None = None, min_freq: int = 1) -> list[dict]:
    """Field-scoped market demand: [{skill, demand, specificity}] sorted by demand.

    Cached per field name (<field>_field_inventory.json). On a cache miss it
    builds from `role_patterns` (from classify_field) or the legacy pattern.
    `specificity` reuses the cross-category IDF (down-weights generic soft skills).
    """
    if field not in _field_cache:
        path = _HERE / f"{field}_field_inventory.json"
        if not path.exists():
            build_field_inventory(field, role_patterns=role_patterns)
        _field_cache[field] = json.loads(path.read_text(encoding="utf-8"))
    idf = _load_inventory().get("idf", {})
    rows = [
        {"skill": s, "demand": int(c), "specificity": float(idf.get(s, 1.0))}
        for s, c in _field_cache[field]["skills"].items()
        if int(c) >= min_freq and s not in _FIELD_EXCLUDE
    ]
    rows.sort(key=lambda r: r["demand"], reverse=True)
    return rows[:top_n] if top_n else rows


# ── Dynamic scope: classify the course → labour-market field (1 qwen3 call) ────
#
# Replaces the manual "AI field / Whole IT" toggle and the hardcoded field key.
# Judges by the course CONTENT (skills/topics), NOT the title — so a course
# titled "Business Information Technology" whose content is calculus is correctly
# flagged FOUNDATIONAL, not scored against IT jobs.

_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "field_name": {"type": "string"},
        "is_foundational": {"type": "boolean"},
        "role_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["field", "field_name", "is_foundational", "role_patterns"],
}

_FIELD_SYS = """\
You classify a university course into the LABOUR-MARKET FIELD it prepares students \
for, so a curriculum-vs-jobs comparison can be scoped to the right postings.

Judge by the COURSE CONTENT (the skills/topics taught), NOT the title — titles can \
be misleading (e.g. a course titled "Business Information Technology" that actually \
teaches calculus is a mathematics course).

Return JSON with:
- field: a short lowercase key for the job field, e.g. "ai", "cybersecurity",
  "software_engineering", "data_engineering", "networking", "web", "devops".
- field_name: a short human-readable name.
- is_foundational: true if the course is foundational/theoretical with NO direct
  labour-market job role (e.g. automata theory, formal languages, pure mathematics,
  calculus, discrete math). The market has no role that hires for these directly.
- role_patterns: lowercase job-title keywords/phrases used to find postings in this
  field, e.g. AI → ["machine learning engineer","data scientist","ai engineer","ml engineer"].
  Use an empty list when is_foundational is true.

Return ONLY the JSON object."""


def classify_field(title: str, skills: list[str], description: str = "") -> dict:
    """qwen3: course → {field, field_name, is_foundational, role_patterns}.

    One schema-locked LLM call. `skills` (the extracted course skills) is the
    strongest signal; title/description are supporting context.
    """
    import json as _json
    user = (f"Course title: {title}\n"
            + (f"Description: {description}\n" if description else "")
            + f"Skills taught: {', '.join(skills)}")
    out = _parse_pdf_mod()._chat("classify_field", _FIELD_SCHEMA, _FIELD_SYS, user)
    data = _json.loads(out)
    data["field"] = (data.get("field") or "").strip().lower().replace(" ", "_")
    data["role_patterns"] = [str(p).strip().lower() for p in data.get("role_patterns", []) if str(p).strip()]
    data["is_foundational"] = bool(data.get("is_foundational"))
    return data


# ── V2: 3-way course gate (FOUNDATIONAL / APPLIED / PROFESSIONAL) ─────────────
#
# Method v2 (propose_pipeline_v2.md §4). One schema-locked qwen3 call classifies
# the course by CONTENT into the type that picks its scoring denominator:
#   FOUNDATIONAL  → gated out, never scored ("not market-facing by design")
#   APPLIED       → scored against the applied-tier core only
#   PROFESSIONAL  → scored against the full core (incl. industry tooling)
# V1's classify_field() is untouched — this is the V2 path.

_COURSE_TYPES = ("foundational", "applied", "professional")

_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "course_type": {"type": "string", "enum": list(_COURSE_TYPES)},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "field": {"type": "string"},
        "field_name": {"type": "string"},
        "role_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["course_type", "confidence", "reason",
                 "field", "field_name", "role_patterns"],
}

_GATE_SYS = """\
You classify a university course for a curriculum-vs-job-market audit. Judge by the
COURSE CONTENT (the skills/topics taught), NOT the title — titles can mislead (a course
titled "Business Information Technology" that teaches calculus is a mathematics course).

Classify into exactly one course_type:

- "foundational": pure theory/formalisms/mathematics with NO direct labour-market
  vocabulary — no job posting lists these as skills. Examples: automata theory, formal
  languages, calculus, discrete mathematics, linear algebra, pure physics.
  TEST: would any job posting list the taught skills verbatim? If no → foundational.

- "applied": teaches transferable technical concepts/techniques with hands-on work,
  intro-to-intermediate level. The market posts the CONCEPTS (machine learning, sql,
  networking) even if the course names no industry tooling. Examples: Introduction to AI,
  Database Systems, Programming, Computer Networks, Web Development basics.

- "professional": teaches industry practice — frameworks, infrastructure, operations,
  workflows; terminal/specialization/capstone courses closest to a job role. Examples:
  MLOps, Cloud Computing with AWS, DevOps, Capstone Project, Cybersecurity Operations.

Edge rules:
- A course teaching theory THROUGH market-posted tools/techniques is "applied", not
  "foundational" (e.g. "Statistics for Data Science" using python/pandas).
- If the syllabus is thin or ambiguous, prefer "applied" with low confidence — NEVER
  default to "foundational" (wrongly gating a course out of the audit is the worse error).

Also return the labour-market field this course belongs to:
- field: short lowercase key, e.g. "ai", "cybersecurity", "software_engineering",
  "data_engineering", "networking", "web", "devops". For foundational courses use the
  closest discipline key anyway (e.g. "computer_science").
- field_name: short human-readable name.
- role_patterns: lowercase job-title keywords used to find this field's postings,
  e.g. ai → ["machine learning engineer","data scientist","ai engineer","ml engineer"].
  Use an empty list when course_type is "foundational".

confidence: 0.0-1.0. reason: ONE short sentence citing the content that decided it.
Return ONLY the JSON object."""


def classify_course_v2(title: str, skills: list[str], description: str = "") -> dict:
    """V2 gate: course → {course_type, confidence, reason, field, field_name,
    role_patterns}. One schema-locked qwen3 call; judged by content, not title.
    `is_foundational` is derived for compatibility with the V1 response shape."""
    import json as _json
    user = (f"Course title: {title}\n"
            + (f"Description: {description}\n" if description else "")
            + f"Skills taught: {', '.join(skills)}")
    out = _parse_pdf_mod()._chat("classify_course_v2", _GATE_SCHEMA, _GATE_SYS, user)
    data = _json.loads(out)
    ct = str(data.get("course_type", "")).strip().lower()
    if ct not in _COURSE_TYPES:
        ct = "applied"                       # never gate out on a malformed answer
        data["confidence"] = 0.0
        data["reason"] = (data.get("reason") or "") + " [malformed course_type → default applied]"
    data["course_type"] = ct
    data["is_foundational"] = ct == "foundational"
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    data["field"] = (data.get("field") or "").strip().lower().replace(" ", "_")
    data["role_patterns"] = [str(p).strip().lower()
                             for p in data.get("role_patterns", []) if str(p).strip()]
    return data


# ── V2: tier-tag the field core (applied vs professional) ─────────────────────
#
# Method v2 (propose_pipeline_v2.md §5). The market core is divided ONCE per field
# into two tiers; the course's type then selects which slice is the scoring
# denominator. Built by one batched qwen3 call, cached to {field}_field_levels.json
# (plain JSON → human-correctable, printable in a thesis appendix).

_TIER_RANK = {"applied": 1, "professional": 2}

_LEVELS_SCHEMA = {
    "type": "object",
    "properties": {
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "tier": {"type": "string", "enum": ["applied", "professional"]},
                },
                "required": ["skill", "tier"],
            },
        }
    },
    "required": ["levels"],
}

_LEVELS_SYS = """\
You tag labour-market skills for a curriculum audit. For EACH skill, decide the LOWEST
university course type that should be expected to teach it:

- "applied": a concept, fundamental technique, or general-purpose language taught in
  regular technical courses. Examples: machine learning, statistics, python, sql, r,
  data analysis, algorithms, networking, deep learning (the concept), natural language
  processing (the concept), computer vision (the concept), java.

- "professional": industry tooling, specific frameworks, cloud platforms, infrastructure,
  or operations practice that belongs in final-year / specialization / capstone courses.
  Examples: pytorch, tensorflow, aws, azure, gcp, docker, kubernetes, spark, hadoop,
  scala, ci/cd, terraform, mlops, data engineering, cloud computing.

Rule of thumb: CONCEPTS and general-purpose languages → "applied";
TOOLS / PLATFORMS / INFRASTRUCTURE / OPS practice → "professional".

Echo each skill VERBATIM. Return one entry per input skill.
Return ONLY JSON: {"levels":[{"skill","tier"}, ...]}."""


def get_field_levels(field: str, skills: list[str]) -> dict[str, str]:
    """Tier tags for a field's core skills: {skill: "applied"|"professional"}.

    Cached to {field}_field_levels.json next to this module. On a cache miss (or
    for skills not yet in the cache) ONE batched qwen3 call tags them and the
    cache is extended. Skills the LLM fails to tag default to "applied" — the
    conservative choice: they stay in every denominator, so a tagging failure can
    only lower a score, never inflate it.
    """
    import json as _json
    path = _HERE / f"{field}_field_levels.json"
    levels: dict[str, str] = {}
    if path.exists():
        try:
            levels = {str(k).lower(): str(v).lower()
                      for k, v in _json.loads(path.read_text(encoding="utf-8")).items()}
        except Exception:
            levels = {}

    todo = [s for s in skills if s.lower() not in levels]
    if todo:
        user = "Skills to tag:\n" + "\n".join(f"- {s}" for s in todo)
        tagged: dict[str, str] = {}
        try:
            out = _parse_pdf_mod()._chat("field_levels", _LEVELS_SCHEMA, _LEVELS_SYS, user)
            for row in _json.loads(out).get("levels", []):
                k = str(row.get("skill", "")).strip().lower()
                v = str(row.get("tier", "")).strip().lower()
                if k and v in _TIER_RANK:
                    tagged[k] = v
        except Exception as e:
            print(f"[field_levels] tagging failed ({e}) — defaulting to 'applied'")
        for s in todo:
            levels[s.lower()] = tagged.get(s.lower(), "applied")
        path.write_text(_json.dumps(levels, indent=1, sort_keys=True), encoding="utf-8")

    return {s: levels.get(s.lower(), "applied") for s in skills}


# ════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Semantic Skill Matching
#
#  Embed once, read the cosine matrix BOTH ways:
#    • course→job — each TAUGHT skill's nearest market skill (→ RELEVANCE:
#      "is what the course teaches in demand?"). Secondary.
#    • job→course — each MARKET skill's nearest taught skill + covered flag
#      (→ COMPLETENESS over the core field: "does it cover the in-demand skills
#      the field actually wants?"). This is the FAIR audit metric.
# ════════════════════════════════════════════════════════════════════════════

def importance(row: dict) -> float:
    """Importance = demand × specificity (missing fields default to neutral)."""
    return float(row.get("demand") or 0) * float(row.get("specificity") or 1.0)


def match_skills(course_skills: list[str], market_skills: list[dict],
                 threshold: float = _DEFAULT_TAU) -> dict:
    """Returns {course, market, metadata}.

      course[]: per TAUGHT skill — nearest market skill, score, in_demand (relevance).
      market[]: per MARKET skill — nearest taught skill, score, covered flag,
                evidence (the taught skill), demand, specificity (completeness).
    """
    market_names = [m["skill"] for m in market_skills]
    if not course_skills or not market_names:
        return {"course": [], "market": [],
                "metadata": {"n_course": len(course_skills),
                             "n_market": len(market_names), "threshold": threshold}}

    emb = _embedder()
    course_vecs = emb.embed(course_skills)        # (C, D) L2-normalized
    market_vecs = emb.embed(market_names)         # (M, D) L2-normalized
    sim = course_vecs @ market_vecs.T             # (C, M) cosine

    # course→job: each taught skill's nearest market skill = its market value
    c_best = sim.max(axis=1)
    c_idx = sim.argmax(axis=1)
    course = []
    for i, cs in enumerate(course_skills):
        m = market_skills[int(c_idx[i])]
        course.append({"skill": cs, "best_job_skill": m["skill"],
                       "score": round(float(c_best[i]), 3),
                       "in_demand": bool(c_best[i] >= threshold),
                       "demand": m.get("demand"), "specificity": m.get("specificity")})

    # job→course: for EACH market skill, is it covered by some taught skill?
    m_best = sim.max(axis=0)
    m_idx = sim.argmax(axis=0)
    market = [
        {"skill": m["skill"], "score": round(float(m_best[j]), 3),
         "covered": bool(m_best[j] >= threshold),
         "evidence": course_skills[int(m_idx[j])],
         "demand": m.get("demand"), "specificity": m.get("specificity")}
        for j, m in enumerate(market_skills)
    ]
    course.sort(key=lambda r: (r["in_demand"], importance(r)), reverse=True)
    return {"course": course, "market": market,
            "metadata": {"n_course": len(course_skills),
                         "n_market": len(market_names), "threshold": threshold}}


# ════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Field Completeness (the fair metric)  +  Stage 4 — label
#
#  PRIMARY  Completeness = Σ Importance(core field skills COVERED)
#                          / Σ Importance(top-K core field skills)
#    "Of the field's K most in-demand skills, how many does the course teach?"
#    Scoped to the field AND to the core skills → neither the all-IT lowball
#    nor the "100% because narrow" leniency.
#  SECONDARY Relevance = Σ Importance(taught skills in demand) / Σ Importance(taught)
#    "Is what the course teaches in demand?" (no-obsolete-content check)
# ════════════════════════════════════════════════════════════════════════════

def _label(coverage: float) -> str:
    if coverage >= TAU_HIGH:
        return "MAINTAIN"
    if coverage >= TAU_LOW:
        return "REVISE"
    return "UPDATE"


def _core(market: list[dict], top_k: int) -> list[dict]:
    """The top-K most-important (demand × specificity) market skills = the field core."""
    return sorted(market, key=importance, reverse=True)[:top_k]


def compute_coverage(match_result: dict, top_k: int = _DEFAULT_TOP_K) -> dict:
    """Field completeness over the top-K core skills → MAINTAIN/REVISE/UPDATE,
    plus the secondary relevance figure."""
    market = match_result.get("market", [])
    course = match_result.get("course", [])

    core = _core(market, top_k)
    imp_cov = sum(importance(m) for m in core if m["covered"])
    imp_core = sum(importance(m) for m in core)
    completeness = (imp_cov / imp_core) if imp_core else 0.0
    n_core_cov = sum(1 for m in core if m["covered"])

    rel_in = sum(importance(r) for r in course if r.get("in_demand"))
    rel_all = sum(importance(r) for r in course)
    relevance = (rel_in / rel_all) if rel_all else 0.0

    return {"coverage": round(completeness, 4), "coverage_pct": round(100 * completeness, 1),
            "label": _label(completeness),
            "completeness_pct": round(100 * completeness, 1),
            "relevance_pct": round(100 * relevance, 1),
            "top_k": top_k, "n_core_covered": n_core_cov, "n_core": len(core)}


# ════════════════════════════════════════════════════════════════════════════
#  Stage 5 — Explainability
# ════════════════════════════════════════════════════════════════════════════

def _explain(match_result: dict, cov: dict) -> dict:
    """Within the top-K core field skills:
       strengths     = core skills the course COVERS (with the taught-skill evidence),
       critical_gaps = core skills it MISSES (the in-demand skills to add)."""
    core = _core(match_result.get("market", []), cov.get("top_k", _DEFAULT_TOP_K))
    covered = [m for m in core if m["covered"]]
    missing = [m for m in core if not m["covered"]]
    strengths = [
        {"skill": m["skill"], "best_job_skill": None, "demand": m["demand"],
         "specificity": round(float(m.get("specificity") or 1.0), 3),
         "evidence": m.get("evidence"), "score": m.get("score"),
         "verified": bool(m.get("verified")), "verify_reason": m.get("verify_reason")}
        for m in covered
    ]
    critical_gaps = [
        {"skill": m["skill"], "demand": m["demand"],
         "specificity": round(float(m.get("specificity") or 1.0), 3),
         "score": m.get("score"),
         # flagged covered by the embedder but rejected by the LLM verifier
         "removed_by_verifier": bool(m.get("verified") and m.get("embed_covered")),
         "verify_reason": m.get("verify_reason")}
        for m in missing
    ]
    return {"coverage_score": cov["coverage"], "coverage_pct": cov["coverage_pct"],
            "completeness_pct": cov["completeness_pct"], "relevance_pct": cov["relevance_pct"],
            "recommendation": cov["label"], "strengths": strengths,
            "critical_gaps": critical_gaps,
            "suggested_additions": [g["skill"] for g in critical_gaps]}


# ════════════════════════════════════════════════════════════════════════════
#  Stage 5b — LLM curation of the "skills to add" list
#
#  Market-mined gaps carry LinkedIn noise that is NOT curriculum-teachable:
#  credentials ("bachelor's degree", "phd"), soft skills ("mentoring",
#  "autonomy"), off-domain terms ("economics"), and near-duplicates ("agile" /
#  "agile methodologies" / "agile development"). One qwen3 call filters + dedups
#  the list down to genuine, field-appropriate skills. It can only KEEP skills
#  already in the gap list (verbatim) — it never invents new ones, so every
#  suggestion stays grounded in real demand evidence.
# ════════════════════════════════════════════════════════════════════════════

_CURATE_SCHEMA = {
    "type": "object",
    "properties": {"keep": {"type": "array", "items": {"type": "string"}}},
    "required": ["keep"],
}

_CURATE_SYS = """\
You advise on which skills to ADD NEXT to a SPECIFIC university course. You are given:
- the FIELD, the COURSE TITLE, and the skills the course ALREADY TEACHES, and
- CANDIDATE skills mined from job postings (sorted by market demand) that the course
  does NOT yet cover.
Select the candidates that are the best, most appropriate NEXT additions for THIS
course — and order them by pedagogical priority (most natural next step first).

Judge each candidate on THREE things:
1. FIELD FIT — it belongs to this field (drop off-field, credentials like "phd"/
   "bachelor's degree", and generic soft skills like "mentoring"/"communication").
2. LEVEL FIT — it matches the course's LEVEL. Infer the level from the title and the
   taught skills:
   • An INTRODUCTORY / foundational course (title says "Introduction", teaches basics)
     should add the NEXT conceptual layer, NOT senior operations tooling. For an
     intro AI course teaching classical ML, GOOD next steps are e.g. deep learning,
     neural networks, a first DL framework, NLP basics, computer vision basics,
     mathematics/statistics. DROP senior MLOps/infra/devops that belong in later or
     specialized courses: docker, kubernetes, cloud computing, aws/azure/gcp, ci/cd,
     monitoring, deployment, data pipelines, git, agile, devops.
   • An APPLIED / advanced course MAY include tooling and infrastructure.
3. BUILDS ON WHAT'S TAUGHT — prefer skills that directly extend the current content
   (a course teaching K-means/regression naturally extends to deep learning, not to
   Kubernetes). Do NOT re-suggest anything already taught.

AVOID REDUNDANCY — the final list must not contain overlapping recommendations:
- Near-synonyms: keep ONLY ONE (e.g. "statistics" vs "statistical analysis" → keep
  "statistics"; "data analysis" vs "data analytics" → keep one).
- Concept + its tools: when you keep a CONCEPT, do not also stack multiple competing
  TOOLS for it. If you suggest "deep learning", keep AT MOST ONE framework — and for
  an INTRODUCTORY course prefer the concept alone (drop both "tensorflow" and
  "pytorch"); only an applied course should name a specific framework.
- Literal variants: "agile methodologies"→"agile"; "continuous integration"→"ci/cd".

Rules:
- Only return skills that appear VERBATIM in the candidate list. Do NOT invent new ones.
- Order by pedagogical priority for THIS course (best next step first).
- Return at most 10, and only skills you are confident are a good next step.
Return JSON: {"keep": ["skill", ...]}."""


def curate_suggestions(gaps: list[str], field_name: str = "", course_title: str = "",
                       taught_skills: list[str] | None = None,
                       max_keep: int = 10) -> list[str]:
    """qwen3 selects, from market-mined gap skills, the best NEXT additions for THIS
    SPECIFIC course — scored on field fit, LEVEL fit (intro vs applied, inferred from
    the title + taught skills), and how well they build on what's already taught.
    Returns a subset of `gaps` (verbatim casing preserved). Falls back to the raw
    list (truncated) if the LLM is unreachable or returns nothing usable."""
    import json as _json
    gaps = [g for g in gaps if g]
    if not gaps:
        return []
    taught = ", ".join(taught_skills) if taught_skills else "(not provided)"
    user = (f"Field: {field_name or 'this field'}\n"
            f"Course title: {course_title or '(untitled)'}\n"
            f"Skills the course ALREADY TEACHES: {taught}\n\n"
            f"Candidate skills to add (demand-sorted): {', '.join(gaps)}")
    try:
        out = _parse_pdf_mod()._chat("curate_suggestions", _CURATE_SCHEMA,
                                     _CURATE_SYS, user)
        keep = _json.loads(out).get("keep", [])
    except Exception:
        return gaps[:max_keep]
    valid = {g.lower(): g for g in gaps}            # case-insensitive membership
    seen, cleaned = set(), []
    for k in keep:
        key = str(k).strip().lower()
        if key in valid and key not in seen:
            seen.add(key)
            cleaned.append(valid[key])              # canonical casing from the gap list
    return (cleaned or gaps)[:max_keep]


# ════════════════════════════════════════════════════════════════════════════
#  Stage 2b — LLM match VERIFICATION  (retrieve-then-verify / neuro-symbolic)
#
#  The embedder gives RECALL: it proposes "covered" whenever a taught skill is
#  cosine-near a market skill. But cosine conflates "related topic" with "actually
#  confers the skill" — e.g. teaching *Gradient Descent* (a math concept) lights up
#  *TensorFlow* (a framework); *Pandas* lights up *Spark*. These false positives
#  inflate completeness.
#
#  This stage adds PRECISION: one batched qwen3 call adjudicates each embedder-
#  proposed match — "does teaching X genuinely confer the job skill Y?" — and flips
#  false positives to covered=False. The pre-verification flag is preserved
#  (`embed_covered`) so the cosine-only vs verified result is a clean ablation.
# ════════════════════════════════════════════════════════════════════════════

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_skill": {"type": "string"},
                    "covered": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["job_skill", "covered", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}

_VERIFY_SYS = """\
You verify curriculum-to-job skill matches. For each pair you are given a TAUGHT
skill (from a university course) and a JOB skill (in market demand) that an
embedding model flagged as similar. Decide whether a student who learned the
TAUGHT skill can legitimately claim the JOB skill (covered=true) or not (false).

The ONLY thing you are filtering out is CROSS-IDENTITY confusion — where the two
are DIFFERENT things the embedder conflated. You are NOT requiring exact equality.

covered=TRUE when the taught skill IS the job skill, or is a major
branch / specialization / direct application of it that genuinely teaches it:
    "Python"               → "python"                 ✓ same
    "Unsupervised Learning"→ "machine learning"       ✓ a core branch you learned
    "Exploratory Data Analysis" → "data analysis"     ✓ EDA is data analysis
    "Gradient Descent"     → "optimization"           ✓ it IS an optimization method
    "Convolutional Neural Networks" → "deep learning" ✓ a core deep-learning topic

covered=FALSE only when they are DIFFERENT things — a different tool/framework, a
mere data artifact, an unrelated subfield, or a pipeline/role the course does not
build:
    "Gradient Descent"  → "tensorflow"      ✗ a concept ≠ a specific framework
    "Pandas"            → "spark"           ✗ single-node lib ≠ distributed engine
    "UCI ML Repository" → "deep learning"   ✗ a dataset host ≠ the skill
    "EDA"               → "data engineering"✗ analysis ≠ building data pipelines
    "EDA"               → "data science"    ✗ EDA alone ≠ the full data-science role

Rule of thumb: if the student clearly DID acquire the job skill (or a real portion
of it), say true. Only say false when teaching the taught skill would NOT let them
do the job skill because it is a different tool, artifact, or subfield.

Return one verdict per input pair, echoing job_skill verbatim, with a SHORT reason.
Return JSON: {"verdicts":[{"job_skill","covered","reason"}, ...]}."""


def _verify_pairs(pairs: list[dict], field_name: str = "",
                  course_title: str = "") -> dict:
    """One batched qwen3 call. pairs: [{job_skill, taught_skill, score}].
    Returns {job_skill.lower(): {"covered": bool, "reason": str}}. On any failure
    returns {} (→ caller keeps the embedder flags: safe degradation)."""
    import json as _json
    if not pairs:
        return {}
    lines = "\n".join(
        f"{i+1}. taught: {p.get('taught_skill','')!r}  →  job: {p['job_skill']!r}"
        for i, p in enumerate(pairs)
    )
    user = (f"Field: {field_name or 'this field'}\n"
            f"Course: {course_title or '(untitled)'}\n\n"
            f"Verify these {len(pairs)} matches:\n{lines}")
    try:
        out = _parse_pdf_mod()._chat("verify_matches", _VERIFY_SCHEMA, _VERIFY_SYS, user)
        verdicts = _json.loads(out).get("verdicts", [])
    except Exception:
        return {}
    return {
        str(v.get("job_skill", "")).strip().lower():
            {"covered": bool(v.get("covered")), "reason": str(v.get("reason", ""))}
        for v in verdicts if v.get("job_skill")
    }


def verify_coverage(match_result: dict, top_k: int = _DEFAULT_TOP_K,
                    field_name: str = "", course_title: str = "") -> dict:
    """Adjudicate the embedder-proposed matches among the top-K core skills and
    flip false positives. Mutates match_result['market'] in place (preserving the
    original flag as `embed_covered`) and returns it."""
    core = _core(match_result.get("market", []), top_k)
    covered = [m for m in core if m.get("covered")]
    if not covered:
        return match_result
    pairs = [{"job_skill": m["skill"], "taught_skill": m.get("evidence", ""),
              "score": m.get("score")} for m in covered]
    verdicts = _verify_pairs(pairs, field_name, course_title)
    for m in covered:
        m["embed_covered"] = m["covered"]          # preserve for the ablation
        v = verdicts.get(m["skill"].lower())
        if v is not None:
            m["covered"] = v["covered"]
            m["verified"] = True
            m["verify_reason"] = v["reason"]
    return match_result


# ── V2: band-based retrieve-then-verify ───────────────────────────────────────
#
# Method v2 (propose_pipeline_v2.md §6). A single hard threshold is brittle
# (model- and vocab-specific), so the threshold only decides the CLEAR cases and
# the LLM adjudicates everything above the band floor:
#
#          0 ──────── band_low(0.55) ─────────── tau(0.70) ─────────── 1
#          auto-reject │   gray zone: LLM may    │ auto-covered: LLM may
#                      │   RESCUE true coverage  │ VETO false positives
#
# One batched qwen3 call covers both jobs (same question per pair: "does teaching
# X genuinely confer job skill Y?"). On LLM failure the embedder flags stand —
# safe degradation, identical to V1's behavior.

def verify_coverage_v2(match_result: dict, top_k: int = _DEFAULT_TOP_K,
                       band_low: float = 0.55, field_name: str = "",
                       course_title: str = "") -> dict:
    """Adjudicate every top-K core pair with cosine ≥ band_low in ONE batched call.
    Mutates match_result['market'] rows in place:
      embed_covered = the pre-verification flag (kept for the ablation)
      band          = "auto" (≥ tau) | "gray" (band_low ≤ cos < tau)
      covered       = the LLM verdict (veto or rescue), else the embedder flag
    """
    core = _core(match_result.get("market", []), top_k)
    cand = [m for m in core if float(m.get("score") or 0.0) >= band_low]
    if not cand:
        return match_result
    pairs = [{"job_skill": m["skill"], "taught_skill": m.get("evidence", ""),
              "score": m.get("score")} for m in cand]
    verdicts = _verify_pairs(pairs, field_name, course_title)
    for m in cand:
        m["embed_covered"] = m["covered"]
        m["band"] = "auto" if m["covered"] else "gray"
        v = verdicts.get(m["skill"].lower())
        if v is not None:
            m["covered"] = v["covered"]
            m["verified"] = True
            m["verify_reason"] = v["reason"]
    return match_result


# ── V2: narrow tool inference (curriculum side) ───────────────────────────────
#
# Method v2 (propose_pipeline_v2.md §6, "python false gap"). Languages/libraries a
# course actually uses are often named ONLY in the References/Tools section
# (e.g. w3schools.com/python, Pandas notebooks) — never in a CLO or lesson row —
# so LLM skill extraction misses them. Deterministic whitelist scan: languages
# and libraries ONLY (high precision, no tutorial-URL noise, no LLM). Re-enabling
# full Resource extraction was rejected as too noisy; this is the narrow fix.

# Boundaries: (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) instead of \b so that
# underscores count as separators ("Pandas_hands_on.ipynb" must match pandas).
_L, _R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
_TOOL_PATTERNS = {
    "python":       rf"{_L}python{_R}",
    "java":         rf"{_L}java{_R}(?!script)",
    "javascript":   rf"{_L}javascript{_R}",
    "c++":          r"c\+\+",
    "c#":           r"\bc#",
    "sql":          rf"{_L}sql{_R}",
    "mysql":        rf"{_L}mysql{_R}",
    "postgresql":   rf"{_L}postgres(?:ql)?{_R}",
    "mongodb":      rf"{_L}mongodb{_R}",
    "r":            rf"{_L}R\s+(?:programming|language){_R}|{_L}rstudio{_R}|{_L}cran{_R}|{_L}r-project{_R}",
    "matlab":       rf"{_L}matlab{_R}",
    "scala":        rf"{_L}scala{_R}",
    "pandas":       rf"{_L}pandas{_R}",
    "numpy":        rf"{_L}numpy{_R}",
    "scikit-learn": rf"{_L}scikit-?learn{_R}|{_L}sklearn{_R}",
    "tensorflow":   rf"{_L}tensorflow{_R}",
    "pytorch":      rf"{_L}pytorch{_R}",
    "keras":        rf"{_L}keras{_R}",
    "matplotlib":   rf"{_L}matplotlib{_R}",
    "jupyter":      rf"{_L}jupyter{_R}|{_L}colab{_R}",
    "excel":        rf"{_L}excel{_R}",
    "tableau":      rf"{_L}tableau{_R}",
    "power bi":     rf"{_L}power\s*bi{_R}",
    "docker":       rf"{_L}docker{_R}",
}

# Lines that are template boilerplate, not actual course usage — the ITC template
# illustrates fields with example lists like "(E.g: QGIS software, AutoCAD,
# Python, …)". A tool named only on such a line must NOT count as taught.
_EXAMPLE_LINE = __import__("re").compile(r"\be\.?\s?g\b|\bfor example\b|\bexample[s]?:",
                                         __import__("re").IGNORECASE)


def infer_tools(markdown: str) -> list[dict]:
    """Scan syllabus markdown for whitelisted languages/libraries.

    Returns [{name, evidence}] — evidence is the first matching line (trimmed),
    so every inferred tool is traceable in the UI. Case-insensitive except the
    R patterns ("R programming"/RStudio/CRAN only — a bare 'r' never matches).
    Matches on template-example lines ("E.g: …, Python, …") are skipped.
    """
    import re
    if not markdown:
        return []
    found = []
    for name, pat in _TOOL_PATTERNS.items():
        flags = 0 if name == "r" else re.IGNORECASE
        for mo in re.finditer(pat, markdown, flags):
            line_start = markdown.rfind("\n", 0, mo.start()) + 1
            line_end = markdown.find("\n", mo.end())
            line = markdown[line_start:line_end if line_end != -1 else None].strip()
            if _EXAMPLE_LINE.search(line):
                continue                      # boilerplate, keep looking
            found.append({"name": name, "evidence": line[:160]})
            break
    return found


# ── V2: tier-filtered scoring + two-axis label ────────────────────────────────
#
# Method v2 (propose_pipeline_v2.md §7-§8).
#   Completeness = Σ Imp(covered core, tier ≤ course tier) / Σ Imp(core, tier ≤ course tier)
#   Relevance    = Σ Imp(taught ∩ in-demand) / Σ Imp(taught)
# Core skills ABOVE the course's tier are reported as next_course_topics, never
# counted as gaps. Credentials/degree labels are excluded from the V2 denominator
# only (V1 behavior unchanged).

_CREDENTIALS = {
    "computer science", "computer engineering", "information systems",
    "bachelor's degree", "bachelor degree", "bachelors degree", "bachelor of science",
    "master's degree", "master degree", "masters degree", "master of science",
    "phd", "ph.d.", "doctorate", "mba", "bsc", "msc", "degree",
}


def _label_v2(completeness: float, relevance: float) -> str:
    """Two-axis decision (quadrant): completeness drives MAINTAIN/REVISE; below
    TAU_LOW, relevance separates FOCUSED (narrow but in-demand) from UPDATE
    (teaches what the market doesn't ask for — the real red flag)."""
    if completeness >= TAU_HIGH:
        return "MAINTAIN"
    if completeness >= TAU_LOW:
        return "REVISE"
    return "FOCUSED" if relevance >= 0.70 else "UPDATE"


def compute_coverage_v2(match_result: dict, course_type: str,
                        levels: dict[str, str], top_k: int = 25) -> dict:
    """V2 scoring: tier-filtered, credential-free denominator → two-axis label.

    course_type: "applied" | "professional" (foundational is gated before scoring).
    levels:      {skill: "applied"|"professional"} from get_field_levels().
    Returns scores + label + strengths / critical_gaps / next_course_topics rows.
    """
    market = match_result.get("market", [])
    course = match_result.get("course", [])
    course_rank = _TIER_RANK.get(course_type, 2)        # unknown → full core (safe)

    pool = [s for s in sorted(market, key=importance, reverse=True)
            if s["skill"].lower() not in _CREDENTIALS][:top_k]
    den = [s for s in pool
           if _TIER_RANK.get(levels.get(s["skill"], "applied"), 1) <= course_rank]
    above = [s for s in pool
             if _TIER_RANK.get(levels.get(s["skill"], "applied"), 1) > course_rank]

    imp_cov = sum(importance(s) for s in den if s["covered"])
    imp_den = sum(importance(s) for s in den)
    completeness = (imp_cov / imp_den) if imp_den else 0.0

    rel_in = sum(importance(r) for r in course if r.get("in_demand"))
    rel_all = sum(importance(r) for r in course)
    relevance = (rel_in / rel_all) if rel_all else 0.0

    label = _label_v2(completeness, relevance)

    def _row(s, covered):
        r = {"skill": s["skill"], "demand": s.get("demand"),
             "specificity": round(float(s.get("specificity") or 1.0), 3),
             "score": s.get("score"), "band": s.get("band"),
             "tier": levels.get(s["skill"], "applied"),
             "verify_reason": s.get("verify_reason")}
        if covered:
            r["evidence"] = s.get("evidence")
        return r

    strengths = [_row(s, True) for s in den if s["covered"]]
    critical_gaps = [_row(s, False) for s in den if not s["covered"]]
    next_course_topics = [
        {"skill": s["skill"], "demand": s.get("demand"),
         "covered": bool(s.get("covered"))} for s in above
    ]

    return {
        "coverage": round(completeness, 4),
        "completeness_pct": round(100 * completeness, 1),
        "relevance_pct": round(100 * relevance, 1),
        "recommendation": label,
        "course_type": course_type,
        "n_core_covered": sum(1 for s in den if s["covered"]),
        "n_core": len(den),
        "top_k": top_k,
        "strengths": strengths,
        "critical_gaps": critical_gaps,
        "suggested_additions": [g["skill"] for g in critical_gaps],
        "next_course_topics": next_course_topics,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Orchestrators
# ════════════════════════════════════════════════════════════════════════════

def recommend_from_skills(course_skills: list[str], category: str = "it",
                          field: str | None = None,
                          role_patterns: list[str] | None = None,
                          top_n: int | None = _DEFAULT_TOP_N,
                          top_k: int = _DEFAULT_TOP_K,
                          threshold: float = _DEFAULT_TAU,
                          verify: bool = True, field_name: str = "",
                          course_title: str = "") -> dict:
    """Steps 2→5 from already-extracted course skills (UI fast path).

    field: if given (e.g. "ai"), score against that FIELD's demand. On a cache
           miss the field inventory is built from `role_patterns` (from
           classify_field). Without `field`, scores against the whole `category`.
    top_k: # of core (most in-demand) field skills the completeness % is over.
    verify: run the Stage-2b LLM verifier to flip embedder false positives before
            scoring (retrieve-then-verify). Off → raw-cosine baseline (ablation).
    """
    market = (get_field_skills(field, role_patterns=role_patterns, top_n=top_n)
              if field else get_market_skills(category, top_n=top_n))
    match = match_skills(course_skills, market, threshold=threshold)

    verify_meta = {"verified": False}
    if verify:
        verify_coverage(match, top_k=top_k, field_name=field_name,
                        course_title=course_title)
        core = _core(match["market"], top_k)
        verify_meta = {
            "verified": True,
            "n_verify_checked": sum(1 for m in core if m.get("verified")),
            "n_verify_removed": sum(1 for m in core if m.get("verified")
                                    and m.get("embed_covered") and not m["covered"]),
        }

    cov = compute_coverage(match, top_k=top_k)
    out = _explain(match, cov)
    out["metadata"] = {"category": category, "field": field,
                       "top_n": top_n, "threshold": threshold, **verify_meta,
                       **match["metadata"],
                       **{k: cov[k] for k in ("top_k", "n_core_covered", "n_core",
                                              "completeness_pct", "relevance_pct")}}
    return out


def recommend_v2(course_skills: list[str], course_title: str = "",
                 description: str = "", markdown: str = "",
                 course_type: str | None = None, field: str | None = None,
                 top_n: int | None = _DEFAULT_TOP_N, top_k: int = 25,
                 threshold: float = _DEFAULT_TAU, band_low: float = 0.55,
                 curate: bool = True) -> dict:
    """Method v2 orchestrator (propose_pipeline_v2.md):

      gate → (foundational? stop) → field core + tier tags → match(skills+tools)
      → band verify → tier-filtered coverage → two-axis label → curated additions

    course_type/field: optional overrides (UI chip) — skip/limit the gate call.
    markdown: full syllabus markdown; used only for narrow tool inference.
    """
    # 1 · gate ──────────────────────────────────────────────────────────────
    classification = None
    if course_type is None or field is None:
        classification = classify_course_v2(course_title, course_skills, description)
        course_type = course_type or classification["course_type"]
        field = field or classification["field"]
    course_type = course_type.strip().lower()
    field_name = (classification or {}).get("field_name", field or "")
    role_patterns = (classification or {}).get("role_patterns")

    tools = infer_tools(markdown)

    if course_type == "foundational":
        return {
            "recommendation": "FOUNDATIONAL",
            "course_type": "foundational",
            "is_foundational": True,
            "field": field, "field_name": field_name,
            "reason": "Foundational/theoretical course — its content does not appear "
                      "as skill words in job postings, so market-demand coverage "
                      "does not apply (not market-facing by design).",
            "classification": classification,
            "tools": tools,
            "metadata": {"n_course": len(course_skills)},
        }

    # 2 · inputs ────────────────────────────────────────────────────────────
    lower = {s.lower() for s in course_skills}
    merged = course_skills + [t["name"] for t in tools if t["name"] not in lower]
    market = get_field_skills(field, role_patterns=role_patterns, top_n=top_n)

    # 3 · match + band verify ───────────────────────────────────────────────
    match = match_skills(merged, market, threshold=threshold)
    verify_coverage_v2(match, top_k=top_k, band_low=band_low,
                       field_name=field_name, course_title=course_title)

    # 4 · tier tags for the scoring pool (cached per field) ─────────────────
    pool = [s["skill"] for s in sorted(match["market"], key=importance, reverse=True)
            if s["skill"].lower() not in _CREDENTIALS][:top_k]
    levels = get_field_levels(field, pool)

    # 5 · score + label ─────────────────────────────────────────────────────
    cov = compute_coverage_v2(match, course_type=course_type,
                              levels=levels, top_k=top_k)

    # 6 · curated additions (LLM selects from real gaps only — never invents) ─
    suggested = cov["suggested_additions"]
    if curate and suggested:
        suggested = curate_suggestions(suggested, field_name=field_name,
                                       course_title=course_title,
                                       taught_skills=merged)

    out = {**cov, "suggested_additions": suggested,
           "is_foundational": False,
           "field": field, "field_name": field_name,
           "classification": classification, "tools": tools,
           "metadata": {"field": field, "top_n": top_n, "top_k": top_k,
                        "threshold": threshold, "band_low": band_low,
                        "n_tools_inferred": len(tools), **match["metadata"]}}
    return out


def recommend(pdf_source, category: str = "it", field: str | None = None,
              role_patterns: list[str] | None = None,
              top_n: int | None = _DEFAULT_TOP_N, top_k: int = _DEFAULT_TOP_K,
              threshold: float = _DEFAULT_TAU) -> dict:
    """Full pipeline Steps 1→5 from a syllabus PDF (path or upload bytes)."""
    course = get_course_skills(pdf_source)
    result = recommend_from_skills(course_skill_names(course), category=category,
                                   field=field, role_patterns=role_patterns,
                                   top_n=top_n, top_k=top_k, threshold=threshold)
    result["course"] = course["course"]
    return result


if __name__ == "__main__":
    ai_course = [
        "Exploratory Data Analysis", "K-Means Clustering", "Gaussian Mixture Models",
        "Supervised Learning", "Linear Regression", "Decision Trees", "Cross-Validation",
        "Logistic Regression", "Classification", "K-Nearest Neighbors", "Gradient Descent",
        "Gradient Boosting", "Model Evaluation",
    ]
    r = recommend_from_skills(ai_course, field="ai", top_n=2000, top_k=30, threshold=0.70)
    md = r["metadata"]
    print(f"\nField=AI · Recommendation: {r['recommendation']}")
    print(f"  Completeness (core field covered): {r['completeness_pct']}%  "
          f"({md['n_core_covered']}/{md['n_core']} core skills)")
    print(f"  Relevance (taught skills in demand): {r['relevance_pct']}%\n")
    print("Covered core AI skills:")
    for s in r["strengths"]:
        print(f"  ✓ {s['skill']:<28} 📊{s['demand']}  ← {s['evidence']}")
    print("\nMissing in-demand AI skills (add these):")
    for g in r["critical_gaps"]:
        print(f"  ✗ {g['skill']:<28} 📊{g['demand']} ×{g['specificity']}")
