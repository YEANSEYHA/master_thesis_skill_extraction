"""Stage 5 — RUNTIME routing: course skills → career domain(s) + cosine scores.

The only stage served by FastAPI (api.py path-loads this file); stages 1–4 are
the offline build that produces domain_profiles.json.

Embeds the course's extracted skills and compares them to a set of CAREER-DOMAIN
profiles, returning the domain(s) the course maps to with REAL cosine scores.

The scores are cosine similarities (course-skill centroid vs domain-profile
centroid) — they are COMPUTED, never produced by the LLM. The LLM only supplies
the 17-bucket `category`; the fine-grained career-domain routing lives here.

DOMAIN_PROFILES is loaded from domain_profiles.json (DATA-DERIVED: the top skills
of each domain's verified postings, built by stage4_profiles.py) so routing stays
in sync with the canonical vocabs and the corpus. _SEED_PROFILES below is only a
fallback if that file is absent. Adding/removing a domain = rebuilding the
profiles file; this stage needs no edit — that is what makes routing dynamic.

Self-contained: only sentence-transformers + numpy (no reach into other folders).
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# Embedder for routing. MiniLM is small + fast; override via V3_EMBED_MODEL.
_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")

# Fallback seed profiles (domain → representative skills) used ONLY when
# domain_profiles.json is missing. The LIVE profiles are loaded from that file.
_SEED_PROFILES: dict[str, list[str]] = {
    "artificial_intelligence": [
        "machine learning", "deep learning", "neural networks", "tensorflow",
        "pytorch", "computer vision", "natural language processing",
        "model training", "supervised learning", "unsupervised learning",
    ],
    "data_science": [
        "data analysis", "statistics", "data visualization", "pandas",
        "exploratory data analysis", "regression", "clustering",
        "feature engineering", "sql", "predictive modeling",
    ],
    "software_engineering": [
        "java", "object-oriented programming", "software design", "git",
        "rest api", "unit testing", "spring", "design patterns",
        "software development", "debugging",
    ],
    "cybersecurity": [
        "network security", "penetration testing", "incident response",
        "digital forensics", "vulnerability assessment", "siem",
        "cryptography", "secure coding", "malware analysis", "threat detection",
    ],
    "cloud_devops": [
        "aws", "docker", "kubernetes", "ci/cd", "terraform", "cloud computing",
        "azure", "infrastructure as code", "devops", "deployment",
    ],
    "networking": [
        "networking", "tcp/ip", "routing", "switching", "network architecture",
        "cisco", "firewall", "vpn", "network administration", "dns",
    ],
}

# the profiles live one level up (recommendation_v3/), written by stage4_profiles.py
_PROFILES_FILE = Path(__file__).resolve().parent.parent / "domain_profiles.json"


def _load_profiles() -> dict[str, list[str]]:
    """Live, DATA-DERIVED career-domain profiles.

    Reads domain_profiles.json {domain: {profile: [top skills], …}} → {domain:
    [skills]}. Falls back to _SEED_PROFILES if the file is absent/unreadable so the
    router always has anchors. This is the dynamic hook: rebuild the profiles file
    (any number of domains) and the router picks them up with no code change.
    """
    try:
        raw = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
        prof = {d: v["profile"] for d, v in raw.items()
                if d != "_meta" and isinstance(v, dict) and v.get("profile")}
        if prof:
            return prof
    except Exception as e:
        print(f"[v3:route] domain_profiles.json unreadable ({e}); using seed profiles",
              flush=True)
    return _SEED_PROFILES


# The domain set the router uses — whatever has been built into the profiles file.
DOMAIN_PROFILES: dict[str, list[str]] = _load_profiles()


def _load_categories() -> dict[str, str]:
    """{domain: job_category} from the profiles file. Profiles built before the
    per-category taxonomy (or the seed fallback) default to 'it'."""
    try:
        raw = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
        return {d: (v.get("category") or "it") for d, v in raw.items()
                if d != "_meta" and isinstance(v, dict) and v.get("profile")}
    except Exception:
        return {d: "it" for d in DOMAIN_PROFILES}


# Which job_category each domain belongs to — lets routing/chips be scoped to the
# course's category (an ITC civil-engineering course never routes to blockchain).
DOMAIN_CATEGORY: dict[str, str] = _load_categories()


def _allowed(category: str | None) -> list[str]:
    """Domain names available for a category (None → every domain)."""
    if not category:
        return list(DOMAIN_PROFILES)
    return [d for d in DOMAIN_PROFILES
            if DOMAIN_CATEGORY.get(d, "it") == category]


def all_domains(category: str | None = None) -> list[dict]:
    """Every career domain the router knows, with posting counts and category.

    Feeds the UI's "Compare by career domain" chip list so the user can pick ANY
    domain, not just the routed one(s). `category` scopes the list to one
    job_category bucket (e.g. only IT domains for an IT course). Falls back to
    the seed-profile names (count=None) if the profiles file is absent.
    """
    try:
        raw = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
        out = [{"name": d, "n_postings": v.get("n_postings"),
                "category": v.get("category") or "it"}
               for d, v in raw.items()
               if d != "_meta" and isinstance(v, dict) and v.get("profile")]
    except Exception:
        out = [{"name": d, "n_postings": None, "category": "it"}
               for d in DOMAIN_PROFILES]
    if category:
        out = [d for d in out if d["category"] == category]
    return out


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
    import torch
    from sentence_transformers import SentenceTransformer
    # Other models loaded earlier in this (shared FastAPI) process can leave torch's
    # DEFAULT device on 'meta' (transformers 5.x lazy loader). Every later load then
    # inherits meta and fails with "Cannot copy out of meta tensor; no data!" — on BOTH
    # mps and cpu. Resetting the default device to cpu before loading is the fix.
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass
    dev = _device()
    try:
        m = SentenceTransformer(_MODEL, device=dev)
        m.encode(["warmup"], show_progress_bar=False)   # force weight materialization
        return m
    except Exception as e:
        print(f"[v3:embedder] {dev} load failed ({e}); falling back to CPU", flush=True)
        return SentenceTransformer(_MODEL, device="cpu")


def _centroid(skills: list[str]) -> np.ndarray:
    """L2-normalized mean of the skills' embeddings."""
    vecs = _embedder().encode(skills, normalize_embeddings=True)
    v = np.asarray(vecs, dtype=np.float32).mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n else v


@lru_cache(maxsize=1)
def _domain_matrix():
    """(names, M) where M[i] = centroid embedding of domain i's seed skills."""
    names = list(DOMAIN_PROFILES)
    M = np.stack([_centroid(DOMAIN_PROFILES[n]) for n in names]).astype(np.float32)
    return names, M


def _matrix_for(category: str | None):
    """(names, M) restricted to a category's domains (None → all)."""
    names, M = _domain_matrix()
    if not category:
        return names, M
    keep = [i for i, n in enumerate(names)
            if DOMAIN_CATEGORY.get(n, "it") == category]
    return [names[i] for i in keep], M[keep]


def route_course(skill_names: list[str], top_k: int = 3,
                 threshold: float = 0.0,
                 category: str | None = None) -> list[dict]:
    """Course skills → [{name, score}] career domains, sorted by cosine desc.

    score = cosine(course-skill centroid, domain-profile centroid). Returns up to
    top_k domains with score >= threshold. `category` restricts candidates to one
    job_category's taxonomy. Empty input / no domains for the category → [].
    """
    skills = [s for s in skill_names if s and s.strip()]
    if not skills:
        return []
    names, M = _matrix_for(category)
    if not names:
        return []
    cvec = _centroid(skills)
    sims = M @ cvec
    order = np.argsort(-sims)
    out = [{"name": names[i], "score": round(float(sims[i]), 3)}
           for i in order if sims[i] >= threshold]
    return out[:top_k]


def route_all(skill_names: list[str], category: str | None = None) -> list[dict]:
    """Like route_course but returns ALL domains (debug / inspection)."""
    return route_course(skill_names, top_k=len(DOMAIN_PROFILES), threshold=-1.0,
                        category=category)


def selected_fields(skill_names: list[str], margin: float = 0.12,
                    floor: float = 0.45, max_n: int = 3,
                    category: str | None = None) -> list[dict]:
    """The career-domain(s) a course actually maps to.

    Selection is RELATIVE (gap from the top), which is robust to the embedder's
    compressed absolute range: keep every domain scoring within `margin` of the
    top domain AND above an absolute `floor`, capped at `max_n`. This cleanly
    picks the primary domain plus any near-tied secondary (e.g. AI + Data
    Science) while dropping the unrelated tail.
    """
    ranked = route_all(skill_names, category=category)
    if not ranked:
        return []
    top = ranked[0]["score"]
    cut = max(floor, top - margin)
    return [d for d in ranked if d["score"] >= cut][:max_n]


# ── Hybrid routing: LLM SELECTS the domains, cosine SCORES them ───────────────
#
# Defensible split (propose.md): the LLM decides WHICH 1–3 career domains a course
# belongs to (a classification it does well), and the displayed number is a
# REPRODUCIBLE cosine — the LLM never invents a score. Falls back to cosine-only
# selection if Ollama is unreachable.

_LLM = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")
_TAU_FIT = float(os.getenv("V3_ROUTE_TAU", "0.45"))   # min TOP cosine for a fixed
                                                      # domain to count as "in scope"

def _route_schema(allowed: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {"type": "string", "enum": allowed},
                "minItems": 1, "maxItems": 3,
            }
        },
        "required": ["fields"],
    }

_ROUTE_SYS = """\
You assign a university course to the 1–3 LABOUR-MARKET CAREER DOMAINS it best
prepares students for. Choose ONLY from the allowed domain keys given by the user.

Judge by the course's SKILLS / topics, not its title (titles can mislead). Pick
the PRIMARY domain first, then add a second (or third) ONLY if the course
genuinely spans it. Do not pad the list — if the course clearly fits one domain,
return just that one.

Return JSON: {"fields": ["<domain_key>", ...]} (1–3 keys, most relevant first)."""


def llm_route(title: str, skill_names: list[str],
              allowed: list[str] | None = None) -> list[str]:
    """qwen3 selects 1–3 domain keys the course belongs to, restricted to
    `allowed` (default: every domain in DOMAIN_PROFILES).

    Returns an ordered list of valid domain keys. On any failure returns [] so the
    caller can fall back to cosine-only selection (safe degradation)."""
    import json
    skills = [s for s in skill_names if s and s.strip()]
    allowed = allowed if allowed is not None else list(DOMAIN_PROFILES)
    if not skills or not allowed:
        return []
    user = (f"Course title: {title or '(untitled)'}\n"
            f"Skills taught: {', '.join(skills)}\n\n"
            f"Allowed domains: {', '.join(allowed)}")
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_route_schema(allowed),
            messages=[{"role": "system", "content": _ROUTE_SYS},
                      {"role": "user", "content": user}],
        )
        chosen = json.loads(resp.message.content).get("fields", [])
    except Exception as e:
        print(f"[v3:llm_route] failed ({e}) — falling back to cosine", flush=True)
        return []
    # keep only valid keys, preserve order, dedup, cap at 3
    valid = set(allowed)
    seen, out = set(), []
    for f in chosen:
        if f in valid and f not in seen:
            seen.add(f)
            out.append(f)
    return out[:3]


# ── Tier 2: OPEN-SET field naming (no fixed domain fits) ──────────────────────
_FIELD_SCHEMA = {"type": "object",
                 "properties": {"field": {"type": "string"}},
                 "required": ["field"]}

_FIELD_SYS = """\
Name, in 1–4 words, the ACADEMIC FIELD or career area a university course belongs
to, judged by its skills/topics — NOT restricted to any list.
  e.g. "Mathematics", "Web Development", "Accounting", "Mechanical Engineering"
Return JSON: {"field": "<short name>"}."""


def llm_name_field(title: str, skill_names: list[str]) -> str:
    """OPEN-SET: qwen3 names the course's real field (NOT limited to the fixed
    domains). Used by Tier 2 when no fixed domain fits. Returns '' on failure."""
    skills = [s for s in skill_names if s and s.strip()]
    if not skills:
        return ""
    user = (f"Course title: {title or '(untitled)'}\n"
            f"Skills taught: {', '.join(skills)}")
    try:
        import ollama
        resp = ollama.chat(
            model=_LLM, think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_FIELD_SCHEMA,
            messages=[{"role": "system", "content": _FIELD_SYS},
                      {"role": "user", "content": user}],
        )
        return str(json.loads(resp.message.content).get("field", "")).strip()
    except Exception as e:
        print(f"[v3:llm_name_field] failed ({e})", flush=True)
        return ""


def route_hybrid(title: str, skill_names: list[str],
                 category: str | None = None) -> list[dict]:
    """Two-tier routing → [{name, score, in_scope}].

    TIER 1 — cosine-score the course against the FIXED domains. If the top score
      ≥ _TAU_FIT a fixed domain genuinely fits: the LLM picks WHICH 1–3 (a
      classification it does well), cosine SCORES them (reproducible). in_scope=True.
    TIER 2 — nothing we have a vocab for fits (top < _TAU_FIT): an OPEN-SET LLM
      names the course's real field. in_scope=False → grouping only, NOT
      market-scored downstream (we have no demand vocab for that field).

    `category` scopes the candidate domains to the course's job_category bucket
    (e.g. an ITC civil-engineering course only sees the engineering taxonomy once
    one is built — and Tier 2 names its field until then).

    Falls back to cosine-only selection if the Tier-1 LLM is unreachable.
    """
    skills = [s for s in skill_names if s and s.strip()]
    if not skills:
        return []
    names, M = _matrix_for(category)
    # no taxonomy built for this category yet → open-set field name (Tier 2)
    if not names:
        field = llm_name_field(title, skills)
        return ([{"name": field, "score": 0.0, "in_scope": False}]
                if field else [])
    cvec = _centroid(skills)
    idx = {n: i for i, n in enumerate(names)}
    sims = M @ cvec
    top = float(sims.max()) if len(sims) else 0.0

    # TIER 2 — no fixed domain fits → open-set field name, flagged out of scope
    if top < _TAU_FIT:
        field = llm_name_field(title, skills) or names[int(np.argmax(sims))]
        return [{"name": field, "score": round(top, 3), "in_scope": False}]

    # TIER 1 — a fixed domain fits: LLM picks which, cosine scores
    chosen = llm_route(title, skills, allowed=names)
    if not chosen:
        out = selected_fields(skills, category=category)   # cosine-only fallback
        for d in out:
            d["in_scope"] = True
        return out
    out = [{"name": n, "score": round(float(M[idx[n]] @ cvec), 3), "in_scope": True}
           for n in chosen if n in idx]
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


if __name__ == "__main__":
    import json
    ai_course = [
        "Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
        "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
        "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
        "Gradient Boosting", "Data Exploration and Analysis (EDA)",
        "AI Application Development", "Algorithm Evaluation", "Data Preparation",
        "Algorithm Selection for Datasets", "AI Sub-Fields",
    ]
    print("ALL domain scores for Intro-to-AI:")
    print(json.dumps(route_all(ai_course), indent=2))
    print("\nTop-3 routed:")
    print(json.dumps(route_course(ai_course, top_k=3, threshold=0.25), indent=2))
