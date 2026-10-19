

input1 = "Introduction into AI"
input2 = "Cybersecurity"


def cats_of(p):
    """Normalize job_category to a LIST of labels. Real data has 4 shapes:
    'IT' · 'Sales / Marketing, Business Administration' (comma-joined multi-label)
    · ['Exec. / Management', …] (actual list) · None (247/2129 postings).
    A posting belongs to EACH of its labels; None → ['(uncategorized)']."""
    raw = p.get("job_category")
    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, str):
        labels = raw.split(",")
    else:
        labels = []
    labels = [c.strip() for c in labels if c and c.strip()]
    return labels or ["(uncategorized)"]


def cat_of(p):
    """Single display string for a posting's categories."""
    return ", ".join(cats_of(p))


def read_json(path, limit=None):
    """Load a JSON or JSONL file and return the list of POSTING dicts.

    limit: max number of postings to return (100, 200, 500, … / None = all).
           Counted AFTER skipping bad records, so you get `limit` usable postings.

    Handles three formats:
      - .jsonl (one posting per line)              → streamed, stops at `limit`
                                                     without reading the rest
      - plain list of postings                     → returned as-is
      - stage2 records {"input": raw_text,
                        "output": {posting}}       → unwrap to the "output" dict
        (records whose output is null/empty — e.g. non-job pages — are skipped)
    """
    import json
    out = []

    def add(r):
        if isinstance(r, dict) and "output" in r:
            if r["output"]:                 # skip null/empty extractions
                out.append(r["output"])
        else:
            out.append(r)
        return limit is not None and len(out) >= limit

    with open(path, encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line and add(json.loads(line)):
                    break
        else:
            for r in json.load(f):
                if add(r):
                    break
    return out


# ── Load the real corpus (data/job_postings_categorized.jsonl, LinkedIn 1.35M) ─
from pathlib import Path

DATA_FILE = (Path(__file__).resolve().parent.parent   # playground root
             / "data" / "job_postings_categorized.jsonl")

# how many postings to use: CLI arg wins (python filter_category.py 500),
# else the default below. None = the whole file.
import sys
N_POSTINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

job_postings = read_json(DATA_FILE, limit=N_POSTINGS)
_untitled = sum(1 for p in job_postings if not p.get("job_title"))
print(f"loaded {len(job_postings)} postings from {DATA_FILE.name} "
      f"({_untitled} without job_title — invisible to the title funnel)")



import numpy as np
from sentence_transformers import SentenceTransformer


# ── HyDE role bridge (Stage 0 of Funnel V3) ───────────────────────────────────
def llm_expand_roles(query):
    """qwen3: field/course name → 5-8 job TITLES. [] on failure (raw fallback)."""
    import json
    try:
        import ollama
        resp = ollama.chat(
            model="qwen3:14b", think=False,
            options={"temperature": 0, "num_ctx": 4096},
            format={"type": "object", "properties": {
                "roles": {"type": "array", "items": {"type": "string"},
                          "minItems": 5, "maxItems": 8}},
                "required": ["roles"]},
            messages=[
                {"role": "system", "content":
                 "Given a field, domain, or course name, list 5-8 JOB TITLES "
                 "commonly posted for that field on job boards. "
                 "Return JSON: {\"roles\": [...]}"},
                {"role": "user", "content": query}],
        )
        return json.loads(resp.message.content).get("roles", [])
    except Exception:
        return []



# ══════════════════════════════════════════════════════════════════════════════
# Funnel V3 — research upgrades on top of V2 (HyDE role bridge kept):
#   #1 JobBERT-v3 (TechWolf, MIT) — bi-encoder trained on 21.1M job-title↔skills
#      pairs → used wherever a JOB TITLE is embedded (steps 1+2). Replaces
#      general-text MiniLM in the title register.
#   #2 cross-encoder rerank (ms-marco-MiniLM-L-6-v2) — jointly reads each
#      (role, title) PAIR with full cross-attention on the shortlist →
#      decisive margins instead of V2's 0.546-vs-τ=0.55 coin flips.
# Selection still touches only category+title — skills stay clean for the
# downstream gap analysis.
# ══════════════════════════════════════════════════════════════════════════════

_JB = _CE = None

def _lazy_models():
    """Load JobBERT-v3 + cross-encoder once (after MiniLM is already resident)."""
    global _JB, _CE
    if _JB is None:
        import torch
        try:
            torch.set_default_device("cpu")   # dodge transformers meta-device leak
        except Exception:
            pass
        from sentence_transformers import CrossEncoder
        _JB = SentenceTransformer("TechWolf/JobBERT-v3")
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _JB, _CE


def funnel_v3(query, cat_top=3, shortlist_k=8, tau_ce=0.5):
    jb, ce = _lazy_models()
    roles = llm_expand_roles(query) or [query]
    R = jb.encode([query] + roles, normalize_embeddings=True)   # HyDE side

    # title funnel can only see TITLED postings (real data: ~23% have none)
    titled = [p for p in job_postings if p.get("job_title")]

    # STEP 1: categories — JobBERT reads the title-list texts (its home register)
    # multi-label: a posting contributes its title to EVERY category it carries
    by_cat = {}
    for p in titled:
        for c in cats_of(p):
            by_cat.setdefault(c, set()).add(p["job_title"])
    cats = sorted(by_cat)
    texts = [f"{c} jobs: {', '.join(sorted(by_cat[c])[:10])}" for c in cats]
    C = jb.encode(texts, normalize_embeddings=True)
    cat_scores = sorted(zip(cats, (C @ R.T).max(axis=1).tolist()),
                        key=lambda t: -t[1])
    keep = {c for c, _ in cat_scores[:cat_top]}

    # STEP 2: bi-encoder SHORTLIST — JobBERT title-vs-role cosine (recall stage)
    # a posting is in the pool if ANY of its labels was kept
    pool = [p for p in titled if any(c in keep for c in cats_of(p))]
    titles = sorted({p["job_title"] for p in pool})
    T = jb.encode(titles, normalize_embeddings=True)
    bscore = dict(zip(titles, (T @ R.T).max(axis=1).tolist()))
    short = sorted(titles, key=lambda t: -bscore[t])[:shortlist_k]

    # STEP 3: cross-encoder RERANK — joint (role, title) pairs, max per title,
    # sigmoid → probability-like score (precision stage)
    qr = [query] + roles
    pairs = [(r, t) for t in short for r in qr]
    logits = np.asarray(ce.predict(pairs)).reshape(len(short), len(qr))
    prob = 1.0 / (1.0 + np.exp(-logits.max(axis=1)))
    cescore = dict(zip(short, prob.tolist()))

    ranked = sorted(((p, bscore[p["job_title"]], cescore[p["job_title"]])
                     for p in pool if p["job_title"] in cescore),
                    key=lambda t: -t[2])
    return cat_scores, roles, ranked, tau_ce


for q in [input1, input2]:
    cat_scores, roles, ranked, tau = funnel_v3(q)
    print(f"\n=== FUNNEL-V3 {q!r} ===")
    print("  step 1 — categories (JobBERT):")
    for c, s in cat_scores[:5]:
        print(f"    {s:.3f}  {c}")
    print(f"  step 2+3 — bi-encoder shortlist → cross-encoder rerank:")
    print(f"    {'bi':>6}  {'ce':>6}")
    for p, b, c in ranked:
        mark = "KEEP" if c >= tau else "drop"
        print(f"    {b:.3f}  {c:.3f}  {mark}  {p['job_title']}  [{cat_of(p)}]")
