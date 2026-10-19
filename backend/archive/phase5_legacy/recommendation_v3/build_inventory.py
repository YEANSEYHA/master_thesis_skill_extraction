"""Recommendation V3 — Phase 2 + 3: classify the 143K IT postings into career
domains, then aggregate per-domain skill demand.

    domain_profiles.json  →  domain centroids (mean-pooled skill embeddings)
            │
    each IT posting → mean-pool its skill embeddings → cosine to the centroids
            │           → soft top-2 assignment (threshold + margin)
            ▼
    aggregate each posting's skills into the domain(s) it was tagged with
            │
            ▼
    domain_inventory.json  ← { domain: {n_postings, top_titles, skills:{skill:freq}} }

Efficiency: the skill VOCABULARY is embedded once (bounded set), then every
posting/domain centroid is a cheap numpy mean over cached vectors — so we never
re-embed the same skill across 143K postings.

Self-contained: sentence-transformers + numpy + the shared corpus only.

Run:  python build_inventory.py
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from domain_routing.stage1_corpus import _EXCLUDE   # same soft-skill/credential exclude set

_HERE = Path(__file__).resolve().parent
_CORPUS = _HERE.parent / "linkedin_job_posting" / "job_postings_categorized.jsonl"
_PROFILES = _HERE / "domain_profiles.json"
_OUT = _HERE / "domain_inventory.json"

_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")
_THRESHOLD = 0.70      # min cosine to join ANY domain (else → "other")
_MARGIN = 0.10         # also keep domains within this of the top score
_MAX_DOMAINS = 2       # soft top-2
_TOP_SKILLS = 1000     # cap skills stored per domain inventory (supports the UI's
                       # "top market skills" selector up to 1k)


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


def _aggregate_by_title(posts: list[tuple[str, list[str]]], profiles: dict,
                        domains: list[str]) -> dict:
    """Title-based field routing: a posting joins a field iff its job TITLE
    contains one of that field's seed titles (word-boundary match), NOT by its
    skill mix. Precise — a 'Senior Software Engineer' no longer leaks into the AI
    field even if its skills look AI-ish. A posting may match several fields.
    Trades recall (smaller, ~title-confirmed samples) for precision (only genuine
    field roles), so each course is compared against postings that truly belong
    to its field."""
    pats = {d: [re.compile(r"\b" + re.escape(s.lower()) + r"\b")
                for s in profiles[d].get("seeds", [])] for d in domains}
    inv_skills = {d: Counter() for d in domains}
    inv_titles = {d: Counter() for d in domains}
    n_post = {d: 0 for d in domains}
    n_other = 0
    for title, sk in posts:
        tl = title.lower()
        matched = [d for d in domains if any(p.search(tl) for p in pats[d])]
        if not matched:
            n_other += 1
            continue
        demand = [s for s in sk if s not in _EXCLUDE]    # distinctive skills only
        for d in matched:
            n_post[d] += 1
            inv_skills[d].update(demand)
            inv_titles[d][title] += 1
    out = {"_meta": {"n_postings_it": len(posts), "n_other": n_other,
                     "route": "title", "embedder": None}}
    for d in domains:
        out[d] = {
            "n_postings": n_post[d],
            "top_titles": [t for t, _ in inv_titles[d].most_common(10)],
            "skills": dict(inv_skills[d].most_common(_TOP_SKILLS)),
        }
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _aggregate_by_cohort(posts: list[tuple[str, list[str]]], profiles: dict,
                         domains: list[str]) -> dict:
    """Cohort routing (retrieve→verify): a posting joins a domain iff its
    normalized title is in that domain's LLM-verified cohort (anchor-bootstrap
    stages 2–3, persisted by stage4_profiles.py). Replaces the manual seed
    lists — the title set is derived from the domain NAME alone and audited by
    the LLM, so precision comes from the verify gate, not hand-written regex."""
    missing = [d for d in domains if not profiles[d].get("cohort")]
    if missing:
        raise SystemExit(f"no 'cohort' in domain_profiles.json for {missing} — "
                         f"rerun domain_routing/stage4_profiles.py "
                         f"(it persists cohorts now)")
    title2doms: dict[str, list[str]] = {}
    for d in domains:
        for t in profiles[d]["cohort"]:
            title2doms.setdefault(t, []).append(d)
    inv_skills = {d: Counter() for d in domains}
    inv_titles = {d: Counter() for d in domains}
    n_post = {d: 0 for d in domains}
    n_other = 0
    for title, sk in posts:
        t = re.sub(r"\s+", " ", title.strip().lower())   # stage1's normalization
        matched = title2doms.get(t)
        if not matched:
            n_other += 1
            continue
        demand = [s for s in sk if s not in _EXCLUDE]    # distinctive skills only
        for d in matched:
            n_post[d] += 1
            inv_skills[d].update(demand)
            inv_titles[d][title] += 1
    out = {"_meta": {"n_postings_it": len(posts), "n_other": n_other,
                     "route": "cohort", "embedder": None}}
    for d in domains:
        out[d] = {
            "n_postings": n_post[d],
            "top_titles": [t for t, _ in inv_titles[d].most_common(10)],
            "skills": dict(inv_skills[d].most_common(_TOP_SKILLS)),
        }
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build(route: str = "cohort") -> dict:
    if not _CORPUS.exists():
        raise FileNotFoundError(f"corpus not found at {_CORPUS}")
    profiles = json.loads(_PROFILES.read_text(encoding="utf-8"))
    domains = [d for d in profiles
               if d != "_meta" and isinstance(profiles[d], dict)]

    # 1) load IT postings (title + lowercased skills) and collect the vocabulary
    posts: list[tuple[str, list[str]]] = []
    vocab: set[str] = set()
    scanned = 0
    with open(_CORPUS, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("job_category") != "it":
                continue
            sk = [s.strip().lower() for s in r.get("skills", []) if s.strip()]
            posts.append((r.get("job_title", "") or "", sk))
            vocab.update(sk)
    print(f"scanned {scanned:,} · {len(posts):,} IT postings · route={route}",
          file=sys.stderr, flush=True)

    # cohort routing (default): LLM-verified title cohorts, no embeddings needed
    if route == "cohort":
        return _aggregate_by_cohort(posts, profiles, domains)

    # title routing needs no embeddings — aggregate by seed-title match and return
    if route == "title":
        return _aggregate_by_title(posts, profiles, domains)

    # ── route == "centroid" (legacy skill-mix routing) ───────────────────────────
    for d in domains:
        vocab.update(s.lower() for s in profiles[d]["profile"])
    vocab = sorted(vocab)
    print(f"  centroid mode · {len(vocab):,} unique skills", file=sys.stderr, flush=True)

    # 2) embed the vocabulary ONCE (cached to disk for fast re-runs)
    _CACHE = _HERE / ".vocab_emb.npz"
    cached = False
    if _CACHE.exists():
        z = np.load(_CACHE, allow_pickle=True)
        if list(z["vocab"]) == vocab:
            V, cached = z["V"], True
            print(f"loaded cached vocab embeddings: {V.shape}", file=sys.stderr, flush=True)
    if not cached:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(_MODEL, device=_device())
        V = np.asarray(emb.encode(vocab, batch_size=512, normalize_embeddings=True,
                                  show_progress_bar=False), dtype=np.float32)
        np.savez(_CACHE, vocab=np.array(vocab, dtype=object), V=V)
        print(f"embedded vocabulary: {V.shape} (cached)", file=sys.stderr, flush=True)
    idx = {s: i for i, s in enumerate(vocab)}

    def centroid(skills: list[str]) -> np.ndarray | None:
        ii = [idx[s] for s in skills if s in idx]
        if not ii:
            return None
        v = V[ii].mean(axis=0)
        n = np.linalg.norm(v)
        return (v / n) if n else v

    # 3) domain centroids (from the data-derived profiles)
    C = np.stack([centroid([s.lower() for s in profiles[d]["profile"]])
                  for d in domains]).astype(np.float32)

    # 4) classify each posting (soft top-2) + aggregate
    inv_skills = {d: Counter() for d in domains}
    inv_titles = {d: Counter() for d in domains}
    n_post = {d: 0 for d in domains}
    n_other = n_empty = 0
    for title, sk in posts:
        clf = [s for s in sk if s not in _EXCLUDE]      # cleaner signal for routing
        v = centroid(clf or sk)
        if v is None:
            n_empty += 1
            continue
        sims = C @ v
        order = np.argsort(-sims)
        top = float(sims[order[0]])
        cut = max(_THRESHOLD, top - _MARGIN)
        chosen = [domains[i] for i in order if sims[i] >= cut][:_MAX_DOMAINS]
        if not chosen:
            n_other += 1
            continue
        demand = [s for s in sk if s not in _EXCLUDE]   # distinctive skills only
        for d in chosen:
            n_post[d] += 1
            inv_skills[d].update(demand)
            inv_titles[d][title] += 1

    # 5) write inventory
    out = {
        "_meta": {"n_postings_it": len(posts), "n_other": n_other,
                  "n_empty": n_empty, "threshold": _THRESHOLD, "margin": _MARGIN,
                  "max_domains": _MAX_DOMAINS, "embedder": _MODEL},
    }
    for d in domains:
        out[d] = {
            "n_postings": n_post[d],
            "top_titles": [t for t, _ in inv_titles[d].most_common(10)],
            "skills": dict(inv_skills[d].most_common(_TOP_SKILLS)),
        }
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 2+3 — build per-field skill inventory")
    ap.add_argument("--route", choices=["cohort", "title", "centroid"],
                    default="cohort",
                    help="cohort = LLM-verified title cohorts (retrieve→verify, "
                         "default); title = legacy manual seed-title match; "
                         "centroid = legacy skill-mix routing")
    a = ap.parse_args()
    out = build(route=a.route)
    m = out["_meta"]
    print(f"\n→ wrote {_OUT.name}  (route={m.get('route')}, "
          f"other={m['n_other']:,}, empty={m.get('n_empty', 0):,})\n")
    for d in [k for k in out if k != "_meta"]:
        p = out[d]
        titles = ", ".join(p["top_titles"][:4])
        skills = ", ".join(list(p["skills"])[:10])
        print(f"■ {d}  — {p['n_postings']:,} postings")
        print(f"   titles: {titles}")
        print(f"   skills: {skills}")
        print()
