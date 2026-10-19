"""Stage 2 — RETRIEVE: anchor-bootstrap the title cohorts (high recall).

    domain name → anchor = MiniLM embedding of the name ("artificial intelligence")
    unique job titles of the category slice → embedded once (one matrix)
    a title joins its BEST domain iff cosine ≥ FLOOR *and* it clears MARGIN over
    the runner-up domain (ambiguous titles join nothing);
    cohort capped at the top-M titles by cosine.

High recall, imperfect precision — stage 3 (LLM verify) restores precision.
"""

import os
import sys
from pathlib import Path

import numpy as np

_MODEL = os.getenv("V3_EMBED_MODEL", "all-MiniLM-L6-v2")

# Cohort rules (high precision beats recall here — the profile only needs a few
# hundred unambiguous postings per domain, not every posting).
_FLOOR = 0.45        # min cosine(title, domain anchor) to join at all
_MARGIN = 0.05       # must beat the runner-up domain by this much
_MAX_TITLES = 500    # cohort cap: top-M unique titles by anchor cosine


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


def embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        torch.set_default_device("cpu")   # dodge transformers-5 meta-device leak
    except Exception:
        pass
    return SentenceTransformer(_MODEL, device=_device())


def retrieve_cohorts(category: str, domains: list[str], titles: list[str],
                     model) -> dict[str, list[str]]:
    """PASS 1 — {domain: [cohort titles, cosine-ordered, capped at _MAX_TITLES]}."""
    anchors = np.asarray(
        model.encode([d.replace("_", " ") for d in domains],
                     normalize_embeddings=True), dtype=np.float32)
    T = np.asarray(
        model.encode(titles, normalize_embeddings=True, batch_size=512,
                     show_progress_bar=True), dtype=np.float32)

    # matmul via torch, NOT numpy — numpy's Accelerate BLAS on macOS intermittently
    # returns garbage (overflow/-inf) on this matmul, silently emptying every cohort
    import torch
    sims = (torch.from_numpy(T) @ torch.from_numpy(anchors).T).numpy()
    if not np.isfinite(sims).all() or sims.max() < _FLOOR:
        raise RuntimeError("degenerate title↔anchor similarities — check embeddings")
    order = np.argsort(-sims, axis=1)
    best = order[:, 0]
    # single-domain categories have no runner-up: margin is always satisfied
    s1 = sims[np.arange(len(titles)), best]
    s2 = (sims[np.arange(len(titles)), order[:, 1]]
          if len(domains) > 1 else np.full(len(titles), -1.0))

    cand: dict[str, list[tuple[float, str]]] = {d: [] for d in domains}
    for i, t in enumerate(titles):
        if s1[i] >= _FLOOR and (s1[i] - s2[i]) >= _MARGIN:
            cand[domains[best[i]]].append((float(s1[i]), t))
    print(f"[{category}] pass-1 candidates: "
          f"{sum(len(v) for v in cand.values()):,} titles of {len(titles):,}",
          file=sys.stderr)
    ordered: dict[str, list[str]] = {}
    for d in domains:
        cand[d].sort(reverse=True)
        ordered[d] = [t for _, t in cand[d][:_MAX_TITLES]]
    return ordered
