"""A2 Step 2 — embed every UNIQUE corpus title ONCE with JobBERT-v3 (cached)."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

IT_SLICE = HERE.parent / "domain_routing" / "output" / "it_slice.json"
CACHE = OUT / "title_emb.npz"

MODEL = "TechWolf/JobBERT-v3"     # title specialist (trained on title↔skills)


# it_slice.json (16,069 postings)
#               │  titles only, deduped
#               ▼
#    ┌──────────────────────────┐   YES   ┌─────────────────┐
#    │  cache title_emb.npz     ├────────▶│  exit — no work │
#    │  up to date ?            │         └─────────────────┘
#    └────────────┬─────────────┘
#                 │ NO
#                 ▼
#    ┌──────────────────────────┐
#    │  JobBERT-v3 encode       │   N unique titles → N vectors
#    │  (normalized, once)      │
#    └────────────┬─────────────┘
#                 ▼
#    ┌──────────────────────────┐
#    │  save title_emb.npz      │──▶ sanity: top-5 neighbours
#    │  {titles, V}             │    of "data scientist"
#    └──────────────────────────┘


def main():
    posts = json.loads(IT_SLICE.read_text())["posts"]
    titles = sorted({t.lower().strip() for t, _ in posts if t.strip()})
    print(f"{len(posts):,} postings → {len(titles):,} unique titles")

    # cache hit? then this file has nothing to do — that's the point of it
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["titles"]) == titles:
            print(f"cache fresh: {CACHE.name}  V = {z['V'].shape} — nothing to do")
            return

    model = SentenceTransformer(MODEL)
    V = model.encode(titles, normalize_embeddings=True, batch_size=256,
                     show_progress_bar=True)
    np.savez_compressed(CACHE, titles=np.array(titles, dtype=object), V=V)
    print(f"saved {CACHE.name}   V shape = {V.shape}")

    # sanity check: nearest neighbours of one title — do they make sense?
    probe = "data scientist"
    i = titles.index(probe) if probe in titles else 0
    sims = V @ V[i]                       # normalized → dot = cosine
    print(f"\nnearest titles to {titles[i]!r}:")
    for j in np.argsort(-sims)[1:6]:
        print(f"   {sims[j]:.3f}  {titles[j]}")




if __name__ == "__main__":
    main()
