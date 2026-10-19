"""Recommendation V3 — ESCO normalization (symmetric: applied to BOTH inputs).

Map a free-text skill to its nearest ESCO *preferred label* so that variants on
the course side and the job side collapse onto a shared, recognized vocabulary
before matching:

    "K-mean" / "k-means clustering" / "kmeans"  →  one ESCO preferred label

Matching surface = every ESCO preferredLabel AND its altLabels (synonyms). A
query maps to the preferred label of its nearest surface; if even the nearest is
below `threshold`, we KEEP THE RAW skill (do not force a wrong ESCO label —
that is the documented source of ESCO noise).

Self-contained: ESCO CSV lives in this folder; reuses route.py's embedder.
First call embeds the ~50k ESCO surfaces once → cached to .esco_emb.npz.
"""

import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_routing import stage5_route as route  # noqa: E402  (reuse the MiniLM embedder)

_HERE = Path(__file__).resolve().parent
_CSV = _HERE / "esco_skills_en.csv"
_CACHE = _HERE / ".esco_emb.npz"

_DEFAULT_TAU = 0.60      # below this, keep the raw skill (avoid forcing a bad label)


@lru_cache(maxsize=1)
def _esco():
    """Return (preferred_per_surface, V) where V[i] embeds surface i and
    preferred_per_surface[i] is the ESCO preferred label that surface maps to."""
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

    if _CACHE.exists():
        z = np.load(_CACHE, allow_pickle=True)
        if len(z["preferred"]) == len(preferred):
            return list(z["preferred"]), z["V"]

    emb = route._embedder()
    V = np.asarray(emb.encode(surfaces, batch_size=512, normalize_embeddings=True,
                              show_progress_bar=False), dtype=np.float32)
    np.savez(_CACHE, preferred=np.array(preferred, dtype=object), V=V)
    return preferred, V


def normalize(skills: list[str], threshold: float = _DEFAULT_TAU) -> list[dict]:
    """Map each skill to its nearest ESCO preferred label (or keep raw if weak).

    Returns [{orig, esco, score, mapped}] — `esco` is the canonical label used for
    matching (= orig when mapped is False)."""
    skills = [s for s in skills if s and s.strip()]
    if not skills:
        return []
    preferred, V = _esco()
    emb = route._embedder()
    Q = np.ascontiguousarray(
        np.asarray(emb.encode(skills, normalize_embeddings=True), dtype=np.float64))
    sims = Q @ np.ascontiguousarray(V.astype(np.float64)).T     # (n, |surfaces|)
    out = []
    for i, skill in enumerate(skills):
        j = int(sims[i].argmax())
        sc = float(sims[i][j])
        mapped = sc >= threshold
        out.append({"orig": skill, "esco": preferred[j] if mapped else skill,
                    "score": round(sc, 3), "mapped": mapped})
    return out


def normalize_map(skills: list[str], threshold: float = _DEFAULT_TAU) -> dict:
    """{orig_skill: esco_label_or_raw} convenience wrapper."""
    return {r["orig"]: r["esco"] for r in normalize(skills, threshold)}


if __name__ == "__main__":
    course = ["Supervised Learning", "Unsupervised Learning", "K-Means",
              "Decision Trees", "Gradient Descent", "Data Exploration and Analysis (EDA)",
              "Logistic Regression", "AI Application Development"]
    jobs = ["python", "machine learning", "sql", "data analysis", "tensorflow",
            "deep learning", "data visualization", "aws"]
    print("=== COURSE skills → ESCO ===")
    for r in normalize(course):
        tag = "" if r["mapped"] else "  [KEPT RAW — weak match]"
        print(f"  {r['orig']:<36} → {r['esco']:<42} ({r['score']}){tag}")
    print("\n=== JOB skills → ESCO ===")
    for r in normalize(jobs):
        tag = "" if r["mapped"] else "  [KEPT RAW — weak match]"
        print(f"  {r['orig']:<36} → {r['esco']:<42} ({r['score']}){tag}")
