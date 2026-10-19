"""
ESCO skill normalizer — maps free-text extracted skills to standardized ESCO
preferred labels via JobBERT cosine similarity.

This is the "normalize to ESCO" step in the Phase-5 pipeline:

    extracted skills  ->  ESCO normalize  ->  canonical ESCO skills  ->  match CLOs

Collapses variant spellings/cases ("communication skills", "communication",
"interpersonal skills") into one ESCO preferred label so demand aggregates
correctly before curriculum matching.

Reuses the exact approach validated in Phase 4
(evaluation/esco_evaluation.py): load preferred + altLabels, JobBERT-embed
every label, map each skill to its nearest label, resolve to the preferred
label. The 100k+ ESCO embeddings are expensive to compute, so they are cached
to disk (.esco_cache/) and reloaded on subsequent runs.

Singleton: model + embeddings load once per process (mirrors embedder.py).
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


# ── Paths / constants ──────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[4]   # …/clo_skill_alignment → repo root
_ESCO_CSV = (
    _REPO_ROOT
    / "backend" / "phase3_skill_extraction"
    / "01_data_collection" / "esco_skills_en.csv"
)
_CACHE_DIR = Path(__file__).parent / ".esco_cache"

_MODEL_NAME = "jjzha/jobbert-base-cased"   # same model as Phase-4 ESCO eval
TAU_ESCO = 0.50                            # min cosine to accept an ESCO mapping
_ENCODE_BATCH = 256

# Serialize model construction + encode() — HF fast tokenizers aren't thread-safe.
_LOCK = threading.RLock()


# ── ESCO taxonomy loading ──────────────────────────────────────────────────────

def _load_esco_labels(csv_path: Path = _ESCO_CSV) -> tuple[dict[str, str], list[str]]:
    """Load ESCO preferred + alt labels.

    Returns:
        label_to_preferred: every label (lowercased) -> its preferred label
        all_labels:         sorted unique lowercased labels (encoding order)
    """
    label_to_preferred: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preferred = (row.get("preferredLabel") or "").strip()
            if not preferred:
                continue
            label_to_preferred[preferred.lower()] = preferred
            alt_labels = row.get("altLabels") or ""
            for alt in alt_labels.split("\n"):
                alt = alt.strip()
                if alt:
                    label_to_preferred.setdefault(alt.lower(), preferred)
    all_labels = sorted(label_to_preferred.keys())
    return label_to_preferred, all_labels


def _csv_signature(csv_path: Path) -> dict:
    st = csv_path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime), "model": _MODEL_NAME}


# ── Singleton normalizer ───────────────────────────────────────────────────────

class EscoNormalizer:
    """Singleton ESCO mapper (JobBERT). Heavy on first init, cheap after."""

    _instance: "EscoNormalizer | None" = None

    def __new__(cls) -> "EscoNormalizer":
        with _LOCK:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        with _LOCK:
            if self._initialized:
                return
            self.model_name = _MODEL_NAME
            self.label_to_preferred: dict[str, str] = {}
            self.all_labels: list[str] = []
            self.esco_emb: np.ndarray | None = None
            print(f"[esco] loading JobBERT ({_MODEL_NAME})…")
            self.model = SentenceTransformer(_MODEL_NAME)
            self._load_or_build()
            self._initialized = True
            print(f"[esco] ready · {len(self.all_labels)} labels · "
                  f"{len(set(self.label_to_preferred.values()))} preferred")

    # ---- cache build / load ----

    def _load_or_build(self) -> None:
        sig = _csv_signature(_ESCO_CSV)
        meta_path = _CACHE_DIR / "meta.json"
        emb_path = _CACHE_DIR / "embeddings.npy"
        labels_path = _CACHE_DIR / "labels.json"

        if meta_path.exists() and emb_path.exists() and labels_path.exists():
            try:
                cached_sig = json.loads(meta_path.read_text())
                if cached_sig == sig:
                    print("[esco] loading embeddings from cache…")
                    payload = json.loads(labels_path.read_text())
                    self.all_labels = payload["all_labels"]
                    self.label_to_preferred = payload["label_to_preferred"]
                    self.esco_emb = np.load(emb_path)
                    return
                print("[esco] cache stale (CSV changed) — rebuilding")
            except Exception as e:
                print(f"[esco] cache unreadable ({e}) — rebuilding")

        self._build_cache(sig, emb_path, labels_path, meta_path)

    def _build_cache(self, sig, emb_path, labels_path, meta_path) -> None:
        t0 = time.time()
        print(f"[esco] building embeddings from {_ESCO_CSV.name} "
              f"(one-time, ~1-2 min)…")
        self.label_to_preferred, self.all_labels = _load_esco_labels()
        with _LOCK:
            self.esco_emb = self.model.encode(
                self.all_labels,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=_ENCODE_BATCH,
            ).astype(np.float32)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, self.esco_emb)
        labels_path.write_text(json.dumps({
            "all_labels": self.all_labels,
            "label_to_preferred": self.label_to_preferred,
        }))
        meta_path.write_text(json.dumps(sig))
        print(f"[esco] cache built in {time.time() - t0:.0f}s → {_CACHE_DIR}")

    # ---- mapping ----

    def map_skills(self, skills: Sequence[str],
                   threshold: float = TAU_ESCO) -> list[dict]:
        """Map each free-text skill to its nearest ESCO preferred label.

        Returns one dict per input skill:
          {original, esco_skill, matched_label, similarity, mapped}
        If best cosine < threshold, the skill is kept raw (mapped=False,
        esco_skill == original).
        """
        skills = list(skills)
        if not skills:
            return []

        with _LOCK:
            skill_emb = self.model.encode(
                skills, normalize_embeddings=True, show_progress_bar=False,
                batch_size=_ENCODE_BATCH,
            ).astype(np.float32)

        results: list[dict] = []
        # Batch the argmax to bound memory (B × ~100k matrix per batch).
        # errstate guards spurious FPE warnings from macOS Accelerate BLAS;
        # both operands are unit-norm so the dot products are well-defined.
        for start in range(0, len(skills), _ENCODE_BATCH):
            batch = skill_emb[start:start + _ENCODE_BATCH]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sim = batch @ self.esco_emb.T        # (B, N_labels)
            best_idx = sim.argmax(axis=1)
            best_score = sim[np.arange(sim.shape[0]), best_idx]
            for j, gidx in enumerate(best_idx):
                skill = skills[start + j]
                score = float(best_score[j])
                matched_label = self.all_labels[int(gidx)]
                preferred = self.label_to_preferred[matched_label]
                mapped = score >= threshold
                results.append({
                    "original": skill,
                    "esco_skill": preferred if mapped else skill,
                    "matched_label": matched_label,
                    "similarity": round(score, 3),
                    "mapped": mapped,
                })
        return results

    def normalize_freq(self, raw_freq: dict[str, int],
                       threshold: float = TAU_ESCO) -> dict:
        """Collapse a {raw_skill: freq} dict into {esco_skill: summed_freq}.

        Returns:
          {
            "esco_freq":   {esco_skill: freq}  (sorted desc),
            "mapping":     [{original, esco_skill, similarity, mapped, freq}],
            "mapped_count": int,   # how many unique raw skills hit an ESCO label
            "total_unique": int,   # unique raw skills in
            "esco_unique":  int,   # unique skills after normalization
          }
        """
        if not raw_freq:
            return {"esco_freq": {}, "mapping": [], "mapped_count": 0,
                    "total_unique": 0, "esco_unique": 0}

        skills = list(raw_freq.keys())
        mappings = self.map_skills(skills, threshold=threshold)

        esco_freq: dict[str, int] = {}
        for m in mappings:
            f = int(raw_freq.get(m["original"], 0))
            m["freq"] = f
            esco_freq[m["esco_skill"]] = esco_freq.get(m["esco_skill"], 0) + f

        esco_freq = dict(sorted(esco_freq.items(), key=lambda kv: kv[1], reverse=True))
        mapping = sorted(mappings, key=lambda m: m["freq"], reverse=True)
        mapped_count = sum(1 for m in mappings if m["mapped"])

        return {
            "esco_freq":    esco_freq,
            "mapping":      mapping,
            "mapped_count": mapped_count,
            "total_unique": len(skills),
            "esco_unique":  len(esco_freq),
        }


# Module-level convenience used by the API ------------------------------------

def normalize_freq(raw_freq: dict[str, int],
                   threshold: float = TAU_ESCO) -> dict:
    """Singleton entry point: ESCO-normalize a {skill: freq} dict."""
    return EscoNormalizer().normalize_freq(raw_freq, threshold=threshold)


# ── CLI eyeball test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "communication skills": 231,
        "communication": 174,
        "interpersonal skills": 132,
        "teamwork": 173,
        "team work": 12,
        "customer service": 162,
        "microsoft excel": 95,
        "DFA minimization": 1,   # should fall below threshold → kept raw
    }
    out = normalize_freq(sample)
    print(f"\n{out['total_unique']} raw → {out['esco_unique']} ESCO "
          f"({out['mapped_count']} mapped ≥ {TAU_ESCO})\n")
    print("ESCO inventory (summed):")
    for s, f in out["esco_freq"].items():
        print(f"  {f:>4}  {s}")
    print("\nmapping:")
    for m in out["mapping"]:
        arrow = "→" if m["mapped"] else "·"
        print(f"  {m['original']:<24} {arrow} {m['esco_skill']:<40} "
              f"({m['similarity']})")
