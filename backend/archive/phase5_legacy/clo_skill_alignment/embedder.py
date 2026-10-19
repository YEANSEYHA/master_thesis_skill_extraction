"""
SBERT embedder wrapper.

Pluggable by design — swap `all-mpnet-base-v2` for `bge-m3`,
`gte-Qwen2-7B-instruct`, etc. via the EMBED_MODEL env var without
touching matcher.py. Instruction prefix is auto-applied for
instruction-tuned models.

Singleton pattern: model loads once per process (~420MB for mpnet,
slow on first call, fast afterwards).
"""

from __future__ import annotations

import os
import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL = "all-mpnet-base-v2"

# Guards model construction (singleton init) and encode() calls. The
# HuggingFace fast tokenizer is not thread-safe — concurrent encode()
# calls raise "Already borrowed" — so we serialize access here.
_LOCK = threading.RLock()


class Embedder:
    """Singleton SBERT wrapper."""

    _instance: "Embedder | None" = None

    def __new__(cls, model_name: str | None = None) -> "Embedder":
        with _LOCK:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        with _LOCK:
            if self._initialized:
                return
            self.model_name = model_name or os.getenv("EMBED_MODEL", _DEFAULT_MODEL)
            print(f"[embedder] loading {self.model_name} (this may take a moment)...")
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            print(f"[embedder] ready · dim={self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: Sequence[str], task: str = "skill") -> np.ndarray:
        """Embed a list of strings → (N, D) L2-normalized matrix.

        For instruction-tuned models, prepends a retrieval instruction
        (adds 3-8 MAP points on retrieval tasks per MTEB benchmarks).
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))

        inputs = list(texts)
        if "instruct" in self.model_name.lower():
            inputs = [
                f"Instruct: Given a course syllabus sentence, retrieve relevant job {task}s.\nText: {t}"
                for t in inputs
            ]

        # Serialize encode() — the fast tokenizer is not thread-safe and
        # concurrent calls raise "Already borrowed" under FastAPI's threadpool.
        with _LOCK:
            return self.model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)


# ---------- CLI eyeball test ----------

if __name__ == "__main__":
    e = Embedder()
    sentences = ["finite automata and regular languages", "Python web development"]
    vecs = e.embed(sentences)
    print(f"shape: {vecs.shape}")
    print(f"L2 norm of vec[0]: {np.linalg.norm(vecs[0]):.4f}  (should be ~1.0)")
    print(f"cosine(s0, s1): {float(vecs[0] @ vecs[1]):.4f}  (should be low)")
