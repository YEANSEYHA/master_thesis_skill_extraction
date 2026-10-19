"""SBERT embedder for the AMC baseline — self-contained.

Wraps sentence-transformers (Qwen/Qwen3-Embedding-0.6B by default). Singleton so
the model (~1.2GB) loads once per process; thread-locked because the HuggingFace
fast tokenizer is not thread-safe under FastAPI's threadpool ("Already borrowed").

Swap the model via the EMBED_MODEL env var without touching callers.
"""

from __future__ import annotations

import os
import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_LOCK = threading.RLock()


class Embedder:
    """Singleton SBERT wrapper → L2-normalized embeddings."""

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
            # `low_cpu_mem_usage=False` forces real weights onto CPU first; with the
            # accelerate default (True) the weights land on the `meta` device and the
            # subsequent `.to("mps")` raises "Cannot copy out of meta tensor; no data!".
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                model_kwargs={"low_cpu_mem_usage": False},
            )
            self._initialized = True
            print(f"[embedder] ready · dim={self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: Sequence[str], task: str = "skill") -> np.ndarray:
        """Embed a list of strings → (N, D) L2-normalized matrix.

        Rows are unit-length, so cosine similarity between any two rows is just
        their dot product. For instruction-tuned models, prepends a retrieval
        instruction (no-op for all-mpnet-base-v2).
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))

        inputs = list(texts)
        if "instruct" in self.model_name.lower():
            inputs = [
                f"Instruct: Given a course syllabus sentence, retrieve relevant job {task}s.\nText: {t}"
                for t in inputs
            ]

        # serialize encode() — fast tokenizer is not thread-safe
        with _LOCK:
            return self.model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)


if __name__ == "__main__":
    e = Embedder()
    vecs = e.embed(["finite automata and regular languages", "Python web development"])
    print(f"shape: {vecs.shape}")
    print(f"L2 norm of vec[0]: {np.linalg.norm(vecs[0]):.4f}  (should be ~1.0)")
    print(f"cosine(s0, s1): {float(vecs[0] @ vecs[1]):.4f}  (should be low)")
