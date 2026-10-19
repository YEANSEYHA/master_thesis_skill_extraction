course_skills = [
  "Supervised Learning",
  "Unsupervised Learning",
  "Clustering",
  "K-Means",
  "Gaussian Mixture",
  "Linear Regression",
  "Decision Trees",
  "Cross-Validation",
  "Logistic Regression",
  "K-Nearest Neighbor",
  "Gradient Descent",
  "Gradient Boosting",
  "Data Exploration and Analysis (EDA)",
  "AI Application Development",
  "Algorithm Evaluation",
  "Data Preparation",
  "Algorithm Selection for Datasets",
  "AI Sub-Fields"
]

market_skills = [
  "python",
  "machine learning",
  "tensorflow",
  "pytorch",
  "deep learning",
  "model deployment",
  "computer vision",
  "data preprocessing",
  "natural language processing",
  "model evaluation",
  "model training",
  "numpy",
  "scikitlearn",
  "jupyter",
  "numba"
]

import os
import numpy as np
import pandas as pd
import ollama

_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
from pathlib import Path
_HERE = Path(__file__).resolve().parent            # backend/archive/playground/esco_retrieval
ESCO_CSV = (_HERE.parents[1] / "phase5_legacy"     # -> backend/archive/phase5_legacy
            / "recommendation_v3" / "esco_skills_en.csv")
CACHE = _HERE / ".esco_labels_qwen3.npz"  # embeddings cached here after run 1


def encode(texts):
    v = np.asarray(ollama.embed(model=_MODEL, input=list(texts)).embeddings,
                   dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

_MEM = None                     # RAM cache: load the big matrix once per process

def load_esco():
    """ESCO labels + their embeddings. Embeds 104k labels ONCE (disk cache),
    then loads from disk once per process (RAM cache)."""
    global _MEM
    if _MEM is not None:
        return _MEM
    labels = pd.read_csv(ESCO_CSV)["preferredLabel"].dropna().unique().tolist()
    if os.path.exists(CACHE):
        E = np.load(CACHE)["E"]
    else:
        chunks = []
        for i in range(0, len(labels), 512):           # batch, or Ollama chokes
            chunks.append(encode(labels[i:i + 512]))
            print(f"  embedded {min(i + 512, len(labels))}/{len(labels)}")
        E = np.vstack(chunks)
        np.savez_compressed(CACHE, E=E)
    _MEM = (labels, E)
    return _MEM
def to_esco(skills, floor=0.80):

    labels, E = load_esco()
    S = encode(skills)                 # (n_skills, 1024)
    sim = S @ E.T                      # (n_skills, 104064)  every skill vs all ESCO
    rows = []
    for i, s in enumerate(skills):
        j = int(sim[i].argmax())       # index of the single nearest ESCO label
        score = float(sim[i][j])
        rows.append({"skill": s,
                     "esco": labels[j] if score >= floor else s,   # keep-raw guard
                     "score": round(score, 3),
                     "mapped": score >= floor})
    return rows


if __name__ == "__main__":
    for r in to_esco(market_skills):
        tag = "->" if r["mapped"] else "x keep raw:"
        print(f"  {r['skill']:28s} {tag} {r['esco']}  ({r['score']:.2f})")