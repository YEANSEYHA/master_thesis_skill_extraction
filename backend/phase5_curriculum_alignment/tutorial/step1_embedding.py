"""LESSON 1 -- What is an embedding?

Run:  python3 tutorial/step1_embedding.py

The encoder turns ANY text into a vector of 768 numbers, built so that
texts with similar MEANING point in similar DIRECTIONS. That is the
whole trick: we never string-match "sql" against "SQL Injection
Prevention" -- we compare directions.

Two facts make the math trivial:
  1. normalize_embeddings=True  -> every vector has length 1.0
  2. for unit vectors, dot product == cosine similarity
so `a @ b` directly gives "how similar in meaning" (roughly 0.4..1.0
for BGE; ~1.0 = same meaning, ~0.5 = unrelated noise floor).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import MODEL_NAME, load_model  # noqa: E402

texts = [
    "python",                 # market-style: short skill tag
    "Python Programming",     # CLO-style: how the curriculum says it
    "java",                   # different language, same domain
    "snake",                  # 'python' the animal -- context test
    "cooking pasta",          # unrelated control
]

print(f"encoder: {MODEL_NAME} (the SAME frozen model as the thesis)")
model = load_model()

emb = model.encode(texts, normalize_embeddings=True,
                   show_progress_bar=False)

# ---- fact 1: each text became 768 numbers of length 1 ---------------
print(f"\nembedding matrix shape: {emb.shape}  "
      f"({len(texts)} texts x {emb.shape[1]} dimensions)")
print("first 6 numbers of 'python':", np.round(emb[0][:6], 3))
print("vector lengths (all 1.0 because normalized):",
      np.round(np.linalg.norm(emb, axis=1), 3))

# ---- fact 2: dot product of unit vectors = cosine similarity --------
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    cos = emb @ emb.T                       # all pairs at once

print("\npairwise cosine similarity:")
w = max(len(t) for t in texts)
print(" " * (w + 2) + "  ".join(f"[{i}]" for i in range(len(texts))))
for i, t in enumerate(texts):
    row = "  ".join(f"{cos[i, j]:.2f}" for j in range(len(texts)))
    print(f"[{i}] {t:<{w}} {row}")

print("""
WHAT TO NOTICE
  - python x Python Programming is HIGH: different strings, same
    meaning -- exactly what string matching cannot see.
  - python x java is mid: related domain, not the same skill.
  - cooking pasta scores ~0.5 against everything: that is BGE's
    noise floor for short texts. Unrelated pairs never score 0 --
    that is why the project needs thresholds (lesson 3) instead of
    'nonzero = match'.
  - python x snake is ~0.64 -- ABOVE the floor! The encoder knows
    python-the-animal. Meaning-tracking is fuzzy at the edges, which
    is why borderline scores need verification (cross-encoder,
    annotation), never blind trust.
""")

# ======================================================================
# EXERCISE 1 -- now YOU write it (model is already loaded above)
#
# SYNTAX reminder (the only two things you need):
#   emb = model.encode(["a", "b"], normalize_embeddings=True)  # (2,768)
#   score = emb[0] @ emb[1]        # dot product = cosine similarity
#
# TASK:
#   1. encode the QUERY "sql"      (tip: encode takes a LIST -> [query])
#   2. encode the 3 CANDIDATES below
#   3. print query-vs-each-candidate score, 2 decimals
#   4. print the WINNER (the candidate with the highest score)
#
# then run:  python3 tutorial/step1_embedding.py
# and paste me your code + output for review.
# ======================================================================

query = "sql"
candidates = [
    "SQL Injection Prevention",
    "Python Programming",
    "Machine Learning Fundamentals",
]

# ---------------- YOUR CODE STARTS HERE ------------------------------

query_emb = model.encode([query], normalize_embeddings=True)
candidate_embs = model.encode(candidates, normalize_embeddings=True)

scores = query_emb[0] @ candidate_embs.T

for candidate, score in zip(candidates, scores):
    print(f"{candidate}: {score:.2f}")

winner_index = scores.argmax()
print(f"Winner: {candidates[winner_index]}")
