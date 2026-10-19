"""LESSON 2 (workbook) -- THE similarity matrix.

Everything in the thesis derives from ONE matrix product:
    sim[i, j] = "how much does CLO j teach market skill i?"

Complete the problems IN ORDER (later ones use your earlier variables),
then run:  python3 tutorial/step2_similarity_matrix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from tiny_data import embed_and_similarity  # noqa: E402

# ---- setup (given): both sides already embedded ----------------------
# 12 market skills and 8 CLO skills, encoded with the frozen encoder.
# The last value is the ready-made matrix -- ignored: you build your own.
names, counts, clo_names, skill_emb, clo_emb, _ = embed_and_similarity()
print(f"setup: {len(names)} market skills x {len(clo_names)} CLO skills")
print(f"skill_emb {skill_emb.shape}, clo_emb {clo_emb.shape}\n")


# PROBLEM 1: Build the matrix
# One matrix product computes all 96 similarities at once.
#
# Steps:
#   1. sim = skill_emb @ clo_emb.T        (note the .T -- why?)
#   2. print(sim.shape)                   expect (12, 8)
#   3. print(f"{sim[0, 0]:.2f}")          python x Python Programming --
#                                         you saw ~0.92 in lesson 1

# Your code below:
sim = skill_emb @ clo_emb.T
print(sim.shape)
print(f"{sim[0,0]:.2f}")


# PROBLEM 2: Print the whole grid
# The real matrix is 141,050 cells; yours is 96 -- print and READ it.
#
# Steps:
#   1. legend first:
#        for j, c in enumerate(clo_names):
#            print(f"C{j + 1}: {c}")
#   2. one line per market skill:
#        for i, name in enumerate(names):
#            row = " ".join(f"{sim[i, j]:.2f}"
#                           for j in range(len(clo_names)))
#            print(f"{name:<18} {row}")
#   3. run it and find: the LOWEST cell in the grid. That value is
#      BGE's noise floor -- "nothing", even though it is not 0.

# Your code below:
for j, c in enumerate(clo_names):
    print(f"C{j + 1}: {c}")

for i, name in enumerate(names):
           row = " ".join(f"{sim[i, j]:.2f}"
                          for j in range(len(clo_names)))
           print(f"{name:<18} {row}")



# PROBLEM 3: Read the matrix in BOTH directions
# Rows answer "is market skill i taught anywhere?" (thesis direction).
# Columns answer "which market skill does CLO j serve best?"
#
# Steps:
#   1. row direction, one line per market skill:
#        best CLO index  = sim[i].argmax()
#        best CLO score  = sim[i].max()
#        print(f"{names[i]:<18} best CLO: {clo_names[...]} (...:.2f)")
#   2. column direction (new axis trick):
#        col_best = sim.argmax(axis=0)     # 8 indexes into names
#        col_max  = sim.max(axis=0)        # 8 scores
#        print one line per CLO: its best market skill + score
#   3. bonus (answer as a comment here): which CLO has the WEAKEST
#      best-match, and what could that mean for the curriculum?

# Your code below:

for i in range(len(names)):
    best_CLO_index  = sim[i].argmax()
    best_CLO_score  = sim[i].max()
    print(f"{names[i]:<18} best CLO: {clo_names[best_CLO_index]} ({best_CLO_score:.2f})")


# After it runs, notice:
#   - sql's best cell is "SQL Injection Prevention" (~0.69) -- the trap
#     from lesson 1, now visible inside the full grid.
#   - No cell is 0; the floor is ~0.45-0.55. Raw cosine is NOT a
#     percentage -- 0.55 means "nothing", not "55% taught".
