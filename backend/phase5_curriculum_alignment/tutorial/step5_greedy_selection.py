"""LESSON 5 (workbook) -- the prescriptive layer: WHICH gaps to add?

Lessons 3-4 DESCRIBE the gap; this layer PRESCRIBES the fix. You call
the REAL production function (pipeline.greedy_skill_selection) -- the
exact code behind the dashboard's "optimized skill plan".

Key idea: "adding skill s" = a new CLO whose embedding is s itself.
Similar market skills gain coverage too (SPILLOVER) through the same
max-rule as everywhere else.

Complete the problems IN ORDER, then run:
    python3 tutorial/step5_greedy_selection.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import (get_thresholds, greedy_skill_selection,  # noqa: E402
                      is_soft_skill)
from tiny_data import embed_and_similarity  # noqa: E402

# ---- setup (given): everything you built in lessons 2-4 ---------------
names, counts, clo_names, skill_emb, clo_emb, sim = embed_and_similarity()
thr = get_thresholds()
hard = [i for i, n in enumerate(names) if not is_soft_skill(n)]
hnames = [names[i] for i in hard]
demand = counts[hard] / counts[hard].sum()
profile = np.clip(sim[hard].max(axis=1), 0.0, None)
print(f"setup: gap today = {(demand * (1 - profile)).sum():.4f}\n")


# PROBLEM 1: The candidate list
# Only non-covered technical skills are candidates for adding.
#
# Steps:
#   1. cand = [i for i in hard if sim[i].max() < thr["covered"]]
#   2. cand_names = [names[i] for i in cand]
#   3. print(cand_names)      expect sql, agile, aws, docker, linux,
#                             scrum, postgresql (7 candidates)

# Your code below:



# PROBLEM 2: Run the real greedy selection
# The spillover matrix is skill-to-skill (NOT skill-to-CLO): adding
# "sql" also lifts "postgresql" because cos(sql, postgresql) = 0.81.
#
# Steps:
#   1. cand_sim = np.clip(skill_emb[cand] @ skill_emb[hard].T, 0.0, None)
#      (rows: candidates, columns: ALL 11 technical market skills)
#   2. steps = greedy_skill_selection(demand, profile, cand_names,
#                                     cand_sim)     # the REAL function
#   3. print each step in order:
#        for r, st in enumerate(steps, 1):
#            print(f"{r}. {st['skill']:<14} gain {st['gain']:.4f}")
#   4. bonus (comment): the gains SHRINK down the list. That property
#      has a name in the thesis (submodularity) -- explain it in one
#      sentence: why does sql being added first make postgresql's
#      later gain smaller?

# Your code below:



# PROBLEM 3: The 80% stopping rule
# The full order is kept; the rule only decides how much to RECOMMEND.
#
# Steps:
#   1. total = sum(st["gain"] for st in steps)
#   2. walk the steps, keeping a running sum `cum` (start 0.0):
#        selected = skills where cum < 0.80 * total  BEFORE adding
#        their own gain (i.e. check first, then cum += st["gain"])
#   3. print the selected plan and the skipped rest
#   4. bonus: change 0.80 to 0.60, rerun -- does the ORDER change,
#      or only where the line is drawn?

# Your code below:



# After it runs, compare with the real thesis result:
#   - real data, same function: add 9 of 13 gaps = 80% of achievable
#     gain; microsoft office suite / azure / devops / kubernetes were
#     skipped by the math (<1% each) -- the numbers, not opinion,
#     drew the line.
#   - spillover on real data: devops drifted 0.66 -> 0.71 without ever
#     being selected, because selected skills covered its neighbors.
