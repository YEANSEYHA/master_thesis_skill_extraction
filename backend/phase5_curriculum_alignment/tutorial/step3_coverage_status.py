"""LESSON 3 (workbook) -- coverage score, thresholds, statuses.

From the matrix (lesson 2) to a VERDICT per market skill:
    coverage = row max;  status = threshold the max.

Complete the problems IN ORDER, then run:
    python3 tutorial/step3_coverage_status.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import get_thresholds, is_soft_skill  # noqa: E402
from tiny_data import embed_and_similarity  # noqa: E402

# ---- setup (given) ----------------------------------------------------
names, counts, clo_names, skill_emb, clo_emb, sim = embed_and_similarity()
thr = get_thresholds()   # {'covered': 0.73, 'borderline': 0.69,
                         #  'host_min': 0.66}  -- BGE, provisional
print(f"thresholds: {thr}\n")


# PROBLEM 1: Soft-skill filter
# Soft skills (communication...) never compete with technical skills --
# they are a teaching-methods question, not a new-module question.
#
# Steps:
#   1. hard = [i for i, n in enumerate(names) if not is_soft_skill(n)]
#   2. soft = [n for n in names if is_soft_skill(n)]
#   3. print the filtered-out soft list, and how many hard remain

# Your code below:



# PROBLEM 2: Coverage score per technical skill
# Coverage = the skill's BEST match over all 8 CLOs (row max).
#
# Steps:
#   1. loop over hard indexes:  for i in hard:
#   2. cov  = sim[i].max()                  # THE coverage score
#      best = sim[i].argmax()               # host CANDIDATE only
#   3. print(f"{names[i]:<18} {int(counts[i]):>8,}  {cov:.2f}  "
#            f"{clo_names[best]}")
#      (counts get thousands separators -- project convention)

# Your code below:



# PROBLEM 3: Status verdicts
# Turn each coverage number into words a dean can read.
#
# Steps:
#   1. inside the same kind of loop, decide:
#        status = "covered"    if cov >= thr["covered"]
#                 "borderline" if cov >= thr["borderline"]
#                 "gap"        otherwise            (if/elif/else)
#   2. host-candidate rule: if cov < thr["host_min"], the argmax CLO
#      is NOISE -- print "no host" instead of the CLO name
#   3. print one line per skill: name, coverage, status, host
#   4. bonus (answer as a comment): sql scores 0.69 on "SQL Injection
#      Prevention". Which status does it get? Why is the cross-encoder
#      still needed before naming that CLO as sql's host?

# Your code below:



# After it runs, check against the thesis rules:
#   - ~0.45-0.55 = noise floor = "not taught"; NEVER read 0.55 as 55%.
#   - aws / agile / linux: best CLO below host_min -> argmax is noise,
#     "no host" (the aws -> 'Pandas' lesson from the real data).
#   - data analysis (~0.79) lands in the 0.73-0.80 zone the LLM judges
#     called "Partly" -- the live recalibration question of the thesis.
