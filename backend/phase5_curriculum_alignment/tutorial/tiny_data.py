"""Shared tiny dataset for the tutorial lessons.

A miniature of the real pipeline inputs:
  MARKET  ~ 01_input_jobmarket  (140,393 postings -> here 10 skills with
            made-up but realistic demand counts)
  CLOS    ~ 02_input_curriculum/clo_skills/*.json  (130 curated
            CLO skills -> here 8, grouped into 3 mini 'courses')

Steps 3-5 import from here so each lesson stays focused on ONE concept.
Step 2 builds the similarity matrix by hand -- that IS its lesson.

IMPORTANT: the tutorial reuses the SAME frozen instrument as the
thesis -- same encoder (BGE), same thresholds -- by importing from the
real pipeline.py one directory up. Small data, real method.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import load_model  # noqa: E402  (the real, frozen encoder)

# ---- market side: (skill, how many postings mention it) -------------
# Counts are invented but keep the real project's proportions.
MARKET = [
    ("python",           20_000),
    ("sql",              18_500),
    ("machine learning", 15_000),
    ("communication",    14_000),   # soft skill -> filtered in step 3
    ("java",             12_000),
    ("agile",             9_000),
    ("aws",               8_000),
    ("data analysis",     7_000),
    ("docker",            6_000),
    ("linux",             5_000),
    ("scrum",             4_000),   # semantic neighbor of agile (0.82)
    ("postgresql",        3_000),   # semantic neighbor of sql (0.81)
]

# ---- curriculum side: (course, CLO skill) ----------------------------
# All 8 strings are real entries from 02_input_curriculum/clo_skills.
CLOS = [
    ("Programming 101",     "Python Programming"),
    ("Programming 101",     "Object-Oriented Programming (OOP)"),
    ("Programming 101",     "Java Programming"),
    ("AI Fundamentals",     "Machine Learning Fundamentals"),
    ("AI Fundamentals",     "Exploratory Data Analysis (EDA)"),
    ("Security & Networks", "SQL Injection Prevention"),
    ("Security & Networks", "Network Troubleshooting"),
    ("Security & Networks", "Virtualization"),
]


def embed_and_similarity():
    """Encode both sides and return THE one matrix everything derives
    from. Mirrors gen_report_data.py line 107:

        sim = market['skill_emb'] @ clo_emb.T

    Returns (names, counts, clo_names, skill_emb, clo_emb, sim) where
    sim[i, j] = cosine similarity of market skill i and CLO skill j.
    """
    model = load_model()
    names = [s for s, _ in MARKET]
    counts = np.array([c for _, c in MARKET], dtype=float)
    clo_names = [c for _, c in CLOS]

    skill_emb = model.encode(names, normalize_embeddings=True,
                             show_progress_bar=False)
    clo_emb = model.encode(clo_names, normalize_embeddings=True,
                           show_progress_bar=False)

    # normalized vectors => matrix product = all cosines at once
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = skill_emb @ clo_emb.T          # shape (10, 8)
    return names, counts, clo_names, skill_emb, clo_emb, sim
