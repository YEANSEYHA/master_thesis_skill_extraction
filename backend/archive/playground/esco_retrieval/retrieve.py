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

# Step 1 : Retrive

import numpy as np
import ollama

_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"

# encode text to vector
def encode(texts):
    v = np.asarray(ollama.embed(model=_MODEL, input=list(texts)).embeddings,
                   dtype=np.float64)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

# Step 0 : Normalize skills to ESCO (symmetric — put both sides in one vocabulary)
from skill_to_esco import to_esco

def normalize(skills):
    """Each raw skill -> its ESCO label (or the raw skill itself if no good match)."""
    return [r["esco"] for r in to_esco(skills)]


def retrieve(course=course_skills, market=market_skills, top_n=2, floor=0.7):
    # Step 0 : ESCO-normalize BOTH sides for matching — but keep the RAW skills,
    # because verify() must judge the raw market skill, NOT the broad ESCO anchor
    # (this is the broad-anchor inflation fix: scikitlearn matches via "machine
    # learning" but is still verified as "scikitlearn").
    course_esco = normalize(course)
    market_esco = normalize(market)

    # Encode the ESCO labels => Vector   (matching happens in shared ESCO space)
    C = encode(course_esco)
    M = encode(market_esco)
    sim = C @ M.T

    shortlist = []
    for i, skill in enumerate(course):              # raw course name for display
        order = np.argsort(-sim[i])
        cands = [{"market": market[j],              # RAW market skill -> verify against THIS
                  "esco": market_esco[j],           # ESCO label it matched through
                  "score": round(float(sim[i][j]), 3)}
                 for j in order if sim[i][j] >= floor][:top_n]
        shortlist.append({"course": skill,
                          "course_esco": course_esco[i],   # so the display can show BOTH
                          "candidates": cands})
    return shortlist




# Coverage (cosine-only baseline) : % of market skills matched by ANY course
# skill at >= threshold. No demand weights — every market skill counts as 1.
# No verify — so this counts RELATEDNESS, not true coverage (inflated).
def coverage(course=course_skills, market=market_skills, threshold=0.7):
    C = encode(normalize(course))
    M = encode(normalize(market))
    sim = C @ M.T                       # (n_course, n_market)
    best = sim.max(axis=0)              # per market skill: best score over ALL course skills
    covered = [m for m, s in zip(market, best) if s >= threshold]
    gaps    = [m for m, s in zip(market, best) if s <  threshold]
    pct = 100 * len(covered) / len(market)
    return {"pct": round(pct, 1), "covered": covered, "gaps": gaps}


def show(skill, esco):
    """'Skill (ESCO == label)' when normalization changed it, else just the skill."""
    return f"{skill} (ESCO == {esco})" if skill != esco else skill


def show_side(title, skills):
    """Numbered list in the note format: 'skill (ESCO = label)' or '(ESCO = kept raw)'."""
    print(title)
    for n, r in enumerate(to_esco(skills), 1):
        tag = r["esco"] if r["mapped"] else "kept raw"
        print(f"  {n:2d}. {r['skill']} (ESCO = {tag})")
    print()


if __name__ == "__main__":
    print("STAGE 0 — ESCO normalization\n")
    show_side("Course Side", course_skills)
    show_side("Market Side", market_skills)

    print("STAGE 1 — retrieve\n")
    shortlist = retrieve(top_n=2, floor=0.7)
    print(f"  {'Course Skill':52s} {'Market Skill'}")
    print(f"  {'-' * 104}")
    for s in shortlist:
        left = show(s["course"], s["course_esco"])
        if not s["candidates"]:
            print(f"  {left:52s} === (none)")
        for c in s["candidates"]:
            right = show(c["market"], c["esco"])
            print(f"  {left:52s} === {right}  {c['score']:.2f}")

    cov = coverage(threshold=0.7)
    print(f"\nCOVERAGE (cosine-only, unweighted, τ=0.7): "
          f"{cov['pct']}%  ({len(cov['covered'])}/{len(market_skills)})")
    print(f"  covered: {', '.join(cov['covered'])}")
    print(f"  gaps   : {', '.join(cov['gaps'])}")

