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
        shortlist.append({"course": skill, "candidates": cands})
    return shortlist

SYS_P = """You judge whether a university COURSE SKILL genuinely teaches a labour-market MARKET SKILL. Answer ONLY 'yes' or 'no'.
Say yes only if a student who learned the course skill has real, usable ability in the market skill.
Rules:
- Related, same field, or often-used-together is NOT enough: answer no.
- A specific course topic covers a broader market skill it is a core part of.
- A broad course topic does NOT cover a more specific market skill, unless teaching it necessarily includes that skill.
- A software tool or library counts only if the course skill implies hands-on use of that exact tool.
- If uncertain, answer no."""


# Step 2 : Verify
def verify(shortlist, model="qwen3:14b"):
    def ask(course_skill, market_skill):
        u = (f"Course skill taught: {course_skill}\n"
             f"Market skill in demand: {market_skill}\n"
             f"Does teaching '{course_skill}' cover '{market_skill}'? yes or no.")
        r = ollama.chat(model=model, think=False,
                        options={"temperature": 0, "num_predict": 3},
                        messages=[{"role": "system", "content": SYS_P},
                                  {"role": "user", "content": u}])
        return r.message.content.strip().lower().startswith("y")

    rows = []
    for item in shortlist:
        verified = [c["market"] for c in item["candidates"]
                    if ask(item["course"], c["market"])]
        rows.append({"course": item["course"], "verified": verified})
    return rows


if __name__ == "__main__":
    print("STAGE 1 — retrieve\n")
    shortlist = retrieve(top_n=2, floor=0.65)
    for s in shortlist:
        cs = ", ".join(
            f"{c['market']} [{c['esco']}] {c['score']:.2f}" if c['market'] != c['esco']
            else f"{c['market']} {c['score']:.2f}"
            for c in s["candidates"])
        print(f"  {s['course']:36s} -> {cs or '(none)'}")

    print("\nSTAGE 2 — verify  (qwen3:14b)\n")
    for r in verify(shortlist):
        print(f"  {r['course']:36s} => {', '.join(r['verified']) or '-'}")