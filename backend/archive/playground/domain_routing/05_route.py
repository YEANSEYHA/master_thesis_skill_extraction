import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent / "output"   # generated results live here

_model = SentenceTransformer("all-MiniLM-L6-v2")


def centroid(skills):
    """L2-normalized mean of the skills' embeddings → one vector for the group."""
    vecs = _model.encode(skills, normalize_embeddings=True)
    v = np.asarray(vecs, dtype=np.float32).mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n else v


def load_domain_matrix():
    """(names, M) where M[i] = centroid of domain i's top-30 profile skills."""
    profiles = json.loads((OUT / "domain_profiles.playground.json").read_text())
    names = list(profiles)
    M = np.stack([centroid(profiles[d]["profile"]) for d in names])
    return names, M

def route_course(course_skills, top_k=3):
    """Course skills → [{name, score}] domains, sorted by cosine desc."""
    names, M = load_domain_matrix()
    cvec = centroid(course_skills)      # the course as one vector
    sims = M @ cvec                     # cosine vs every domain (both normalized)
    order = np.argsort(-sims)
    return [{"name": names[i], "score": round(float(sims[i]), 3)}
            for i in order[:top_k]]


# course skills          course centroid        cosine vs each          routed
# [k-means, regression,  ──(1 mean vector)──▶    domain centroid   ──▶   data_science 0.77
#  decision trees, ...]                          (built in Stage 4)      artificial_int 0.76 …



if __name__ == "__main__":
    ai_course = [
        "Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
        "Linear Regression", "Decision Trees", "Cross-Validation",
        "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
        "Data Exploration and Analysis (EDA)", "Algorithm Evaluation",
    ]
    print("Course skills:")
    print("  " + ", ".join(ai_course) + "\n")
    print("Routed domains (cosine of course centroid vs domain profile):")
    for d in route_course(ai_course, top_k=5):
        bar = "█" * int(d["score"] * 40)
        print(f"  {d['score']:.3f}  {d['name']:<26} {bar}")