import json
import sys
from pathlib import Path

# ── make the REAL stage5_route.py importable (we reuse it, we don't copy it) ──
STAGE_DIR = (Path(__file__).resolve().parents[2]      # backend/
             / "phase5_legacy" / "recommendation_v3" / "domain_routing")
sys.path.insert(0, str(STAGE_DIR))

import stage5_route as route   # noqa: E402  the actual Stage-5 module


# A sample course's extracted skills (the ITC "Introduction to AI" syllabus).
# In production these come out of the PDF parser; here we hard-code them so you
# can see routing in isolation.
AI_COURSE = [
    "Supervised Learning", "Unsupervised Learning", "Clustering", "K-Means",
    "Gaussian Mixture", "Linear Regression", "Decision Trees", "Cross-Validation",
    "Logistic Regression", "K-Nearest Neighbor", "Gradient Descent",
    "Gradient Boosting", "Data Exploration and Analysis (EDA)",
    "Algorithm Evaluation", "Data Preparation",
]


def main():
    print("Skills we are routing:")
    print("  " + ", ".join(AI_COURSE) + "\n")

    # 1) EVERY domain, scored. Shows the full ranking so you see the separation
    #    between the winning domain and the tail.
    print("=" * 70)
    print("route_all()  —  cosine of the course centroid vs EVERY domain centroid")
    print("=" * 70)
    for d in route.route_all(AI_COURSE):
        bar = "█" * int(d["score"] * 40) if d["score"] > 0 else ""
        print(f"  {d['score']:+.3f}  {d['name']:<28} {bar}")

    # 2) The clean top-k answer with a threshold — what the API returns.
    print("\n" + "=" * 70)
    print("route_course(top_k=3, threshold=0.25)  —  the routed answer")
    print("=" * 70)
    print(json.dumps(route.route_course(AI_COURSE, top_k=3, threshold=0.25),
                     indent=2))


if __name__ == "__main__":
    main()
