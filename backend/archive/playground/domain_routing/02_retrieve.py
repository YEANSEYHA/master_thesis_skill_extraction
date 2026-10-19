import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output"   # generated results live here
OUT.mkdir(exist_ok=True)

FLOOR = 0.45        # min cosine(title, domain anchor) to join at all
MARGIN = 0.05       # must beat the runner-up domain by this much
MAX_TITLES = 500    # cohort cap: top-M titles by cosine

DOMAINS = [
    "artificial_intelligence", "data_science", "software_engineering",
    "cybersecurity", "cloud_devops", "networking", "data_engineering",
    "data_analytics", "qa_testing", "business_analysis", "web_frontend",
    "mobile_development", "embedded_systems", "database_administration",
    "it_operations", "it_support", "project_product_management",
    "blockchain", "ui_ux_design", "game_development",
]

def retrieve_cohorts(titles):
    """{domain: [cohort titles, best-first]} via anchor-cosine + floor/margin."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1) the ANCHOR: the embedding of the domain name itself ("cloud devops")
    anchors = np.asarray(model.encode(
        [d.replace("_", " ") for d in DOMAINS], normalize_embeddings=True))

    # 2) every unique title, embedded once
    T = np.asarray(model.encode(titles, normalize_embeddings=True,
                                batch_size=512, show_progress_bar=True))

    # 3) cosine of every title vs every anchor -> matrix (n_titles x 20)
    sims = T @ anchors.T

    # 4) best + runner-up domain per title
    order = np.argsort(-sims, axis=1)
    best = order[:, 0]
    s1 = sims[np.arange(len(titles)), best]         # best score
    s2 = sims[np.arange(len(titles)), order[:, 1]]  # runner-up score

    # 5) FLOOR + MARGIN -> candidates;  6) CAP -> cohort
    cand = {d: [] for d in DOMAINS}
    for i, t in enumerate(titles):
        if s1[i] >= FLOOR and (s1[i] - s2[i]) >= MARGIN:
            cand[DOMAINS[best[i]]].append((float(s1[i]), t))
    kept = sum(len(v) for v in cand.values())
    print(f"pass-1 candidates: {kept:,} of {len(titles):,} titles "
          f"({len(titles) - kept:,} dropped as unrelated/ambiguous)")

    cohorts = {}
    for d in DOMAINS:
        cand[d].sort(reverse=True)                  # best cosine first
        cohorts[d] = [t for _, t in cand[d][:MAX_TITLES]]
    return cohorts



# 10,071 unique titles
#    │
#    ├──▶  2,400 sorted into the 20 groups   (24%)
#    │
#    └──▶  7,671 DROPPED                      (76%)
#           ├─ failed FLOOR   → not close to any domain ("ct technologist")
#           ├─ failed MARGIN  → torn between 2 domains ("full stack data engineer")
#           └─ failed CAP     → related, but not in the domain's top-500

if __name__ == "__main__":
    titles = json.loads((OUT / "it_slice.json").read_text())["titles"]
    print(f"{len(titles):,} unique titles to sort\n")

    cohorts = retrieve_cohorts(titles)

    print("\ncohort sizes + top-3 titles (best cosine first):")
    for d, ts in cohorts.items():
        print(f"  {d:<28} {len(ts):4}  {ts[:3]}")

    (OUT / "cohorts.json").write_text(json.dumps(cohorts, indent=1))
    print("\nsaved output/cohorts.json  (Stage 3 will audit these)")


    