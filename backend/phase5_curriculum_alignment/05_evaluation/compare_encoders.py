# Encoder stability experiment (thesis fix #6): rerun the full pipeline
# with a modern embedding model and compare against the MiniLM baseline.
# Raw cosine scales differ between encoders, so the comparison reports
# ORDERINGS (rank correlation, top-N overlap, agreement), not thresholds.
#
# Run from the project root:
#   python3 05_evaluation/compare_encoders.py

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import (
    BASE_DIR, BASELINE_MODEL, MODEL_NAME, build_market, is_soft_skill,
    load_model, kl_divergence,
)

CHALLENGER = MODEL_NAME          # active encoder (BGE)
BASELINE = BASELINE_MODEL        # MiniLM reference
CLO_DIR = BASE_DIR / "02_input_curriculum" / "clo_skills"
OUT = Path(__file__).resolve().parent / "encoder_comparison.json"
TOP_N = 25


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def run_pipeline(model_name: str) -> dict:
    print(f"\n##### encoder: {model_name}")
    model = load_model(model_name)
    market = build_market(model, model_name)

    clo_texts, clo_owner = [], []
    for f in sorted(CLO_DIR.glob("*.json")):
        name = f.stem.split("_", 1)[1].replace("_", " ").title()
        for item in json.loads(f.read_text()):
            clo_texts.append(item["skill"])
            clo_owner.append(name)

    clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                           show_progress_bar=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = market["skill_emb"] @ clo_emb.T

    base, counts = market["skill_base"], market["counts"]
    hard = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top = sorted(hard, key=lambda i: -counts[base[i]])[:TOP_N]

    coverage = {base[i]: float(sim[i].max()) for i in top}
    best_clo = {base[i]: clo_texts[int(sim[i].argmax())] for i in top}

    demand = market["demand_dist"][hard]
    demand = demand / demand.sum()
    profile = np.clip(sim[hard].max(axis=1), 0.0, None)
    kl, _ = kl_divergence(demand, profile / profile.sum())

    # priority ranking of non-obviously-covered skills (order only)
    share = {base[i]: counts[base[i]] / market["n_jobs"] for i in top}
    priority = {s: share[s] * (1.0 - coverage[s]) for s in coverage}
    ranking = sorted(priority, key=lambda s: -priority[s])

    return {"model": model_name, "n_jobs": market["n_jobs"],
            "skill_base_size": len(base), "coverage": coverage,
            "best_clo": best_clo, "kl": kl, "gap_ranking": ranking}


def main() -> None:
    a = run_pipeline(BASELINE)         # baseline (MiniLM)
    b = run_pipeline(CHALLENGER)       # active encoder

    shared = sorted(set(a["coverage"]) & set(b["coverage"]))
    cov_a = np.array([a["coverage"][s] for s in shared])
    cov_b = np.array([b["coverage"][s] for s in shared])
    rho = spearman(cov_a, cov_b)
    overlap = len(shared) / TOP_N
    same_clo = sum(a["best_clo"].get(s) == b["best_clo"].get(s)
                   for s in shared) / len(shared)
    top5_a, top5_b = a["gap_ranking"][:5], b["gap_ranking"][:5]

    print("\n=== STABILITY TABLE (baseline vs challenger) ===")
    print(f"top-{TOP_N} demanded-skill overlap : {overlap:.0%}")
    print(f"coverage Spearman rho (shared)  : {rho:.3f}")
    print(f"same best-matching CLO          : {same_clo:.0%}")
    print(f"top-5 gap ranking baseline      : {top5_a}")
    print(f"top-5 gap ranking challenger    : {top5_b}")
    print(f"KL baseline={a['kl']:.4f}  challenger={b['kl']:.4f} "
          f"(not directly comparable — different score scales)")

    print(f"\n{'skill':<26} {'MiniLM':>7} {'bge':>7}  same best CLO?")
    for s in sorted(shared, key=lambda s: -a['coverage'][s]):
        mark = "yes" if a["best_clo"][s] == b["best_clo"][s] else "NO"
        print(f"{s:<26} {a['coverage'][s]:>7.2f} {b['coverage'][s]:>7.2f}  "
              f"{mark}")

    OUT.write_text(json.dumps({
        "baseline": a, "challenger": b,
        "stability": {"top_overlap": overlap, "coverage_spearman": rho,
                      "same_best_clo": same_clo},
    }, indent=1))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
