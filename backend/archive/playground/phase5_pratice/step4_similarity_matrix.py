"""Step 4 — the similarity matrix.

Production stage 4 of 13.  One line in `gen_report_data.main()`:

    sim = market["skill_emb"] @ clo_emb.T

That is the whole step. Both sides are unit vectors (step 3), so the dot
product IS the cosine — no norms, no division. sim[i, j] = how close
market skill i is to course outcome j.

This matrix is the single source for everything after it:

    sim ─┬─▶ step 5   per-course grid  -> the heatmap
         ├─▶ step 6   thresholds       -> covered / borderline / gap
         ├─▶ step 7   rerank shortlist -> top-3 candidates per skill
         └─▶ KL       max per row      -> one diagnostic number

Nothing recomputes similarity after this point.

Run:  python3 step4_similarity_matrix.py
"""

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)


def build_matrix(mkt, clo_emb):
    """sim[i, j] = cosine(market skill i, course outcome j)."""
    return mkt["skill_emb"] @ clo_emb.T


def coverage_profile(p5, sim, hard_idx):
    """Best evidence per market skill: the MAX across all 226 outcomes.

    MAX, not mean: a skill is taught if ONE outcome teaches it. Averaging
    over 226 outcomes would drown every real match in noise.
    """
    return np.clip(sim[hard_idx].max(axis=1), 0.0, None)


def main() -> None:
    banner("STEP 4", "the similarity matrix",
           "one cosine for every (market skill, course outcome) pair")

    p5 = engine()
    mkt = market()
    base, counts = mkt["skill_base"], mkt["counts"]

    _, clo_texts, _ = curriculum_units(read_curriculum())
    clo_emb = model().encode(clo_texts, normalize_embeddings=True,
                             show_progress_bar=False)

    sim = build_matrix(mkt, clo_emb)
    print(f"sim = skill_emb @ clo_emb.T")
    print(f"    = {sim.shape[0]} market skills x {sim.shape[1]} outcomes"
          f" = {sim.size:,} cosine values\n")

    # the hard-skill segment (step 2) is applied here as an INDEX, not by
    # re-embedding — same vectors, a different view of them
    hard_idx = [i for i, s in enumerate(base) if not p5.is_soft_skill(s)]
    top = sorted(hard_idx, key=lambda i: -counts[base[i]])[:30]

    profile = coverage_profile(p5, sim, hard_idx)
    demand = mkt["demand_dist"][hard_idx]
    demand = demand / demand.sum()
    kl, contrib = p5.kl_divergence(demand, profile / profile.sum())

    print(f"KL over all {len(hard_idx)} hard skills : {kl:.4f}")
    print(f"   for scale: a curriculum teaching NOTHING scores ~0.3988,")
    print(f"   so this sits {(0.3988 - kl) / 0.3988:.1%} below the null.\n")

    top_set = set(top)
    top_prof = profile[[hard_idx.index(i) for i in top]]
    tail_prof = np.array([profile[k] for k, i in enumerate(hard_idx)
                          if i not in top_set])
    print(f"the 30 heatmap rows : best evidence mean {top_prof.mean():.2f}")
    print(f"the {len(tail_prof)} tail rows      : best evidence mean "
          f"{tail_prof.mean():.2f}")
    print("   ^ only ~0.07 apart, and the tail is 95% of what KL reads.")
    print("     That is the flattening: the tail sits at the encoder's")
    print("     noise floor and never moves, whatever the school teaches.\n")

    order = np.argsort(-contrib)[:5]
    print("top-5 KL contributors (its biggest complaints):")
    for k in order:
        skill = base[hard_idx[k]]
        print(f"   {contrib[k]:.4f}  {skill:32s} {counts[skill]:3d} jobs,"
              f" best evidence {profile[k]:.2f}")

    print("\n   ^ look at 'database management': best evidence 0.91, the")
    print("     BEST-COVERED skill in the programme — and KL still lists it")
    print("     as a top complaint. KL is not asking 'is this taught?'; it")
    print("     asks 'does this skill's SHARE of teaching-evidence equal its")
    print("     SHARE of market demand?'. Different question. That is why")
    print("     demand-weighted coverage (step 6) is the headline metric and")
    print("     KL is kept only as a diagnostic.")


if __name__ == "__main__":
    main()
