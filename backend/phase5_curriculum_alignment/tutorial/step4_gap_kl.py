"""LESSON 4 (short) -- KL divergence.

WHAT IT IS
  One number for how differently two distributions are SHAPED --
  here: does the curriculum emphasize skills the way the market
  demands them?  0 = same shape; bigger = more mismatch.
  (Real thesis headline: KL = 0.6952.)

  Formula sum q*log(q/p) punishes hardest where q is big but p is
  small: market wants it a lot, curriculum barely offers it.

SYNTAX
  kl, contrib = kl_divergence(demand, profile / profile.sum())
  # q = demand (market shares, sums to 1)
  # p = profile normalized into shares  <- must also sum to 1
  # contrib = each skill's piece of the total (+ or -)

Run:  python3 tutorial/step4_gap_kl.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import is_soft_skill, kl_divergence  # noqa: E402
from tiny_data import embed_and_similarity  # noqa: E402

# ---- setup (given) ----------------------------------------------------
names, counts, clo_names, skill_emb, clo_emb, sim = embed_and_similarity()
hard = [i for i, n in enumerate(names) if not is_soft_skill(n)]
hnames = [names[i] for i in hard]
demand = counts[hard] / counts[hard].sum()            # q: market shares
profile = np.clip(sim[hard].max(axis=1), 0.0, None)   # coverage per skill
print(f"setup: {len(hard)} technical skills, "
      f"demand sums to {demand.sum():.3f}\n")


# PROBLEM: Compute KL and find who causes it
#
# Steps:
#   1. kl, contrib = kl_divergence(demand, profile / profile.sum())
#   2. print(f"KL = {kl:.4f}")
#   3. top-3 contributors:
#        for k in np.argsort(-contrib)[:3]:
#            print(f"{hnames[k]:<18} +{contrib[k]:.4f}")
#   4. bonus (answer as a comment): KL measures SHAPE, the gap sum
#      measures AMOUNT. Which of the two could be ZERO even if every
#      skill were only half-taught?

# Your code below:

kl, contrib = kl_divergence(demand, profile / profile.sum())
print(f"KL = {kl:.4f}")
for k in np.argsort(-contrib)[:3]:
    print(f"{hnames[k]:<18} +{contrib[k]:.4f}")