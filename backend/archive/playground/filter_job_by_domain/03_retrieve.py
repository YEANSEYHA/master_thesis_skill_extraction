"""A2 Step 3 — retrieve: roles.json (Step 1) × title_emb.npz (Step 2) → shortlist."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

FLOOR = 0.70     # min cosine for a title to enter the shortlist
TOP_N = 40       # cap per domain — keeps Step 4's LLM bill tiny

MODEL = "TechWolf/JobBERT-v3"    # MUST be the same model as Step 2!


# INPUT 1 — roles.json                    INPUT 2 — title bank (10,071 titles)
# ─────────────────────                   ─────────────────────────────────────
# "Network": [                            "network engineer"
#   "network engineer",                   "senior network administrator"
#   "network administrator",              "data scientist"
#   "noc technician", ...                 "restaurant manager"
# ]                                       "html engineer" ...

#             │                                   │
#             └───────── compare every ───────────┘
#                     role × title (cosine)

#                  score = best match with ANY role
#                  ┌─────────────────────────────────────┐
#                  │ "network engineer"            0.97  │
#                  │ "senior network administrator" 0.88 │
#                  │ "noc technician l2"           0.81  │
#                  │ "systems & network admin"     0.76  │
#                  │ ──────── FLOOR 0.70 ✂ ───────────── │
#                  │ "it support specialist"       0.64  │  dropped
#                  │ "data scientist"              0.41  │  dropped
#                  │ "restaurant manager"          0.12  │  dropped
#                  └─────────────────────────────────────┘
#                                 │
#                                 ▼  keep top 40 above the line

# OUTPUT — shortlist.json
# ───────────────────────
# "Network": [
#   ["network engineer",             0.97],
#   ["senior network administrator", 0.88],
#   ["noc technician l2",            0.81],
#   ["systems & network admin",      0.76]
# ]


def main():
    roles = json.loads((OUT / "roles.json").read_text())
    z = np.load(OUT / "title_emb.npz", allow_pickle=True)
    titles, V = list(z["titles"]), z["V"]
    model = SentenceTransformer(MODEL)

    shortlist = {}
    for domain, role_list in roles.items():
        # queries = the domain name itself + its HyDE roles
        R = model.encode([domain] + role_list, normalize_embeddings=True)
        sims = V @ R.T                 # (10,071 titles × n queries) in one shot
        best = sims.max(axis=1)        # title matches if close to ANY query
        order = np.argsort(-best)

        kept = [(titles[i], round(float(best[i]), 3))
                for i in order[:TOP_N] if best[i] >= FLOOR]
        shortlist[domain] = kept

        print(f"\n■ {domain} — {len(kept)} candidates ≥ {FLOOR}")
        for t, s in kept[:15]:
            print(f"   {s:.3f}  {t}")
        print(f"   ─── the floor ({FLOOR}) cuts here — first 3 below it: ───")
        for i in [i for i in order if best[i] < FLOOR][:3]:
            print(f"   {best[i]:.3f}  {titles[i]}   ✂")

    (OUT / "shortlist.json").write_text(json.dumps(shortlist, indent=2))
    print("\nsaved output/shortlist.json   (Step 4 verifies these)")


if __name__ == "__main__":
    main()
