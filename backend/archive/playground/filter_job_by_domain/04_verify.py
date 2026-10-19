"""A2 Step 4 — verify: qwen judges each shortlisted title → cohort.json."""

import json
from pathlib import Path

import ollama

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

MODEL = "qwen3:14b"

# INPUT — shortlist.json (Step 3)
# ────────────────────────────────
# "Network": [
#   ["network engineer",            0.97],
#   ["senior network administrator", 0.88],
#   ["noc technician l2",           0.81],
#   ["network cabling installer",   0.74],   ← plausible but wrong
#   ["it support specialist",       0.71]    ← borderline
# ]
#             │
#             ▼   one title at a time
# ┌─────────────────────────────────────────────┐
# │  qwen3:14b                                  │
# │                                             │
# │  "domain: Network                           │
# │   job title: network cabling installer"     │
# │                                             │
# │  → {"is_match": false}     temp=0,          │
# │                            boolean only     │
# └──────────────┬──────────────────────────────┘
#                │  verdict per title
#                ▼
#    ✓ network engineer             true
#    ✓ senior network administrator true
#    ✓ noc technician l2            true
#    ✗ network cabling installer    false  ── cosine liked it (0.74),
#    ✗ it support specialist        false     the LLM read it and said no
#                │
#      ┌─────────┴──────────┐
#      ▼                    ▼
# OUTPUT 1 — cohort.json   OUTPUT 2 — rejected.json
# ────────────────────────  ─────────────────────────
# "Network": [              "Network": [
#   "network engineer",       "network cabling installer",
#   "senior network           "it support specialist"
#      administrator",      ]
#   "noc technician l2"     (audit material — shows the
# ]                          verifier earning its keep)
# (the DYNAMIC seed list —
#  Step 5 filters with this)



def verify(title, domain):
    """One strict yes/no: is this job title a {domain} role? (title only)"""
    resp = ollama.chat(
        model=MODEL, think=False,
        options={"temperature": 0, "num_ctx": 2048},
        format={"type": "object",
                "properties": {"is_match": {"type": "boolean"}},
                "required": ["is_match"]},
        messages=[
            {"role": "system", "content":
             "You judge job titles. Reply ONLY with JSON {\"is_match\": true/false}. "
             "true = the job title is a role that belongs to the given career "
             "domain. Judge by the TITLE alone; a generic role or a role from "
             "another field is false."},
            {"role": "user", "content": f"domain: {domain}\njob title: {title}"}],
    )
    return json.loads(resp.message.content)["is_match"]


if __name__ == "__main__":
    shortlist = json.loads((OUT / "shortlist.json").read_text())
    cohort, rejected = {}, {}

    for domain, cands in shortlist.items():
        print(f"\n■ {domain} — verifying {len(cands)} candidates ...")
        keep, drop = [], []
        for title, score in cands:
            ok = verify(title, domain)
            (keep if ok else drop).append(title)
            print(f"   {'✓' if ok else '✗'}  {score:.3f}  {title}")
        cohort[domain], rejected[domain] = keep, drop
        print(f"   → kept {len(keep)} / rejected {len(drop)}")

    (OUT / "cohort.json").write_text(json.dumps(cohort, indent=2))
    (OUT / "rejected.json").write_text(json.dumps(rejected, indent=2))
    print("\nsaved output/cohort.json + output/rejected.json")
