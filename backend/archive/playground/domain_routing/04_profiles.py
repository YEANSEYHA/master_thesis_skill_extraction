
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"   # generated results live here
OUT.mkdir(exist_ok=True)

TOP_N = 30    # how many top skills define a profile



def build_profiles():
    # the postings: [(title, [skills])]  — the raw skill source
    slice_ = json.loads((OUT / "it_slice.json").read_text())
    posts = slice_["posts"]

    # the clean cohorts: {domain: [titles]}  — after LLM verify
    verified = json.loads((OUT / "verified.json").read_text())

    # invert: title -> its domain  (only verified titles map to a domain)
    title2dom = {t: d for d, titles in verified.items() for t in titles}

    # walk every posting; if its title belongs to a domain, pool its skills
    counts = {d: Counter() for d in verified}
    n_post = {d: 0 for d in verified}
    for title, skills in posts:
        d = title2dom.get(title)
        if d is None:                 # title wasn't in any verified cohort
            continue
        n_post[d] += 1
        counts[d].update(skills)

    # top-30 skills = the profile
    profiles = {}
    for d in verified:
        top = counts[d].most_common(TOP_N)
        profiles[d] = {
            "n_postings": n_post[d],
            "n_titles": len(verified[d]),
            "profile": [s for s, _ in top],
            "skill_freq": {s: c for s, c in top},
        }
    return profiles

# The goal of Stage 4
# For each domain: count how often every skill appears across its postings, then keep the top 30.


# count skill frequency  →  sort by count  →  keep top 30  =  the PROFILE

if __name__ == "__main__":
    profiles = build_profiles()
    for d, p in profiles.items():
        print(f"\n■ {d}  ({p['n_postings']} postings, {p['n_titles']} titles)")
        print("   " + ", ".join(p["profile"][:12]) + " …")

    (OUT / "domain_profiles.playground.json").write_text(json.dumps(profiles, indent=1))
    print("\nsaved output/domain_profiles.playground.json  (Stage 5 routes against this)")