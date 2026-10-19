"""Step 2 — drop soft skills, take the top-30.

Production stage 2 of 13.  These two lines at the top of
`gen_report_data.main()`:

    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top      = sorted(hard_idx, key=lambda i: -counts[base[i]])[:TOP_N]

WHY drop soft skills. Teamwork and communication are taught by teaching
METHOD (group work, presentations), not by adding course content — so a
"gap" in teamwork is not actionable as a curriculum change. Leaving them
in would also push real technical skills off the top-30, because in this
corpus soft skills take 4 of the top 6 places.

NOTE the order: filter FIRST, then take the top. Taking the top-30 and
filtering afterwards would leave fewer than 30 technical rows.

Two slices come out of this step and they are NOT the same:
    top-30  -> the heatmap rows; coverage, gaps, recommendations
    all hard -> what KL reads (the long tail included)

Run:  python3 step2_top_skills.py
"""

from phase5_pratice import banner, engine, market


def split_skills(p5, mkt, top_n=None):
    """Return (hard, soft, top) — the engine's own is_soft_skill decides."""
    top_n = top_n or 30
    base, counts = mkt["skill_base"], mkt["counts"]
    hard = [s for s in base if not p5.is_soft_skill(s)]
    soft = [s for s in base if p5.is_soft_skill(s)]
    top = sorted(hard, key=lambda s: -counts[s])[:top_n]
    return hard, soft, top


def main() -> None:
    banner("STEP 2", "drop soft skills, take the top-30",
           "the 30 rows of the heatmap + the hard-skill segment KL reads")

    p5 = engine()
    mkt = market()
    counts = mkt["counts"]
    hard, soft, top = split_skills(p5, mkt)

    total = sum(counts[s] for s in mkt["skill_base"])
    soft_demand = sum(counts[s] for s in soft)
    hard_demand = total - soft_demand
    top_demand = sum(counts[s] for s in top)

    print(f"corpus                 : {mkt['n_jobs']} postings")
    print(f"vocabulary (step 1)    : {len(mkt['skill_base'])} skills")
    print(f"soft skills removed    : {len(soft)}"
          f"  ({soft_demand / total:.1%} of all demand)")
    print(f"hard-skill segment     : {len(hard)}   <- what KL reads")
    print(f"top-30 (heatmap rows)  : {len(top)}\n")

    print("the 30 rows, most-demanded first:")
    for rank, s in enumerate(top, 1):
        print(f"   {rank:2d}. {counts[s]:4d} jobs  {s}")

    print(f"\nthose {len(top)} rows carry {top_demand / hard_demand:.1%} of "
          f"technical demand;")
    print(f"the remaining {len(hard) - len(top)} hard skills carry "
          f"{1 - top_demand / hard_demand:.1%}.")
    print("   ^ remember this split — it is why KL saturates. A curriculum")
    print("     can only move the 30 rows, but KL is scored over all "
          f"{len(hard)}.")

    print("\nremoved soft skills, most-demanded first:")
    for s in sorted(soft, key=lambda s: -counts[s])[:8]:
        print(f"   {counts[s]:4d} jobs  {s}")
    print("   ^ these outrank every technical skill in this corpus, which is")
    print("     the 'soft-skill-skewed' limitation noted in the thesis.")


if __name__ == "__main__":
    main()
