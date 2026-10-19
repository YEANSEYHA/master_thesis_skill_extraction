"""Step 8 — recommendations: rank every non-covered skill, and route it.

Production stage 10 of 13.  The `recs` / `recommendations` block in
`gen_report_data.main()`.

This is the first PRESCRIPTIVE step. Steps 1-7 describe the curriculum's
state; from here the pipeline says what to change.

Two decisions per skill.

1. HOW URGENT — the priority score:

       priority = share * (1 - coverage)

   `share` is the skill's slice of total demand (jobs / n_jobs), and
   (1 - coverage) is how much of it is still missing. So a skill only
   ranks high when it is BOTH widely demanded AND poorly taught. A
   heavily-demanded skill that is already covered scores ~0; a totally
   untaught skill nobody asks for also scores ~0.

2. WHERE IT GOES — the host, decided by the CROSS-ENCODER, not cosine:

       SUPPORTED -> host = that course   ("extend an existing outcome")
       otherwise -> host = None          ("new content needed")

   This is the one place a rerank verdict changes an OUTPUT rather than
   just adding a badge. Note the asymmetry with step 6: the verdict never
   moves a skill's covered/gap status, but it does decide whether we are
   allowed to name a host course for it. Naming the wrong host is worse
   than naming none — it would send a lecturer to edit a course that has
   nothing to do with the skill.

Run:  python3 step8_recommendations.py   (loads the 2.1 GB cross-encoder)
"""

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)
from step5_course_grid import course_grid
from step6_coverage_status import coverage_metrics
from step7_rerank_verify import verify_hosts


def build_recommendations(p5, top, program, counts, n_jobs, statuses,
                          verify, grid, course_names):
    """Every non-covered skill, most urgent first, with its host or None."""
    host_min = p5.get_thresholds()["host_min"]
    rows = []
    for r, (skill, cov, status) in enumerate(zip(top, program, statuses)):
        if status == "covered":
            continue
        share = counts[skill] / n_jobs
        priority = share * max(0.0, 1.0 - cov)

        v = verify.get(skill)
        if v:                       # cross-encoder governs the host decision
            hosted = v["verdict"] == "SUPPORTED"
            host = v["rerank_best"]["course"] if hosted else None
            host_clo = v["rerank_best"]["clo"] if hosted else None
        else:                       # no verification: fall back to a cosine gate
            hosted = cov >= host_min
            c = int(grid[r].argmax())
            host = course_names[c] if hosted else None
            host_clo = None

        rows.append({"skill": skill, "jobs": counts[skill], "coverage": cov,
                     "status": status, "priority": priority,
                     "host": host, "host_clo": host_clo,
                     "argmax_course": course_names[int(grid[r].argmax())]})
    return sorted(rows, key=lambda x: -x["priority"])


def main() -> None:
    banner("STEP 8", "recommendations — rank the non-covered, route each one",
           "an ordered action list: what to add, and to which course")

    p5 = engine()
    mkt = market()
    base, counts = mkt["skill_base"], mkt["counts"]

    course_names, clo_texts, clo_owner = curriculum_units(read_curriculum())
    clo_emb = model().encode(clo_texts, normalize_embeddings=True,
                             show_progress_bar=False)
    sim = mkt["skill_emb"] @ clo_emb.T

    hard_idx = [i for i, s in enumerate(base) if not p5.is_soft_skill(s)]
    top_idx = sorted(hard_idx, key=lambda i: -counts[base[i]])[:30]
    top = [base[i] for i in top_idx]
    grid, program = course_grid(sim, base, top_idx, clo_owner, course_names)
    m = coverage_metrics(p5, top, program, counts)

    print("loading cross-encoder for the host decision ...\n")
    verify = verify_hosts(p5, sim, base, top_idx, clo_texts, clo_owner)

    recs = build_recommendations(p5, top, program, counts, mkt["n_jobs"],
                                 m["status"], verify, grid, course_names)

    print(f"{len(recs)} non-covered skills "
          f"({m['gap']} gap + {m['borderline']} borderline), most urgent "
          f"first:\n")
    print(f"{'#':>3s}  {'skill':30s}{'jobs':>5s}{'cov':>6s}"
          f"{'priority':>10s}  action")
    print("-" * 96)
    for n, r in enumerate(recs, 1):
        if r["host"]:
            action = f"EXTEND {r['host']}"
        else:
            action = f"NEW content (argmax {r['argmax_course']} unverified)"
        print(f"{n:3d}. {r['skill'][:30]:30s}{r['jobs']:5d}"
              f"{r['coverage']:6.2f}{r['priority']:10.4f}  {action}")

    print("\nread the priority formula on the top row:")
    t = recs[0]
    share = t["jobs"] / mkt["n_jobs"]
    print(f"   {t['skill']}: {t['jobs']}/{mkt['n_jobs']} postings = "
          f"{share:.3f} share")
    print(f"   still missing = 1 - {t['coverage']:.2f} = "
          f"{1 - t['coverage']:.2f}")
    print(f"   priority = {share:.3f} x {1 - t['coverage']:.2f} = "
          f"{t['priority']:.4f}")

    hosted = [r for r in recs if r["host"]]
    print(f"\n{len(hosted)} of {len(recs)} got a named host course; "
          f"{len(recs) - len(hosted)} need NEW content.")
    if hosted:
        print("hosted (a verified outcome already sits nearby):")
        for r in hosted:
            print(f"   {r['skill']:28s} -> {r['host']}  "
                  f"(\"{r['host_clo']}\")")
    print("\n   ^ everything else says 'new content' because the")
    print("     cross-encoder refused to confirm the nearest outcome.")
    print("     Cosine would have named a host for several of them —")
    print("     naming the WRONG course is worse than naming none, since")
    print("     it sends a lecturer to edit an unrelated course.")

    print("\nnote: this list is every non-covered skill. Step 9 decides")
    print("which of them actually earn a course slot.")


if __name__ == "__main__":
    main()
