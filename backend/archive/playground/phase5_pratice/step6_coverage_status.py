"""Step 6 — thresholds, status, and the headline metric.

Production stages 8 and 9 of 13.  Two things happen here:

    status = ("covered"    if program >= 0.73 else
              "borderline" if program >= 0.69 else "gap")

    demand_weighted_coverage = jobs[covered] / jobs[top_n]

This is the ONLY step in the pipeline that decides anything. Steps 1-5
measure and organise; they produce continuous numbers with no opinion
attached. Step 6 turns them into verdicts, and every claim in the thesis
starts here.

Note where 0.73 comes from: NOT from the data. It is external knowledge
imported into the pipeline — anchor-mapped from the calibrated MiniLM
thresholds onto BGE's scale, then validated against LLM-judge silver
labels. That is why this step is the one marked PROVISIONAL until the
human annotation lands, and why a change of threshold re-derives every
classification without re-running anything upstream.

Two numbers come out, and the gap between them is diagnostic:

    coverage_rate     = covered / 30           "how many boxes ticked"
    demand-weighted   = jobs[covered] / jobs[top-30]
                                               "how much of what employers
                                                actually ask for"

Run:  python3 step6_coverage_status.py
"""

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)
from step5_course_grid import course_grid


def classify(p5, program_score: float) -> str:
    """One skill's status from its best evidence anywhere in the programme."""
    t = p5.get_thresholds()
    if program_score >= t["covered"]:
        return "covered"
    if program_score >= t["borderline"]:
        return "borderline"
    return "gap"


def coverage_metrics(p5, top, program, counts) -> dict:
    """Statuses + both coverage numbers + the half-credit sensitivity bound."""
    status = [classify(p5, p) for p in program]
    top_jobs = sum(counts[s] for s in top)
    covered_jobs = sum(counts[s] for s, st in zip(top, status)
                       if st == "covered")
    borderline_jobs = sum(counts[s] for s, st in zip(top, status)
                          if st == "borderline")
    n_cov = status.count("covered")
    return {
        "status": status,
        "covered": n_cov,
        "borderline": status.count("borderline"),
        "gap": status.count("gap"),
        "coverage_rate": n_cov / len(top),
        "demand_weighted": covered_jobs / top_jobs,
        "demand_weighted_half": (covered_jobs + 0.5 * borderline_jobs)
        / top_jobs,
        "covered_jobs": covered_jobs,
        "top_jobs": top_jobs,
    }


def main() -> None:
    banner("STEP 6", "thresholds, status, and the headline metric",
           "covered / borderline / gap + demand-weighted coverage")

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
    _, program = course_grid(sim, base, top_idx, clo_owner, course_names)

    t = p5.get_thresholds()
    m = coverage_metrics(p5, top, program, counts)

    print(f"thresholds (PROVISIONAL): covered >= {t['covered']}, "
          f"borderline >= {t['borderline']}\n")

    mark = {"covered": "*", "borderline": "~", "gap": "!"}
    print(f"{'market skill':30s}{'jobs':>5s}{'best':>7s}  status")
    print("-" * 56)
    for skill, p, st in zip(top, program, m["status"]):
        print(f"{skill[:30]:30s}{counts[skill]:5d}{p:7.2f}  "
              f"{mark[st]} {st}")

    print(f"\ncovered {m['covered']} / borderline {m['borderline']} / "
          f"gap {m['gap']}\n")
    print(f"  coverage_rate             {m['coverage_rate']:6.1%}"
          f"   ({m['covered']} of {len(top)} skills — supporting stat)")
    print(f"  demand-weighted coverage  {m['demand_weighted']:6.1%}"
          f"   <- THE HEADLINE  ({m['covered_jobs']} of {m['top_jobs']} "
          f"job-mentions)")
    print(f"  ... borderline at half credit "
          f"{m['demand_weighted_half']:6.1%}   (sensitivity bound)")

    # what the two numbers disagreeing actually tells you
    cov = [s for s, st in zip(top, m["status"]) if st == "covered"]
    avg_cov = sum(counts[s] for s in cov) / len(cov)
    avg_all = sum(counts[s] for s in top) / len(top)
    direction = ("HIGHER" if m["demand_weighted"] > m["coverage_rate"]
                 else "LOWER")
    print(f"\ncovered skills average {avg_cov:.1f} jobs vs {avg_all:.1f} "
          f"across all {len(top)} rows,")
    print(f"so the weighted number reads {direction} than the plain count.")
    print("   ^ the DIRECTION is the diagnostic. Weighted above count means")
    print("     the programme covers the big-demand skills; below would mean")
    print("     it ticks boxes that employers rarely ask for.")

    print("\nthe band boundaries, live — DISPLAY vs DECISION:")
    print(f"   {'skill':30s}{'shown':>8s}{'raw':>10s}  status")
    for skill, p, st in zip(top, program, m["status"]):
        if abs(p - t["borderline"]) < 0.011 or abs(p - t["covered"]) < 0.011:
            print(f"   {skill:30s}{p:8.2f}{p:10.4f}  {st}")
    print(f"\n   ^ borderline is {t['borderline']} <= x < {t['covered']} "
          f"(STRICT), and the bar is tested")
    print("     against the RAW float, never the rounded one. That is why")
    print("     'process improvement' shows 0.73 but is borderline: its raw")
    print("     score is just under the bar and the 2-decimal display rounds")
    print("     it up. Production does the same — per_course is rounded for")
    print("     the heatmap, program_cov is kept raw for the status.")
    print("     A reader checking the figure by eye CANNOT reproduce every")
    print("     status from the printed cells; the stored raw scores are")
    print("     the authority (and are what a threshold change re-derives).")


if __name__ == "__main__":
    main()
