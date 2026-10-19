"""Step 10 — draw the figure.

Production stage 12 of 13.  `04_report/heatmap.py draw()`, called by
`_write_heatmap()` at the end of every run so each archived run ships its
own figure.

Nothing is computed here. The figure renders three earlier results in one
frame — that is the entire job:

    cell colour + number   = the step-5 grid (cosine evidence)
    row order + job counts = the step-1/2 demand ranking
    the ringed cell        = each row's max, i.e. the PROGRAM column
                             that step 6 thresholds into a status

The practice does NOT reimplement the drawing: it assembles the minimal
report dict that production's build_matrix()/draw() expect, then calls
the REAL heatmap.py. If this PNG matches the one in outputs/<run_id>/,
every step of the practice matched production, end to end.

Run:  python3 step10_draw_heatmap.py     -> writes practice_heatmap.png
"""

import importlib.util
import sys

from phase5_pratice import (HERE, PHASE5, banner, curriculum_units, engine,
                            market, model, read_curriculum)
from step5_course_grid import course_grid
from step6_coverage_status import coverage_metrics

OUT = HERE / "practice_heatmap.png"


def load_heatmap_module():
    """Load production's 04_report/heatmap.py by file path (same reason
    as engine(): a numbered folder can never be an importable package)."""
    path = PHASE5 / "04_report" / "heatmap.py"
    spec = importlib.util.spec_from_file_location("p5_heatmap", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p5_heatmap"] = mod
    spec.loader.exec_module(mod)
    return mod


def assemble_report(p5, top, program, grid, counts, course_names,
                    kl, covered) -> dict:
    """The minimal report dict heatmap.build_matrix()/draw() read:
    courses, thresholds, per-skill rows, and the title-line stats
    (kl / covered / top_n / source / cohort)."""
    skills = []
    for r, skill in enumerate(top):
        skills.append({
            "skill": skill,
            "jobs": counts[skill],
            "per_course": {c: round(float(grid[r, ci]), 2)
                           for ci, c in enumerate(course_names)},
        })
    return {"courses": course_names, "skills": skills,
            "thresholds": p5.get_thresholds(),
            "kl": kl, "covered": covered, "top_n": len(top),
            "source": p5.JOB_SOURCE, "cohort": "Y5 (practice)"}


def main() -> None:
    banner("STEP 10", "draw the figure",
           "practice_heatmap.png — rendered by PRODUCTION heatmap.py")

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

    # the title-line KL, same as step 4
    import numpy as np
    profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)
    demand = mkt["demand_dist"][hard_idx]
    demand = demand / demand.sum()
    kl, _ = p5.kl_divergence(demand, profile / profile.sum())

    report = assemble_report(p5, top, program, grid, counts, course_names,
                             kl, m["covered"])

    hm = load_heatmap_module()
    out = hm.draw(report, top=30, out=OUT)
    print(f"figure -> {out}")

    print("\nwhat the figure encodes (all computed in earlier steps):")
    print("   colour ramp   white -> amber -> teal maps the cosine bands;")
    print("                 the teal end starts at the covered bar")
    print("   ringed cell   each row's max — the PROGRAM value that step 6")
    print("                 turned into covered/borderline/gap")
    print("   row order     demand rank from steps 1-2 (jobs in the label)")
    print(f"\nsummary it should agree with: covered {m['covered']} / "
          f"borderline {m['borderline']} / gap {m['gap']}, "
          f"demand-weighted {m['demand_weighted']:.1%}")
    print("\ncheck: open practice_heatmap.png next to the frozen run's")
    print("outputs/<run_id>/heatmap.png — matching figures mean the whole")
    print("practice matched production, end to end.")
    print("\n(one thing the PNG lacks vs the dashboard: the cross-encoder")
    print(" badges from step 7 — the static figure is cosine-only.)")


if __name__ == "__main__":
    main()
