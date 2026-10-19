"""Step 5 — collapse 226 outcome columns into 8 course columns.

Production stage 7 of 13.  The per_course / program_cov block inside the
skills loop of `gen_report_data.main()`:

    for c in courses:
        idx = [k for k, o in enumerate(clo_owner) if o == c]
        per_course[c] = round(float(row[idx].max()), 2)

The 619 x 226 matrix is too big to look at; a human needs 30 x 8. Two
levels of MAX get us there, and each answers a different question:

    max over ONE course's columns  ->  "does THIS course teach it?"
    max over ALL courses           ->  "does the PROGRAM teach it?"

Both are MAX, not mean, for the same reason as step 4: a course teaches a
skill if ANY of its outcomes teaches it. A course with 55 outcomes must
not be punished for the 54 that are about something else.

NOTE what is and is not new here. The PROGRAM column is just the row max
of `sim`, which step 4 already computed as the coverage profile — nothing
new. The 8 per-course cells ARE new, and they are what lets a gap be
routed to a host course later. Step 4 measures; step 5 organises.

Run:  python3 step5_course_grid.py
"""

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)


def course_grid(sim, base, top_idx, clo_owner, course_names):
    """Return (grid, program).

    grid[r, c] = best evidence for top skill r inside course c
    program[r] = best evidence anywhere in the programme
    """
    cols = {c: [k for k, o in enumerate(clo_owner) if o == c]
            for c in course_names}
    grid = np.zeros((len(top_idx), len(course_names)))
    for r, i in enumerate(top_idx):
        row = sim[i]
        for c, course in enumerate(course_names):
            grid[r, c] = row[cols[course]].max()
    return grid, grid.max(axis=1)


def _abbrev(name: str, width: int = 7) -> str:
    """'Introduction to Artificial Intelligence' -> 'ITAI'."""
    parts = [p for p in name.split() if p[:1].isupper() or len(p) > 3]
    tag = "".join(p[0] for p in parts).upper() if len(parts) > 1 else name
    return tag[:width]


def main() -> None:
    banner("STEP 5", "collapse 226 outcome columns into 8 course columns",
           "the 30 x 8 grid that IS the heatmap")

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
    print(f"{sim.shape[1]} outcome columns -> {len(course_names)} course "
          f"columns, for the top {len(top)} skills\n")

    header = "".join(f"{_abbrev(c):>8s}" for c in course_names)
    print(f"{'market skill':28s}{'jobs':>5s}{header}{'PROGRAM':>9s}")
    print("-" * (28 + 5 + 8 * len(course_names) + 9))
    for r, skill in enumerate(top[:12]):
        cells = "".join(f"{grid[r, c]:8.2f}" for c in range(len(course_names)))
        print(f"{skill[:28]:28s}{counts[skill]:5d}{cells}{program[r]:9.2f}")
    print(f"... ({len(top) - 12} more rows)\n")
    print("columns: " + ",  ".join(f"{_abbrev(c)} = {c}" for c in course_names))

    print("\nwhy the per-course split matters — the PROGRAM number alone")
    print("cannot tell you WHICH course to change:")
    for r, skill in enumerate(top):
        if program[r] < 0.73:
            continue
        c = int(grid[r].argmax())
        runner_up = float(np.sort(grid[r])[-2])
        print(f"   {skill:30s} {program[r]:.2f} lives in "
              f"{course_names[c]:40s} (next best {runner_up:.2f})")

    print("\n   ^ each covered skill traces to ONE course, with a big drop to")
    print("     the runner-up. That gap is the signal the match is real and")
    print("     localised rather than diffuse noise — and it is what makes")
    print("     routing a gap to a host course possible in step 8.")

    flat = [(s, grid[r].max() - grid[r].min()) for r, s in enumerate(top)]
    worst = min(flat, key=lambda x: x[1])
    print(f"\n   ^ contrast '{worst[0]}': spread of only {worst[1]:.2f} across")
    print("     all 8 courses — no peak anywhere. THAT is what a genuine gap")
    print("     looks like at cell level: not one low number, but the absence")
    print("     of any high one.")


if __name__ == "__main__":
    main()
