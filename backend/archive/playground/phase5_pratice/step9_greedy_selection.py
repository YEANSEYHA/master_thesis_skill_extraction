"""Step 9 — greedy selection: which gaps actually earn a course slot.

Production stage 11 of 13.  `pipeline.greedy_skill_selection()` plus the
CUM_FRAC cut in `gen_report_data.main()`. This produces the dashboard's
"10 / 14 gaps worth adding" card, and it is the only real OPTIMISATION in
the pipeline — everything else is measurement.

THE PROBLEM. Step 8 ranked 14 candidates by priority. But a curriculum
cannot add 14 skills, and the ranking is greedy-blind: two candidates may
be near-duplicates, so adding both buys little more than adding one.
Which subset gives the most coverage per slot?

THE OBJECTIVE — facility location:

    f(S) = sum_i  demand_i * max( coverage_i , max_{s in S} sim(s, i) )

Read it as: after adding the set S, every market skill i is covered by
whichever is better — what the curriculum already had, or the closest
newly-added skill. Weight by demand and sum over the WHOLE hard segment,
not just the top-30, so a new skill also gets credit for the tail skills
it drags along ("spillover").

WHY GREEDY IS ENOUGH. f is monotone submodular (adding a skill never
hurts, and helps less the more you already have), so picking the biggest
marginal gain each round is provably within (1 - 1/e) ~ 63% of the exact
optimum — Nemhauser, Wolsey & Fisher (1978). The engine's own
eval_selection.py checks the achieved plan against a brute-force optimum.

WHERE IT STOPS. Marginal gains are non-increasing, so the plan is a
prefix: keep adding until CUM_FRAC (80%) of the total achievable gain is
banked. The rest cost a slot each for very little.

Run:  python3 step9_greedy_selection.py
"""

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)
from step5_course_grid import course_grid
from step6_coverage_status import coverage_metrics


def run_selection(p5, mkt, hard_idx, top_idx, top, statuses, profile,
                  cum_frac=0.80):
    """Greedy-order the non-covered skills and mark the 80% prefix."""
    demand = mkt["demand_dist"][hard_idx]
    demand = demand / demand.sum()

    cand_idx = [i for i, st in zip(top_idx, statuses) if st != "covered"]
    cand_names = [mkt["skill_base"][i] for i in cand_idx]

    # "adding skill s" is modelled as a new outcome whose embedding IS s,
    # so similar market skills gain coverage through the same max-rule
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cand_sim = np.clip(mkt["skill_emb"][cand_idx]
                           @ mkt["skill_emb"][hard_idx].T, 0.0, None)

    steps = p5.greedy_skill_selection(demand, profile, cand_names, cand_sim)

    total_gain = sum(s["gain"] for s in steps)
    gap0 = float((demand * (1.0 - profile)).sum())
    pos = {g: p for p, g in enumerate(hard_idx)}
    top_pos = [pos[i] for i in top_idx]

    out, cum, prev = [], 0.0, profile
    for st in steps:
        selected = cum < cum_frac * total_gain
        cum += st["gain"]
        prof = st["profile"]
        kl_after, _ = p5.kl_divergence(demand, prof / prof.sum())
        covered_after = int(sum(prof[p] >= p5.get_thresholds()["covered"]
                                for p in top_pos))
        # which OTHER top-30 skills did this addition drag upward?
        spill = [top[k] for k, p in enumerate(top_pos)
                 if top[k] != st["skill"] and float(prof[p] - prev[p]) >= 0.05]
        out.append({"skill": st["skill"], "gain": st["gain"],
                    "gain_pct": st["gain"] / gap0 * 100,
                    "selected": selected, "kl_after": kl_after,
                    "covered_after": covered_after, "spillover": spill})
        prev = prof
    return out, total_gain, gap0


def main() -> None:
    banner("STEP 9", "greedy selection — which gaps earn a course slot",
           "the ordered plan behind the '10 / 14' dashboard card")

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
    m = coverage_metrics(p5, top, program, counts)

    profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)
    plan, total_gain, gap0 = run_selection(
        p5, mkt, hard_idx, top_idx, top, m["status"], profile)

    n_sel = sum(1 for e in plan if e["selected"])
    print(f"candidates : {len(plan)}  ({m['gap']} gap + {m['borderline']} "
          f"borderline — borderlines compete too)")
    print(f"selected   : {n_sel}   <- the dashboard's "
          f"'{n_sel} / {len(plan)}' card")
    print(f"stop rule  : 80% of the total achievable gain\n")

    print(f"{'#':>3s} {'sel':>4s}  {'skill':30s}{'closes':>8s}"
          f"{'KL after':>10s}{'covered':>9s}  spillover")
    print("-" * 92)
    for n, e in enumerate(plan, 1):
        mark = " [x]" if e["selected"] else "  . "
        spill = ", ".join(e["spillover"][:2]) if e["spillover"] else "—"
        print(f"{n:3d}{mark}  {e['skill'][:30]:30s}{e['gain_pct']:7.1f}%"
              f"{e['kl_after']:10.4f}{e['covered_after']:9d}  {spill}")

    print("\nnote how the 'closes' column falls monotonically — that is")
    print("submodularity, and it is why the 80% cut is a clean PREFIX of")
    print("the list rather than a scattered subset.")

    first, last_sel = plan[0], plan[n_sel - 1]
    tail = plan[n_sel:]
    print(f"\n   #1 {first['skill']} alone closes {first['gain_pct']:.1f}% "
          f"of the demand-weighted gap.")
    print(f"   the {n_sel} selected close "
          f"{sum(e['gain_pct'] for e in plan[:n_sel]):.1f}% between them;")
    if tail:
        print(f"   the {len(tail)} rejected would add only "
              f"{sum(e['gain_pct'] for e in tail):.1f}% for "
              f"{len(tail)} more course slots.")

    print(f"\nprojected end state if the whole plan were taught:")
    print(f"   covered {m['covered']} -> {plan[-1]['covered_after']} of 30, "
          f"KL {plan[0]['kl_after']:.4f} -> {plan[-1]['kl_after']:.4f}")
    print("   ^ MODELLED, never achieved. It assumes each added skill is")
    print("     taught perfectly (its own embedding as the new outcome), so")
    print("     it is an UPPER BOUND. Always label it 'projected'.")

    spillers = [e for e in plan if e["spillover"]]
    if spillers:
        e = spillers[0]
        print(f"\nspillover, e.g. adding '{e['skill']}' also lifts "
              f"{', '.join(e['spillover'][:3])}")
        print("   ^ this is why greedy beats the step-8 priority ranking:")
        print("     it credits a skill for everything it drags along, so it")
        print("     never spends two slots on near-duplicates.")


if __name__ == "__main__":
    main()
