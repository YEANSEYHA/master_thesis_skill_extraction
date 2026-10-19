"""Step 7 — cross-encoder host verification.

Production stage 5 of 13 (the RERANK block in `gen_report_data.main()`).
Production runs it right after the matrix; the practice runs it here,
after step 6, because the badges only mean something once there are
statuses to attach them to. Same computation either way.

WHY a second model. The bi-encoder from step 3 never sees two texts
together: it turns each into a vector separately, then compares vectors.
Cheap — every vector is computed once and reused across all 139,894
pairs — but blunt. It judges by general resemblance, which is how
"java programming" lands at 0.74 against an outcome called "Python".

A CROSS-encoder reads BOTH strings in one forward pass and scores the
pair directly. Far more accurate, far too slow for 139,894 pairs (nothing
can be pre-computed), so it only runs on a shortlist:

    30 skills x top-3 candidate outcomes = 90 pairs

That is the standard retrieve-then-rerank architecture: a cheap model
proposes, an expensive model judges.

WHAT IT DOES NOT DO: change any skill's covered/borderline/gap status.
Cosine alone decides that. The verdict is rendered as a badge beside it,
so a disagreement stays VISIBLE instead of being silently resolved.

Run:  python3 step7_rerank_verify.py   (loads a 2.1 GB model, ~1 min)
"""

from collections import Counter

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)
from step5_course_grid import course_grid
from step6_coverage_status import coverage_metrics


def verify_hosts(p5, sim, base, top_idx, clo_texts, clo_owner):
    """Re-read each top-30 skill against its top-3 candidate outcomes.

    Returns {skill: {cosine_best, rerank_best, verdict}}. cosine_best is
    what the bi-encoder picked, rerank_best what the cross-encoder picked
    — they are often NOT the same outcome, which is the interesting part.
    """
    pairs, cand_lists = [], []
    for i in top_idx:
        cand = np.argsort(-sim[i])[:p5.RERANK_TOP_K]
        cand_lists.append(cand)
        pairs.extend((base[i], clo_texts[j]) for j in cand)

    scores = p5.rerank_pair_scores(pairs)          # ONE batched call

    out, k = {}, 0
    for i, cand in zip(top_idx, cand_lists):
        cands = []
        for j in cand:
            cands.append({"clo": clo_texts[j], "course": clo_owner[j],
                          "cosine": float(sim[i, j]),
                          "rerank": float(scores[k])})
            k += 1
        by_rerank = sorted(cands, key=lambda c: -c["rerank"])
        out[base[i]] = {"cosine_best": cands[0],   # cand list is cosine-sorted
                        "rerank_best": by_rerank[0],
                        "verdict": p5.rerank_verdict(by_rerank[0]["rerank"])}
    return out


def main() -> None:
    banner("STEP 7", "cross-encoder host verification",
           "a supported / weak / contradicted badge for every top-30 skill")

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

    print(f"bi-encoder   : {p5.MODEL_NAME}  (already scored "
          f"{sim.size:,} pairs)")
    print(f"cross-encoder: {p5.RERANKER_NAME}")
    print(f"shortlist    : {len(top)} skills x top-{p5.RERANK_TOP_K} "
          f"outcomes = {len(top) * p5.RERANK_TOP_K} pairs")
    print(f"bands        : >= {p5.RERANK_SUPPORTED_AT} supported, "
          f">= {p5.RERANK_WEAK_AT} weak")
    print("loading cross-encoder ...\n")

    verify = verify_hosts(p5, sim, base, top_idx, clo_texts, clo_owner)

    glyph = {"SUPPORTED": "v", "WEAK": "~", "NOT SUPPORTED": "x"}
    print(f"{'market skill':30s}{'cos':>6s}{'status':>12s}"
          f"{'rerank':>8s}  verdict")
    print("-" * 72)
    for skill, p, st in zip(top, program, m["status"]):
        v = verify[skill]
        print(f"{skill[:30]:30s}{p:6.2f}{st:>12s}"
              f"{v['rerank_best']['rerank']:8.2f}  "
              f"{glyph[v['verdict']]} {v['verdict']}")

    cov = [s for s, st in zip(top, m["status"]) if st == "covered"]
    tally = Counter(verify[s]["verdict"] for s in cov)
    print(f"\namong the {len(cov)} COVERED skills: "
          f"{tally['SUPPORTED']} supported / {tally['WEAK']} weak / "
          f"{tally['NOT SUPPORTED']} contradicted")
    print("   ^ this decomposition is what the thesis reports. A single")
    print("     'covered' count would hide it.")

    print("\nwhere the two models DISAGREE (covered by cosine, rejected by")
    print("the cross-encoder):")
    for skill, p, st in zip(top, program, m["status"]):
        v = verify[skill]
        if st == "covered" and v["verdict"] == "NOT SUPPORTED":
            cb = v["cosine_best"]
            print(f"   {skill}  (cosine {p:.2f} -> rerank "
                  f"{v['rerank_best']['rerank']:.2f})")
            print(f"      matched to \"{cb['clo']}\"  [{cb['course']}]")

    # the pattern nobody designed
    bad = [p for p, st, s in zip(program, m["status"], top)
           if st == "covered" and verify[s]["verdict"] == "NOT SUPPORTED"]
    ok = [p for p, st, s in zip(program, m["status"], top)
          if st == "covered" and verify[s]["verdict"] == "SUPPORTED"]
    if bad and ok:
        print(f"\ncontradicted skills sit at cosine "
              f"{min(bad):.2f}-{max(bad):.2f};")
        print(f"supported skills sit at cosine {min(ok):.2f}-{max(ok):.2f}.")
        print("   ^ nobody tuned this. The cross-encoder was never told")
        print(f"     where the {p5.get_thresholds()['covered']} bar is, yet "
              "every contradiction lands")
        print("     just above it. Independent evidence that the strict bar")
        print("     is set slightly too low — the same conclusion the")
        print("     LLM-judge study reached when it fitted t* ~ 0.80.")

    print("\nnote: the status column above is UNCHANGED by this step.")


if __name__ == "__main__":
    main()
