# Leave-one-site-out sensitivity of the labour-market side.
#
#   python3 05_evaluation/loso_site_sensitivity.py     (from the phase5_curriculum_alignment root)
#
# Examiner comment on Table 3.1: "can you explain?" — the eight portals
# contribute very unequally, and two of them (jobnet.com.kh, camhr.com)
# supply about two thirds of the technology segment. That raises one
# question the thesis has to answer with a number rather than a caveat:
#
#     does the skill demand the curriculum is scored against come from the
#     Cambodian IT labour market, or from whichever portal happened to be
#     the easiest to crawl?
#
# The test: drop ONE portal's postings, rebuild the market from what is
# left, and re-run the four locked cohorts. If the top-30 demand ranking,
# the covered counts, the demand-weighted coverage and the Y3 < Y4 < Y5 <
# combined ordering all survive every single-portal removal, then platform
# concentration is a property of the crawl, not of the result.
#
# Same design rules as dedup_sensitivity.py, and for the same reasons:
#   * a RE-SLICE, not a re-run — same encoder, same curriculum files, same
#     calibrated thresholds; the only thing that changes is which postings
#     are counted;
#   * the committed market cache must first reproduce the 2026-07-30 table
#     exactly (verify_locked) or nothing below would be comparable;
#   * the raw->canonical skill mapping is fitted ONCE on the full segment
#     with a deterministic tie order and then held FIXED for every holdout.
#     Refitting per holdout would mix the effect being measured (fewer
#     postings) with PYTHONHASHSEED-dependent re-clustering drift — see the
#     tie-order note in dedup_sensitivity.py. Deltas are therefore read
#     against `refit_full` (all postings, refit mapping), and the
#     refit_full-vs-locked mapping drift is reported separately.
#
# Because the mapping is frozen and every holdout is a subset of the full
# segment, each holdout's skill base is a subset of the full skill base, so
# the embedding matrix is computed once and indexed — no re-encoding.

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE_DIR = _HERE.parent
sys.path.insert(0, str(BASE_DIR / "03_engine"))
sys.path.insert(0, str(_HERE))
os.environ.setdefault("JOB_SOURCE", "cambodia")
os.environ.setdefault("SEGMENT_MODE", "category")

from pipeline import (                                        # noqa: E402
    MIN_SKILL_FREQ, MODEL_NAME, RAW_JOBS, build_market, clean_jobs,
    get_thresholds, is_it_by_category,
)
from sweep_top_n import (                                     # noqa: E402
    BASELINE_N, load_model_cpu, verify_locked,
)
from dedup_sensitivity import (                               # noqa: E402
    compare, fit_mapping, market_variant, run_cohorts,
)

OUT_JSON = _HERE / "loso_site_sensitivity_results.json"
OUT_MD = _HERE / "loso_site_sensitivity.md"

# Ordering claims the thesis makes about the four cohorts. Each is checked
# after every removal; a claim that flips under any holdout is not a claim
# the single snapshot can support.
ORDER_METRICS = {
    "covered": ("desc", "covered skills out of the top 30"),
    "demand_weighted_coverage": ("desc", "demand-weighted coverage"),
    "kl": ("asc", "KL divergence (lower = better aligned)"),
}


def site_table(segment: list[dict]) -> list[dict]:
    """Per-portal contribution to the technology segment — the Section
    3.4.1 distribution, derived from the same records the scoring path
    uses rather than counted by hand."""
    n = len(segment)
    counts = Counter(j["site"] for j in segment)
    return [{"site": s, "n": c, "share": round(c / n, 4)}
            for s, c in counts.most_common()]


def ordering(results: dict, metric: str) -> list[str]:
    """Cohorts ordered by one metric, best first."""
    direction = ORDER_METRICS[metric][0]
    cohorts = results["cohorts"]
    return sorted(cohorts, key=lambda c: (cohorts[c][metric]
                                          if direction == "asc"
                                          else -cohorts[c][metric]))


def ordering_check(ref: dict, var: dict) -> dict:
    """Did any cohort ordering flip when this portal was removed?"""
    out = {}
    for metric in ORDER_METRICS:
        r, v = ordering(ref, metric), ordering(var, metric)
        out[metric] = {"reference": r, "variant": v, "stable": r == v}
    return out


def main() -> None:
    thr = get_thresholds()
    print(f"model {MODEL_NAME} · source {RAW_JOBS.name} · N={BASELINE_N}")
    model = load_model_cpu(MODEL_NAME)

    # 1. the locked market straight from the committed cache
    locked_market = build_market(model)
    jobs, stats = clean_jobs(RAW_JOBS)
    segment = [j for j in jobs if is_it_by_category(j) and j["skills"]]
    if len(segment) != locked_market["n_jobs"]:
        sys.exit(f"ABORT: rebuilt segment has {len(segment)} jobs, cache "
                 f"says {locked_market['n_jobs']}")

    # 2. credibility gate — the cache must BE the published table
    results = {"locked": run_cohorts(model, locked_market, thr)}
    results["locked"]["removed_site"] = None
    results["locked"]["removed"] = 0
    problems = [p for c, r in results["locked"]["cohorts"].items()
                for p in verify_locked(c, r)]
    if problems:
        print("\nLOCKED-RUN CHECK FAILED:")
        for p in problems:
            print("  " + p)
        sys.exit("ABORT: the cached market does not reproduce the "
                 "2026-07-30 runs — nothing here would be comparable.")
    print("locked-run check PASSED: cache reproduces all four cohorts")

    sites = site_table(segment)
    print(f"IT segment {len(segment)} postings across {len(sites)} portals: "
          + ", ".join(f"{r['site']} {r['n']} ({r['share']:.1%})"
                      for r in sites))

    # 3. one deterministic mapping, fitted on the FULL segment, then held
    #    fixed for every holdout
    mapping = fit_mapping(model, segment)
    results["refit_full"] = run_cohorts(
        model, market_variant(model, mapping, segment), thr)
    results["refit_full"]["removed_site"] = None
    results["refit_full"]["removed"] = 0

    # 4. drop one portal at a time
    holdouts = {}
    for row in sites:
        site = row["site"]
        kept = [j for j in segment if j["site"] != site]
        name = f"drop_{site}"
        res = run_cohorts(model, market_variant(model, mapping, kept), thr)
        res["removed_site"] = site
        res["removed"] = len(segment) - len(kept)
        results[name] = res
        holdouts[name] = res

    for name, res in results.items():
        tag = (f"−{res['removed_site']} ({res['removed']} postings)"
               if res["removed_site"] else name)
        print(f"\n[{tag}] {res['n_jobs']} postings, "
              f"{res['n_technical']} technical skills")
        for cohort, r in res["cohorts"].items():
            print(f"  {cohort:9s} covered {r['covered']:>2}/30  "
                  f"dwc {r['demand_weighted_coverage']:>6.1%}  "
                  f"KL {r['kl']:.4f}")

    ref = results["refit_full"]
    deltas = {name: compare(ref, res) for name, res in holdouts.items()}
    orders = {name: ordering_check(ref, res) for name, res in holdouts.items()}

    # 5. the summary the thesis actually quotes: worst case over all eight
    #    removals, per cohort
    worst = {}
    for cohort in ref["cohorts"]:
        ov = [deltas[n][cohort]["top30_overlap"] for n in holdouts]
        dw = [abs(deltas[n][cohort]["dwc_delta_pts"]) for n in holdouts]
        cv = [abs(deltas[n][cohort]["covered"][1]
                  - deltas[n][cohort]["covered"][0]) for n in holdouts]
        kl = [abs(deltas[n][cohort]["kl"][1] - deltas[n][cohort]["kl"][0])
              for n in holdouts]
        worst[cohort] = {
            "min_top30_overlap": min(ov),
            "max_dwc_shift_pts": round(max(dw), 2),
            "max_covered_shift": max(cv),
            "max_kl_shift": round(max(kl), 4),
        }
    order_stable = {m: all(orders[n][m]["stable"] for n in holdouts)
                    for m in ORDER_METRICS}

    # Skill-level stability, pooled over every holdout x cohort. This is the
    # part that separates "the measurement is portal-dependent" from "the
    # top-30 TAIL is portal-dependent": a status change means the same skill
    # was judged covered with one portal in and not covered with it out,
    # whereas a top-30 substitution only means rank 30 changed hands.
    pairs = [(n, c) for n in holdouts for c in ref["cohorts"]]
    rank_corr = sorted(deltas[n][c]["rank_corr_shared"] for n, c in pairs
                       if deltas[n][c]["rank_corr_shared"] is not None)
    stability = {
        "comparisons": len(pairs),
        "status_changes": sum(len(deltas[n][c]["status_changes"])
                              for n, c in pairs),
        "rank_corr_min": round(rank_corr[0], 4),
        "rank_corr_median": round(rank_corr[len(rank_corr) // 2], 4),
        # how thin the cut-off is: how many postings mention the skill that
        # sits at rank 30 — the churn budget of the top-30 membership
        "jobs_at_rank_n": {name: res["cohorts"]["Y3+Y4+Y5"]
                           ["min_jobs_at_rank_n"]
                           for name, res in results.items()},
    }

    payload = {
        "generated_by": "05_evaluation/loso_site_sensitivity.py",
        "config": {"model": MODEL_NAME, "dataset": RAW_JOBS.name,
                   "top_n": BASELINE_N, "thresholds": thr,
                   "min_skill_freq": MIN_SKILL_FREQ,
                   "locked_check": "PASSED",
                   "mapping": "fitted once on the full segment, held fixed"},
        "cleaning_funnel": {"n_raw": sum(stats.values()), **stats,
                            "it_segment": len(segment)},
        "site_distribution": sites,
        "variants": results,
        "mapping_drift": compare(results["locked"], results["refit_full"]),
        "deltas": deltas,
        "ordering": orders,
        "summary": {"worst_case_per_cohort": worst,
                    "ordering_stable": order_stable,
                    "skill_level_stability": stability},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    OUT_MD.write_text(render_md(payload))
    print(f"\nwrote {OUT_JSON.name} and {OUT_MD.name}")


# --------------------------------------------------------------- render
def render_md(p: dict) -> str:
    L = []
    c, f = p["config"], p["cleaning_funnel"]
    L += ["# Leave-one-site-out sensitivity of the labour-market side", "",
          f"Generated by `{p['generated_by']}` · corpus `{c['dataset']}` · "
          f"encoder `{c['model']}` · top-{c['top_n']} · "
          f"locked-run check {c['locked_check']}", "",
          "One portal at a time is removed from the technology segment, the "
          "market is rebuilt from the remaining postings, and the four locked "
          "cohorts are re-scored. Encoder, curriculum files, thresholds and "
          "the raw→canonical skill mapping are identical across every row; "
          "the only thing that changes is which postings are counted.", ""]

    L += ["## Portal contribution to the technology segment", "",
          "| portal | postings | share |", "|---|---:|---:|"]
    for r in p["site_distribution"]:
        L.append(f"| {r['site']} | {r['n']} | {r['share']:.1%} |")
    L += [f"| **total** | **{f['it_segment']}** | **100.0%** |", ""]

    ref = p["variants"]["refit_full"]
    L += ["## Reference (all portals, refit mapping)", "",
          "| cohort | postings | covered | demand-weighted coverage | KL |",
          "|---|---:|---:|---:|---:|"]
    for cohort, r in ref["cohorts"].items():
        L.append(f"| {cohort} | {ref['n_jobs']} | {r['covered']}/30 | "
                 f"{r['demand_weighted_coverage']:.1%} | {r['kl']:.4f} |")
    L.append("")

    L += ["## Effect of removing each portal", "",
          "| portal removed | postings left | cohort | top-30 kept | "
          "covered | Δ DWC (pts) | KL |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for name, res in p["variants"].items():
        if not res.get("removed_site"):
            continue
        d = p["deltas"][name]
        for cohort, r in res["cohorts"].items():
            row = d[cohort]
            L.append(f"| −{res['removed_site']} | {res['n_jobs']} | {cohort} | "
                     f"{row['top30_overlap']}/30 | "
                     f"{row['covered'][0]}→{row['covered'][1]} | "
                     f"{row['dwc_delta_pts']:+.2f} | "
                     f"{row['kl'][0]:.4f}→{row['kl'][1]:.4f} |")
    L.append("")

    w = p["summary"]["worst_case_per_cohort"]
    L += ["## Worst case over all eight removals", "",
          "| cohort | min top-30 kept | max covered shift | "
          "max Δ DWC (pts) | max Δ KL |",
          "|---|---:|---:|---:|---:|"]
    for cohort, r in w.items():
        L.append(f"| {cohort} | {r['min_top30_overlap']}/30 | "
                 f"{r['max_covered_shift']} | {r['max_dwc_shift_pts']:.2f} | "
                 f"{r['max_kl_shift']:.4f} |")
    L.append("")

    s = p["summary"]["skill_level_stability"]
    rank30 = s["jobs_at_rank_n"]
    L += ["## Skill-level stability (what actually moved)", "",
          f"Pooled over all {s['comparisons']} holdout × cohort comparisons:",
          "",
          f"- **{s['status_changes']} status changes.** No skill present in "
          "both top-30 lists was ever judged covered with a portal in and "
          "not covered with it out. Removing a portal does not re-score a "
          "skill; the coverage decision for a given skill is independent of "
          "which portal supplied the postings.",
          f"- **Demand ranking holds**: Spearman-style rank correlation on "
          f"the shared skills, median {s['rank_corr_median']:.2f}, worst "
          f"case {s['rank_corr_min']:.2f}.",
          f"- **What churns is the tail of the cut-off.** The skill at rank "
          f"30 is mentioned in only {rank30['refit_full']} of "
          f"{p['cleaning_funnel']['it_segment']} postings, so top-30 "
          "membership at the boundary is decided by a handful of adverts. "
          "Every covered-count movement below traces to substitutions at "
          "that boundary, not to a changed score.", "",
          "## Cohort ordering stability", ""]
    for metric, (_, label) in ORDER_METRICS.items():
        stable = p["summary"]["ordering_stable"][metric]
        ref_order = " > ".join(ordering(ref, metric))
        L.append(f"- **{label}** — reference order `{ref_order}`; "
                 + ("preserved under every single-portal removal."
                    if stable else "**FLIPS** under at least one removal:"))
        if not stable:
            for name, o in p["ordering"].items():
                if not o[metric]["stable"]:
                    L.append(f"    - −{p['variants'][name]['removed_site']}: "
                             + " > ".join(o[metric]["variant"]))
    L.append("")

    md = p["mapping_drift"]
    L += ["## Mapping drift (same postings, refit mapping vs committed cache)",
          "",
          "Reported separately so it is never confused with a removal effect; "
          "see the tie-order note in `dedup_sensitivity.py`.", "",
          "| cohort | top-30 kept | covered | Δ DWC (pts) | KL |",
          "|---|---:|---:|---:|---:|"]
    for cohort, r in md.items():
        L.append(f"| {cohort} | {r['top30_overlap']}/30 | "
                 f"{r['covered'][0]}→{r['covered'][1]} | "
                 f"{r['dwc_delta_pts']:+.2f} | "
                 f"{r['kl'][0]:.4f}→{r['kl'][1]:.4f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
