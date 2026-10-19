# TOP_N sensitivity sweep for the journal revision.
#
#   python3 05_evaluation/sweep_top_n.py            (from the phase5_curriculum_alignment root)
#
# Reviewer question: "why exactly the top 30 market skills — why not
# 10, 20, 40 or 50?". This script answers it empirically: it recomputes
# the ENTIRE reported metric set at N in {10, 20, 30, 40, 50} for all
# four locked curriculum configurations (Y3, Y4, Y5, programme Y3+Y4+Y5)
# and reports whether the paper's conclusions survive the choice of N.
#
# It is a RE-SLICE, not a re-run: the encoder, the market cache, the
# curriculum files, the calibrated thresholds and the top-by-demand rule
# are all identical to the locked 2026-07-30 runs. Only the cut-off rank
# moves. Consequently the sweep MUST reproduce the locked table at N=30,
# and it refuses to publish results if it does not (see verify_locked).
#
# What does NOT depend on N (stated so the write-up cannot overclaim):
#   - KL divergence: computed over ALL hard skills in the segment, so it
#     is constant across N by construction (verified here, not assumed).
#   - the similarity matrix, thresholds, and every per-pair score.
# What DOES depend on N: which skills enter the comparison set, hence
# covered/N, demand-weighted coverage, and the recommendation candidates.
#
# The cross-encoder host verification is deliberately SKIPPED: it governs
# recommendation routing/explanations only, never a coverage status,
# a coverage number or the selection order (paper §"Cross-encoder
# verification affected only the explanation and routing").

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE_DIR = _HERE.parent
sys.path.insert(0, str(BASE_DIR / "03_engine"))

# The locked runs are the Cambodian corpus segmented on job_category.
# Set before importing pipeline: both are resolved at module import.
os.environ.setdefault("JOB_SOURCE", "cambodia")
os.environ.setdefault("SEGMENT_MODE", "category")

from pipeline import (                                        # noqa: E402
    JOB_SOURCE, MODEL_NAME, RAW_JOBS, SEGMENT_MODE, build_market,
    get_thresholds, greedy_skill_selection, is_soft_skill, kl_divergence,
)

# Encode the curriculum on the CPU, NOT via pipeline.load_model() (which
# prefers Apple MPS). Measured: MPS and CPU agree on every coverage status,
# covered count and demand-weighted number, but differ in the 4th decimal of
# KL for one cohort (Y5: 0.376988 on MPS vs 0.376798 on CPU, i.e. 1.9e-4).
# The locked 2026-07-30 runs match the CPU value for all four cohorts, so the
# CPU path is what makes this sweep byte-comparable with the published table.
SWEEP_DEVICE = os.environ.get("SWEEP_DEVICE", "cpu")


def load_model_cpu(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=SWEEP_DEVICE)

CLO_DIR = BASE_DIR / "02_input_curriculum" / "clo_skills_db"
OUT_JSON = _HERE / "sweep_top_n_results.json"
OUT_MD = _HERE / "sweep_top_n.md"
# single self-contained file to hand to whoever writes the revision — it
# repeats the context a reader outside this repo would otherwise lack
OUT_HANDOFF = (BASE_DIR.parent.parent / "docs" / "journal"
               / "TOPN_SENSITIVITY_HANDOFF.md")

TOP_NS = [10, 20, 30, 40, 50]
BASELINE_N = 30           # the value used by the locked runs / the paper
CUM_FRAC = 0.80           # selection stopping rule, as in gen_report_data.py

# Cohorts as file-number ranges over clo_skills_db/ (verified byte-identical
# to the four locked runs' inputs.json sha256 lists — see verify_locked).
COHORTS = {
    "Y3":          range(1, 10),
    "Y4":          range(10, 18),
    "Y5":          range(18, 26),
    "Y3+Y4+Y5":    range(1, 26),
}
# locked reference runs (2026-07-30, git 0f226bb1, config e54e7e...)
LOCKED = {
    "Y3":       "20260730T063316Z-e54e7e",
    "Y4":       "20260730T063353Z-e54e7e",
    "Y5":       "20260730T063430Z-e54e7e",
    "Y3+Y4+Y5": "20260730T063522Z-e54e7e",
}


def course_name(path: Path) -> str:
    """Same display name as gen_report_data.course_name — the numeric
    prefix is stripped, so a course keeps its name across cohorts even
    though the prefix differs between per-cohort exports."""
    return path.stem.split("_", 1)[1].replace("_", " ").title()


def cohort_files(prefixes: range) -> list[Path]:
    """CLO files whose numeric prefix falls in the cohort's range, in the
    same sorted-by-filename order gen_report_data.py uses (the order only
    affects argmax tie-breaking, never a score)."""
    wanted = {str(p) for p in prefixes}
    files = [f for f in sorted(CLO_DIR.glob("*.json"))
             if f.stem.split("_", 1)[0] in wanted]
    return files


def load_cohort(files: list[Path]):
    courses, clo_texts, clo_owner = [], [], []
    for f in files:
        name = course_name(f)
        courses.append(name)
        for item in json.loads(f.read_text()):
            clo_texts.append(item["skill"])
            clo_owner.append(name)
    return courses, clo_texts, clo_owner


def tie_analysis(sim, market, ranked, n, thr):
    """"Top-N by posting frequency" is not always a unique set: skills can
    tie on the job count at the cut-off, and which of them lands inside is
    decided by the sort's tie-break (alphabetical, since hard_idx is in
    skill_base order) rather than by evidence.

    Enumerates EVERY way the boundary tie could have been resolved and
    returns the range of covered / demand-weighted coverage across them.
    A reviewer asks this about any frequency cut-off; the honest answer is
    the range, not the single number the alphabet happened to produce."""
    base, counts = market["skill_base"], market["counts"]
    cut = counts[base[ranked[n - 1]]]
    tied = [i for i in ranked if counts[base[i]] == cut]
    inside = [i for i in ranked[:n] if counts[base[i]] == cut]
    fixed = [i for i in ranked[:n] if counts[base[i]] != cut]
    if len(tied) == len(inside):                 # clean cut, nothing to vary
        return {"boundary_jobs": cut, "n_tied": len(tied),
                "n_tied_inside": len(inside), "n_resolutions": 1,
                "covered_range": None, "dwc_range": None}
    covs, dwcs = [], []
    for combo in itertools.combinations(tied, len(inside)):
        top = fixed + list(combo)
        sc = [float(sim[i].max()) for i in top]
        jb = [counts[base[i]] for i in top]
        covs.append(sum(s >= thr["covered"] for s in sc))
        dwcs.append(sum(j for j, s in zip(jb, sc)
                        if s >= thr["covered"]) / sum(jb))
    return {"boundary_jobs": cut, "n_tied": len(tied),
            "n_tied_inside": len(inside),
            "n_resolutions": len(covs),
            "tied_skills": [base[i] for i in tied],
            "covered_range": [min(covs), max(covs)],
            "dwc_range": [round(min(dwcs), 4), round(max(dwcs), 4)]}


def evaluate(sim, market, hard_idx, ranked, demand, profile, kl, n, thr):
    """Every reported metric at cut-off n. Mirrors gen_report_data.main()
    exactly for the parts that depend on N."""
    base, counts = market["skill_base"], market["counts"]
    top = ranked[:n]

    rows = []
    for i in top:
        program_cov = float(sim[i].max())
        status = ("covered" if program_cov >= thr["covered"] else
                  "borderline" if program_cov >= thr["borderline"] else "gap")
        rows.append({"skill": base[i], "jobs": counts[base[i]],
                     "program": program_cov, "status": status})

    covered = sum(r["status"] == "covered" for r in rows)
    borderline = sum(r["status"] == "borderline" for r in rows)
    top_jobs = sum(r["jobs"] for r in rows)
    cov_jobs = sum(r["jobs"] for r in rows if r["status"] == "covered")
    bor_jobs = sum(r["jobs"] for r in rows if r["status"] == "borderline")
    # share of the WHOLE segment's technical demand that the top-n captures —
    # the coverage denominator's own representativeness, and the core
    # argument for where to cut
    all_jobs = sum(counts[base[i]] for i in hard_idx)

    # --- prescriptive layer: how many additions the plan recommends at n
    cand_glob = [i for i, r in zip(top, rows) if r["status"] != "covered"]
    n_selected, kl_after = 0, kl
    if cand_glob:
        cand_names = [base[i] for i in cand_glob]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cand_sim = np.clip(market["skill_emb"][cand_glob]
                               @ market["skill_emb"][hard_idx].T, 0.0, None)
        steps = greedy_skill_selection(demand, profile, cand_names, cand_sim)
        total_gain = sum(st["gain"] for st in steps)
        cum, plan_profile = 0.0, profile
        for st in steps:
            if cum >= CUM_FRAC * total_gain:
                break
            cum += st["gain"]
            plan_profile = st["profile"]
            n_selected += 1
        kl_after, _ = kl_divergence(demand, plan_profile / plan_profile.sum())

    return {
        "top_n": n,
        "covered": covered,
        "borderline": borderline,
        "gap": n - covered - borderline,
        "coverage_rate": round(covered / n, 4),
        "demand_weighted_coverage": round(cov_jobs / top_jobs, 4),
        "demand_weighted_coverage_half_credit":
            round((cov_jobs + 0.5 * bor_jobs) / top_jobs, 4),
        "top_n_job_mentions": top_jobs,
        "demand_mass_captured": round(top_jobs / all_jobs, 4),
        "min_jobs_at_rank_n": rows[-1]["jobs"],
        "kl": round(kl, 4),
        "n_recommended": n_selected,
        "kl_after_plan": round(kl_after, 4),
        "skills": [{"skill": r["skill"], "jobs": r["jobs"],
                    "score": round(r["program"], 4), "status": r["status"]}
                   for r in rows],
    }


def verify_locked(cohort: str, res: dict) -> list[str]:
    """The sweep is only credible if its N=30 slice IS the locked run.
    Compares against outputs/<run_id>/report.json, which is immutable."""
    path = BASE_DIR / "outputs" / LOCKED[cohort] / "report.json"
    if not path.exists():
        return [f"{cohort}: locked run {LOCKED[cohort]} not found"]
    ref = json.loads(path.read_text())
    errs = []
    for key in ("covered", "demand_weighted_coverage",
                "demand_weighted_coverage_half_credit", "kl"):
        if abs(float(res[key]) - float(ref[key])) > 1e-4:
            errs.append(f"{cohort}.{key}: sweep {res[key]} != locked {ref[key]}")
    ref_skills = [s["skill"] for s in ref["skills"]]
    if [s["skill"] for s in res["skills"]] != ref_skills:
        errs.append(f"{cohort}: top-30 skill set differs from the locked run")
    return errs


def main() -> None:
    thr = get_thresholds()
    print(f"model {MODEL_NAME} · source {JOB_SOURCE} · segment {SEGMENT_MODE}")
    print(f"thresholds covered {thr['covered']} / borderline "
          f"{thr['borderline']}")
    model = load_model_cpu(MODEL_NAME)
    market = build_market(model)
    base, counts = market["skill_base"], market["counts"]
    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    print(f"market: {market['n_jobs']} segment jobs, {len(base)} skills, "
          f"{len(hard_idx)} technical (soft filtered)")
    if max(TOP_NS) > len(hard_idx):
        sys.exit(f"ABORT: only {len(hard_idx)} technical skills available")

    demand = market["demand_dist"][hard_idx]
    demand = demand / demand.sum()
    # one canonical ranking, reused by every cohort and every N (the top-N
    # sets are nested prefixes of it)
    ranked = sorted(hard_idx, key=lambda i: -counts[base[i]])

    results, problems = {}, []
    for cohort, prefixes in COHORTS.items():
        files = cohort_files(prefixes)
        courses, clo_texts, clo_owner = load_cohort(files)
        clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                               show_progress_bar=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = market["skill_emb"] @ clo_emb.T
        # KL is over the full technical segment, so it never sees N
        profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)
        kl, _ = kl_divergence(demand, profile / profile.sum())

        per_n = []
        for n in TOP_NS:
            row = evaluate(sim, market, hard_idx, ranked, demand, profile,
                           kl, n, thr)
            row["boundary_tie"] = tie_analysis(sim, market, ranked, n, thr)
            per_n.append(row)
        results[cohort] = {
            "courses": courses, "n_courses": len(courses),
            "n_clo_skills": len(clo_texts), "kl": round(kl, 4),
            "locked_run": LOCKED[cohort],
            "by_top_n": {str(r["top_n"]): r for r in per_n},
        }
        baseline = next(r for r in per_n if r["top_n"] == BASELINE_N)
        problems += verify_locked(cohort, baseline)
        print(f"\n{cohort}: {len(courses)} courses, {len(clo_texts)} CLO "
              f"skills, KL={kl:.4f}")
        for r in per_n:
            print(f"  top-{r['top_n']:<3} covered {r['covered']:>2}/{r['top_n']}"
                  f" ({r['coverage_rate']:>6.1%})  dwc {r['demand_weighted_coverage']:>6.1%}"
                  f"  half {r['demand_weighted_coverage_half_credit']:>6.1%}"
                  f"  demand mass {r['demand_mass_captured']:>6.1%}"
                  f"  min jobs {r['min_jobs_at_rank_n']:>3}"
                  f"  recs {r['n_recommended']:>2}")

    if problems:
        print("\nLOCKED-RUN CHECK FAILED:")
        for p in problems:
            print("  " + p)
        sys.exit("ABORT: N=30 slice does not reproduce the locked runs — "
                 "the sweep would not be comparable with the paper.")
    print(f"\nlocked-run check PASSED: N={BASELINE_N} reproduces all four "
          f"2026-07-30 runs (covered, dwc, half-credit, KL, top-30 set)")

    # ---- what the choice of N does and does not change -----------------
    kls = {c: {n: results[c]["by_top_n"][str(n)]["kl"] for n in TOP_NS}
           for c in COHORTS}
    kl_invariant = all(len(set(v.values())) == 1 for v in kls.values())

    # The substantive claim of the paper is an ORDERING (a later academic
    # year covers more market demand than an earlier one, and the pooled
    # programme covers most). Whether that ordering survives is the test
    # that matters for choosing N — not whether a number moves.
    order = ["Y3", "Y4", "Y5", "Y3+Y4+Y5"]
    metrics = ("demand_weighted_coverage", "coverage_rate",
               "demand_weighted_coverage_half_credit")
    ordering_by_n = {}
    for n in TOP_NS:
        ordering_by_n[str(n)] = {
            m: all(a <= b + 1e-9 for a, b in
                   zip([results[c]["by_top_n"][str(n)][m] for c in order],
                       [results[c]["by_top_n"][str(n)][m] for c in order][1:]))
            for m in metrics}

    # spread of each cohort's headline number across N: how much the choice
    # of N is worth in percentage points
    spread = {c: round(max(results[c]["by_top_n"][str(n)]
                           ["demand_weighted_coverage"] for n in TOP_NS)
                       - min(results[c]["by_top_n"][str(n)]
                             ["demand_weighted_coverage"] for n in TOP_NS), 4)
              for c in COHORTS}
    # marginal demand evidence bought by widening the comparison set
    base_n = results["Y3+Y4+Y5"]["by_top_n"]
    marginal = []
    prev_mass = 0.0
    for n in TOP_NS:
        mass = base_n[str(n)]["demand_mass_captured"]
        marginal.append({"top_n": n, "demand_mass": mass,
                         "added_vs_previous": round(mass - prev_mass, 4),
                         "min_postings_at_rank_n":
                             base_n[str(n)]["min_jobs_at_rank_n"]})
        prev_mass = mass

    summary = {
        "kl_invariant_to_top_n": kl_invariant,
        "cohort_ordering_holds_by_n": ordering_by_n,
        "cohort_ordering_holds_for_all_metrics_at":
            [n for n in TOP_NS if all(ordering_by_n[str(n)].values())],
        "dwc_spread_across_n": spread,
        "demand_mass_by_n": marginal,
    }
    print(f"\nKL invariant to N: {kl_invariant}")
    for n in TOP_NS:
        flags = ordering_by_n[str(n)]
        print(f"  top-{n:<3} cohort ordering Y3<=Y4<=Y5<=programme holds: "
              + ", ".join(f"{m.split('_')[0]}={v}" for m, v in flags.items()))
    print("  DWC spread across N: "
          + ", ".join(f"{c} {v:.1%}" for c, v in spread.items()))

    # boundary ties: printed, not only stored, so a pasted terminal log is
    # enough to check the whole result without opening the json
    print("\nBOUNDARY TIES (every resolution enumerated) — dwc range:")
    print(f"  {'N':>3} {'cut':>4} {'tied':>5} {'in':>3} {'ways':>5}  "
          + "  ".join(f"{c:^15}" for c in order))
    for n in TOP_NS:
        t = results[order[0]]["by_top_n"][str(n)]["boundary_tie"]
        cells = []
        for c in order:
            rng = results[c]["by_top_n"][str(n)]["boundary_tie"]["dwc_range"]
            cells.append(f"{'clean cut':^15}" if rng is None
                         else f"{rng[0]:>6.1%}-{rng[1]:<6.1%}"[:15].center(15))
        print(f"  {n:>3} {t['boundary_jobs']:>4} {t['n_tied']:>5} "
              f"{t['n_tied_inside']:>3} {t['n_resolutions']:>5}  "
              + "  ".join(cells))

    payload = {
        "generated_by": "05_evaluation/sweep_top_n.py",
        "config": {
            "model": MODEL_NAME, "job_source": JOB_SOURCE,
            "segment_mode": SEGMENT_MODE, "dataset": RAW_JOBS.name,
            "n_segment_jobs": market["n_jobs"],
            "n_technical_skills": len(hard_idx),
            "device": SWEEP_DEVICE,
            "thresholds": thr, "cum_frac": CUM_FRAC,
            "top_ns": TOP_NS, "baseline_n": BASELINE_N,
            "locked_check": "PASSED",
        },
        "summary": summary,
        "cohorts": results,
    }
    _check_note_numbers(payload)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    OUT_MD.write_text(render_md(payload))
    OUT_HANDOFF.write_text(render_handoff(payload))
    print(f"\nwritten {OUT_JSON.relative_to(BASE_DIR)} and "
          f"{OUT_MD.relative_to(BASE_DIR)}")
    print(f"handoff for the journal revision -> {OUT_HANDOFF}")


def render_md(p: dict) -> str:
    cfg, out = p["config"], []
    out.append("# Sensitivity of the results to the top-N comparison set\n")
    out.append(f"Encoder `{cfg['model']}` · corpus `{cfg['dataset']}` "
               f"({cfg['n_segment_jobs']} technology postings, "
               f"{cfg['n_technical_skills']} recurrent technical skills) · "
               f"thresholds covered {cfg['thresholds']['covered']} / "
               f"borderline {cfg['thresholds']['borderline']}.\n")
    out.append(f"Only the cut-off rank N varies; the encoder, market, "
               f"curriculum, thresholds and selection rule are those of the "
               f"locked runs. The N={cfg['baseline_n']} slice reproduces all "
               f"four locked runs exactly (checked at runtime).\n")
    out.append("\n## Per-configuration results by N\n")
    out.append("| Cohort | N | Covered | Coverage rate | Demand-weighted "
               "coverage | Half-credit | Demand mass in top-N | Postings at "
               "rank N | Recommended additions | KL |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c, d in p["cohorts"].items():
        for n in cfg["top_ns"]:
            r = d["by_top_n"][str(n)]
            out.append(
                f"| {c} | {n} | {r['covered']}/{n} | {r['coverage_rate']:.1%} "
                f"| {r['demand_weighted_coverage']:.1%} "
                f"| {r['demand_weighted_coverage_half_credit']:.1%} "
                f"| {r['demand_mass_captured']:.1%} "
                f"| {r['min_jobs_at_rank_n']} | {r['n_recommended']} "
                f"| {r['kl']:.4f} |")
    out.append("\n## What the comparison set buys\n")
    s = p["summary"]
    out.append("| N | Share of technical demand in the top-N | Added vs "
               "previous N | Postings supporting the skill at rank N |")
    out.append("|---:|---:|---:|---:|")
    for m in s["demand_mass_by_n"]:
        out.append(f"| {m['top_n']} | {m['demand_mass']:.1%} "
                   f"| +{m['added_vs_previous']:.1%} "
                   f"| {m['min_postings_at_rank_n']} |")
    out.append("\n## Stability of the conclusions\n")
    out.append(f"- KL divergence is identical at every N "
               f"(`{s['kl_invariant_to_top_n']}`) — it is computed over the "
               f"full technical segment, not over the top-N, so it cannot be "
               f"used to argue for any particular N.")
    out.append("- Cohort ordering (Y3 <= Y4 <= Y5 <= programme):\n")
    out.append("| N | Demand-weighted coverage | Unweighted coverage rate "
               "| Half-credit |")
    out.append("|---:|---|---|---|")
    for n in p["config"]["top_ns"]:
        f = s["cohort_ordering_holds_by_n"][str(n)]
        out.append(f"| {n} | {f['demand_weighted_coverage']} "
                   f"| {f['coverage_rate']} "
                   f"| {f['demand_weighted_coverage_half_credit']} |")
    out.append(f"\n- The ordering holds on all three measures at N = "
               f"{s['cohort_ordering_holds_for_all_metrics_at']}.")
    out.append("- Demand-weighted coverage varies across N by "
               + ", ".join(f"{c} {v:.1f} pt" for c, v in
                           ((c, v * 100) for c, v in
                            s["dwc_spread_across_n"].items())) + ".")
    out.append(MANUSCRIPT_NOTE)
    return "\n".join(out) + "\n"


# Written for the journal revision (reviewer: "why exactly 30?"). Kept in the
# generator, not in a hand-edited file, so the prose and the numbers above can
# never drift apart. Numbers quoted here are asserted in _check_note_numbers().
MANUSCRIPT_NOTE = """
## Text for the manuscript

The comparison set was fixed at the thirty most frequently demanded
technical skills. To establish that the reported findings are not an
artefact of that choice, the complete analysis was repeated with the
comparison set truncated at 10, 20, 30, 40 and 50 skills, holding the
encoder, the labour-market segment, the curriculum inventories, the
human-calibrated thresholds and the recommendation rule constant.

Three observations follow. First, the ranking of the curriculum
configurations is stable: Year 3 < Year 4 < Year 5 < complete programme
holds on unweighted coverage, demand-weighted coverage and the
half-credit measure for every comparison set of twenty skills or more,
and programme-level demand-weighted coverage remains between 79.7% and
87.1% throughout. Second, the ranking does not hold at ten skills, where
Year 3 (57.4%) exceeds Year 4 (39.7%). At that width the comparison set
consists mainly of foundational competencies already taught in Year 3,
and a single skill accounts for ten percentage points of the unweighted
rate, so the measure reflects the composition of a very small set rather
than curriculum breadth. Third, widening the set yields diminishing
evidential return: the top ten skills account for 12.2% of recurrent
technical demand and the top thirty for 25.9%, but each further increment
adds less demand while admitting progressively rarer skills, from sixteen
supporting postings at rank thirty to twelve at rank fifty.

Thirty was therefore retained: it lies inside the range over which the
conclusions are invariant, every skill in the set is supported by at
least sixteen postings (6.3% of the 254 technology-related
advertisements), and it provides a larger candidate pool for the
recommendation stage than a narrower set. Kullback-Leibler divergence is
unchanged at every value of N, since it is computed over all 574
recurrent technical skills rather than over the comparison set.
"""


ORDER = ["Y3", "Y4", "Y5", "Y3+Y4+Y5"]
LABELS = {"Y3": "Year 3", "Y4": "Year 4", "Y5": "Year 5",
          "Y3+Y4+Y5": "Programme (Y3+Y4+Y5)"}


def _grid(p, key, fmt):
    """One markdown table: rows = N, columns = the four cohorts."""
    ns = p["config"]["top_ns"]
    rows = ["| N | Year 3 | Year 4 | Year 5 | Programme (Y3+Y4+Y5) |",
            "|---:|---:|---:|---:|---:|"]
    for n in ns:
        cells = [fmt(p["cohorts"][c]["by_top_n"][str(n)], n) for c in ORDER]
        mark = "**" if n == p["config"]["baseline_n"] else ""
        rows.append(f"| {mark}{n}{mark} | "
                    + " | ".join(f"{mark}{x}{mark}" for x in cells) + " |")
    return "\n".join(rows)


def render_handoff(p: dict) -> str:
    """The paste-into-a-writing-assistant file. Everything a writer needs
    and nothing they have to look up in the repo: context, frozen config,
    all five comparison widths, the robustness checks, the corrections the
    manuscript needs, and an explicit list of claims NOT to make."""
    cfg, s = p["config"], p["summary"]
    c = p["cohorts"]
    ns = cfg["top_ns"]
    tie30 = c["Y3"]["by_top_n"][str(cfg["baseline_n"])]["boundary_tie"]

    out = []
    out.append("# Top-N sensitivity analysis — material for the journal "
               "revision\n")
    out.append("**How to use this file:** it contains every number needed to "
               "answer the reviewer's question about the comparison-set size, "
               "plus the corrections the current manuscript needs. All values "
               "are machine-generated from the analysis run, not transcribed. "
               "Do not invent numbers that are not in this file; if something "
               "needed is missing, say so instead of estimating it.\n")

    out.append("\n## 1. What the reviewer asked\n")
    out.append("The study compares a university curriculum against labour-"
               "market demand, using the N most frequently demanded technical "
               "skills as the comparison set. The manuscript used N = 30. The "
               "reviewer asked why exactly 30 rather than 10, 20 or 50, and "
               "asked for multiple values to be reported so the results table "
               "shows a comparison across settings.\n")
    out.append("In response the complete analysis was repeated at N = "
               + ", ".join(str(n) for n in ns) + ". The encoder, the "
               "labour-market corpus and segment, the curriculum inventories, "
               "the human-calibrated similarity thresholds and the "
               "recommendation rule were all held constant; only the cut-off "
               "rank changed. The N = "
               f"{cfg['baseline_n']} results reproduce the previously reported "
               "runs exactly, so the sweep extends the published analysis "
               "rather than replacing it.\n")

    out.append("\n## 2. Frozen configuration (unchanged across all five "
               "settings)\n")
    out.append(f"- Encoder: `{cfg['model']}` (bi-encoder, normalised "
               f"768-dimensional vectors)")
    out.append(f"- Labour-market corpus: {cfg['n_segment_jobs']} "
               f"technology-related postings")
    out.append(f"- Recurrent technical skills in the segment: "
               f"{cfg['n_technical_skills']}")
    out.append(f"- Thresholds: covered >= {cfg['thresholds']['covered']}, "
               f"related/borderline >= {cfg['thresholds']['borderline']}, "
               f"gap below that")
    out.append(f"- Curriculum configurations: Year 3 "
               f"({c['Y3']['n_courses']} courses, {c['Y3']['n_clo_skills']} "
               f"skill phrases), Year 4 ({c['Y4']['n_courses']}, "
               f"{c['Y4']['n_clo_skills']}), Year 5 ({c['Y5']['n_courses']}, "
               f"{c['Y5']['n_clo_skills']}), programme "
               f"({c['Y3+Y4+Y5']['n_courses']}, "
               f"{c['Y3+Y4+Y5']['n_clo_skills']})")

    out.append("\n## 3. Main result — demand-weighted coverage by "
               "comparison-set width\n")
    out.append("This is the table to put in the paper. Bold row = the "
               "adopted setting.\n")
    out.append(_grid(p, None, lambda r, n:
                     f"{r['demand_weighted_coverage']:.1%}"))
    out.append("\nTwo columns that depend only on N, not on the curriculum, "
               "can be appended to the same table:\n")
    out.append("| N | Share of all recurrent technical demand inside the "
               "top-N | Postings supporting the skill at rank N |")
    out.append("|---:|---:|---:|")
    for m in s["demand_mass_by_n"]:
        out.append(f"| {m['top_n']} | {m['demand_mass']:.1%} "
                   f"| {m['min_postings_at_rank_n']} |")

    out.append("\n## 4. Supporting tables\n")
    out.append("### 4.1 Skills classified as covered (count out of N)\n")
    out.append(_grid(p, None, lambda r, n: f"{r['covered']}/{n}"))
    out.append("\n### 4.2 Unweighted coverage rate\n")
    out.append(_grid(p, None, lambda r, n: f"{r['coverage_rate']:.1%}"))
    out.append("\n### 4.3 Half-credit sensitivity measure "
               "(borderline skills counted at 0.5)\n")
    out.append(_grid(p, None, lambda r, n:
                     f"{r['demand_weighted_coverage_half_credit']:.1%}"))
    out.append("\n### 4.4 Number of recommended skill additions produced\n")
    out.append(_grid(p, None, lambda r, n: f"{r['n_recommended']}"))

    out.append("\n## 5. Robustness checks\n")
    out.append("### 5.1 Ordering of the curriculum configurations\n")
    out.append("The substantive claim is an ordering: a later academic year "
               "covers more market demand than an earlier one, and the pooled "
               "programme covers most. Whether that ordering survives is the "
               "test that matters.\n")
    out.append("| N | Demand-weighted coverage | Unweighted rate "
               "| Half-credit |")
    out.append("|---:|---|---|---|")
    for n in ns:
        f = s["cohort_ordering_holds_by_n"][str(n)]
        cells = [f["demand_weighted_coverage"], f["coverage_rate"],
                 f["demand_weighted_coverage_half_credit"]]
        out.append(f"| {n} | "
                   + " | ".join("holds" if x else "**FAILS**" for x in cells)
                   + " |")
    out.append(f"\nThe ordering holds on all three measures at N = "
               f"{', '.join(str(n) for n in s['cohort_ordering_holds_for_all_metrics_at'])}"
               f". It fails at N = 10, where Year 3 "
               f"({c['Y3']['by_top_n']['10']['demand_weighted_coverage']:.1%}) "
               f"exceeds Year 4 "
               f"({c['Y4']['by_top_n']['10']['demand_weighted_coverage']:.1%}). "
               f"At that width the comparison set consists mainly of "
               f"foundational competencies already taught in Year 3, and a "
               f"single skill is worth ten percentage points of the "
               f"unweighted rate.\n")

    out.append("### 5.2 Ties at the cut-off rank\n")
    out.append("A frequency cut-off does not always define a unique set: "
               "skills can tie on posting count at the boundary, and which of "
               "them falls inside is then decided by sort order rather than "
               "by evidence. Every possible resolution of that tie was "
               "enumerated.\n")
    out.append(f"At N = {cfg['baseline_n']}, {tie30['n_tied']} skills tie at "
               f"{tie30['boundary_jobs']} postings "
               f"({', '.join(tie30['tied_skills'])}) and only "
               f"{tie30['n_tied_inside']} fit inside the cut, giving "
               f"{tie30['n_resolutions']} possible comparison sets. Across "
               f"all of them:\n")
    out.append("| Configuration | Reported value | Range across all "
               "tie resolutions |")
    out.append("|---|---:|---:|")
    for k in ORDER:
        r = c[k]["by_top_n"][str(cfg["baseline_n"])]
        t = r["boundary_tie"]
        out.append(f"| {LABELS[k]} | {r['demand_weighted_coverage']:.1%} "
                   f"| {t['dwc_range'][0]:.1%} – {t['dwc_range'][1]:.1%} |")
    # where the reported value sits inside its own tie range, DERIVED rather
    # than asserted (an earlier hand-written version of this sentence claimed
    # "at the top in every configuration", which is false for Year 4)
    at_max, worst = [], 0.0
    for k in ORDER:
        r = c[k]["by_top_n"][str(cfg["baseline_n"])]
        lo, hi = r["boundary_tie"]["dwc_range"]
        if abs(r["demand_weighted_coverage"] - hi) < 5e-4:
            at_max.append(LABELS[k])
        worst = max(worst, hi - lo)
    # ordering under EVERY tie resolution: ranges must not overlap
    disjoint_at = []
    for n in ns:
        rngs = [c[k]["by_top_n"][str(n)]["boundary_tie"]["dwc_range"]
                or [c[k]["by_top_n"][str(n)]["demand_weighted_coverage"]] * 2
                for k in ORDER]
        if all(a[1] < b[0] for a, b in zip(rngs, rngs[1:])):
            disjoint_at.append(n)
    out.append(f"\nTwo things must both be stated. First, the reported value "
               f"is the most favourable resolution of the tie in "
               f"{len(at_max)} of the {len(ORDER)} configurations "
               f"({', '.join(at_max)}), so those figures are the optimistic "
               f"end of a band up to {worst * 100:.1f} percentage points "
               f"wide. "
               f"Second, the ranges do not overlap between configurations at "
               f"N = {', '.join(str(n) for n in disjoint_at)}, so the "
               f"ordering of the configurations holds under every possible "
               f"resolution of the tie — the conclusion is unaffected and "
               f"only the final digit is arbitrary. Ties also occur at "
               f"N = 20, 40 and 50; N = 10 is the only clean cut.\n")

    out.append("### 5.3 Kullback-Leibler divergence\n")
    out.append(f"KL divergence is identical at every N "
               f"(Year 3 {c['Y3']['kl']:.4f}, Year 4 {c['Y4']['kl']:.4f}, "
               f"Year 5 {c['Y5']['kl']:.4f}, programme "
               f"{c['Y3+Y4+Y5']['kl']:.4f}), "
               f"because it is computed over all "
               f"{cfg['n_technical_skills']} recurrent technical skills "
               f"rather than over the comparison set. It therefore cannot be "
               f"used to argue for any particular N, and it is reported as a "
               f"secondary distributional statistic only.\n")

    out.append("\n## 6. Corrections the manuscript needs\n")
    out.append(f"1. The manuscript states that the segment contains 571 "
               f"recurrent technical skills. The correct figure for the "
               f"reported results is **{cfg['n_technical_skills']}**. The "
               f"571 figure predates a rebuild of the labour-market index "
               f"and is inconsistent with the reported runs.")
    out.append("2. Any statement that the top-30 skills carry 25.8% of "
               "technical demand should read **25.9%**.")
    out.append("3. The sensitivity analysis in section 3 above is new "
               "material and needs a short subsection in the results, plus "
               "one sentence in the methods explaining that N = 30 was "
               "checked against 10, 20, 40 and 50.")

    out.append("\n## 7. Claims that are supported, and claims to avoid\n")
    out.append("**Supported:**")
    out.append("- The reported findings are not an artefact of the "
               "comparison-set size; the ordering of curriculum "
               "configurations is invariant for N between 20 and 50.")
    out.append(f"- Programme-level demand-weighted coverage stays between "
               f"{min(c['Y3+Y4+Y5']['by_top_n'][str(n)]['demand_weighted_coverage'] for n in ns):.1%} "
               f"and "
               f"{max(c['Y3+Y4+Y5']['by_top_n'][str(n)]['demand_weighted_coverage'] for n in ns):.1%} "
               f"across all five settings.")
    out.append("- A comparison set of ten skills is too narrow to support "
               "the analysis, and this is shown empirically rather than "
               "asserted.")
    out.append("\n**Avoid:**")
    out.append("- Do not claim that 30 is an optimal or uniquely correct "
               "value. The defensible claim is that the conclusions are "
               "stable across 20 to 50 and that 30 lies inside that range "
               "while keeping at least "
               f"{s['demand_mass_by_n'][2]['min_postings_at_rank_n']} "
               "supporting postings per skill.")
    out.append("- Do not present the KL divergence as evidence for the "
               "choice of N; it does not vary with N.")
    out.append("- Do not describe similarity scores as percentages; they are "
               "cosine similarities.")
    out.append("- Do not report a top-10 configuration as a result without "
               "also reporting that the ordering fails there.")

    out.append("\n## 8. Draft text (rewrite freely, keep the numbers)\n")
    out.append(MANUSCRIPT_NOTE.split("## Text for the manuscript\n", 1)[1]
               .strip())
    return "\n".join(out) + "\n"


def _check_note_numbers(p: dict) -> None:
    """Guard the hand-written manuscript paragraph against silent drift:
    if a re-run moves any number quoted in it, fail loudly rather than
    ship prose that disagrees with the table above it."""
    c = p["cohorts"]
    prog = [c["Y3+Y4+Y5"]["by_top_n"][str(n)]["demand_weighted_coverage"]
            for n in p["config"]["top_ns"]]
    claims = {
        "programme dwc min 0.797": abs(min(prog) - 0.797) < 5e-4,
        "programme dwc max 0.871": abs(max(prog) - 0.871) < 5e-4,
        "Y3@10 dwc 0.574":
            abs(c["Y3"]["by_top_n"]["10"]["demand_weighted_coverage"]
                - 0.574) < 5e-4,
        "Y4@10 dwc 0.397":
            abs(c["Y4"]["by_top_n"]["10"]["demand_weighted_coverage"]
                - 0.397) < 5e-4,
        # compared as RENDERED strings: both values sit on a rounding
        # boundary (0.1225, 0.2595), so a float tolerance would disagree
        # with the percentage printed in the table right above the prose
        "mass@10 renders 12.2%":
            f"{c['Y3']['by_top_n']['10']['demand_mass_captured']:.1%}"
            == "12.2%",
        "mass@30 renders 25.9%":
            f"{c['Y3']['by_top_n']['30']['demand_mass_captured']:.1%}"
            == "25.9%",
        "postings at rank 30 == 16":
            c["Y3"]["by_top_n"]["30"]["min_jobs_at_rank_n"] == 16,
        "postings at rank 50 == 12":
            c["Y3"]["by_top_n"]["50"]["min_jobs_at_rank_n"] == 12,
        "ordering holds for N >= 20":
            p["summary"]["cohort_ordering_holds_for_all_metrics_at"]
            == [20, 30, 40, 50],
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        sys.exit("ABORT: the manuscript paragraph quotes numbers this run no "
                 "longer produces: " + "; ".join(bad))


if __name__ == "__main__":
    main()
