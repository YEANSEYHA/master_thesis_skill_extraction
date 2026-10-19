"""Export one archived run into the thesis bundle.

Reads only what run_store.py already wrote — outputs/<run_id>/{params,
report,inputs}.json — and emits flat, citable artifacts beside them:

    outputs/<run_id>/thesis/
        run_summary.json      the cohort's headline record + provenance
        course_manifest.csv   exactly which courses fed the run
        skill_status.csv      one row per market skill (the results table)
        heatmap_cells.csv     skill x course long format (the figure's data)
        recommendations.csv   greedy plan, selected AND skipped

    outputs/thesis_registry.csv   one row per run, rebuilt from every bundle

Nothing here recomputes anything: the bundle is a re-shaping of the run,
so re-exporting is always safe and always agrees with report.json.
See THESIS_EXPORT_FORMAT.md for the field-by-field spec.

Usage:
    python3 04_report/export_run.py <run_id>
    python3 04_report/export_run.py --latest
    python3 04_report/export_run.py --all
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
REGISTRY = OUTPUTS / "thesis_registry.csv"

# recorded verbatim in every bundle: the counts below are threshold-dependent
# and the thresholds have not been calibrated against human labels yet
THRESHOLDS_STATUS = (
    "PROVISIONAL — linearly mapped from MiniLM; the LLM-judge silver study "
    "fits strict `covered` at ~0.80. Awaiting human gold (remaining #1) -> "
    "calibration (#2). Every covered/borderline/gap count and every heatmap "
    "cell colour moves if the bar moves."
)
MODELLED_NOTE = (
    "Simulated by greedy facility-location: 'adding' a skill inserts a "
    "synthetic CLO that embeds the skill itself. An upper bound, not a "
    "forecast. Never report as an achieved result."
)
EVAL_NOTE = (
    "Precision/recall/F1 are properties of the INSTRUMENT (the 150-pair "
    "annotation study), not of a cohort run. Do not fill per-run."
)
KNOWN_LIMITS = [
    "KL is a point estimate with no confidence interval (remaining #6).",
    "KL is encoder-bound: the curriculum profile is raw unthresholded "
    "max-cosine and BGE's floor for unrelated pairs is ~0.5, so KL is "
    "systematically damped.",
    "KL is computed over the full ~571-skill segment, not just top_n. At "
    "the coverage magnitudes a single cohort can reach (~12-17/30), KL sits "
    "within ~6% of the 'teaches nothing' baseline (measured: null=0.3988, "
    "cohorts 0.374-0.378) and does not usefully separate cohorts. "
    "demand_weighted_coverage_percent is the primary cross-cohort metric; "
    "KL is a secondary distributional stat, not a ranking metric.",
    "KL is not comparable across corpora of different size; only "
    "cross-cohort comparisons on the same market segment are valid.",
    "Coverage is max cosine, not a percentage. coverage_rate_percent is the "
    "SHARE OF SKILLS covered (e.g. 12/30), never a cosine as a percent.",
    "demand_weighted_coverage_percent = share of top_n's job-count mass "
    "that is covered (Sigma jobs[covered] / Sigma jobs[top_n]), i.e. how "
    "much of what employers ask for (within the top_n) is taught, not how "
    "many of top_n skills are ticked. Inherits the covered_at threshold "
    "(0.73, provisional; silver LLM-judge study suggests the true 'fully "
    "covered' bar is closer to 0.80) -- report the half-credit variant "
    "alongside it as a sensitivity bound.",
]

VERDICT = {"SUPPORTED": "SUPPORTED", "WEAK": "WEAK",
           "NOT SUPPORTED": "NOT_SUPPORTED"}


def _load(run_dir: Path) -> tuple[dict, dict, dict, list]:
    def read(name, default=None):
        p = run_dir / name
        if not p.is_file():
            if default is None:
                raise SystemExit(f"{run_dir.name}: missing {name}")
            return default
        return json.loads(p.read_text())
    # db_courses.json is written by the API when the run came from saved
    # courses; a bare CLI run has no course codes and the column stays empty
    return (read("report.json"), read("params.json"),
            read("inputs.json", {}), read("db_courses.json", []))


def _rows_skill_status(report: dict) -> list[dict]:
    rows = []
    for rank, s in enumerate(report.get("skills", []), 1):
        v = s.get("verify") or {}
        rows.append({
            "rank": rank,
            "market_skill": s.get("skill", ""),
            "job_count": s.get("jobs", ""),
            "program_score": s.get("program", ""),
            "status": s.get("status", ""),
            "best_course": s.get("best_course") or "",
            "best_clo": s.get("best_clo") or "",
            "rerank_score": v.get("best_rerank", ""),
            "rerank_verdict": VERDICT.get(v.get("verdict"), v.get("verdict") or ""),
            "host_ok": str(bool(s.get("host_ok"))).lower(),
            "priority": s.get("priority", ""),
        })
    return rows


def _rows_heatmap(report: dict, run_id: str, cohort: str) -> list[dict]:
    covered_at = (report.get("thresholds") or {}).get("covered", 0.73)
    border_at = (report.get("thresholds") or {}).get("borderline", 0.69)
    rows = []
    for rank, s in enumerate(report.get("skills", []), 1):
        per = s.get("per_course") or {}
        best = s.get("best_course")
        for course, cos in per.items():
            rows.append({
                "run_id": run_id,
                "cohort": cohort,
                "rank": rank,
                "market_skill": s.get("skill", ""),
                "job_count": s.get("jobs", ""),
                "course_name": course,
                "cosine": cos,
                "is_row_max": str(course == best).lower(),
                "cell_status": ("covered" if cos >= covered_at
                                else "borderline" if cos >= border_at
                                else "gap"),
            })
    return rows


def _rows_recommendations(report: dict) -> list[dict]:
    rows = []
    for step, p in enumerate((report.get("selection") or {}).get("plan", []), 1):
        route = p.get("route") or {}
        rel = p.get("relationship") or {}
        act = p.get("recommended_action") or {}
        spill = ";".join(f"{x.get('skill')}:{x.get('to')}"
                         for x in (p.get("spillover") or []))
        rows.append({
            "step": step,
            "skill": p.get("skill", ""),
            "job_count": p.get("jobs", ""),
            "selected": str(bool(p.get("selected"))).lower(),
            "gain_pct": p.get("gain_pct", ""),
            "kl_after": p.get("kl_after", ""),
            "covered_after": p.get("covered_after", ""),
            "route_kind": route.get("kind") or "",
            "route_course": route.get("where") or "",
            "route_clo": route.get("clo") or "",
            "route_score": route.get("score", ""),
            "relationship_kind": rel.get("kind") or "",
            "action_kind": act.get("kind") or "",
            "action_title": act.get("title") or "",
            "spillover": spill,
        })
    return rows


def _rows_course_manifest(report: dict, params: dict, inputs: dict,
                          db_courses: list) -> list[dict]:
    """Join report.courses (display names, sorted-glob order) to
    inputs.curriculum.courses (file + sha256, same order) and, when the run
    came from the DB, to the course codes captured at staging time.

    The DB join is keyed on FILENAME, never on position: staged files are
    named `<n>_<slug>.json` and sort lexicographically, so with 10+ courses
    (Y3+Y4, Y3-Y5) `10_…` precedes `2_…` and an index join would silently
    attach the wrong code to the wrong course."""
    run_id = params.get("run_id", "")
    cohort = params.get("cohort") or report.get("cohort") or ""
    names = report.get("courses", [])
    files = ((inputs.get("curriculum") or {}).get("courses") or [])
    by_file = {c.get("file"): c for c in db_courses if c.get("file")}
    rows = []
    for i, name in enumerate(names):
        f = files[i] if i < len(files) else {}
        db = by_file.get(f.get("file"), {})
        rows.append({
            "run_id": run_id,
            "cohort": cohort,
            "course_code": db.get("course_code") or "",
            "course_name": name,
            "study_year": db.get("year") or "",
            "n_clo_skills": f.get("n_skills", db.get("n_skills", "")),
            "source_file": f.get("file", ""),
            "sha256": f.get("sha256", ""),
        })
    return rows


def _summary(report: dict, params: dict, inputs: dict) -> dict:
    p = params.get("params", {})
    thr = report.get("thresholds") or {}
    sel = report.get("selection") or {}
    plan = sel.get("plan", [])
    chosen = [s for s in plan if s.get("selected")]

    statuses = [s.get("status") for s in report.get("skills", [])]
    verdicts = [((s.get("verify") or {}).get("verdict")) for s in report.get("skills", [])]
    hosts = [bool(s.get("host_ok")) for s in report.get("skills", [])]
    top_n = report.get("top_n") or len(statuses) or 1

    # demand_weighted_coverage lives in report.json for runs after this field
    # was added; derive it from the skill rows for older bundles so re-export
    # never requires a re-run.
    skill_rows = report.get("skills", [])
    top_n_jobs = sum(s.get("jobs", 0) for s in skill_rows)
    dwc = report.get("demand_weighted_coverage")
    dwc_half = report.get("demand_weighted_coverage_half_credit")
    if dwc is None and top_n_jobs:
        dwc = sum(s.get("jobs", 0) for s in skill_rows
                  if s.get("status") == "covered") / top_n_jobs
    if dwc_half is None and top_n_jobs:
        dwc_half = sum(
            s.get("jobs", 0) * (1.0 if s.get("status") == "covered"
                                 else 0.5 if s.get("status") == "borderline" else 0.0)
            for s in skill_rows) / top_n_jobs

    kl0 = sel.get("kl0", report.get("kl"))
    kl_final = chosen[-1].get("kl_after") if chosen else kl0
    cov_final = chosen[-1].get("covered_after") if chosen else sel.get("covered0")
    abs_impr = round(kl0 - kl_final, 4) if (kl0 and kl_final) else None
    rel_impr = (round(abs_impr / kl0 * 100, 1)
                if (abs_impr is not None and kl0) else None)

    started = params.get("started_at") or ""
    return {
        "run_id": params.get("run_id", ""),
        "cohort": params.get("cohort") or report.get("cohort") or "",
        "label": params.get("label", ""),
        "experiment_date": started[:10],
        "started_at": started,
        "duration_s": params.get("duration_s"),

        "provenance": {
            "git_sha": params.get("git_sha", ""),
            "config_hash": params.get("config_hash", ""),
            "input_hash": params.get("input_hash", ""),
            "cache_version": p.get("cache_version", ""),
        },
        "curriculum": {
            "source": (inputs.get("curriculum") or {}).get("mode", ""),
            "n_courses": len(report.get("courses", [])),
            "n_clo_skills": report.get("n_clo_skills"),
            "dir": (inputs.get("curriculum") or {}).get("dir", ""),
        },
        "job_market": {
            "dataset": report.get("dataset", ""),
            "source": report.get("source", ""),
            "segment_mode": report.get("segment_mode", ""),
            "n_jobs": report.get("n_jobs"),
            "top_n": report.get("top_n"),
            "min_skill_freq": p.get("min_skill_freq"),
        },
        "configuration": {
            "bi_encoder": p.get("encoder", ""),
            "cross_encoder": p.get("reranker", report.get("reranker", "")),
            "skill_merge_threshold": p.get("skill_merge_threshold"),
            "covered_at": thr.get("covered", p.get("covered_at")),
            "borderline_at": thr.get("borderline", p.get("borderline_at")),
            "host_min": thr.get("host_min", p.get("host_min")),
            "rerank_supported_at": p.get("rerank_supported_at"),
            "rerank_weak_at": p.get("rerank_weak_at"),
            "rerank_top_k": p.get("rerank_top_k"),
            "eps": p.get("eps"),
            "cum_frac": p.get("cum_frac"),
            "spill_min": p.get("spill_min"),
            "thresholds_status": THRESHOLDS_STATUS,
        },
        "measured_results": {
            "kl_divergence": report.get("kl"),
            "covered": statuses.count("covered"),
            "borderline": statuses.count("borderline"),
            "gap": statuses.count("gap"),
            "coverage_rate_percent": round(statuses.count("covered") / top_n * 100, 1),
            "demand_weighted_coverage_percent":
                round(dwc * 100, 1) if dwc is not None else None,
            "demand_weighted_coverage_half_credit_percent":
                round(dwc_half * 100, 1) if dwc_half is not None else None,
            "host_verified": hosts.count(True),
            "host_unverified": hosts.count(False),
            "rerank_supported": verdicts.count("SUPPORTED"),
            "rerank_weak": verdicts.count("WEAK"),
            "rerank_not_supported": verdicts.count("NOT SUPPORTED"),
        },
        "modelled_results": {
            "_note": MODELLED_NOTE,
            "gaps_considered": len(plan),
            "n_selected": len(chosen),
            "stop_rule": f"{sel.get('cum_frac')} of achievable gain",
            "kl_final": kl_final,
            "covered_final": cov_final,
            "kl_absolute_improvement": abs_impr,
            "kl_relative_improvement_percent": rel_impr,
            "routes": _count(plan, lambda s: (s.get("route") or {}).get("kind")),
            "actions": _count(plan, lambda s: (s.get("recommended_action") or {}).get("kind")),
        },
        "evaluation_ref": {
            "_note": EVAL_NOTE,
            "calibration_record": "05_evaluation/llm_judge_itc_results.json",
            "gold_status": "silver only (LLM judges, Cohen k=0.83); human gold 0/150",
        },
        "known_limits": KNOWN_LIMITS,
    }


def _count(items, key) -> dict:
    out: dict = {}
    for it in items:
        k = key(it)
        if k:
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    # csv defaults to \r\n; thesis_registry.csv is git-tracked, so force LF
    # or every save shows up as a whole-file change
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


REGISTRY_COLS = [
    "cohort", "run_id", "experiment_date", "git_sha", "config_hash",
    "input_hash", "n_courses", "n_clo_skills", "n_jobs", "top_n", "encoder",
    "covered", "borderline", "gap", "coverage_rate_percent",
    "demand_weighted_coverage_percent",
    "demand_weighted_coverage_half_credit_percent", "kl",
    "n_selected", "kl_final_modelled", "covered_final_modelled",
]


def build_registry() -> Path:
    """Rebuild the cross-cohort table from every exported bundle."""
    rows = []
    for d in sorted(OUTPUTS.glob("*/thesis/run_summary.json")):
        s = json.loads(d.read_text())
        m, mo = s["measured_results"], s["modelled_results"]
        rows.append({
            "cohort": s["cohort"], "run_id": s["run_id"],
            "experiment_date": s["experiment_date"],
            "git_sha": s["provenance"]["git_sha"],
            "config_hash": s["provenance"]["config_hash"][:8],
            "input_hash": s["provenance"]["input_hash"][:8],
            "n_courses": s["curriculum"]["n_courses"],
            "n_clo_skills": s["curriculum"]["n_clo_skills"],
            "n_jobs": s["job_market"]["n_jobs"],
            "top_n": s["job_market"]["top_n"],
            "encoder": s["configuration"]["bi_encoder"],
            "covered": m["covered"], "borderline": m["borderline"],
            "gap": m["gap"],
            "coverage_rate_percent": m["coverage_rate_percent"],
            "demand_weighted_coverage_percent":
                m.get("demand_weighted_coverage_percent"),
            "demand_weighted_coverage_half_credit_percent":
                m.get("demand_weighted_coverage_half_credit_percent"),
            "kl": m["kl_divergence"], "n_selected": mo["n_selected"],
            "kl_final_modelled": mo["kl_final"],
            "covered_final_modelled": mo["covered_final"],
        })
    # cohort order first (Y3, Y4, Y5, combined), then run time
    rows.sort(key=lambda r: (len(r["cohort"]), r["cohort"], r["run_id"]))
    _write_csv(REGISTRY, [{c: r.get(c, "") for c in REGISTRY_COLS} for r in rows])
    return REGISTRY


def export(run_id: str) -> Path:
    run_dir = OUTPUTS / run_id
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_id}")
    report, params, inputs, db_courses = _load(run_dir)
    out = run_dir / "thesis"
    out.mkdir(exist_ok=True)
    cohort = params.get("cohort") or report.get("cohort") or ""

    summary = _summary(report, params, inputs)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out / "course_manifest.csv",
               _rows_course_manifest(report, params, inputs, db_courses))
    _write_csv(out / "skill_status.csv", _rows_skill_status(report))
    _write_csv(out / "heatmap_cells.csv",
               _rows_heatmap(report, params.get("run_id", run_id), cohort))
    _write_csv(out / "recommendations.csv", _rows_recommendations(report))
    build_registry()
    return out


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    arg = argv[0]
    if arg == "--all":
        ids = [d.name for d in sorted(OUTPUTS.iterdir())
               if d.is_dir() and (d / "report.json").is_file()]
    elif arg == "--latest":
        cands = [d for d in OUTPUTS.iterdir()
                 if d.is_dir() and (d / "report.json").is_file()]
        if not cands:
            raise SystemExit("no runs in outputs/")
        ids = [max(cands, key=lambda d: d.name).name]
    else:
        ids = [arg]

    for rid in ids:
        out = export(rid)
        s = json.loads((out / "run_summary.json").read_text())
        m = s["measured_results"]
        print(f"{rid}  {s['cohort'] or '-':<6} "
              f"KL={m['kl_divergence']}  covered={m['covered']}/"
              f"{s['job_market']['top_n']}  -> {out.relative_to(OUTPUTS.parent)}/")
    print(f"registry -> {REGISTRY.relative_to(OUTPUTS.parent)} "
          f"({len(ids)} run(s) exported)")


if __name__ == "__main__":
    main(sys.argv[1:])
