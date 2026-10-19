# Compute the program-level gap analysis (market cache x all curated CLO
# files in 02_input_curriculum/clo_skills/) and write dashboard/report_data.js
# for the React page (dashboard/index.html).
#
#   python3 04_report/gen_report_data.py       (from the phase5_curriculum_alignment root)
#
# Every run ALSO lands in outputs/<run_id>/ as an immutable record of both
# sides of the experiment — the resolved parameters and the results (see
# run_store.py). report_data.js keeps being overwritten in place because
# index.html script-tags it; the run directory is what you keep.

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 04_report/ (run_store, heatmap) is this script's own folder; the engine lives
# in the sibling 03_engine/. Numbered folders can't be Python packages, so both
# go on sys.path and are imported by bare module name.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "03_engine"))

import run_store

from pipeline import (
    BORDERLINE_AT, COVERED_AT, DEFAULT_SOURCE, JOB_SOURCE, MODEL_NAME, RAW_JOBS,
    RERANK_SUPPORTED_AT, RERANK_TOP_K, RERANK_WEAK_AT, RERANKER_NAME,
    SEGMENT_MODE, as_text, build_market, greedy_skill_selection, is_soft_skill,
    load_model, load_raw_postings, kl_divergence, rerank_pair_scores,
    rerank_verdict,
)

# phase5 root — this script lives in 04_report/, so the data paths below stay
# anchored on the project root exactly as they were before the folder reorg
BASE_DIR = _HERE.parent
# CLO_SKILLS_DIR env var overrides the curated dir (the API sets it when the
# user runs on courses pulled from PostgreSQL instead of the curated files).
CLO_DIR = Path(os.environ.get("CLO_SKILLS_DIR")
               or BASE_DIR / "02_input_curriculum" / "clo_skills")
# Per-source output: the default (linkedin) keeps the historical filenames
# (committed baseline); extra corpora get a .<source> suffix so both coexist.
_SRC_SUFFIX = "" if JOB_SOURCE == DEFAULT_SOURCE else f".{JOB_SOURCE}"
# REPORT_OUT env var redirects output (the API sets it to a gitignored path
# for live runs so the committed report_data.js baseline is never rewritten).
OUT = Path(os.environ.get("REPORT_OUT")
           or BASE_DIR / "dashboard" / f"report_data{_SRC_SUFFIX}.js")
# preview_data.js is regenerated alongside the report for non-default sources
# (the committed linkedin preview is left untouched unless explicitly asked).
# PREVIEW_OUT env var redirects it, same as REPORT_OUT — the API points both
# at out/ so a live run never rewrites a committed file.
PREVIEW_OUT = Path(os.environ.get("PREVIEW_OUT")
                   or BASE_DIR / "dashboard" / f"preview_data{_SRC_SUFFIX}.js")
# Study-year cohort of the curriculum in this run (e.g. "Y3", "Y3+Y4"),
# derived by the API from the ACTUALLY-selected DB courses and passed through
# so the record can never disagree with what ran. None for a bare
# `python3 gen_report_data.py` on the curated files (year is not in those).
COHORT = os.environ.get("RUN_COHORT") or None
# optional manual relationship decisions (presentation/routing only —
# never affects ranking, selection, gains, KL or covered counts)
ADJUDICATION_FILE = BASE_DIR / "05_evaluation" / "relationship_adjudication.json"
TOP_N = 30  # heatmap now offers a top-20/25/30 view selector; carry 30 rows
# Selection stopping rule — module scope (not main()-local) so run_store
# can record them: they change the plan, so they are parameters.
CUM_FRAC = 0.80   # keep adding skills until 80% of achievable gain
SPILL_MIN = 0.05  # min coverage lift to report as spillover
from pipeline import get_thresholds
HOST_MIN_SIM = get_thresholds()["host_min"]  # below: host is noise


def course_name(path: Path) -> str:
    return path.stem.split("_", 1)[1].replace("_", " ").title()


def load_adjudication() -> dict:
    """Optional per-skill manual relationship decisions. Missing or
    invalid file -> empty dict (defaults apply, never crashes)."""
    try:
        data = json.loads(ADJUDICATION_FILE.read_text())
        decisions = data.get("decisions", {})
        return decisions if isinstance(decisions, dict) else {}
    except (OSError, ValueError):
        return {}


def relationship_for(skill: str, route: dict, adjudication: dict):
    """Additive presentation fields: what related curriculum content
    exists (relationship) vs what must be added (recommended_action).
    Defaults derive from the existing route; adjudication entries merge
    over them. A related foundation is NOT coverage of the target —
    addition_required stays true for every recommended (non-covered)
    skill."""
    course, clo, score = route["where"], route["clo"], route["score"]
    if route["kind"] == "extend":
        rel = {"kind": "content_exists", "target_status": "partial",
               "reason": f'Verified related content exists in {course} '
                         f'("{clo}"), but the target skill is still '
                         f"below coverage — explicit additions are "
                         f"required."}
        act = {"kind": "extend",
               "title": f"Extend {course} with explicit {skill} content",
               "detail": f'Add {skill} outcomes alongside "{clo}" in '
                         f"{course}."}
    elif route["kind"] == "partial":
        rel = {"kind": "foundation", "target_status": "missing",
               "reason": f'Related foundation exists in {course} '
                         f'("{clo}"), but broader {skill} outcomes are '
                         f"not explicitly evidenced."}
        act = {"kind": "add_module",
               "title": f"Add {skill} content building on {course}",
               "detail": f"Introduce explicit {skill} outcomes on top "
                         f"of the existing foundation; do not duplicate "
                         f"{course}."}
    else:
        rel = {"kind": "none", "target_status": "missing",
               "reason": "No existing CLO shows verified evidence of "
                         "teaching this skill."}
        act = {"kind": "new_content",
               "title": f"Add new {skill} content",
               "detail": f"Introduce a CLO or module that explicitly "
                         f"teaches {skill}."}
    rel.update({"addition_required": True, "course": course, "clo": clo,
                "score": score,
                "evidence_source": ("cross_encoder" if score is not None
                                    else "fallback")})
    adj = adjudication.get(skill, {})
    rel = {**rel, **adj.get("relationship", {})}
    act = {**act, **adj.get("recommended_action", {})}
    return rel, act


def write_preview() -> None:
    """Regenerate preview_data.<source>.js — the Job-Market Dataset Preview
    (first 5 usable postings) + the curriculum preview the V8 tab reads.
    The curriculum side is source-independent; only the job side differs."""
    raw = load_raw_postings(RAW_JOBS)
    jobs_preview = []
    for out in raw:
        title = as_text(out.get("job_title")).strip()
        skills = [str(s).strip() for s in (out.get("skills") or []) if str(s).strip()]
        if not title or not skills:
            continue
        jobs_preview.append({
            "job_title": title,
            "company_name": as_text(out.get("company_name")).strip(),
            "location": as_text(out.get("location")).strip(),
            "skills": skills[:4],
        })
        if len(jobs_preview) >= 5:
            break

    courses_preview = []
    for f in sorted(CLO_DIR.glob("*.json")):
        items = json.loads(f.read_text())
        all_skills = [it["skill"] for it in items if it.get("skill")]
        courses_preview.append({
            "course": course_name(f),
            "file": f.name,
            "n_skills": len(all_skills),
            "sample_skills": all_skills[:5],
            "all_skills": all_skills,
        })

    PREVIEW_OUT.write_text(
        "window.JOB_PREVIEW = " + json.dumps(jobs_preview, indent=2) + ";\n"
        "window.JOB_TOTAL = " + str(len(raw)) + ";\n"
        "window.CURRICULUM_PREVIEW = " + json.dumps(courses_preview, indent=2)
        + ";\n")
    print(f"written {PREVIEW_OUT.name}: {len(jobs_preview)} preview jobs, "
          f"{len(courses_preview)} courses, job_total={len(raw)}")


def main() -> dict:
    print(f"Loading model '{MODEL_NAME}' + market (cached) ...")
    model = load_model(MODEL_NAME)
    market = build_market(model)

    courses, clo_texts, clo_owner = [], [], []
    for f in sorted(CLO_DIR.glob("*.json")):
        name = course_name(f)
        courses.append(name)
        for item in json.loads(f.read_text()):
            clo_texts.append(item["skill"])
            clo_owner.append(name)
    print(f"Curriculum: {len(courses)} courses, {len(clo_texts)} CLO skills")

    clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                           show_progress_bar=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = market["skill_emb"] @ clo_emb.T   # (market skills, CLO items)

    base, counts = market["skill_base"], market["counts"]
    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top = sorted(hard_idx, key=lambda i: -counts[base[i]])[:TOP_N]

    # cross-encoder host verification, computed INLINE against this run's
    # own curriculum and top skills (verifies WHICH CLO can host a skill —
    # coverage statuses stay cosine-based). Replaces the
    # 05_evaluation/rerank_results.json sidecar, which went stale whenever the
    # curriculum changed (it carried five-course verdicts into DB runs).
    # RERANK=0 skips it (e.g. first-run model download unwanted).
    verify, bands, reranker = {}, None, None
    if os.environ.get("RERANK", "1") != "0":
        try:
            pairs, cand_lists = [], []
            for i in top:
                cand = np.argsort(-sim[i])[:RERANK_TOP_K]
                cand_lists.append(cand)
                pairs.extend((base[i], clo_texts[j]) for j in cand)
            scores = rerank_pair_scores(pairs)
            k = 0
            for i, cand in zip(top, cand_lists):
                cands = []
                for j in cand:
                    cands.append({"clo": clo_texts[j],
                                  "course": clo_owner[j],
                                  "cosine": round(float(sim[i, j]), 2),
                                  "rerank": round(float(scores[k]), 2)})
                    k += 1
                by_rr = sorted(cands, key=lambda c: -c["rerank"])
                verify[base[i]] = {"cosine_best": cands[0],
                                   "rerank_best": by_rr[0],
                                   "verdict": rerank_verdict(
                                       by_rr[0]["rerank"])}
            bands = {"supported": RERANK_SUPPORTED_AT,
                     "weak": RERANK_WEAK_AT,
                     "note": "PROVISIONAL until task #1 annotation"}
            reranker = RERANKER_NAME
            print(f"Host verification: {len(verify)} skills x "
                  f"{RERANK_TOP_K} CLOs, inline ({reranker})")
        except Exception as e:      # degrade to no badges, never crash a run
            verify, bands, reranker = {}, None, None
            print(f"(host verification skipped: {e})")

    demand = market["demand_dist"][hard_idx]
    demand = demand / demand.sum()
    profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)
    kl, _ = kl_divergence(demand, profile / profile.sum())

    skills = []
    for i in top:
        row = sim[i]
        j = int(row.argmax())
        program_cov = float(row[j])
        per_course = {}
        # which CLO inside each course produced that course's max — the UI
        # shows it on cell click so a score is always traceable to the text
        # it was compared against. Display only; changes no metric.
        per_course_clo = {}
        for c in courses:
            idx = [k for k, o in enumerate(clo_owner) if o == c]
            kbest = max(idx, key=lambda k: row[k])
            per_course[c] = round(float(row[kbest]), 2)
            per_course_clo[c] = clo_texts[kbest]
        status = ("covered" if program_cov >= COVERED_AT else
                  "borderline" if program_cov >= BORDERLINE_AT else "gap")
        share = counts[base[i]] / market["n_jobs"]
        v = verify.get(base[i])
        skills.append({
            "skill": base[i],
            "jobs": counts[base[i]],
            "per_course": per_course,
            "per_course_clo": per_course_clo,
            "program": round(program_cov, 2),
            "status": status,
            "host_ok": program_cov >= HOST_MIN_SIM,
            "best_course": clo_owner[j],
            "best_clo": clo_texts[j],
            "priority": round(share * max(0.0, 1.0 - program_cov), 4),
            "verify": ({
                "argmax_rerank": v["cosine_best"]["rerank"],
                "best_clo": v["rerank_best"]["clo"],
                "best_course": v["rerank_best"]["course"],
                "best_rerank": v["rerank_best"]["rerank"],
                "verdict": v["verdict"],
            } if v else None),
        })

    # ALL non-covered skills, ranked by priority — the UI renders the
    # first 3 as action cards and the rest as a compact ranked table
    recs = sorted((s for s in skills if s["status"] != "covered"),
                  key=lambda s: -s["priority"])
    recommendations = []
    for r in recs:
        v = r["verify"]
        if v:                       # cross-encoder governs the host decision
            hosted = v["verdict"] == "SUPPORTED"
            host, host_clo = ((v["best_course"], v["best_clo"]) if hosted
                              else (None, None))
        else:                       # no verification data: cosine gate
            hosted = r["host_ok"]
            host, host_clo = ((r["best_course"], r["best_clo"]) if hosted
                              else (None, None))
        recommendations.append({
            "skill": r["skill"], "jobs": r["jobs"],
            "coverage": r["program"], "status": r["status"],
            "no_host": not hosted, "host": host, "best_clo": host_clo,
            "argmax_clo": r["best_clo"], "argmax_course": r["best_course"],
            "verify": v,
        })

    # ---- prescriptive selection: WHICH gaps are worth adding? --------
    # Greedy facility-location over the whole demand distribution.
    # Not every gap earns a slot: keep adding skills until CUM_FRAC of
    # the total achievable gain is reached (80/20 budget rule; the full
    # ranked list stays in the report for the sensitivity discussion).
    cand = [(i, s) for i, s in zip(top, skills) if s["status"] != "covered"]
    selection = None
    if cand:
        cand_glob = [i for i, _ in cand]
        cand_names = [s["skill"] for _, s in cand]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cand_sim = np.clip(market["skill_emb"][cand_glob]
                               @ market["skill_emb"][hard_idx].T, 0.0, None)
        steps = greedy_skill_selection(demand, profile, cand_names,
                                       cand_sim)
        pos = {g: p for p, g in enumerate(hard_idx)}
        top_pos = [pos[i] for i in top]
        tpos = {s["skill"]: p for p, s in zip(top_pos, skills)}
        by_name = {s["skill"]: s for s in skills}
        gap0 = float((demand * (1.0 - profile)).sum())
        total_gain = sum(st["gain"] for st in steps)
        adjudication = load_adjudication()
        # decisions are authored against a specific curriculum — drop any
        # that reference a course not in this run (same staleness class
        # as the old rerank sidecar)
        adjudication = {s: d for s, d in adjudication.items()
                        if d.get("relationship", {}).get("course")
                        in set(courses)}
        if adjudication:
            print(f"Relationship adjudication: {len(adjudication)} "
                  f"manual decisions (presentation only)")
        prev, plan, cut_profile, cum = profile, [], None, 0.0
        for st in steps:
            selected = cum < CUM_FRAC * total_gain
            cum += st["gain"]
            prof = st["profile"]
            kl_s, _ = kl_divergence(demand, prof / prof.sum())
            spill = []
            for p, sk in zip(top_pos, skills):
                if sk["skill"] == st["skill"]:
                    continue
                if float(prof[p] - prev[p]) >= SPILL_MIN:
                    spill.append({
                        "skill": sk["skill"],
                        "to": round(float(prof[p]), 2),
                        "flipped": bool(prev[p] < COVERED_AT <= prof[p]),
                    })
            # three-level route (legacy field, kept for compatibility):
            # extend = verified related content; partial = weak
            # foundation exists; new = no verified host found.
            v = by_name[st["skill"]]["verify"]
            if v and v["verdict"] == "SUPPORTED":
                route = {"kind": "extend", "where": v["best_course"],
                         "clo": v["best_clo"], "score": v["best_rerank"]}
            elif v and v["verdict"] == "WEAK":
                route = {"kind": "partial", "where": v["best_course"],
                         "clo": v["best_clo"], "score": v["best_rerank"]}
            else:
                route = {"kind": "new", "where": None, "clo": None,
                         "score": v["best_rerank"] if v else None}
            rel, act = relationship_for(st["skill"], route, adjudication)
            plan.append({
                "skill": st["skill"],
                "jobs": by_name[st["skill"]]["jobs"],
                "gain_pct": round(st["gain"] / gap0 * 100, 1),
                "kl_after": round(kl_s, 4),
                "covered_after": int(sum(prof[p] >= COVERED_AT
                                         for p in top_pos)),
                "selected": bool(selected),
                "spillover": spill[:4],
                "route": route,
                "relationship": rel,
                "recommended_action": act,
            })
            if selected:
                cut_profile = prof
            prev = prof
        # why each skipped skill was left out
        for e, st in zip(plan, steps):
            if e["selected"]:
                continue
            pv = float(cut_profile[tpos[e["skill"]]])
            e["cov_after_plan"] = round(pv, 2)
            e["absorbed_by"] = None
            if pv >= COVERED_AT:
                for e2, st2 in zip(plan, steps):
                    if not e2["selected"]:
                        break
                    if float(st2["profile"][tpos[e["skill"]]]) >= COVERED_AT:
                        e["absorbed_by"] = e2["skill"]
                        break
        selection = {
            "cum_frac": CUM_FRAC,
            "n_selected": sum(e["selected"] for e in plan),
            "kl0": round(kl, 4),
            "covered0": int(sum(s["status"] == "covered" for s in skills)),
            "plan": plan,
        }

    # Demand-weighted coverage: share of TOP_N's own job-count mass that is
    # covered, vs the plain count-of-skills coverage_rate. KL is computed over
    # the full 571-skill segment and saturates near the "teaches nothing"
    # baseline at this scale (see readme.md); this is the metric that
    # actually separates cohorts by how much of what employers ask for (in
    # the top-30) is taught, not just how many of 30 boxes are ticked.
    top_n_jobs = sum(s["jobs"] for s in skills)
    covered_jobs = sum(s["jobs"] for s in skills if s["status"] == "covered")
    borderline_jobs = sum(s["jobs"] for s in skills if s["status"] == "borderline")
    demand_weighted_coverage = covered_jobs / top_n_jobs if top_n_jobs else 0.0
    demand_weighted_coverage_half_credit = (
        (covered_jobs + 0.5 * borderline_jobs) / top_n_jobs if top_n_jobs else 0.0)

    report = {
        "dataset": RAW_JOBS.name,
        "source": JOB_SOURCE,
        "cohort": COHORT,
        "segment_mode": SEGMENT_MODE,
        "n_jobs": market["n_jobs"],
        "courses": courses,
        "n_clo_skills": len(clo_texts),
        "kl": round(kl, 4),
        "covered": sum(s["status"] == "covered" for s in skills),
        "demand_weighted_coverage": round(demand_weighted_coverage, 4),
        "demand_weighted_coverage_half_credit":
            round(demand_weighted_coverage_half_credit, 4),
        "top_n": TOP_N,
        "skills": skills,
        "recommendations": recommendations,
        "thresholds": {"covered": COVERED_AT, "borderline": BORDERLINE_AT,
                       "host_min": HOST_MIN_SIM},
        "reranker": reranker,
        "rerank_bands": bands,
        "selection": selection,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.GAP_REPORT = "
                   + json.dumps(report, indent=1) + ";\n")
    print(f"written {OUT.name}: covered {report['covered']}/{TOP_N}, "
          f"demand-weighted {demand_weighted_coverage*100:.1f}%, KL={kl:.4f}")
    # regenerate the matching preview for extra corpora; the committed
    # linkedin preview_data.js is deliberately left untouched
    if JOB_SOURCE != DEFAULT_SOURCE:
        write_preview()
    if selection:
        print(f"\nSELECTION (greedy facility-location, stop at "
              f"{CUM_FRAC:.0%} of achievable gain): "
              f"add {selection['n_selected']}/{len(selection['plan'])} gaps")
        for e in selection["plan"]:
            tag = "ADD " if e["selected"] else "skip"
            lift = ", ".join(f"{s['skill']}->{s['to']}"
                             for s in e["spillover"][:3])
            rel, act = e["relationship"], e["recommended_action"]
            print(f"  {tag} {e['skill']:<24} closes {e['gain_pct']:>4}% "
                  f"KL->{e['kl_after']:.4f} covered->{e['covered_after']} "
                  f"[{rel['kind']} -> {act['kind']}: {act['title']}]"
                  + (f"  lifts: {lift}" if lift else ""))
    return report


def run() -> Path:
    """Run the analysis and archive it as outputs/<run_id>/.

    Resolves the parameters BEFORE the work so the record is built from
    the same values the pipeline is about to use, rather than
    reconstructed afterwards from whatever the report happened to carry.
    """
    run_store.ensure_gitignore()
    params = run_store.resolve_params(TOP_N, CUM_FRAC, SPILL_MIN)
    inputs = run_store.resolve_inputs(CLO_DIR, RAW_JOBS, ADJUDICATION_FILE)
    # stamp the derived study-year cohort onto the curriculum record (kept out
    # of input_hash: it is a label for the same files, not a distinct input)
    inputs["curriculum"]["cohort"] = COHORT
    cfg, inp = run_store.config_hash(params), run_store.input_hash(inputs)
    meta = {
        "run_id": run_store.new_run_id(cfg),
        "label": os.environ.get("RUN_LABEL") or None,
        "cohort": COHORT,
        "config_hash": cfg,
        "input_hash": inp,
        "git_sha": run_store.git_sha(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_data_js": str(OUT),
    }

    # tee, never redirect: the API streams this process's stdout line by
    # line into the V8 tab's terminal view
    tee = run_store.Tee(sys.stdout)
    sys.stdout = tee
    t0 = time.time()
    try:
        print(f"run {meta['run_id']}  config {run_store.short(cfg)} "
              f"input {run_store.short(inp)}")
        earlier = run_store.find_run(cfg, inp)
        if earlier:
            print(f"note: identical params+inputs already ran as {earlier} "
                  f"— recomputing anyway (direct invocation)")
        report = main()
    finally:
        sys.stdout = tee.stream

    meta["duration_s"] = round(time.time() - t0, 1)
    run_dir = run_store.write_run(report, params, inputs, meta,
                                  tee.buf.getvalue())
    print(f"run archived -> outputs/{meta['run_id']}/ "
          f"(report.md, report.json, params.json, inputs.json, log.txt)")
    _write_heatmap(report, run_dir)
    return run_dir


def _write_heatmap(report: dict, run_dir: Path) -> None:
    """Drop a heatmap.png beside the run's report.json so every run ships its
    figure — no separate `heatmap.py` call needed. The report dict is
    passed straight in (same source as report.json). Never fatal: the run is
    already archived, so a missing matplotlib only skips the picture.
    """
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "phase6_heatmap", _HERE / "heatmap.py")   # sibling in 04_report/
        hm = _ilu.module_from_spec(spec)
        spec.loader.exec_module(hm)
        hm.draw(report, TOP_N, run_dir / "heatmap.png")
        print(f"heatmap  -> outputs/{run_dir.name}/heatmap.png")
    except Exception as e:                                       # noqa: BLE001
        print(f"heatmap skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    run()
