# Data-scope + duplicate-posting sensitivity for the thesis revision.
#
#   python3 05_evaluation/dedup_sensitivity.py        (from the phase5_curriculum_alignment root)
#
# Examiner comment on Table 3.1: "address the scope of data clearly, for
# which duration the jobs are active and how you prevented duplications of
# the same job across different platforms — your DWC is defined on these
# weights". Two things are needed and this script produces both:
#
#   1. DATA SCOPE — per-site posting-date profile of the crawl (what the
#      snapshot actually contains), from the `posted_date` field of the
#      Phase-3 records. Some portals expose a closing date rather than a
#      posting date; that is reported, not hidden.
#   2. DUPLICATE SENSITIVITY — the locked pipeline deduplicates on content
#      (title + extracted skill set, pipeline.clean_jobs). That misses a job
#      cross-posted on two portals when the two HTML pages yield different
#      skill lists. So: rebuild the market under STRICTER duplicate rules
#      keyed on (title, employer) and re-run the four locked cohorts. If the
#      top-30 set, covered counts and demand-weighted coverage barely move,
#      duplicates cannot be what drives the weights.
#
# Like sweep_top_n.py this is a RE-SLICE with one thing changed (the set of
# postings that count), not a re-run: same encoder, same curriculum files,
# same calibrated thresholds. The `locked` variant is served from the
# committed market cache and must reproduce the 2026-07-30 table exactly or
# the script refuses to publish.
#
# One caveat that shapes the design. The raw→canonical skill mapping in
# pipeline.normalize_skills is greedy over `Counter.most_common()`, whose
# tie order is the Counter's insertion order, which comes from iterating
# `set(j["skills"])` — i.e. it depends on PYTHONHASHSEED. The committed
# cache therefore cannot be re-derived bit-for-bit from the raw records
# (verified: seeds 0/1/2 give 1470/1472/1474 canonical entries; the same
# seed twice gives the identical mapping). So the mapping is refitted ONCE
# here with a deterministic tie order and held fixed across every variant,
# and the deltas are read against `refit_locked` (same postings as the
# published run, refit mapping). `refit_locked` vs `locked` is reported
# separately as the mapping drift, so the two effects are never confused.

import collections
import datetime as dt
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE_DIR = _HERE.parent
sys.path.insert(0, str(BASE_DIR / "03_engine"))
os.environ.setdefault("JOB_SOURCE", "cambodia")
os.environ.setdefault("SEGMENT_MODE", "category")

from pipeline import (                                        # noqa: E402
    MIN_SKILL_FREQ, MODEL_NAME, RAW_JOBS, build_market, clean_jobs,
    get_thresholds, is_it_by_category, is_soft_skill, kl_divergence,
    normalize_skills,
)
from sweep_top_n import (                                     # noqa: E402
    BASELINE_N, COHORTS, LOCKED, cohort_files, evaluate, load_cohort,
    load_model_cpu, tie_analysis, verify_locked,
)

OUT_JSON = _HERE / "dedup_sensitivity_results.json"
OUT_MD = _HERE / "dedup_sensitivity.md"

CRAWL_DAY = dt.date(2026, 7, 15)   # the single-day snapshot in Table 3.1

# The eight portals of Table 3.1 and how their date field should be read.
# Verified by hand on the crawl: bongthom and hrincjobs print a closing
# date (every value falls AFTER the crawl day); cambojob keeps expired
# adverts online (dates back to 2020); camupjob and jobnet expose no
# machine-readable date on most pages.
DATE_FIELD = {
    "bongthom.com":  "closing date",
    "cambojob.com":  "posting date (site does not expire adverts)",
    "camhr.com":     "posting date",
    "camupjob.com":  "mostly absent",
    "hrincjobs.com": "closing date",
    "jobify.works":  "posting date",
    "jobnet.com.kh": "mostly absent / relative ('1 week ago')",
    "workingna.com": "posting date",
}

_FMTS = ("%d-%b-%Y", "%b-%d-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y",
         "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d")


def parse_date(value: str):
    s = (value or "").strip()
    for fmt in _FMTS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ----------------------------------------------------------- data scope
def date_scope(jobs: list[dict]) -> dict:
    """Per-site posting-date profile of the cleaned corpus (all 4,174
    records, not just the IT segment — the scope statement is about the
    crawl)."""
    per_site = collections.defaultdict(list)
    nulls, unparsed = Counter(), Counter()
    for j in jobs:
        site = j["site"]
        if not j["posted_date"]:
            nulls[site] += 1
            continue
        d = parse_date(j["posted_date"])
        if d is None:
            unparsed[site] += 1
        else:
            per_site[site].append(d)

    def pct(ds, f):
        return ds[min(len(ds) - 1, int(f * len(ds)))].isoformat()

    rows = []
    for site in sorted(set(per_site) | set(nulls) | set(unparsed)):
        ds = sorted(per_site[site])
        row = {"site": site, "n": sum(1 for j in jobs if j["site"] == site),
               "dated": len(ds), "no_date": nulls[site],
               "unparsed": unparsed[site], "field": DATE_FIELD.get(site, "?")}
        if ds:
            row.update(min=ds[0].isoformat(), median=pct(ds, 0.5),
                       max=ds[-1].isoformat(),
                       within_30d=sum(CRAWL_DAY - dt.timedelta(30) <= d
                                      <= CRAWL_DAY for d in ds),
                       after_crawl=sum(d > CRAWL_DAY for d in ds))
        rows.append(row)

    # posting-date sites only: the window the snapshot really samples
    posting_sites = [s for s, f in DATE_FIELD.items()
                     if f.startswith("posting date")]
    ds = sorted(d for s in posting_sites for d in per_site[s])
    win = {
        "sites": posting_sites, "dated": len(ds),
        "within_30d": sum(CRAWL_DAY - dt.timedelta(30) <= d <= CRAWL_DAY
                          for d in ds),
        "within_90d": sum(CRAWL_DAY - dt.timedelta(90) <= d <= CRAWL_DAY
                          for d in ds),
        "older_than_1y": sum(d < CRAWL_DAY - dt.timedelta(365) for d in ds),
        "median": pct(ds, 0.5) if ds else None,
    }
    return {"crawl_day": CRAWL_DAY.isoformat(), "per_site": rows,
            "posting_date_window": win}


# ------------------------------------------------------ duplicate rules
def dedup_variants(segment: list[dict]) -> dict[str, list[dict]]:
    """Three nested views of the IT segment. Each keeps the FIRST record
    of a duplicate group (file order = crawl order), so a variant is
    always a subset of `locked` and the canonical mapping stays valid."""
    key = lambda j: (norm(j["title"]), norm(j["company"]))   # noqa: E731

    cross, seen_sites = [], collections.defaultdict(set)
    for j in segment:
        k = key(j)
        if k[0] and k[1] and seen_sites[k] and j["site"] not in seen_sites[k]:
            continue                       # same advert seen on another portal
        seen_sites[k].add(j["site"])
        cross.append(j)

    reposts, seen = [], set()
    for j in segment:
        k = key(j)
        if k[0] and k[1] and k in seen:
            continue                       # any repeat, same portal or not
        seen.add(k)
        reposts.append(j)

    return {"refit_locked": segment, "cross_site": cross,
            "all_reposts": reposts}


def duplicate_audit(segment: list[dict]) -> dict:
    """What the (title, employer) key finds inside the IT segment, so the
    thesis can state the duplicate rate rather than assert it."""
    groups = collections.defaultdict(list)
    for j in segment:
        groups[(norm(j["title"]), norm(j["company"]))].append(j)
    multi = {k: v for k, v in groups.items() if len(v) > 1 and k[0] and k[1]}
    cross = {k: v for k, v in multi.items()
             if len({x["site"] for x in v}) > 1}
    fmt = lambda v: [{"site": x["site"], "posted_date": x["posted_date"],  # noqa: E731
                      "url": x["url"]} for x in v]
    return {
        "n_segment": len(segment),
        "n_title_employer_keys": len(groups),
        "repeat_groups": len(multi),
        "repeat_records": sum(len(v) for v in multi.values()),
        "cross_site_groups": len(cross),
        "cross_site_records": sum(len(v) for v in cross.values()),
        "cross_site_examples": [{"title": k[0], "employer": k[1],
                                 "records": fmt(v)} for k, v in cross.items()],
        "same_site_examples": [{"title": k[0], "employer": k[1],
                                "records": fmt(v)}
                               for k, v in multi.items() if k not in cross],
    }


def fit_mapping(model, segment: list[dict]) -> dict:
    """pipeline.normalize_skills with a DETERMINISTIC tie order: phrases are
    inserted into the Counter in sorted order, so most_common() breaks
    count ties alphabetically instead of by hash seed."""
    raw = Counter(s for j in segment for s in set(j["skills"]))
    ordered = Counter()
    for s in sorted(raw):
        ordered[s] = raw[s]
    return normalize_skills(model, ordered)


def market_variant(model, mapping: dict, segment: list[dict]) -> dict:
    """build_market's counting tail over `segment` with a FIXED mapping —
    mirrors pipeline.build_market from `counts = Counter()` onward."""
    counts = Counter()
    for j in segment:
        counts.update({mapping.get(s, s) for s in j["skills"]})
    min_freq = max(MIN_SKILL_FREQ, len(segment) // 500)
    base = sorted(s for s, c in counts.items() if c >= min_freq)
    demand = np.array([counts[s] for s in base], dtype=float)
    return {
        "n_jobs": len(segment),
        "skill_base": base,
        "counts": counts,
        "demand_dist": demand / demand.sum(),
        "skill_emb": model.encode(base, normalize_embeddings=True,
                                  show_progress_bar=False),
    }


def run_cohorts(model, market: dict, thr: dict) -> dict:
    base, counts = market["skill_base"], market["counts"]
    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    demand = market["demand_dist"][hard_idx]
    demand = demand / demand.sum()
    ranked = sorted(hard_idx, key=lambda i: -counts[base[i]])
    out = {"n_jobs": market["n_jobs"], "n_skills": len(base),
           "n_technical": len(hard_idx), "cohorts": {}}
    for cohort, prefixes in COHORTS.items():
        courses, clo_texts, _ = load_cohort(cohort_files(prefixes))
        clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                               show_progress_bar=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = market["skill_emb"] @ clo_emb.T
        profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)
        kl, _ = kl_divergence(demand, profile / profile.sum())
        row = evaluate(sim, market, hard_idx, ranked, demand, profile, kl,
                       BASELINE_N, thr)
        row["boundary_tie"] = tie_analysis(sim, market, ranked, BASELINE_N,
                                           thr)
        row["n_courses"] = len(courses)
        out["cohorts"][cohort] = row
    return out


def compare(ref: dict, var: dict) -> dict:
    """Locked vs variant, per cohort: what moved."""
    diff = {}
    for cohort, r in ref["cohorts"].items():
        v = var["cohorts"][cohort]
        r_set = [s["skill"] for s in r["skills"]]
        v_set = [s["skill"] for s in v["skills"]]
        r_rank = {s: i for i, s in enumerate(r_set)}
        v_rank = {s: i for i, s in enumerate(v_set)}
        shared = [s for s in r_set if s in v_rank]
        rho = None
        if len(shared) > 2:
            a = np.array([r_rank[s] for s in shared], dtype=float)
            b = np.array([v_rank[s] for s in shared], dtype=float)
            rho = float(np.corrcoef(a, b)[0, 1])
        status_changes = [
            {"skill": s["skill"], "locked": rs["status"], "variant": s["status"]}
            for s in v["skills"]
            for rs in r["skills"] if rs["skill"] == s["skill"]
            and rs["status"] != s["status"]]
        diff[cohort] = {
            "top30_overlap": len(shared),
            "left_top30": [s for s in r_set if s not in v_rank],
            "entered_top30": [s for s in v_set if s not in r_rank],
            "rank_corr_shared": round(rho, 4) if rho is not None else None,
            "covered": [r["covered"], v["covered"]],
            "dwc": [r["demand_weighted_coverage"],
                    v["demand_weighted_coverage"]],
            "dwc_delta_pts": round(100 * (v["demand_weighted_coverage"]
                                          - r["demand_weighted_coverage"]), 2),
            "half_credit": [r["demand_weighted_coverage_half_credit"],
                            v["demand_weighted_coverage_half_credit"]],
            "kl": [r["kl"], v["kl"]],
            "status_changes": status_changes,
        }
    return diff


# ------------------------------------------------------------------ main
def main() -> None:
    thr = get_thresholds()
    print(f"model {MODEL_NAME} · source {RAW_JOBS.name} · N={BASELINE_N}")
    model = load_model_cpu(MODEL_NAME)

    # 1. the locked market (from cache) + the same segment rebuilt from raw
    locked_market = build_market(model)
    jobs, stats = clean_jobs(RAW_JOBS)
    segment = [j for j in jobs if is_it_by_category(j) and j["skills"]]
    if len(segment) != locked_market["n_jobs"]:
        sys.exit(f"ABORT: rebuilt segment has {len(segment)} jobs, cache "
                 f"says {locked_market['n_jobs']}")

    # 2. the published numbers, straight from the committed cache — the
    #    credibility gate for everything below
    results = {"locked": run_cohorts(model, locked_market, thr)}
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

    # 3. scope + audit
    scope = date_scope(jobs)
    audit = duplicate_audit(segment)
    print(f"IT segment {audit['n_segment']}: {audit['repeat_groups']} "
          f"(title, employer) repeat groups / {audit['repeat_records']} "
          f"records; cross-portal {audit['cross_site_groups']} groups / "
          f"{audit['cross_site_records']} records")

    # 4. one deterministic mapping, then every variant under it
    mapping = fit_mapping(model, segment)
    variants = dedup_variants(segment)
    for name, seg in variants.items():
        res = run_cohorts(model, market_variant(model, mapping, seg), thr)
        res["removed"] = len(segment) - len(seg)
        results[name] = res
    for name, res in results.items():
        print(f"\n[{name}] {res['n_jobs']} postings, "
              f"{res['n_technical']} technical skills")
        for cohort, r in res["cohorts"].items():
            print(f"  {cohort:9s} covered {r['covered']:>2}/30  "
                  f"dwc {r['demand_weighted_coverage']:>6.1%}  "
                  f"half {r['demand_weighted_coverage_half_credit']:>6.1%}  "
                  f"KL {r['kl']:.4f}  rank-30 jobs {r['min_jobs_at_rank_n']}")

    payload = {
        "generated_by": "05_evaluation/dedup_sensitivity.py",
        "config": {"model": MODEL_NAME, "dataset": RAW_JOBS.name,
                   "top_n": BASELINE_N, "thresholds": thr,
                   "min_skill_freq": MIN_SKILL_FREQ,
                   "locked_check": "PASSED"},
        "cleaning_funnel": {"n_raw": sum(stats.values()), **stats,
                            "it_segment": len(segment)},
        "data_scope": scope,
        "duplicate_audit": audit,
        "variants": results,
        # mapping drift: same postings, refit mapping vs the committed cache
        "mapping_drift": compare(results["locked"], results["refit_locked"]),
        # the actual question: stricter duplicate rules vs the same mapping
        "deltas": {name: compare(results["refit_locked"], results[name])
                   for name in variants if name != "refit_locked"},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    OUT_MD.write_text(render_md(payload))
    print(f"\nwrote {OUT_JSON.name} and {OUT_MD.name}")


# --------------------------------------------------------------- render
def render_md(p: dict) -> str:
    L = []
    f = p["cleaning_funnel"]
    L += ["# Data scope and duplicate-posting sensitivity", "",
          f"Generated by `{p['generated_by']}` · corpus `{p['config']['dataset']}`"
          f" · encoder `{p['config']['model']}` · top-{p['config']['top_n']}"
          f" · locked-run check {p['config']['locked_check']}", "",
          "## Cleaning funnel (pipeline.clean_jobs, pooled over all portals)", "",
          "| step | records |", "|---|---:|",
          f"| structured records from Phase 3 | {f['n_raw']:,} |",
          f"| − Khmer-dominant text (> 15 % Khmer characters) | {f.get('skipped_khmer', 0):,} |",
          f"| − non-English prose | {f.get('skipped_non_english', 0):,} |",
          f"| − non-job pages (tenders, calls for proposals) | {f.get('dropped_non_job', 0):,} |",
          f"| − content duplicates (same title + same extracted skill set) | {f.get('dropped_duplicate', 0):,} |",
          f"| = cleaned records | {f['kept']:,} |",
          f"| → technology segment (job_category / title rule, ≥ 1 skill) | {f['it_segment']:,} |",
          ""]

    s = p["data_scope"]
    L += [f"## Posting-date profile of the snapshot (crawl day {s['crawl_day']})", "",
          "| portal | cleaned records | dated | no date | unparsed | date field | min | median | max | within 30 d before crawl | after crawl day |",
          "|---|---:|---:|---:|---:|---|---|---|---|---:|---:|"]
    for r in s["per_site"]:
        L.append(f"| {r['site']} | {r['n']} | {r['dated']} | {r['no_date']} | "
                 f"{r['unparsed']} | {r['field']} | {r.get('min', '—')} | "
                 f"{r.get('median', '—')} | {r.get('max', '—')} | "
                 f"{r.get('within_30d', '—')} | {r.get('after_crawl', '—')} |")
    w = s["posting_date_window"]
    L += ["",
          f"Portals with a true posting date ({', '.join(w['sites'])}): "
          f"{w['dated']:,} dated records, median {w['median']}; "
          f"{w['within_30d']:,} ({w['within_30d'] / max(w['dated'], 1):.1%}) "
          f"posted within the 30 days before the crawl, "
          f"{w['within_90d']:,} ({w['within_90d'] / max(w['dated'], 1):.1%}) "
          f"within 90 days, {w['older_than_1y']:,} older than one year "
          f"(cambojob.com keeps expired adverts online).", ""]

    a = p["duplicate_audit"]
    L += ["## Duplicate audit inside the technology segment ((title, employer) key)", "",
          f"- {a['n_segment']} postings → {a['n_title_employer_keys']} distinct (title, employer) keys",
          f"- repeat groups: {a['repeat_groups']} ({a['repeat_records']} records)",
          f"- of which cross-portal: {a['cross_site_groups']} groups ({a['cross_site_records']} records)", ""]
    for e in a["cross_site_examples"]:
        L.append(f"  - cross-portal: *{e['title']}* — {e['employer']}: "
                 + "; ".join(f"{r['site']} ({r['posted_date'] or 'no date'})"
                             for r in e["records"]))
    L.append("")
    L.append("Same-portal repeats (re-advertised roles, distinct advert IDs):")
    for e in a["same_site_examples"]:
        L.append(f"  - *{e['title']}* — {e['employer']}: "
                 + "; ".join(f"{r['site']} ({r['posted_date'] or 'no date'})"
                             for r in e["records"]))
    L.append("")

    def delta_lines(dl: dict) -> list[str]:
        out = []
        for cohort, d in dl.items():
            out.append(
                f"- **{cohort}**: covered {d['covered'][0]}→{d['covered'][1]}, "
                f"DWC {d['dwc'][0]:.1%}→{d['dwc'][1]:.1%} "
                f"({d['dwc_delta_pts']:+.1f} pts); top-30 overlap "
                f"{d['top30_overlap']}/30"
                + (f", left: {d['left_top30']}" if d["left_top30"] else "")
                + (f", entered: {d['entered_top30']}" if d["entered_top30"] else "")
                + (f"; status changes: {d['status_changes']}"
                   if d["status_changes"] else "; no status change"))
        return out

    L += ["## Sensitivity of the locked results to the duplicate rule", "",
          "`locked` = the published 2026-07-30 runs served from the committed "
          "market cache. `refit_locked` = the same 254 postings with the "
          "canonical-skill mapping refitted deterministically (see the "
          "header comment: the cached mapping is hash-seed dependent and "
          "cannot be re-derived). The duplicate variants are read against "
          "`refit_locked`, so the mapping drift and the duplicate effect are "
          "never mixed.", "",
          "| variant | postings | removed | technical skills | cohort | covered | DWC | half-credit | KL | top-30 overlap | Δ DWC (pts) |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, res in p["variants"].items():
        for cohort, r in res["cohorts"].items():
            d = (p["mapping_drift"] if name == "refit_locked"
                 else p["deltas"].get(name, {})).get(cohort)
            ov = f"{d['top30_overlap']}/30" if d else "—"
            dd = f"{d['dwc_delta_pts']:+.1f}" if d else "—"
            L.append(f"| {name} | {res['n_jobs']} | {res['removed']} | "
                     f"{res['n_technical']} | {cohort} | {r['covered']}/30 | "
                     f"{r['demand_weighted_coverage']:.1%} | "
                     f"{r['demand_weighted_coverage_half_credit']:.1%} | "
                     f"{r['kl']:.4f} | {ov} | {dd} |")
    L += ["", "### mapping drift (refit_locked vs locked — same postings)"]
    L += delta_lines(p["mapping_drift"]) + [""]
    for name, dl in p["deltas"].items():
        L += [f"### {name} (vs refit_locked — same mapping)"]
        L += delta_lines(dl) + [""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
