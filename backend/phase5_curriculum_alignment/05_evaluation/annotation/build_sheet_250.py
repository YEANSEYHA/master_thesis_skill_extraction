# Build the frozen 250-pair human/LLM annotation sheet for the V8 threshold
# calibration study (thesis §4.9.1).
#
#   JOB_SOURCE=cambodia python3 05_evaluation/annotation/build_sheet_250.py
#
# Design (recorded here so the sheet is a describable procedure, not an edit):
#   - CENSUS: every skill x course cell of the locked programme run
#     (20260727T105055Z-475eb3, 30 skills x 25 courses = 750 cells) whose
#     full-precision cosine is >= the 0.66 host floor. These are ALL the
#     pairs whose covered/borderline/gap classification depends on where
#     the threshold sits — annotating all of them makes the study a census
#     of the decision region, not a sample of it.
#   - BOUNDARY: the cells whose full-precision cosine is just below the
#     floor but ROUNDS to 0.66 — i.e. every cell the published heatmap
#     displays at the floor. The census is defined on full precision
#     (like the pipeline's decisions); these display-boundary cells are
#     kept as their own labelled stratum so the study covers everything
#     a reader of the heatmap would take to be at the floor.
#   - CONTROLS: 30 seeded-random cells below the floor (boundary cells
#     excluded), stratified 10 x (<0.50 clear negatives / 0.50-0.60 /
#     0.60-0.66 near-misses). Clear negatives double as attention checks.
#   - ANCHOR BLOCK: the first 30 printed rows — 15 controls (5 per stratum)
#     + 15 borderline (0.69-0.73) cells — every rater is asked to complete
#     at least this block; the rest is seeded-shuffled.
#   - BLIND: the printed sheet and annotator CSV carry no cosine, band,
#     job count or system decision; those live only in the master JSON.
#   - Each pair shows the ARGMAX curriculum skill for that course — the
#     exact unit the pipeline's max-reduction used — so humans judge the
#     same object the system decided on.
#
# The script re-runs the frozen alignment (deterministic given the committed
# market cache) and REFUSES to build the sheet unless every recomputed
# rounded cell equals the locked report.json — the sheet provably belongs
# to the locked experiment.

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # 05_evaluation/annotation
PHASE5 = _HERE.parent.parent
sys.path.insert(0, str(PHASE5 / "03_engine"))

os.environ.setdefault("JOB_SOURCE", "cambodia")

from pipeline import (MODEL_NAME, build_market, is_soft_skill,  # noqa: E402
                      load_model, get_thresholds)

LOCKED_RUN = PHASE5 / "outputs" / "20260727T105055Z-475eb3"
CLO_DIR = PHASE5 / "02_input_curriculum" / "clo_skills_db"
COHORT_RUNS = {"Y3": "20260727T104658Z-475eb3",
               "Y4": "20260727T104742Z-475eb3",
               "Y5": "20260727T104822Z-475eb3"}
TOP_N = 30
SEED = 20260728          # frozen; changing it is a new sheet
N_CONTROLS_PER_STRATUM = 10
ANCHOR_CONTROLS_PER_STRATUM = 5
ANCHOR_BORDERLINE = 15

THR = get_thresholds()
HOST_MIN = THR["host_min"]          # 0.66
BORDER_AT = THR["borderline"] if "borderline" in THR else 0.69
COVER_AT = THR["covered"] if "covered" in THR else 0.73

OUT_MASTER = _HERE / "sheet_250_master.json"
OUT_CSV = _HERE / "sheet_250_annotator.csv"
OUT_HTML = _HERE / "sheet_250.html"
OUT_PDF = _HERE / "curriculum_annotation_250_score.pdf"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Reference-only glossary for market skills a rater may not recognise.
GLOSSARY = {
    "ci/cd pipelines": "automated build-test-deploy chains (e.g. GitHub "
                       "Actions, Jenkins) that ship code continuously",
    "restful api development": "building web APIs that follow the REST "
                               "style (HTTP endpoints returning JSON)",
    "api integration": "connecting software to other systems through "
                       "their APIs (payment, mapping, messaging services)",
    "docker containerization": "packaging applications into portable "
                               "Docker containers",
    "vmware virtualization": "running many virtual machines on one "
                             "physical server with VMware",
    "aws cloud platforms": "Amazon Web Services — cloud servers, storage "
                           "and managed services",
    "git version control": "tracking and merging source-code changes "
                           "with Git",
    "incident response": "the organised process of detecting, containing "
                         "and recovering from a security incident",
    "security compliance": "meeting security standards/policies "
                           "(ISO 27001, PCI-DSS) and passing audits",
    "security monitoring": "watching logs/alerts (SIEM) to detect "
                           "attacks in progress",
    "firewall configuration": "setting up network firewalls and their "
                              "traffic rules",
    "disaster recovery planning": "preparing backup systems and "
                                  "procedures so IT services survive a "
                                  "major failure",
    "it infrastructure management": "operating an organisation's servers, "
                                    "networks and core IT services",
    "performance tuning": "measuring and improving the speed of systems, "
                          "databases or code",
    "postgresql database": "PostgreSQL — an open-source relational "
                           "database system",
    "mysql database": "MySQL — an open-source relational database system",
    "code reviews": "systematically reading a colleague's code changes "
                    "before they are merged",
    "system integration": "making separate software systems work "
                          "together as one",
    "technical documentation": "writing manuals, runbooks and design "
                               "documents for technical systems",
    "microsoft office proficiency": "practical use of Word, Excel and "
                                    "PowerPoint",
}


def course_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ").title()


def build_cells():
    """Recompute the frozen alignment; return full-precision cells."""
    locked = json.loads((LOCKED_RUN / "report.json").read_text())
    locked_pc = {s["skill"]: s["per_course"] for s in locked["skills"]}
    top_skills = [s["skill"] for s in locked["skills"]]
    jobs = {s["skill"]: s["jobs"] for s in locked["skills"]}

    model = load_model(MODEL_NAME)
    market = build_market(model)

    courses, clo_texts, clo_owner = [], [], []
    for f in sorted(CLO_DIR.glob("*.json")):
        name = course_name(f)
        courses.append(name)
        for item in json.loads(f.read_text()):
            clo_texts.append(item["skill"])
            clo_owner.append(name)

    clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                           show_progress_bar=False)
    sim = market["skill_emb"] @ clo_emb.T

    base, counts = market["skill_base"], market["counts"]
    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top = sorted(hard_idx, key=lambda i: -counts[base[i]])[:TOP_N]
    got = [base[i] for i in top]
    if got != top_skills:
        sys.exit(f"ABORT: recomputed top-{TOP_N} differs from locked run\n"
                 f"locked: {top_skills}\n  got: {got}")

    owner_idx = {c: [k for k, o in enumerate(clo_owner) if o == c]
                 for c in courses}
    cells, mismatches = [], 0
    for i in top:
        skill = base[i]
        row = sim[i]
        for c in courses:
            idx = owner_idx[c]
            j = idx[int(np.argmax(row[idx]))]
            score = float(row[j])
            if round(score, 2) != locked_pc[skill][c]:
                mismatches += 1
                print(f"MISMATCH {skill} x {c}: "
                      f"{round(score, 2)} vs locked {locked_pc[skill][c]}")
            cells.append({"skill": skill, "jobs": jobs[skill],
                          "course": c, "clo": clo_texts[j],
                          "cos": score})
    if mismatches:
        sys.exit(f"ABORT: {mismatches} cells disagree with the locked run — "
                 "sheet NOT built")
    print(f"Reproduction check: all {len(cells)} cells match "
          f"the locked report.json (rounded)")
    return cells


def cohort_map():
    m = {}
    for cohort, rid in COHORT_RUNS.items():
        r = json.loads((PHASE5 / "outputs" / rid / "report.json").read_text())
        for c in r["courses"]:
            m[c] = cohort
    return m


def band_of(cos: float) -> str:
    if cos >= 0.85:
        return "A >=0.85"
    if cos >= 0.78:
        return "B 0.78-0.85"
    if cos >= COVER_AT:
        return "C 0.73-0.78"
    if cos >= BORDER_AT:
        return "D 0.69-0.73 borderline"
    if cos >= HOST_MIN:
        return "E 0.66-0.69 host floor"
    if cos >= 0.60:
        return "F 0.60-0.66 near-miss"
    if cos >= 0.50:
        return "G 0.50-0.60"
    return "H <0.50 clear negative"


def assemble(cells):
    rng = random.Random(SEED)
    census = [c for c in cells if c["cos"] >= HOST_MIN]
    below = [c for c in cells if c["cos"] < HOST_MIN]
    # cells the heatmap DISPLAYS at the floor (round to 0.66) but whose
    # full-precision score is below it — own stratum, all included
    boundary = [c for c in below if round(c["cos"], 2) == HOST_MIN]
    pool_below = [c for c in below if round(c["cos"], 2) != HOST_MIN]
    strata = {"near": [c for c in pool_below if c["cos"] >= 0.60],
              "mid": [c for c in pool_below if 0.50 <= c["cos"] < 0.60],
              "clear": [c for c in pool_below if c["cos"] < 0.50]}
    controls = []
    for name in ("clear", "mid", "near"):        # fixed draw order
        pool = sorted(strata[name], key=lambda c: (c["skill"], c["course"]))
        controls += rng.sample(pool, N_CONTROLS_PER_STRATUM)
    for c in census:
        c["set"] = "census"
    for c in boundary:
        c["set"] = "boundary"
    for c in controls:
        c["set"] = "control"

    # anchor block: 5 controls per stratum + 15 borderline census cells
    anchor = []
    for name in ("clear", "mid", "near"):
        pick = [c for c in controls
                if band_of(c["cos"]).startswith(
                    {"clear": "H", "mid": "G", "near": "F"}[name])]
        anchor += rng.sample(pick, ANCHOR_CONTROLS_PER_STRATUM)
    borderline = sorted((c for c in census
                         if BORDER_AT <= c["cos"] < COVER_AT),
                        key=lambda c: (c["skill"], c["course"]))
    anchor += rng.sample(borderline, ANCHOR_BORDERLINE)
    anchor_keys = {(c["skill"], c["course"]) for c in anchor}

    rest = [c for c in census + boundary + controls
            if (c["skill"], c["course"]) not in anchor_keys]
    rng.shuffle(anchor)
    rng.shuffle(rest)
    ordered = anchor + rest
    for n, c in enumerate(ordered, 1):
        c["row"] = n
        c["anchor"] = n <= len(anchor)
        c["band"] = band_of(c["cos"])
    return ordered, len(census), len(boundary), len(controls)


def content_hash(ordered) -> str:
    payload = json.dumps([[c["row"], c["skill"], c["clo"], c["course"]]
                          for c in ordered]).encode()
    return hashlib.sha256(payload).hexdigest()


def write_outputs(ordered, n_census, n_boundary, n_controls, cohorts):
    sheet_hash = content_hash(ordered)
    locked = json.loads((LOCKED_RUN / "params.json").read_text())
    master = {
        "description": ("Frozen 250-pair annotation sheet for V8 threshold "
                        "calibration: complete census of every skill x "
                        f"course cell >= {HOST_MIN} host floor in the locked "
                        "programme run (full precision), plus every cell "
                        "the heatmap displays at the floor (rounds to "
                        f"{HOST_MIN}), plus 30 stratified sub-floor "
                        "controls. Blind columns only on the printed sheet; "
                        "cos/band/jobs/set/anchor live here."),
        "locked_run": locked["run_id"],
        "config_hash": locked["config_hash"],
        "input_hash": locked["input_hash"],
        "seed": SEED,
        "sheet_hash": sheet_hash,
        "scoring": ("0-10: 0 = not taught here; 5 = related/foundation, "
                    "not this skill itself; 10 = fully teaches this exact "
                    "skill. Optional mapping: 0-3 No, 4-6 Partly, 7-10 Yes."),
        "counts": {"census": n_census, "boundary": n_boundary,
                   "controls": n_controls, "total": len(ordered),
                   "anchor_block": sum(1 for c in ordered if c["anchor"])},
        "pairs": [{**{k: c[k] for k in ("row", "skill", "jobs", "course",
                                        "clo", "cos", "band", "set",
                                        "anchor")},
                   "cohort": cohorts[c["course"]]} for c in ordered],
    }
    OUT_MASTER.write_text(json.dumps(master, indent=1))

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "market_skill", "course_learning_outcome",
                    "course", "year", "score_0_10", "note"])
        for c in ordered:
            w.writerow([c["row"], c["skill"], c["clo"], c["course"],
                        cohorts[c["course"]], "", ""])
    return master


def write_pdf(master, cohorts):
    rows = []
    for p in master["pairs"]:
        rows.append(
            f"<tr><td class='n'>{p['row']}</td>"
            f"<td class='s'>{p['skill']}</td>"
            f"<td>{p['clo']}</td>"
            f"<td class='sc'>0&nbsp; 1&nbsp; 2&nbsp; 3&nbsp; 4&nbsp; "
            f"5&nbsp; 6&nbsp; 7&nbsp; 8&nbsp; 9&nbsp; 10</td>"
            f"<td class='note'></td>"
            f"<td class='c'>{p['course']} ({p['cohort']})</td></tr>")
    gloss = "".join(f"<tr><td class='s'>{k}</td><td>{v}</td></tr>"
                    for k, v in sorted(GLOSSARY.items()))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
@page {{ size: A4 landscape; margin: 9mm; }}
body {{ font: 8.5px/1.35 -apple-system, Helvetica, Arial, sans-serif;
       color:#111; }}
h1 {{ font-size: 15px; margin: 0 0 4px; }}
h2 {{ font-size: 12px; margin: 14px 0 4px; page-break-before: always; }}
p  {{ margin: 2px 0; font-size: 8.5px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 6px; }}
th {{ text-align: left; border-bottom: 1.2px solid #111; padding: 2px 4px;
     font-size: 8px; }}
td {{ border-bottom: 0.4px solid #bbb; padding: 2.2px 4px;
     vertical-align: top; }}
tr {{ page-break-inside: avoid; }}
thead {{ display: table-header-group; }}
td.n {{ color:#777; width: 18px; }}
td.s {{ font-weight: 600; width: 150px; }}
td.sc {{ font-family: Menlo, monospace; font-size: 8px; width: 168px;
        white-space: nowrap; }}
td.note {{ width: 90px; }}
td.c {{ color:#555; width: 170px; }}
.small {{ color:#555; font-size: 7.5px; }}
.anchor-note {{ background:#f2f2f2; padding:4px 6px; margin-top:4px; }}
</style></head><body>
<h1>Curriculum Coverage Annotation — Human Evaluation Sheet (0–10)</h1>
<p><b>Instructions:</b> each item pairs a skill demanded in Cambodian IT job
postings with one learning outcome from an ITC course (Years 3–5, Software
Engineering programme). Judge from the text alone: <b>how well does this
course outcome teach this skill?</b> The course each outcome belongs to is
shown in the last column for context. A glossary is on the last page.</p>
<p><b>Score 0–10:</b>&nbsp; 0 = not taught here &nbsp;·&nbsp;
5 = related / foundation, but not this skill itself &nbsp;·&nbsp;
10 = fully teaches this exact skill.</p>
<p class="anchor-note"><b>Rows 1–30 are the core block — please complete at
least these 30.</b> After that, continue as far as you have time; every
additional rating helps.</p>
<p>250 pairs &nbsp;·&nbsp; annotator: ____________________ &nbsp;·&nbsp;
date: __________ &nbsp;·&nbsp; role (circle): &nbsp;lecturer /
senior student / classmate / industry</p>
<table>
<thead><tr><th>#</th><th>Market Skill</th><th>Course Learning Outcome</th>
<th>Score (0–10) — circle one</th><th>Note</th><th>Course (Year)</th></tr>
</thead><tbody>{''.join(rows)}</tbody></table>
<p class="small">Blind sheet: similarity scores, bands and job-demand counts
are hidden to prevent priming (kept in the master file). Optional mapping:
0–3 = No · 4–6 = Partly · 7–10 = Yes. &nbsp; sheet
{master['sheet_hash'][:12]} · run {master['locked_run']}</p>
<h2>Glossary — market-skill terms (reference only)</h2>
<p class="small">If a skill is clearly unrelated to the course outcome, a
low score (0–2) is the expected answer.</p>
<table><thead><tr><th>Term</th><th>Meaning</th></tr></thead>
<tbody>{gloss}</tbody></table>
</body></html>"""
    OUT_HTML.write_text(html)
    subprocess.run([CHROME, "--headless", "--disable-gpu",
                    f"--print-to-pdf={OUT_PDF}", "--no-pdf-header-footer",
                    str(OUT_HTML)], check=True, capture_output=True)
    print(f"PDF: {OUT_PDF}")


def main():
    cells = build_cells()
    cohorts = cohort_map()
    ordered, n_census, n_boundary, n_controls = assemble(cells)
    master = write_outputs(ordered, n_census, n_boundary, n_controls,
                           cohorts)
    write_pdf(master, cohorts)

    by_band, by_cohort, by_set = {}, {}, {}
    for p in master["pairs"]:
        by_band[p["band"]] = by_band.get(p["band"], 0) + 1
        by_cohort[p["cohort"]] = by_cohort.get(p["cohort"], 0) + 1
        by_set[p["set"]] = by_set.get(p["set"], 0) + 1
    print(f"\nsheet_hash: {master['sheet_hash']}")
    print(f"census {n_census} + boundary {n_boundary} + controls "
          f"{n_controls} = {len(master['pairs'])} pairs")
    print("bands:", dict(sorted(by_band.items())))
    print("cohorts:", dict(sorted(by_cohort.items())))
    print("sets:", by_set)


if __name__ == "__main__":
    main()
