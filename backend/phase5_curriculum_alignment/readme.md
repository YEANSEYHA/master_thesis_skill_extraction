# Phase 5 — Curriculum–Market Gap Analysis

Measures the gap between the ITC curriculum and job-market skill demand, and
recommends which skills to add to which course. The folders are numbered in
**pipeline order** so the flow reads top to bottom.

```
01 job market  ─┐
                ├─▶  03 engine  ─▶  04 report  ─▶  outputs/<run_id>/  +  dashboard/
02 curriculum  ─┘                                          │
                                                           └─▶  05 evaluation
```

## Structure

| Folder | Stage | What's inside |
|--------|-------|---------------|
| `01_input_jobmarket/` | **input 1 — demand** | `it_job_linkedin.json` (140,393 US LinkedIn postings, used to validate the method; not distributed, see `DATA.md`). The Cambodian corpus is read from Phase 3 (`JOB_SOURCE=cambodia`) |
| `02_input_curriculum/` | **input 2 — supply** | `curriculum_parse.py` (PDF/.docx → skills), the syllabus PDFs, `clo_skills/` (5 hand-curated courses, validation runs), `clo_skills_db/` (the 25 Year 3–5 ITC courses used in the thesis runs) |
| `03_engine/` | **engine** | `pipeline.py` — cleaning, classification, skill normalization, market build, thresholds, KL, greedy selection |
| `04_report/` | **report** | `gen_report_data.py` (the runnable entry point), `run_store.py` (immutable run archive), `heatmap.py` (figure) |
| `05_evaluation/` | **evaluate** | External benchmarks + validation: `bench_uniskill.py`, `bench_skillmatch.py`, `compare_encoders.py`, `eval_selection.py`, `llm_judge_itc.py`, `judge_250.py`, `rerank_borderline.py`, `sweep_top_n.py`, `dedup_sensitivity.py`, `loso_site_sensitivity.py`, their `*_results.json`, `annotation/` (human calibration study), vendored datasets |
| `dashboard/` | front-end | `index.html` + `index.js` (static React, no build step) and the generated `report_data*.js` / `preview_data*.js` |
| `cache/` | cache | `market_cache_*.npz` — the built market, keyed by model + dataset version (rebuilt on first run, not distributed) |
| `outputs/` | results | one immutable dir per run (`report.md/.json`, `params.json`, `inputs.json`, `log.txt`, `heatmap.png`) + the `index.md` ledger. The eight runs reported in the thesis are included and listed in `thesis_registry.csv` |
| `out/` | live runs | gitignored output of API-triggered runs, so the committed baseline never churns |
| `tutorial/` | learning | step1–step5, the method rebuilt from scratch on tiny data |

## How the imports work

Numbered folders can't be Python packages (`import 03_engine` is a syntax
error), so cross-folder imports go through `sys.path`, not dotted paths:

- Everything that needs the engine adds `03_engine/` to `sys.path` and imports
  by bare name (`from pipeline import ...`) — that's `04_report/`,
  `05_evaluation/`, and `tutorial/`.
- **Modules anchor data paths on the phase-5 root**
  (`BASE_DIR = Path(__file__).resolve().parent.parent`), so
  `01_input_jobmarket/`, `02_input_curriculum/`, `cache/`, `dashboard/` and
  `outputs/` resolve the same no matter which folder the script lives in.
- `gen_report_data.py` also puts its own folder on `sys.path` for its
  siblings `run_store.py` and `heatmap.py`.

## Running

```bash
# the pipeline — writes dashboard/report_data.js + outputs/<run_id>/
python3 04_report/gen_report_data.py

# leave the committed baseline alone (what the API does for live runs)
REPORT_OUT=out/report_data.js PREVIEW_OUT=out/preview_data.js \
  python3 04_report/gen_report_data.py

# other corpus / other curriculum / named run
JOB_SOURCE=cambodia python3 04_report/gen_report_data.py
CLO_SKILLS_DIR=02_input_curriculum/clo_skills_db python3 04_report/gen_report_data.py
RUN_LABEL="ablation: covered=0.75" python3 04_report/gen_report_data.py
RERANK=0 python3 04_report/gen_report_data.py        # skip host verification

# figure for any archived run
python3 04_report/heatmap.py --run <run_id> --top 25

# evaluation / benchmarks
python3 05_evaluation/bench_uniskill.py
python3 05_evaluation/compare_encoders.py
python3 05_evaluation/eval_selection.py

# the method explained step by step on tiny data
python3 tutorial/step1_embedding.py       # ... through step5
```

Front-end: open `dashboard/index.html` directly (CDN React, no server). The
Next.js **V8 tab** reads the same files through `GET /api/v8/report`.

**A direct run overwrites the committed `dashboard/report_data.js`** — that is
the frozen reference. `git checkout` it afterwards unless you meant to move the
baseline, or use `REPORT_OUT=` as above.

## Thesis results (Table 4.7)

The four calibrated runs reported in the thesis (`config_hash e54e7e17`,
thresholds covered 0.7368 / borderline 0.6903, 254 Cambodian technology
postings, top-30 market skills). Each has a `report.md`, `report.json`,
`heatmap.png` and a `thesis/` folder of CSVs under `outputs/<run_id>/`.

| cohort | run_id | courses | covered | demand-weighted coverage | KL |
|---|---|---:|---:|---:|---:|
| Y3 | `20260730T063316Z-e54e7e` | 9 | 12/30 | 44.9% | 0.3810 |
| Y4 | `20260730T063353Z-e54e7e` | 8 | 16/30 | 51.4% | 0.3771 |
| Y5 | `20260730T063430Z-e54e7e` | 8 | 17/30 | 56.3% | 0.3768 |
| Y3+Y4+Y5 | `20260730T063522Z-e54e7e` | 25 | 25/30 | 85.3% | 0.3689 |

The `20260727T…-475eb3` runs are the same inputs scored with the provisional,
pre-calibration thresholds (0.73 / 0.69); they form the "before calibration"
side of the threshold comparison.

## Reading the numbers

- Coverage is a **max cosine, not a percentage**. Around 0.25–0.35 is the
  noise floor ("not taught"); a score of 0.32 is never "32%".
- Thresholds are **encoder-specific** and live in `pipeline.THRESHOLDS`
  (`03_engine/pipeline.py`). The BGE values were calibrated against the
  250-pair human rating study in `05_evaluation/annotation/`.
- Raw cosines are not comparable across encoders or across pair
  distributions; compare rankings, not absolute values.
- Demand-weighted coverage is the primary cross-cohort measure. KL divergence
  is reported as a secondary distributional statistic.
- Soft skills are filtered out of the technical rankings.
- The cross-encoder (`BAAI/bge-reranker-v2-m3`) only decides which course
  should host a recommended skill; coverage statuses stay cosine-based.
