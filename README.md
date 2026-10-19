# An LLM-Based Approach for Extracting In-Demand Job Skills to Enable AI-Powered Curriculum Enhancement

Source code for the Master's thesis of **YEAN Seyha**, Master's Degree of
Engineering in Computer Science, Graduate School, Institute of Technology of
Cambodia (ITC), September 2026. Supervisor: Dr. VALY Dona. Co-supervisor:
Prof. MICHEL Verleysen.

The system collects job postings from Cambodian recruitment websites, extracts
the requested skills with a large language model, and compares them with the
skills documented in ITC Computer Science course specifications. It reports
which in-demand skills the curriculum covers, a skill × course heatmap, and a
prioritised list of skills to add.

## Pipeline

```
Phase 1  Job-site discovery        find Cambodian recruitment websites (LLM + web search)
   │
Phase 2  Crawling                  6,607 HTML pages from 8 sites (Crawl4AI)
   │
Phase 3  Skill extraction          job-page classification (Llama 3.1 8B)
   │                               → structured extraction of job fields and skills (Qwen3-14B)
   │
Phase 4  Extraction evaluation     SkillSpan benchmark: Qwen3-14B vs KeyBERT
   │
Phase 5  Curriculum alignment      course documents → curriculum skills
                                   → BGE semantic matching against 254 technology postings
                                   → coverage, heatmap, KL divergence, greedy recommendation plan
                                   → evaluation (human threshold calibration, LLM judges,
                                     public benchmarks, sensitivity analyses)
```

## Repository layout

```
.
├── README.md                    this file
├── DATA.md                      which data is included, which is not, and how to request it
├── start.sh / stop.sh           start or stop the API server and web application
├── backend/
│   ├── requirements.txt         Python dependencies (Python 3.11)
│   ├── .env.example             configuration template (API keys, database)
│   ├── phase1_job_site_discovery/   job-site discovery + api.py, the FastAPI server for all phases
│   ├── phase2_crawling/             Crawl4AI crawler
│   ├── phase3_skill_extraction/     01 collect → 02 classify → 03 extract → 04 trending → 05 evaluation
│   ├── phase4_extraction_evaluation/  SkillSpan benchmark, KeyBERT baseline, ESCO evaluation
│   ├── phase5_curriculum_alignment/   01 job market → 02 curriculum → 03 engine → 04 report → 05 evaluation
│   │   ├── outputs/                 the frozen runs reported in the thesis
│   │   ├── dashboard/               static results dashboard (open index.html)
│   │   └── tutorial/                the method rebuilt step by step on tiny data
│   ├── course_syllabus/         ITC course specifications (not included, see DATA.md)
│   └── archive/                 earlier approaches and experiments (see archive/README.md)
├── frontend/                    Next.js web application (port 3000)
└── web_annotate/                web app used for the 250-pair human rating study
```

Phase 3 and Phase 5 each have their own `readme.md` describing their stages
and commands in detail.

## Where the thesis results come from

| Thesis | Code | Output |
|---|---|---|
| Table 3.1, data collection | `phase2_crawling/main.py` | raw HTML (not included, see `DATA.md`) |
| Table 4.1, job-page classification and extraction | `phase3_skill_extraction/02_classify/classify_corpus.py`, `phase3_skill_extraction/03_extract/extract_corpus.py` | `stage2_result.json` (not included; sample in `phase3_skill_extraction/samples/`) |
| Table 3.2, cleaning and technology-posting selection | `03_engine/pipeline.py` (`clean_jobs`, `classify_jobs`) | `outputs/<run_id>/report.json` |
| Tables 4.3–4.4, SkillSpan benchmark | `phase4_extraction_evaluation/skillspan_benchmark.py`, `skill_evaluation_metric.py`, `keybert_benchmark.py`, `sweep_match_threshold.py` | `phase4_extraction_evaluation/outputs/skillspan_benchmark/` |
| Curriculum skill extraction | `phase5_curriculum_alignment/02_input_curriculum/curriculum_parse.py` | `02_input_curriculum/clo_skills_db/` |
| Matching, coverage, KL, recommendation plan | `phase5_curriculum_alignment/03_engine/pipeline.py`, `04_report/gen_report_data.py` | `outputs/<run_id>/` |
| Table 4.5, LLM silver-standard evaluation | `05_evaluation/llm_judge_itc.py`, `05_evaluation/judge_250.py` | `05_evaluation/*_results.json` |
| Table 4.6, human threshold calibration | `05_evaluation/annotation/`, `web_annotate/` | `05_evaluation/annotation/EVALUATION_RESULTS.md` |
| Table 4.7, final frozen runs | `04_report/run_store.py`, `04_report/export_run.py` | `outputs/thesis_registry.csv` |
| Table 4.8, top-N sensitivity | `05_evaluation/sweep_top_n.py` | `05_evaluation/sweep_top_n.md` |
| Tables 4.9–4.11, cohort and programme results, heatmaps | `04_report/heatmap.py` | `outputs/<run_id>/report.md`, `heatmap.png`, `thesis/*.csv` |
| Post-defence revisions: data scope, duplicate and per-site sensitivity, annotator panel | `05_evaluation/dedup_sensitivity.py`, `loso_site_sensitivity.py`, `annotation/rater_profile.py` | the matching `.md` and `_results.json` files |

Paths that start with `02_` to `05_` or `outputs/` are relative to
`backend/phase5_curriculum_alignment/`.

## Running

Requirements: Python 3.11 (a conda environment named `thesis`), Node.js 20,
PostgreSQL, and API keys for the LLM providers you use. A local
[Ollama](https://ollama.com) server is needed only for the local Qwen3-14B paths.

```bash
# 1. Python environment
conda create -n thesis python=3.11
conda activate thesis
pip install -r backend/requirements.txt
playwright install chromium

# 2. Configuration
cp backend/.env.example backend/.env      # then fill in the keys you need

# 3. Web application
(cd frontend && npm install)
./start.sh            # API on http://localhost:8000, web app on http://localhost:3000
```

The API server takes one to two minutes to start because it loads the
embedding models. The **Curriculum–Market Gap Analysis** tab is the final method; the
other tabs belong to Phases 1–3 and to the earlier approaches.

Without the database or the web application, the Phase 5 results can be
inspected and partly regenerated from the command line:

```bash
cd backend/phase5_curriculum_alignment
python3 tutorial/step1_embedding.py        # the method on tiny data, steps 1–5
python3 04_report/heatmap.py --run 20260730T063522Z-e54e7e --top 30 --out /tmp/heatmap.png
open dashboard/index.html                  # static dashboard, no server needed
```

## Data

See [`DATA.md`](DATA.md). Raw job postings, the LinkedIn dataset, caches and
identifiable annotation data are not included; they are available from the
author on request.
