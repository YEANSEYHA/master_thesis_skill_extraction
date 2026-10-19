# Data

This repository contains the source code, the curriculum inputs, and the
results reported in the thesis. The raw job-posting data and other large or
restricted files are not included. Contact the author to request access to restricted research data.
Identifiable participant data and credentials are never distributed.

## Included

| Path | Content |
|---|---|
| `backend/phase5_curriculum_alignment/02_input_curriculum/clo_skills_db/` | Skills extracted from the 25 Year 3–5 courses, one JSON file per course: the curriculum input to the thesis runs |
| `backend/phase5_curriculum_alignment/02_input_curriculum/clo_skills/` | Five hand-curated course skill sets used in the method-validation runs |
| `backend/phase5_curriculum_alignment/outputs/` | The eight frozen thesis runs (`report.md`, `report.json`, `heatmap.png`, `params.json`, `inputs.json`, `thesis/*.csv`) and `thesis_registry.csv` |
| `backend/phase5_curriculum_alignment/dashboard/` | Generated report data for the static dashboard (`report_data*.js`) |
| `backend/phase5_curriculum_alignment/05_evaluation/` | Results of every evaluation script (`*_results.json`, `*.md`) and the vendored public benchmarks UniSkill (Apache-2.0) and SkillMatch-1K (MIT) |
| `backend/phase5_curriculum_alignment/05_evaluation/annotation/` | The frozen 250-pair rating sheet, the pre-registration, and the **pseudonymised** human ratings (`human_ratings_250_anon.*`, raters coded R01–R13) |
| `backend/phase4_extraction_evaluation/outputs/`, `backend/phase3_skill_extraction/outputs/skillspan_benchmark/` | SkillSpan benchmark predictions and scores for Qwen3-14B and KeyBERT |
| `backend/phase3_skill_extraction/samples/stage2_result.sample.json` | 10 extracted technology postings showing the record format (contact details removed) |
| `backend/phase1_job_site_discovery/output/` | Job-site discovery results per country |
| `backend/phase3_skill_extraction/01_data_collection/esco_skills_en.csv` | ESCO skill labels (European Commission, CC BY 4.0) |
| `web_annotate/backend/data/pairs.json` | The 250 blinded skill × course pairs shown to raters |

## Not included

| Data | Size | Expected location | Reason |
|---|---|---|---|
| ITC Computer Science course specifications (Year 3 to Year 5, `.docx` / `.xlsx` / `.pdf`) and their full-text conversions | ~155 MB | `backend/course_syllabus/`, `backend/phase5_curriculum_alignment/02_input_curriculum/*.pdf` | Contain lecturers' contact details; the skills extracted from them are included (`clo_skills_db/`) |
| Raw crawled HTML: 6,607 pages from eight Cambodian recruitment sites, collected 15 July 2026 | ~2 GB | `backend/phase2_crawling/outputs/2026-07-15/` | Size; third-party page content. Archived in Cloudflare R2. |
| Phase 3 page classifications and the extracted job records (`stage2_result.json`, 4,876 records) | ~110 MB | `backend/phase3_skill_extraction/outputs/` | Size; contains full posting text, including employer contact details |
| LinkedIn job postings (Kaggle), the 140,393-posting IT subset derived from it, and per-posting records derived from it in `archive/` | 5.7 GB / 90 MB | `backend/phase5_curriculum_alignment/01_input_jobmarket/it_job_linkedin.json` | Third-party dataset, not ours to redistribute |
| Built market and embedding caches (`*.npz`, `.esco_cache/`) | ~2 GB | `backend/phase5_curriculum_alignment/cache/`, archive folders | Rebuilt automatically on first run |
| Identifiable rating exports and the rater code map | small | `05_evaluation/annotation/` | Privacy: the pre-registration commits to never publishing them |
| Labelled and preprocessed HTML pages used to develop the page classifier and to test extraction, and the fine-tuned adapter weights | ~140 MB | `backend/phase3_skill_extraction/01_data_collection/`, `02_classify/preprocessing/outputs/`, `fine_tuned/` | Size; third-party page content |
| Output of the earlier Playwright crawler (24,200 HTML files) | ~4 GB | `backend/archive/phase2_playwright_crawler/outputs/` | Size; superseded |

## What can be reproduced without the missing data

- The cohort results (Tables 4.7 and 4.9–4.11) can be checked directly from
  `phase5_curriculum_alignment/outputs/<run_id>/`. For example, a heatmap is
  regenerated from the stored report (run from `phase5_curriculum_alignment/`):
  `python3 04_report/heatmap.py --run 20260730T063522Z-e54e7e --top 30 --out /tmp/heatmap.png`.
- The method tutorial (`phase5_curriculum_alignment/tutorial/`) runs on built-in
  tiny data.
- The SkillSpan scores (Phase 4) are recomputed from the stored predictions, which
  include the human-annotated skills (`skill_evaluation_metric.py --input
  outputs/skillspan_benchmark/skill5-15.json`, `sweep_match_threshold.py`).
- Re-running the full pipeline needs the Phase 3 `stage2_result.json` (on
  request) and the PostgreSQL database configured in `backend/.env`.
