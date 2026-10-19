# Archive: earlier approaches and experiments

Code from earlier iterations of the project. It is kept so that the
development history described in the thesis can be traced, but **none of it
produces the final thesis results**. Those come from the five `phaseN_*`
folders in `backend/`.

| Folder | What it is | Why it was superseded |
|---|---|---|
| `phase2_playwright_crawler/` | First crawler: BFS crawl with a Playwright → undetected-browser → FlareSolverr rendering fallback and LLM-assisted pagination. | Replaced by the Crawl4AI crawler in `phase2_crawling/`, which collected the thesis corpus. |
| `phase5_legacy/` | Curriculum-recommendation versions V1–V7: syllabus PDF parsing, ESCO normalisation, coverage/market-value scoring, the Ahadi et al. (2022) lexical baseline, domain routing and the MAINTAIN / REVISE / UPDATE labels. | Replaced by the segment-matched, threshold-calibrated method in `phase5_curriculum_alignment/`. |
| `playground/` | One-question experiments (encoder comparison, ESCO retrieval, domain routing, job-title filtering, syllabus extraction model comparison). Its own `README.md` maps each experiment to where its answer went. | Findings were folded into the phases. |
| `pilot_annotation_scoring/` | Scoring of the first 200-pair annotation sheet by GPT and by the method. | Pilot only; replaced by the 250-pair multi-rater study in `phase5_curriculum_alignment/05_evaluation/annotation/`. |

The web application still mounts some `phase5_legacy` endpoints (the older
Recommendation tabs), and `api.py` loads the Playwright crawler for the
Pipeline tab, so these folders must stay in place for the server to start.

Large data files and caches these scripts read (the 1.35 M-posting LinkedIn
dump, embedding caches, raw HTML) are not included. See `DATA.md` at the
repository root.
