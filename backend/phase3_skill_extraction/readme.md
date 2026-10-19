# Phase 3 — LLM HTML → Structured Job Skills

Turns crawled job-posting HTML into a structured skill inventory, then evaluates
and analyses it. The folders are numbered in **pipeline order** so the flow reads
top to bottom.

```
crawled HTML  →  01 collect  →  02 classify  →  03 extract  →  04 trending
                                                     │
                                                     └→  05 evaluation
```

## Structure

| Folder | Stage | What's inside |
|--------|-------|---------------|
| `01_data_collection/` | **collect** | Scraping (`scrape_process.py`, Playwright + FlareSolverr), raw `htmls/`, `data/`, `esco_skills_en.csv` |
| `02_classify/` | **classify** | `stage1_classify.py` (single_job_detail vs non_job) + runners `classify_corpus.py`, `classify_bakeoff.py`, `it_filter.py`, `preprocessing/` |
| `03_extract/` | **extract** | `stage2_skill_extraction.py` (qwen3 skill extraction) + runners `extract_corpus.py`, `bakeoff.py` |
| `04_trending/` | **trending** | `stage5_trending_skills.py` + trend-detection methods (`Kleinberg/`, `Mann-Kendall/`, `TF-IDF/`, `google_trends/`) |
| `05_evaluation/` | **evaluate** | SkillSpan benchmark (`skillspan_benchmark.py`, `skill_evaluation_metric.py`, `esco_evaluation.py`, `keybert_benchmark.py`) |
| `lib/` | shared | `utils.py` (`read_all_htmls`, …) — imported by the stage modules |
| `tools/` | utilities | `upload_stage2_to_r2.py` (push results to R2), `visual_data.py` |
| `visualization/` | dashboards | `app.py`, `app_stage2.py`, `skill_span.html` |
| `fine_tuned/` | model | classifier fine-tuning script + outputs |
| `outputs/` | results | per-stage run outputs (classify / stage2 / bakeoff …) |
| `main.py` | orchestrator | runs stage 1 + stage 2 end to end |

## How the imports work

Numbered folders can't be Python packages (`import 02_classify` is a syntax
error), so cross-folder imports go through `sys.path`, not dotted paths:

- **Runners** add their own folder to `sys.path` and import the sibling stage by
  bare name (`from stage1_classify import …`).
- **Stage modules and runners anchor data paths on the phase-3 root**
  (`HERE = Path(__file__).resolve().parent.parent`), so `outputs/`, the crawl
  corpus, and `backend/.env` resolve exactly as they did before the reorg.
- `stage2_skill_extraction.py` adds `lib/` to `sys.path` for `from utils import …`.
- `main.py` (at the root) adds `02_classify/`, `03_extract/`, and `lib/`.

## Running

```bash
# End to end (classify + extract) on a crawl run
python3 main.py --run-id <crawl_run_id>
python3 main.py --classify-only --run-id <id>     # stage 1 only
python3 main.py --extract-only                     # stage 2 only

# Individual stages / experiments
python3 02_classify/classify_corpus.py             # stage 1 over a corpus
python3 03_extract/extract_corpus.py               # stage 2 over stage-1 output
python3 03_extract/bakeoff.py                       # model bake-off for extraction

# Evaluation (see 05_evaluation for the full command set)
python3 05_evaluation/skillspan_benchmark.py --model qwen3:14b --skills 5-15
python3 05_evaluation/skill_evaluation_metric.py --input <benchmark.json>
```

## FlareSolverr (for `01_data_collection`)

To bypass Cloudflare when scraping:

```bash
docker run -d --name=flaresolverr -p 8191:8191 \
  -e LOG_LEVEL=info --restart unless-stopped \
  ghcr.io/flaresolverr/flaresolverr:latest
```

Available at `http://localhost:8191`.
