# Playground — method experiments

Throwaway-by-design scripts that answered one question each before the answer
was folded into a phase. Nothing here is imported by the API or the pipelines;
they exist so a result in the thesis can be traced back to the experiment that
produced it.

Unlike `phase3/` and `phase5_curriculum_alignment/`, these folders are **not numbered** — the
experiments are independent, not a pipeline. Numbering appears *inside* an
experiment when its own steps run in order (`domain_routing/00_…05_`).

## Experiments

| Folder | Question it answered | Where the answer went |
|--------|----------------------|------------------------|
| `esco_retrieval/` | Can a skill be mapped to ESCO labels by cosine, and does reranking fix the wrong ones? `skill_to_esco.py` (map) → `retrieve.py` (top-k) → `retrieve_and_rerank.py` (+cross-encoder) → `test_coverage_score.py` (the V6 coverage fixture) | phase5_legacy `recommendation_v3/esco*.py`; the retrieve-then-rerank design in `phase5_curriculum_alignment/03_engine` |
| `domain_routing/` | Given a course, which job segment should it be scored against? Steps `00_route_demo` → `01_corpus` → `02_retrieve` → `03_verify` → `04_profiles` → `05_route`, plus `model_compare.py` / `bigger_compare.py` / `sweep_params.py` | the career-domain cohorts + LLM-verified routing |
| `filter_job_by_domain/` | Can job *titles* alone segment the corpus? `00_load_data` → `01_expand_roles` → `02_embed_titles` → `03_retrieve` → `04_verify` → `05_filter` | the title-embedding segment filter |
| `category_filter/` | How reliable is the raw `job_category` field for segmenting? (4 different shapes in the real data, 247/2129 null) | `SEGMENT_MODE="category"` in `phase5_curriculum_alignment/03_engine/pipeline.py` |
| `coverage_score/` | The V6 verdict-card flow — coverage + the suggest-add panel (`main.py`, `suggest_add.py`) | the V7 suggest-add panel |
| `embedder_compare/` | Which encoder separates skill pairs best? (`test_embbeding_model.py`) | BGE as the primary encoder; MiniLM kept as baseline |
| `extract_skill_from_curriculum/` | Parsing the real ITC `.docx` syllabi (Year 3 / Year 4, Computer Sciences) | `phase5_curriculum_alignment/02_input_curriculum/curriculum_parse.py` (docling branch) |
| `audit_samples/` | Manual precision audit of AI + cyber matches (2 CSVs) | the precision numbers quoted in the write-up |

## Shared

- `data/` — the corpora every experiment reads. `job_postings_categorized.jsonl`
  is the LinkedIn 1.35M set (969 MB, **gitignored**); `job_posting.json` is the
  smaller Cambodian sample. Pinned by path in the repo `.gitignore`, so don't
  move this folder without updating it.
- `figures/` — pipeline diagrams exported for the thesis.

## Running

Each experiment is standalone; run a script directly and it anchors its own
paths on `__file__`, so the working directory does not matter:

```bash
python3 esco_retrieval/skill_to_esco.py
python3 esco_retrieval/retrieve_and_rerank.py
python3 category_filter/filter_category.py 500      # optional posting limit
python3 domain_routing/01_corpus.py                 # steps run in order
python3 filter_job_by_domain/00_load_data.py
```

Outputs land in each experiment's own `output/` directory.
