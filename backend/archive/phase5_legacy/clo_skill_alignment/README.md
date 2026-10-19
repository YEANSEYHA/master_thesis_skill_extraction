# CLO-Skill Alignment

The core Phase-5 methodology: align a course's **Learning Outcomes (CLOs)**
against the **labour-market skill demand** extracted from job postings, and
emit a per-course recommendation — **MAINTAIN / REVISE / UPDATE**.

It is an **open-stack adaptation of Ahadi et al. (EDM 2022)**, "Skills Taught
vs Skills Sought", replacing the closed/lexical components with an open,
semantic pipeline:

| Ahadi 2022                          | This work (CLO-Skill Alignment)                       |
|-------------------------------------|-------------------------------------------------------|
| EMSI / Burning Glass (paid, closed) | qwen3:14b LLM extraction + **ESCO** normalization     |
| Exact lexical set match             | **JobBERT / SBERT cosine** similarity + threshold     |
| Course description as unit          | **Course Learning Outcomes (CLOs)** — finer grained   |
| Unweighted gap counts               | **Frequency-weighted** coverage Σf(matched)/Σf(all)   |
| Gap inventory                       | **3-class action label** (MAINTAIN/REVISE/UPDATE)     |
| US / EU context                     | **Cambodian** higher-education context (ITC)          |

## Pipeline

```
extracted job skills
   │  ESCO normalize (JobBERT → preferred labels)   [esco_normalizer]
   ▼
canonical ESCO skill inventory  S, with frequency f(s)
   │
course CLOs + lessons  ──► composite text T(c)
   │  embed both sides (SBERT)                       [embedder]
   ▼
cos(CLO, skill) ≥ τ_sim → matched(c)                 [matcher]
   │
coverage(c) = Σ f(s) for matched / Σ f(s) over S
   │  threshold                                       [runner]
   ▼
coverage ≥ τ_high → MAINTAIN
coverage ≥ τ_low  → REVISE
else              → UPDATE
+ top-K gaps (highest-demand unmet skills)
```

## Formal algorithm

```
Inputs
  C = { c_1, ..., c_n }    courses, each with CLOs + lessons
  S = { s_1, ..., s_m }    ESCO-normalized skill inventory
  f : S → ℕ                skill frequency across all job postings

Parameters (defaults)
  τ_sim  = 0.60   match threshold
  τ_high = 0.66   MAINTAIN cutoff
  τ_low  = 0.33   UPDATE cutoff
  K_gaps = 5      gaps reported per course

For each course c ∈ C:
  T(c)        = concat( synopsis(c), CLO_text(c), lesson_outcomes(c) )
  matched(c)  = { s ∈ S : cos( e(T(c)), e(s) ) ≥ τ_sim }
  coverage(c) = Σ_{s ∈ matched(c)} f(s)  /  Σ_{s ∈ S} f(s)
  label(c)    = MAINTAIN if coverage ≥ τ_high
                REVISE   if coverage ≥ τ_low
                UPDATE   otherwise
  gaps(c)     = top-K_gaps of (S \ matched(c)) by f(s) desc

Output  Report = [ ( c, coverage(c), label(c), gaps(c) ) ]
```

## Modules (this package)

- `esco_normalizer.py` — raw skills → ESCO preferred labels (JobBERT, cached in `.esco_cache/`)
- `embedder.py`        — SBERT singleton (all-mpnet-base-v2) for CLO↔skill matching
- `matcher.py`         — cosine matching + max-pool → matched / gaps
- `runner.py`          — per-course orchestration + MAINTAIN/REVISE/UPDATE
- `curriculum_runner.py` — programme-level aggregation
- `aggregate_skills.py` — build the {skill: frequency} inventory from Phase-3
- `clo_skill_extractor.py` — (optional) CLO → atomic-skill decomposition
- `pdf_parser.py`      — ITC course PDF → {course, CLOs, lessons}
- data: `job_freq.json`, `dummy_job_skills.json`; cache: `.esco_cache/` (gitignored)

> Consumed by `backend/phase1_job_site_discovery/api.py`, which loads these via importlib
> from this package directory (`phase5_curricula/clo_skill_alignment/`) and puts
> it on `sys.path` so the sibling imports (e.g. `from matcher import match`)
> resolve.

## Known limitations

- **ESCO mapping uses English JobBERT** — Khmer-script skills garbage-map to a
  single arbitrary label; "X skills" phrases over-collapse. See the top-level
  CLAUDE.md "Known issues".
- **Matcher uses SBERT (mpnet)** while the paper claims JobBERT for matching —
  consistency gap to resolve.
- **Corpus is soft-skill dominated** (general Cambodian job boards), so pure
  technical courses score low coverage regardless of tuning.
