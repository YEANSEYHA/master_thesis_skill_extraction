# Lexical Baseline (Ahadi EDM 2022 exact-match)

A **baseline / ablation** of the `clo_skill_alignment/` method. It answers the
reviewer-style question: *how much does the semantic + ESCO machinery actually
buy us over a plain lexical match?*

It re-runs the **exact same pipeline** and changes **only the matching step**:

| Stage | `clo_skill_alignment` (main) | `lexical_baseline` (this) |
|-------|------------------------------|---------------------------|
| PDF → CLOs | pdf_parser.py | *same* (reused) |
| jobs → {skill: freq} | aggregate_skills.py | *same* |
| **ESCO normalize** | ✅ JobBERT → preferred labels | ❌ **off** (raw skill strings) |
| **Representation** | embed → 768-dim vectors (SBERT) | tokenize → **word sets** |
| **"Covered?" rule** | `cos(e(CLO), e(skill)) ≥ τ_sim` | **word containment** (all skill tokens ∈ a course unit) |
| coverage | Σf(matched)/Σf(S') | *same formula* |
| MAINTAIN/REVISE/UPDATE | τ_high=0.66, τ_low=0.33 | *same thresholds* |

Everything except the matcher (and ESCO) is held constant, so any coverage
difference is attributable **only** to "semantic + ESCO."

## How the lexical match works

```
   skill  →  lowercase → strip punctuation → tokenize → drop stopwords → token set
   course →  (per CLO / lesson)  same normalization → token set

   COVERED if  skill_tokens ⊆ course_unit_tokens   (overlap ratio = 1.0)
   score    =  max over units of |skill ∩ unit| / |skill|
```

No embeddings, no ESCO, no similarity threshold to tune — just surface-form
overlap. `score` (0–1) is reported so partial overlaps are visible in the gap
list, but a match requires full containment (true exact-match baseline).

## What it reveals (vs semantic)

- **Semantic wins** on paraphrase / synonym / abbreviation:
  `"ML"` ≈ `"machine learning"`, `"SQL"` ≈ `"query databases"` → semantic
  matches, lexical **misses**.
- **Lexical can false-match** on coincidental shared tokens (e.g. a CLO with
  "analyze" lexically overlapping "data **analysis**" after stemming-like
  token sharing) — here mitigated by requiring *full* containment.
- **Lexical needs no GPU/model** and is fully deterministic.

## Files

- `lexical_matcher.py` — tokenize + stopword + set-containment; same return
  shape as `clo_skill_alignment/matcher.py:match`
- `lexical_runner.py` — coverage + classify + CLI (built-in Automata sample)
- `__init__.py`

## Run

```bash
cd backend/phase5_curricula/lexical_baseline
python3 lexical_runner.py                    # built-in Automata sample
python3 lexical_runner.py --job-freq f.json  # real raw {skill: freq}
python3 lexical_runner.py --pdf course.pdf    # parse a real syllabus (needs Ollama)
```

## Caveat for the comparison

On the **current general-job corpus**, real top-N skills are soft-skill
dominated, so BOTH methods score ~0 % for a theory course — the delta is
invisible. The lexical-vs-semantic comparison is only meaningful on a
**tech-filtered corpus** (or with skill inventories that actually share
vocabulary with the course), where semantic matching's paraphrase coverage
shows up as a measurable gain. The built-in sample includes automata skills
precisely so the baseline's matching behavior is demonstrable in isolation.
