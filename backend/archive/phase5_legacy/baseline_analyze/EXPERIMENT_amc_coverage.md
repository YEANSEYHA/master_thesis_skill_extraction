# Experiment — AMC semantic coverage (course vs IT labour-market demand)

AMC-style baseline (Almaleh et al. 2019, *Align My Curriculum*) reproduced on the
LinkedIn IT corpus. A course's multi-granular skill set (CLOs + lessons + activities)
is embedded with SBERT and compared to the aggregated IT demand table; a market
skill is "covered" when some course skill is within cosine τ of it. Coverage is
reported both unweighted (% of skills) and demand-weighted (% of postings' demand),
then thresholded into MAINTAIN / REVISE / UPDATE.

- Date run: 2026-06-02
- Corpus: LinkedIn `job_postings_categorized.jsonl`, category = `it`
- Demand table: 688,150 distinct skills → lowercase-merged; **top-N by frequency**
  (top-200 = 814,160 · top-500 = 1,126,919 postings' demand)
- Threshold: τ_sim = 0.60 · labels τ_high = 0.66 / τ_low = 0.33 *(borrowed paper
  defaults — NOT yet calibrated for this metric; see Open issue below)*
- Code: `baseline_analyze/main.py` → `compute_coverage()`

## Results (same corpus, τ = 0.60)

### top-200

| Course | Type | Label | Coverage (unweighted) | Coverage (demand-weighted) |
|---|---|---|---|---|
| **Automata Theory** (GICI3AT) | Foundational/theory | 🔴 UPDATE | 11.5 % (23/200) | 16.3 % (119K/729K) |
| **Introduction to AI** (AIECS031) | Applied | 🔴 UPDATE | **19.5 % (39/200)** | **28.1 % (229K/814K)** |

### top-500

| Course | Type | Label | Coverage (unweighted) | Coverage (demand-weighted) |
|---|---|---|---|---|
| **Automata Theory** (GICI3AT) | Foundational/theory | 🔴 UPDATE | 7.8 % (39/500) | 13.5 % (152K/1,127K) |
| **Introduction to AI** (AIECS031) | Applied | 🔴 UPDATE | **14.0 % (70/500)** | **23.4 % (264K/1,127K)** |

### What Intro-to-AI matched that Automata could not

| Market skill | 📊 demand | course evidence | cosine |
|---|---|---|---|
| `python` | 19,994 | "python programming" | 0.788 |
| `project management` | 22,661 | "project implementation" | 0.746 |
| `data analysis` | 14,188 | "data analysis" | 1.000 |

Both courses also pick up the soft-skill head every course matches:
`communication` (0.634), `teamwork` (0.675), `problem solving` (0.913),
`collaboration` (1.000), `analytical skills` (0.797).

## Discussion

### 1. The applied course discriminates — the pipeline works

Intro-to-AI nearly **doubles** the demand-weighted coverage of Automata Theory
(23.4 % vs 13.5 %) and matches the **technical** demand the theory course
structurally cannot reach (python is a top-3 IT skill and was a hard gap for
Automata). This is the intended contrast: a market-demand matcher should rank an
applied course above a foundational one, and it does.

### 2. Why both still land on UPDATE

Both courses fall below the τ_low = 0.33 cutoff at every top-N, so the label stays
UPDATE even for the applied course (best case 28.1 % at top-200). The remaining
gaps are **genuine**, not noise: `sql`, `troubleshooting`, `leadership`, `java`,
`aws` are high-demand IT skills an intro ML course legitimately does not teach. A
single 3-credit course covering a third of *all* IT demand was never a realistic
bar — this is the "penalized against the whole market" effect (denominator = the
whole market, not the course's own skills).

## ⚠️ OPEN ISSUE — label thresholds not yet calibrated

The coverage **measurement** is done and working (it cleanly ranks applied >
foundational). What is **not** done is the MAINTAIN/REVISE/UPDATE **labeling**:
τ_high = 0.66 / τ_low = 0.33 are borrowed paper defaults and are miscalibrated for
the "coverage of the whole IT market" denominator — no single course exceeds 0.33,
so every course degenerates to UPDATE and the label carries no signal.

This is a **denominator + calibration** problem, not a matcher bug. Candidate fixes
(decision pending):

1. **Domain-conditioned coverage** — restrict the market to the course's job-category
   segment first (qwen3 classifier already in `clo_skill_alignment/domain_router.py`),
   then compute coverage. Coverage lands in a usable range so 0.66/0.33 regain meaning.
2. **Market-value-of-taught score** — flip the denominator to the course's own skills
   (Σ demand of matched / over course skills). Invariant to market size; removes the
   "penalized for not covering all of IT" unfairness.
3. **Recalibrate absolute cutoffs** — empirically set realistic τ_high/τ_low for this
   metric. Fast but arbitrary on only 2 courses.
4. **Percentile across courses** — relative rank (top/mid/bottom tercile). Principled
   but needs a batch of courses (only 2 so far).

### 3. Coverage drops as the market grows (top-200 → top-500)

The metric is a ratio against the market. The course's matchable footprint
saturates near the demand head (generic soft skills) by ~rank 200; ranks 201-500
are mostly specific tools the course can't touch, so adding them grows the
denominator far faster than the numerator and the ratio falls. This is expected,
not a regression — and motivates a complementary **market-value-of-taught** score
(denominator = course skills) that is invariant to how much of the market is loaded.

### 4. Two-input asymmetry still holds

The curriculum side is authoritative (school-issued, parsed verbatim); the
limiting factor is the labour-market side. Here that side is the **global LinkedIn
IT corpus** (US/UK), not Cambodian — an augmentation/robustness corpus, not the
local-novelty corpus.

## Known data-quality issues (visible in the gaps)

- `problemsolving` (📊11,766) is a no-space variant of `problem solving` —
  lowercase-merge fixed casing but not spacing/phrasing.
- `communication skills` vs `communication` — same phrasing-variant split.
- A stronger normalization (strip non-alphanumerics) would collapse these, but
  was left out of this run to keep the demand table verbatim.

## Reproduce

```bash
cd backend/phase5_curricula/baseline_analyze && python main.py
# switch course: set _ACTIVE_COURSE = _AUTOMATA_COURSE | _INTRO_AI_COURSE in main.py
# switch market size: top_n=200 | 500 in the compute_coverage() call in __main__
```
