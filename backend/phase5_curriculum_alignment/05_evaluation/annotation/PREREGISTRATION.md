# Pre-registered analysis plan — 250-pair human rating study

Declared **2026-07-28, before any participant data was collected**. The git
commit that introduces this file is its timestamp. These rules are applied as
written; any deviation must be reported as a deviation in the thesis.

## Instrument

- The 250 frozen skill × course pairs of `sheet_250_master.json`
  (sheet `78dfd3fe9b637a36…`, locked run `20260727T105055Z-475eb3`,
  config `475eb38e`, input `a7c8b24a`, seed 20260728).
- Collected through the `web_annotate/` app. Raters are **blinded**: the
  payload carries no cosine, band, stratum, system decision, or job-ad
  frequency. Every rater sees the 30 anchor pairs first in fixed order, then
  their own random permutation of the remaining 220 (fixed at signup).
- Scale: integers 0–10 with printed anchors 0 = not taught, 5 = the basics
  only, 10 = fully taught. There is no "unable to assess" button; a rater who
  cannot judge a pair leaves it unscored (a *skip*). Skips are missing data:
  excluded from that pair's mean, never imputed.

## Data collected and anonymisation

Per rating: score, optional free-text note, response time (ms), position in
the rater's sequence. Per rater: name, optional student/staff ID, role,
familiarity, browser user agent, consent flag. **Names and IDs exist only for
login-resume. In the thesis and any publication, raters appear as codes
(R01, R02, …); the code↔name mapping is kept off-repo and never published.**

## Declared analysis rules

1. **Primary reliability statistic: Krippendorff's α with ordinal weights**,
   computed on the 30-pair anchor block over all included raters (α handles
   the missing cells that skips create). ICC(2,k) on the same block is
   reported as a supplementary check only.

2. **Categorical mapping** — used *only* to compare against the completed
   150-pair LLM silver study: 0–3 = No, 4–6 = Partly, 7–10 = Yes.

3. **Threshold selection (single method).** For each pair, the human score is
   the **mean over included raters**. The reference label is
   **covered ⇔ human mean ≥ 5.0** (the scale's printed midpoint: the course
   teaches at least the basics of the skill). The adopted cosine threshold is
   the value that **maximises Youden's J on the ROC** of the frozen pipeline's
   cosine against this label. The fitted threshold is adopted **whatever value
   it takes**, and the four cohorts are re-run with it. Pre-declared
   sensitivity analyses (reported, never headline): (a) strict label at human
   mean ≥ 7.0 (the "Yes" band); (b) the provisional 0.73 bar.

4. **Rater exclusion — applied before any analysis; all exclusions reported
   with reasons.** A rater is excluded entirely if any of:
   - (a) median response time over their rated pairs **< 2 seconds**;
   - (b) Spearman ρ between their anchor-block scores and the
     leave-one-out mean of the other included raters **< 0.20**;
   - (c) fewer than **20 rated pairs** in total (too little to quality-check).

5. **Role weighting.** If **≥ 5 raters with role lecturer or teaching
   assistant** complete the anchor block, the lecturer/assistant mean is the
   primary reference and the all-rater mean is the robustness check;
   otherwise the all-rater mean is primary. Student and industry ratings are
   supplementary in either case, with subgroup agreement reported. (This
   conditional rule is itself declared here, before recruitment.)

6. **Coverage / stop rule.** Random per-rater ordering gives unbiased
   *expected* coverage, not guaranteed equal coverage under early stopping.
   Collection continues until **every one of the 250 pairs has ≥ 3 valid
   ratings** and **≥ 10 raters have completed all 250 pairs**, monitored via
   the admin dashboard (`min_raters`). 10 complete raters is the adequacy
   floor; 15–20 is the target for subgroup comparisons.

## Where the blind is lifted

`npm run export` writes ratings **still blinded**. The analysis script's join
back to `sheet_250_master.json` on `row` is the first and only place the human
score and the system cosine meet.
