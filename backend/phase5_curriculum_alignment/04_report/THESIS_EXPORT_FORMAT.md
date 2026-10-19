# Thesis export format — one bundle per cohort run

Finalized artifact spec for the Y3 / Y4 / Y5 / Y3–Y5 experiments.
All samples below are the **real values** from run `20260726T023922Z-475eb3`
(Y3, Cambodia corpus, 2026-07-26).

## Principle

`outputs/<run_id>/` is canonical and is never edited by hand. Everything
in this document is **derived** from the three JSONs already written there:

    params.json   run identity, resolved params, config/input hashes, git sha
    report.json   kl, covered, skills[30], recommendations, selection.plan
    inputs.json   curriculum files + sha256 per course, corpus, rerank data

No number in a thesis table is ever typed by a human. If a threshold is
recalibrated (remaining #2), re-run the export and every table updates.

## Layout

    outputs/<run_id>/              ← canonical, written by run_store.py
        report.json  params.json  inputs.json  report.md  log.txt  heatmap.png
        thesis/                    ← generated, safe to delete and rebuild
            run_summary.json
            course_manifest.csv
            skill_status.csv
            heatmap_cells.csv
            recommendations.csv

    outputs/thesis_registry.csv    ← one row per cohort, built from all runs

Keep the run_id as the only key. Do not copy bundles into a parallel
`final_experiments/` tree — two copies drift.

---

## 1. `run_summary.json` — the cohort's headline record

One flat object per run. Every field traces to a source file; the
`_provenance` block records where, so a reader can audit any number.

```json
{
  "run_id": "20260726T023922Z-475eb3",
  "cohort": "Y3",
  "label": "V8 tab · cambodia · 9 saved courses · Y3",
  "experiment_date": "2026-07-26",
  "started_at": "2026-07-26T02:39:23+00:00",
  "duration_s": 15.3,

  "provenance": {
    "git_sha": "7ecb9cb4",
    "config_hash": "475eb38e6ff0a1eaabd0f377e56dcf53ca500fe19beb84aca06a771af0a90eb6",
    "input_hash": "f051a7b241a1bd0445f45720a69938109f301fa44f80495092b14c412794ed3e",
    "cache_version": "v3"
  },

  "curriculum": {
    "source": "db",
    "n_courses": 9,
    "n_clo_skills": 218,
    "dir": "02_input_curriculum/clo_skills_db"
  },

  "job_market": {
    "dataset": "stage2_result.json",
    "source": "cambodia",
    "segment_mode": "category",
    "n_jobs": 254,
    "top_n": 30,
    "min_skill_freq": 2
  },

  "configuration": {
    "bi_encoder": "BAAI/bge-base-en-v1.5",
    "cross_encoder": "BAAI/bge-reranker-v2-m3",
    "skill_merge_threshold": 0.80,
    "covered_at": 0.73,
    "borderline_at": 0.69,
    "host_min": 0.66,
    "rerank_supported_at": 0.50,
    "rerank_weak_at": 0.10,
    "rerank_top_k": 3,
    "eps": 1e-06,
    "cum_frac": 0.80,
    "spill_min": 0.05,
    "thresholds_status": "PROVISIONAL — linearly mapped from MiniLM; LLM-judge silver study fits strict `covered` at ~0.80. Awaiting human gold (remaining #1) → calibration (#2)."
  },

  "measured_results": {
    "kl_divergence": 0.3784,
    "covered": 12,
    "borderline": 5,
    "gap": 13,
    "coverage_rate_percent": 40.0,
    "demand_weighted_coverage_percent": 45.1,
    "demand_weighted_coverage_half_credit_percent": 52.4,
    "host_verified": 22,
    "host_unverified": 8,
    "rerank_supported": 10,
    "rerank_weak": 3,
    "rerank_not_supported": 17
  },

  "modelled_results": {
    "_note": "Simulated by greedy facility-location: 'adding' a skill inserts a synthetic CLO that embeds the skill itself. Upper bound, not a forecast. Never report as an achieved result.",
    "gaps_considered": 18,
    "n_selected": 13,
    "stop_rule": "80% of achievable gain (cum_frac=0.80)",
    "kl_final": 0.3474,
    "covered_final": 28,
    "kl_absolute_improvement": 0.0310,
    "kl_relative_improvement_percent": 8.2,
    "routes": {"new": 14, "extend": 3, "partial": 1},
    "actions": {"new_content": 14, "extend": 3, "add_module": 1}
  },

  "evaluation_ref": {
    "_note": "Precision/recall/F1 are properties of the INSTRUMENT (the 150-pair annotation study), not of a cohort run. Do not fill per-run.",
    "calibration_record": "05_evaluation/llm_judge_itc_results.json",
    "gold_status": "silver only (LLM judges, Cohen κ=0.83); human gold 0/150"
  },

  "known_limits": [
    "KL is a point estimate with no confidence interval (remaining #6).",
    "KL is encoder-bound: the profile is raw unthresholded max-cosine and BGE's floor for unrelated pairs is ~0.5, so KL is systematically damped.",
    "KL is computed over the full ~571-skill segment, not just top_n. At the coverage magnitudes a single cohort can reach (~12-17/30), KL sits within ~6% of the 'teaches nothing' baseline (measured: null=0.3988, cohorts 0.374-0.378) and does not usefully separate cohorts. demand_weighted_coverage_percent is the primary cross-cohort metric; KL is a secondary distributional stat, not a ranking metric.",
    "KL is not comparable across corpora of different size; only cross-cohort comparisons on this same 254-posting segment are valid here.",
    "Coverage is max cosine, not a percentage. coverage_rate_percent is the SHARE OF SKILLS covered (12/30), never a cosine rendered as a percent.",
    "demand_weighted_coverage_percent = share of top_n's job-count mass that is covered (Σjobs[covered] / Σjobs[top_n]) -- how much of what employers ask for is taught, not how many boxes are ticked. Inherits the covered_at threshold (0.73, provisional); report the half-credit variant alongside it as a sensitivity bound."
  ]
}
```

---

## 2. `course_manifest.csv` — exactly which courses fed the run

`report.json` stores course *names* only. This pins codes, phrase counts
and the file hash, because DB course selection changes between uploads.

```csv
run_id,cohort,course_code,course_name,study_year,n_clo_skills,source_file,sha256
20260726T023922Z-475eb3,Y3,GICI31ACI,Information System Analysis And Design,Y3,20,1_information_system_analysis_and_design.json,5b3a0578783ecf55
20260726T023922Z-475eb3,Y3,GICI31ALP,Algorithm And Programming,Y3,17,2_algorithm_and_programming.json,5b871bd85a667e71
20260726T023922Z-475eb3,Y3,GICI31CSL,Combinational And Sequential Logics,Y3,26,3_combinational_and_sequential_logics.json,84c4d09605009ee3
20260726T023922Z-475eb3,Y3,GICI31CSN,Introduction To Computer Systems And Networks,Y3,27,4_introduction_to_computer_systems_and_networks.json,976897cfcf0186d7
20260726T023922Z-475eb3,Y3,GICI31SS,Soft Skills,Y3,2,5_soft_skills.json,6cc36f2aa25a6102
20260726T023922Z-475eb3,Y3,GICI32OOP,Object Oriented Programming,Y3,20,6_object_oriented_programming.json,f5df60a76f2c56ec
20260726T023922Z-475eb3,Y3,GICI3DM,Discrete Mathematics,Y3,38,7_discrete_mathematics.json,565f2b29180bc42d
20260726T023922Z-475eb3,Y3,GICI3ELE,Electronics,Y3,30,8_electronics.json,a488b02307c3590b
20260726T023922Z-475eb3,Y3,GICI3RM,Research Methodology,Y3,38,9_research_methodology.json,<sha>
```

Course codes come from the DB row; the export joins them on course name.
Phrase counts sum to `n_clo_skills` (20+17+26+27+2+20+38+30+38 = 218 ✓) —
use that as the export's self-check.

---

## 3. `skill_status.csv` — the thesis results table (30 rows)

One row per market skill. This is the table you cite in the results
chapter.

```csv
rank,market_skill,job_count,program_score,status,best_course,best_clo,rerank_score,rerank_verdict,host_ok,priority
1,technical documentation,49,0.78,covered,Information System Analysis And Design,Data Integrity and Data Quality Documentation,0.03,NOT_SUPPORTED,true,0.0418
2,database management,43,0.91,covered,Information System Analysis And Design,Database Management Systems (DBMS),0.99,SUPPORTED,true,0.0158
3,java programming,41,0.87,covered,Object Oriented Programming,Java,1.00,SUPPORTED,true,0.0202
```

- `rank` = descending `job_count` (demand order), matching the heatmap rows.
- `program_score` = max cosine over all courses × all their CLO skills.
- `status` ∈ `covered` (≥0.73) · `borderline` ([0.69, 0.73)) · `gap` (<0.69).
- `rerank_verdict` ∈ `SUPPORTED` (≥0.50) · `WEAK` ([0.10, 0.50)) · `NOT_SUPPORTED` (<0.10).
- Cosine and rerank scores are **different scales** — never average or compare them.

Row 1 is the case worth discussing in the write-up: *technical
documentation* is `covered` at 0.78 but the cross-encoder scores its best
CLO at 0.03. Cosine argmax ≠ evidence of teaching. In Y3, 17 of 30 skills
are NOT_SUPPORTED while only 13 are `gap` — that divergence is a finding,
not a bug.

---

## 4. `heatmap_cells.csv` — the figure's underlying data (30 × n_courses)

Long format, one row per skill × course. 270 rows for Y3. This is what
`heatmap.png` draws, exported so the figure is reproducible without rerunning.

```csv
run_id,cohort,rank,market_skill,job_count,course_name,cosine,is_row_max,cell_status
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Information System Analysis And Design,0.78,true,covered
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Algorithm And Programming,0.63,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Combinational And Sequential Logics,0.66,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Introduction To Computer Systems And Networks,0.65,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Soft Skills,0.52,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Object Oriented Programming,0.68,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Discrete Mathematics,0.63,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Electronics,0.63,false,gap
20260726T023922Z-475eb3,Y3,1,technical documentation,49,Research Methodology,0.67,false,gap
20260726T023922Z-475eb3,Y3,2,database management,43,Information System Analysis And Design,0.91,true,covered
```

`cell_status` applies the same thresholds per cell; `is_row_max` marks the
ringed cell in the figure. Cell colours are painted by the PROVISIONAL
thresholds — say so in the figure caption.

---

## 5. `recommendations.csv` — the greedy skill plan (18 rows)

One row per candidate gap, in greedy order, selected and skipped alike.
Keeping the skipped rows is what makes the 80% stopping rule auditable.

```csv
step,skill,job_count,selected,gain_pct,kl_after,covered_after,route_kind,route_course,route_clo,route_score,relationship_kind,action_kind,action_title,spillover
1,project management,37,true,2.4,0.3742,13,new,,,0.02,none,new_content,Add new project management content,
2,security monitoring,20,true,2.0,0.373,15,new,,,0.02,none,new_content,Add new security monitoring content,security compliance:0.78
4,technical support,27,true,1.8,0.3701,17,extend,Object Oriented Programming,,,content_exists,extend,Extend Object Oriented Programming with explicit technical support content,
6,mysql database,30,true,1.6,0.3624,21,partial,Information System Analysis And Design,,,foundation,add_module,Add mysql database content building on Information System Analysis And Design,postgresql database:0.8
14,incident response,19,false,0.8,0.3455,29,extend,Object Oriented Programming,,,content_exists,extend,Extend Object Oriented Programming with explicit incident response content,
```

- `route_kind` ∈ `new` · `extend` · `partial` — the three-level route label
  from the cross-encoder joint read (✓ extend ≥0.50 / ◑ builds-on 0.10–0.50 / ＋ new <0.10).
- `spillover` = `skill:new_score` pairs that a selection lifts for free
  (e.g. adding *security monitoring* lifts *security compliance* to 0.78).
- `gain_pct` = share of today's demand-weighted gap over the WHOLE segment,
  not just the top-30.
- `kl_after` / `covered_after` are **modelled**, not measured.
- **The candidate set is gaps AND borderlines.** Y3's 18 rows = 13 `gap` +
  5 `borderline`, so the dashboard's "13 / 18 gaps worth adding (of all gaps
  found)" is loose wording: 18 is not the gap count (13 is). Write it as
  "13 of 18 candidate skills (13 gap + 5 borderline)" in the thesis.

---

## 6. `outputs/thesis_registry.csv` — the cross-cohort table

One row per cohort, generated by scanning every `outputs/*/thesis/run_summary.json`.
This is the comparison table in the experiments chapter.

```csv
cohort,run_id,experiment_date,git_sha,config_hash,input_hash,n_courses,n_clo_skills,n_jobs,top_n,encoder,covered,borderline,gap,coverage_rate_percent,demand_weighted_coverage_percent,demand_weighted_coverage_half_credit_percent,kl,n_selected,kl_final_modelled,covered_final_modelled
Y3,20260726T031810Z-475eb3,2026-07-26,08231876,475eb38e,f051a7b2,9,218,254,30,bge-base-en-v1.5,12,5,13,40.0,45.1,52.4,0.3784,13,0.3474,28
Y4,20260726T031858Z-475eb3,2026-07-26,08231876,475eb38e,54f4eaed,8,146,254,30,bge-base-en-v1.5,17,6,7,56.7,54.9,64.7,0.3741,9,0.3486,27
Y5,20260726T031941Z-475eb3,2026-07-26,08231876,475eb38e,917674e9,8,226,254,30,bge-base-en-v1.5,16,6,8,53.3,54.1,63.4,0.3743,10,0.3484,28
Y3-Y5,,,,,,,,254,30,bge-base-en-v1.5,,,,,,,,,
```

**`demand_weighted_coverage_percent` is the primary cross-cohort number.**
KL barely moves across these three real runs (0.3784 / 0.3741 / 0.3743 — all
within ~6% of a measured "teaches nothing" null of 0.3988, because it's
computed over the full ~571-skill segment and a ~15-skill curriculum can't
shift a distribution dominated by an untouched 541-skill tail). Demand-weighted
coverage separates them cleanly (45.1% / 54.9% / 54.1%) because it only looks
at the top_n skills the report is actually about. Lead the results section
with this column; report KL next to its null baseline as a secondary stat,
never as a cohort ranking.

`config_hash` identical across rows + `input_hash` differing = the cohorts
differ only in curriculum, which is exactly the claim the comparison rests
on. Show both columns in the thesis; they are the reproducibility argument.

---

## Reporting rules for these tables

1. **`n_jobs` is deliberately held constant at 254** for all four cohorts —
   same market segment, different curriculum slices. State this. The
   fair-matrix rule normally matches unit to segment; holding the segment
   fixed is what makes the cohorts comparable to each other.
2. **The Y3–Y5 row will mechanically beat every single year.** More
   curriculum = more chances to match, so lower KL and higher coverage are
   expected by construction. It is not evidence that the program is better
   than its parts.
3. **Never write a cosine as a percentage.** `coverage_rate_percent` = 40.0
   means 12 of 30 skills, not "40% similarity". Program scores stay as
   two-decimal cosines.
4. **Measured vs modelled.** KL = 0.3784 is measured. KL = 0.3474 is what
   the greedy simulation predicts after 13 additions. Keep the words
   `measured_` and `modelled_` in every table header.
5. **KL does not separate single-cohort curricula — do not use it to rank
   cohorts.** Report `demand_weighted_coverage_percent` as the headline
   comparison metric instead (it does separate them); state KL next to its
   measured null baseline (0.3988 — a curriculum that teaches nothing) as a
   secondary distributional stat.
6. **Every threshold-dependent count is provisional** until remaining #1
   → #2 land. Covered/borderline/gap counts and all cell colours move if
   the strict bar goes to ~0.80. Footnote the tables now so you do not have
   to rewrite the prose later.
7. **KL has no error bar yet** (remaining #6). Do not report differences
   between cohort KLs as significant until the bootstrap CI exists.
