# Phase 5 — Curriculum Recommendation: Implementation Task Breakdown

> **Goal.** Implement the end-to-end curriculum-job alignment system formally
> specified in Section 2.4 of the Scientific Day manuscript, and run it on
> the full ITC 2025–2026 curriculum to produce per-course
> **maintain / revise / update** recommendations.
>
> **Status.** Algorithm DESIGNED + SPECIFIED + DEMONSTRATED on one course
> (Automata Theory). Full implementation + evaluation is post-May-29 work
> for the journal extension.

---

## 0. Visual Flow Summary (TL;DR)

### One-frame overview

```
   📚 CURRICULUM            🤖 RECOMMENDATION         🎯 ACTION
   (Skills Taught)          ENGINE                    LABEL

   📄 PDF                                             🟢 MAINTAIN
     +              ───▶   embed → cosine     ───▶   🟡 REVISE
   📊 Job-market           → score → classify         🔴 UPDATE
   skills
   (Skills Sought)                                    + gap list
```

### Two streams meeting in the middle

```
   📚 SKILLS TAUGHT (Left)              💼 SKILLS SOUGHT (Right)
   ────────────────────                 ────────────────────────

   📄 ITC course PDF                    🌐 Cambodian job portals
        │                                       │ (Phase 1 + 2)
        ▼                                       ▼
   pdf_parser.py                         stage2_extraction
   PDF → markdown                        HTML → skills (qwen3:14b)
   → CLOs JSON                           (Phase 3)
   (Stages 1 + 2) ✅                     ✅
        │                                       │
        ▼                                       ▼
   clo_skill_extractor.py                aggregate_skills.py
   CLOs → skill phrases                  Counter → {skill: freq}
   (Stage 3) ✅                          (Stage 5) ✅
        │                                       │
        └────────────┬──────────────────────────┘
                     ▼

              ╔════════════════╗
              ║   matcher.py   ║
              ║   + runner.py  ║
              ╠════════════════╣
              ║ • embed both   ║
              ║ • cosine sim   ║
              ║ • threshold    ║
              ║ • coverage     ║
              ║ • classify     ║
              ║ • gap list     ║
              ║ (Stages 6+7) ✅║
              ╚═══════╤════════╝
                      ▼

   🎯 PER-COURSE RECOMMENDATION REPORT
      { label, matched[], gaps[], coverage, metadata }
```

### Algorithm in plain math

```
   FOR EACH COURSE c:

     T(c)        = synopsis(c) + CLOs(c) + lessons(c)
     e(c)        = JobBERT( T(c) )           ∈ R^768

     FOR EACH skill s in job market:
       sim(c, s) = cos( e(c), JobBERT(s) )

     matched(c)  = { s : sim(c, s) ≥ τ_sim }    τ_sim = 0.60

                       Σ_{s ∈ matched} freq(s)
     coverage(c) = ─────────────────────────
                       Σ_{s ∈ all} freq(s)

     IF   coverage(c) ≥ 0.66 → 🟢 MAINTAIN
     ELIF coverage(c) ≥ 0.33 → 🟡 REVISE
     ELSE                    → 🔴 UPDATE

     gaps(c) = top-K skills NOT matched, sorted by freq desc
```

### Backend → API → Frontend flow

```
   📂 BACKEND (FastAPI :8000)        🌐 FRONTEND (Next.js :3000)
   ─────────────────────────         ─────────────────────────────

   POST /api/curricula/upload  ◀───  📤 Drop PDF (Step 1)
        │
        ▼
   pdf_parser.py
        │
        ▼
   { course, lessons, CLOs }  ───▶   📋 Display (Step 2)

   POST /api/curricula/analyze ◀───  ⚙️ Generate (Step 3)
        │
        ▼
   matcher.py + runner.py
        │
        ▼
   { label, matched, gaps,    ───▶   🎯 Renders big label
     coverage, evidence }              + matched table
                                       + gap table
                                       + Dummy/Real toggle
```

### File map (what is built)

```
   📂 backend/phase5_curricula/
     pdf_parser.py            Stages 1+2 ✅
     clo_skill_extractor.py   Stage 3    ✅
     aggregate_skills.py      Stage 5    ✅
     matcher.py               Stage 6    ✅
     runner.py                Stage 7    ✅
     embedder.py              SBERT wrapper ✅
     dummy_job_skills.json    Test data  ✅
     job_freq.json            Real data  ✅

   📂 backend/phase1_job_site_discovery/api.py
     POST /api/curricula/upload    ✅
     GET  /api/curricula/skills    ✅
     POST /api/curricula/analyze   ✅  (use_dummy flag)

   📂 frontend/src/app/components/CurriculaTab.js
     Step 1: Upload PDF                       ✅
     Step 2: Configure topN + threshold       ✅
     Step 3: Recommendation + Dummy/Real toggle ✅
```

### 30-second elevator pitch

> Our recommendation engine adapts Ahadi et al.'s
> Skills-Taught-vs-Skills-Sought framework.
>
> Left side: PDF → markdown → CLOs → skill phrases
> (pymupdf4llm + qwen3:14b).
>
> Right side: Cambodian job postings → LLM-extracted skills
> → frequency aggregation (Phase 3 + aggregate_skills.py).
>
> Match: both sides go through JobBERT (or SBERT) cosine
> similarity, instead of Ahadi's exact lexical alignment.
> This captures paraphrased skills like "ML" ≈ "machine
> learning".
>
> Classify: frequency-weighted coverage thresholded into
> 🟢 MAINTAIN / 🟡 REVISE / 🔴 UPDATE.
>
> Output: per-course action label + 5-skill gap list +
> per-match evidence (which lesson/CLO matched).
>
> Demo: end-to-end on Automata Theory in 2.6 seconds,
> running in the Curricula tab on localhost:3000.

---

## 0.5. Step-by-Step Visual Process (as built)

End-to-end walkthrough of the **currently implemented** pipeline, including the
ESCO normalization layer. The methodology code now lives in
`phase5_curricula/clo_skill_alignment/`.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                  CLO-SKILL ALIGNMENT — FULL PROCESS                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### STEP 1 — Upload course syllabus  📤

```
   User drops ITC course PDF  (Curricula tab, Step 1)
        │  POST /api/curricula/upload
        ▼
   pdf_parser.py        PDF ──pymupdf4llm──▶ markdown
                         markdown ──RAG──▶ qwen3:14b ──▶ structured
        ▼
   { course, synopsis, learning_outcomes[CLOs], lessons[] }
```

### STEP 2 — Build the labour-market skill inventory  💼

```
   Phase 3 outputs  (stage2_result.json × 3,216 postings)
        │  GET /api/curricula/skills   (aggregate_skills logic)
        ▼
   Counter over all extracted skills, lowercased
        ▼
   raw {skill: freq}      5,098 unique  ·  e.g. "communication skills":252
```

### STEP 3 — Normalize to ESCO  🗂️   ← the new layer

```
   raw {skill: freq}
        │  esco_normalizer.py  (JobBERT, 99,624 ESCO labels, disk-cached)
        ▼
   for each unique skill:  cos(JobBERT(skill), every ESCO label) → argmax
        │                  if sim ≥ τ_esco (0.50) → ESCO preferred label
        │                  else                    → keep raw
        ▼
   group by ESCO label, SUM frequencies
        ▼
   ESCO {skill: freq}     5,098 → 2,258 unique   (variants merged)
        ▲
        │  Step 2 card toggle:  [ Raw | ESCO ]   ESCO = default
        └────────────────────────────────────────────────────────
```

### STEP 4 — Select the comparison set  🎚️

```
   ESCO inventory S, sorted by freq desc
        │  Top-N slider (default 15)
        ▼
   S' = top-N most in-demand ESCO skills   ← what CLOs are matched against
```

### STEP 5 — Embed both sides  🧮

```
   COURSE side                          SKILL side
   ───────────                          ──────────
   T(c) = synopsis + CLOs + lessons     each s ∈ S'
        │ embedder.py (SBERT mpnet)          │ embedder.py (SBERT mpnet)
        ▼                                    ▼
   e(c) ∈ R^768  (one vec per             e(s) ∈ R^768
   lesson + per CLO)
```

### STEP 6 — Match (cosine + threshold)  🔗

```
   sim(c,s) = cos( e(c), e(s) )   for every (CLO/lesson × skill)
        │  max-pool per skill (best evidence)        matcher.py
        ▼
   matched(c) = { s ∈ S' : sim ≥ τ_sim (0.60) }
   gaps(c)    = S' \ matched(c)
```

### STEP 7 — Score coverage  📊

```
                   Σ  freq(s)  for s ∈ matched(c)
   coverage(c) = ───────────────────────────────
                   Σ  freq(s)  for s ∈ S'
```

### STEP 8 — Classify + report  🎯

```
   coverage(c) ≥ 0.66  → 🟢 MAINTAIN     runner.py
   coverage(c) ≥ 0.33  → 🟡 REVISE       (curriculum_runner.py for
   else                → 🔴 UPDATE         programme-level aggregate)
        ▼
   { label, coverage, matched[], gaps[top-K], evidence, metadata }
        │  rendered in Curricula tab, Step 3
        ▼
   🎯  big label  +  matched table  +  gap table  +  [Course|Curriculum] [Dummy|Real]
```

### Whole pipeline on one line

```
   PDF ─▶ CLOs ─┐
                ├─▶ embed ─▶ cosine ─▶ τ_sim ─▶ coverage ─▶ MAINTAIN/REVISE/UPDATE
   jobs ─▶ raw skills ─▶ ESCO normalize ─▶ top-N ─┘
```

### Where each step lives  📂

```
   backend/phase5_curricula/clo_skill_alignment/
     pdf_parser.py          Step 1   PDF → CLOs
     aggregate_skills.py    Step 2   jobs → {skill: freq}
     esco_normalizer.py     Step 3   raw → ESCO (JobBERT, .esco_cache/)
     embedder.py            Step 5   SBERT singleton
     matcher.py             Step 6   cosine + threshold
     runner.py              Steps 7-8 coverage + classify (per course)
     curriculum_runner.py   Steps 7-8 programme-level aggregate

   backend/phase1_job_site_discovery/api.py
     POST /api/curricula/upload            Step 1
     GET  /api/curricula/skills?normalize  Steps 2-3  (esco|none)
     POST /api/curricula/analyze           Steps 4-8  (esco_normalize flag)
     POST /api/curricula/analyze-curriculum  programme view

   frontend/src/app/components/CurriculaTab.js
     Step 1 Upload  ·  Step 2 Configure [Raw|ESCO] + Top-N + threshold
     Step 3 Recommend  [Course|Curriculum] [Dummy|Real]
```

---

## 0.6. Results & Discussion — Automata Theory case study

Real end-to-end run on the actual ITC syllabus `automa_theory_course.pdf`
(GICI3AT, Computer Sciences), skill source `phase3+esco`, Top-N = 15,
τ_sim = 0.65.

```
   Course:    GICI3AT — Automata Theory (3 CLOs, 8 lessons)
   Inventory: 5,098 raw → 2,258 ESCO skills (from 3,216 postings)
   Result:    🔴 UPDATE · coverage 0.0% · 0 / 15 skills matched
   Gaps:      teach communication methods (252), communication (186),
              teamwork principles (182), customer service (163),
              organise information, objects and resources (134)
```

### 1. The headline result

The recommendation is **UPDATE** with **0.0 % frequency-weighted coverage** —
none of the top-15 ESCO-normalized labour-market skills are matched by any of
the course's CLOs/lessons. The reported gaps are all generic/soft skills. On
the surface this reads as *"the Computer Sciences curriculum is completely
misaligned — add communication, teamwork, and customer-service courses,"* which
is **algorithmically correct but substantively misleading**. Unpacking why is
the point of this discussion.

### 2. The result is driven by a corpus mismatch, not a curriculum defect

The inventory `S` is aggregated from **general Cambodian job portals**, dominated
by sales, hospitality, admin, and customer-facing roles. The top of the demand
distribution is therefore **entirely soft/transversal skills**; there are
essentially **no theoretical-CS skills** (automata, formal languages,
computability) near the top-15.

Automata Theory is a **foundational theory course** (DFA, NFA, PDA, Turing
machines, CFGs). Its CLOs are about *analysing, designing, and evaluating
computational models* — content a general job board never advertises as a
"skill." So 0 % is **structurally guaranteed**: a pure-theory course compared
against a soft-skill-dominated corpus cannot match, regardless of threshold or
Top-N. **The UPDATE verdict reflects the sampled labour market, not a
deficiency in the course.**

This is exactly Ahadi et al.'s **REDUNDANT / "Taught − Sought"** quadrant:
knowledge taught but not directly sought. The framework measures *market
alignment* and is structurally **blind to foundational/long-horizon value** —
automata theory underpins compilers, parsing, and verification, but those
second-order benefits are invisible to a frequency count of job-ad phrases.

### 3. The comparison set is also partly corrupted by ESCO artifacts

Inspecting the top-15 ESCO skills:

- **"issue fines" (418×) — the #1 "in-demand skill" — is a pure artifact.**
  Every raw skill feeding it is **Khmer script** (ភាសាអង់គ្លេស = "English
  language", ដោះស្រាយបញ្ហា = "problem solving"…), all mapped at an identical
  0.739. JobBERT is **English-only** and cannot embed Khmer, so it dumps all
  Khmer skills into one arbitrary label.
- **"advertising techniques" (241×)** absorbs negotiation/organizational/
  presentation/computer skills — JobBERT over-weights the token "skills".
- **"teach communication methods" (265×)** is "communication skills" mis-mapped.

So roughly **half of the visible top skills are ESCO mapping errors**, not real
demand. **Robustness note:** this noise corrupts the *gap labels' readability*
but **not the 0 % verdict** — a theory course matches neither clean soft skills
nor artifacts. The verdict is robust to ESCO noise; only the interpretability of
the gap list degrades.

### 4. Threats to validity

| # | Threat | Effect on this result |
|---|--------|----------------------|
| 1 | **Corpus representativeness** — general job board, not CS/IT roles | Guarantees ~0 % for any technical course; dominant threat |
| 2 | **ESCO mapping quality** — English JobBERT on bilingual corpus; "X skills" over-merge; τ_esco=0.50 forces 5,097/5,098 to map | Pollutes comparison set + gap names ("issue fines" as #1) |
| 3 | **Embedder inconsistency** — ESCO uses JobBERT, matching uses SBERT mpnet | Two semantic spaces in one pipeline |
| 4 | **Granularity mismatch** — short Bloom-tagged CLOs vs short skill phrases | Legitimate overlaps may fall below τ_sim |
| 5 | **Single-course "programme"** — programme view holds only one course | Programme coverage = course coverage; not a real curriculum test |

### 5. Is the recommendation actionable?

No — not as literally rendered. Suggesting a **theoretical CS programme** add
*communication / teamwork / customer service* courses is face-invalid. This
demonstrates the **boundary of the Skills-Taught-vs-Sought paradigm**: apt for
*vocationally-oriented, market-facing* programmes, misleading for
*foundational/theoretical* ones. The correct reading here is meta-level: the
method ran end-to-end on a real ITC syllabus and correctly surfaced that this
course–corpus pairing is a **degenerate (negative-control) case**.

### 6. What would make this a fair, informative test

1. **Filter the job corpus to IT/SE/data roles** so `S` contains technical
   skills; then an applied course (Software Engineering, Databases) should
   produce REVISE/MAINTAIN — the missing positive control.
2. **Fix the ESCO layer** — keep-raw/drop non-Latin (Khmer) skills; retune
   τ_esco so "issue fines"-type junk disappears.
3. **Unify the embedder** (JobBERT for both mapping and matching).
4. **Use an applied course for the headline demo**; keep Automata Theory as the
   documented negative control.

### 7. Take-home

> Automata Theory → UPDATE / 0 % is **internally correct but externally invalid
> as a recommendation**. It is driven primarily by a **corpus mismatch**
> (general Cambodian job market vs theoretical CS) and secondarily by **ESCO
> mapping artifacts** (Khmer collapse, soft-skill over-merge). The verdict is
> robust to the ESCO noise; the gap labels are not. Frame it in the paper as a
> **negative control / limitation study** motivating (i) a domain-filtered
> corpus, (ii) a Khmer-aware ESCO step, and (iii) embedder unification — with an
> applied course as the positive demonstration.

---

## 0.7. Root-Cause Diagnosis & Solution Options

After the lexical-vs-semantic experiment (see
`lexical_baseline/EXPERIMENT_lexical_vs_semantic.md`), the recurring ~0 % /
UPDATE result was traced to a single root cause. This section records the
diagnosis and the candidate fixes.

### Which is the problem — arithmetic, method, or data?

| Suspect | Guilty? | Evidence |
|---------|---------|----------|
| **Arithmetic / formula** | ❌ No | `coverage = Σf(matched)/Σf(all)` is correct; 0 % and 6.8 % are computed right. |
| **Method** (semantic / ESCO) | ⚠️ Minor | τ_sim = 0.65 too strict + ESCO mis-maps; real but secondary calibration issues. |
| **Data** (job corpus) | ✅ **Root cause** | General Cambodian job boards — communication, teamwork, customer service, sales — contain **almost no technical CS skills**. |

### Proof it is the data, not the method

The three-way run on the same Automata PDF:

```
   LEXICAL          → UPDATE  ~7 %  (1 match)
   SEMANTIC + ESCO  → UPDATE   0 %  (0 matches)
   SEMANTIC (raw)   → UPDATE   0 %  (0 matches)
```

All three methods **agree** on ~0 %. A method bug would make them *disagree*;
agreement means the limitation is **upstream of every matcher** — i.e. in the
shared input (the job-skill data). You cannot match a theory course (DFA,
Turing machines) against a list of *communication / customer service / Excel*,
no matter how good the matcher is.

> Analogy: grading a **swimming** class by counting how often students **ride a
> bicycle**. The counting is correct, the scoring is fine — the **reference set
> is the wrong domain**, so everyone scores 0.

### Solution options (the fix must change the DATA)

**❌ Not a solution — switch algorithm.** All algorithms read the same corpus;
if it has no tech skills, all score ~0 %. Ruled out empirically above.

| # | Option | Effort | Notes |
|---|--------|--------|-------|
| 1 | **Filter existing postings to IT/tech roles** (by title/category) and rebuild the inventory | Low (no re-crawl) | Only viable if enough tech postings already exist — **measure first** |
| 2 | **Crawl IT-specific portals / IT category** (re-run Phase 1–3) | Medium–High | Correct, domain-matched corpus |
| 3 | **Augment with external tech-skill dataset** (ESCO ICT links / intl. IT postings) | Medium | Guarantees tech coverage; weakens the "Cambodian-local" novelty |
| 4 | **Scope the tool to vocational programmes**; keep theory courses as negative controls | None (framing) | A scoping decision, not a data fix |

### Recommended order

1. **Measure first** — count how many of the ~3,216 stage-2 postings are
   IT/tech (by job title/category).
   - Enough (e.g. > 200) → **Option 1** (filter + re-run today).
   - Too few → **Option 2** (must crawl IT jobs).
2. Only *after* the corpus is domain-matched, revisit the **method calibration**
   (lower τ_sim, Khmer-aware ESCO) — these are secondary and currently masked by
   the data problem.

The algorithm is **not broken**: it is faithfully reporting *"this theory course
does not match this general-jobs corpus."* That statement is true; the fix is to
feed it the right job data.

---

## 1. Methodological Framing

Adaptation of **Skills Taught vs Skills Sought** (Ahadi et al., EDM 2022),
replacing closed lexical components with an open semantic pipeline.

```
   📚 INPUT 1: Skills TAUGHT (from school curriculum CLOs)
   💼 INPUT 2: Skills SOUGHT (from aggregated job postings)

   PROCESS:
     1. Embed both sides via JobBERT
     2. Cosine similarity (course CLO ↔ each skill)
     3. Threshold: keep matches above τ_sim = 0.60
     4. Score: coverage = Σ freq(matched) / Σ freq(all)
     5. Classify per course:
          ≥ 0.66 → 🟢 MAINTAIN
          0.33–0.66 → 🟡 REVISE
          < 0.33 → 🔴 UPDATE
     6. Gaps = top-K skills NOT matched by the course

   OUTPUT: per-course recommendation report (action + gap list)
```

---

## 1.A. Core Pipeline — Two Inputs → Algorithm → Recommendations

This section drills into the **central computation**: how two distinct
inputs are combined through the Skills Taught vs Skills Sought algorithm
to produce the final recommendation report.

### 1.A.1. Inputs

```
   📊 INPUT 1 — Aggregated Skill Inventory (Skills SOUGHT)
   ────────────────────────────────────────────────────
   Source     : stage2_result.json (Phase 3 output)
                aggregated across all crawled job postings
   Shape      : { ESCO_label : frequency }
   Produced by: T2 (aggregate_skills.py)

   Example:
     {
       "Python programming":      287,
       "SQL querying":            142,
       "stakeholder communication": 120,
       "machine learning":         89,
       "project management":       78,
       "REST API development":     65,
       "data analysis":            61,
       "cloud computing (AWS)":    54,
       ...
     }
```

```
   📚 INPUT 2 — Parsed Curriculum Document (Skills TAUGHT)
   ──────────────────────────────────────────────────────
   Source     : ITC course-specification PDFs (one per course)
                school_program/itc_cs_2025_2026/*.pdf
   Shape      : [ { course, CLOs, lessons } , ... ]
   Produced by: pdf_parser.py (existing component)

   Example (Automata Theory PDF):
     {
       "course": {
         "code": "GICI3AT",
         "title": "Automata Theory",
         "credits": 2,
         "programme": "Computer Sciences",
         "semester": 2
       },
       "synopsis": "Introduces formal languages and...",
       "learning_outcomes": [
         {"id": "CLO1", "domain": "K",
          "text": "Analyze formal languages and computational
                   models, including finite automata..."},
         {"id": "CLO2", "domain": "S",
          "text": "Design computational models and formal
                   representations..."},
         {"id": "CLO3", "domain": "A",
          "text": "Evaluate problem-solving approaches with
                   respect to logical rigor..."}
       ],
       "lessons": [
         {"week": "1-2", "topic": "Introduction to Automata
                                   Theory, Languages..."},
         {"week": "3-4", "topic": "Deterministic Finite Automata
                                   (DFA)"},
         ...
       ]
     }
```

### 1.A.2. The Skills Taught vs Skills Sought Algorithm

```
   ALGORITHM: CLO-Level Curriculum-Job Alignment
   (adapted from Ahadi et al. 2022 by replacing exact-match
    set operations with JobBERT cosine similarity +
    frequency-weighted coverage scoring)

   ── Setup ──
     S      = { s_1, ..., s_m }   # set of aggregated skills
     C      = { c_1, ..., c_n }   # set of parsed courses
     f      : S → ℕ                # market frequency function
     E      : text → R^768         # JobBERT embedding function

     τ_sim  = 0.60   # match threshold
     τ_high = 0.66   # MAINTAIN cutoff
     τ_low  = 0.33   # UPDATE cutoff
     K_gaps = 5      # gap report depth

   ── Steps (for each course c ∈ C) ──

   STEP 1 — Compose course representation
     T(c) := concat( synopsis(c),
                     ⊕ CLO_text(c),
                     ⊕ lesson_outcomes(c) )

   STEP 2 — Embed course
     e(c) := E( T(c) )                              ∈ R^768

   STEP 3 — Embed skills (cache across courses)
     For each s ∈ S:
       e(s) := E( s )                                ∈ R^768

   STEP 4 — Cosine similarity matrix
     For each s ∈ S:
       sim(c, s) := cos( e(c), e(s) )

   STEP 5 — Threshold match
     matched(c) := { s ∈ S : sim(c, s) ≥ τ_sim }

   STEP 6 — Frequency-weighted coverage
                       Σ_{s ∈ matched(c)} f(s)
     coverage(c) := ─────────────────────────
                          Σ_{s ∈ S} f(s)

   STEP 7 — Action classification
     If   coverage(c) ≥ τ_high : label(c) := MAINTAIN
     Elif coverage(c) ≥ τ_low  : label(c) := REVISE
     Else                       : label(c) := UPDATE

   STEP 8 — Gap inventory
     gaps(c) := top-K_gaps  s ∈ (S \ matched(c))
                sorted by f(s) descending
```

### 1.A.3. Output Shape

```
   📋 OUTPUT — Per-Course Recommendation Report
   ────────────────────────────────────────────
   Shape      : [ { course_code, course_title, coverage,
                    label, matched_skills[], gaps[] }, ... ]
   Saved to   : outputs/phase5/<timestamp>/recommendations.json
   Also       : aggregated CSV for human review

   Example row (predicted for Automata Theory):
     {
       "course_code":  "GICI3AT",
       "course_title": "Automata Theory",
       "coverage":     0.18,
       "label":        "UPDATE",
       "matched_skills": [
         {"skill": "algorithm design", "sim": 0.71,
          "freq": 42},
         {"skill": "mathematical modelling", "sim": 0.66,
          "freq": 28}
       ],
       "gaps": [
         {"skill": "Python programming",         "freq": 287},
         {"skill": "SQL querying",               "freq": 142},
         {"skill": "stakeholder communication",  "freq": 120},
         {"skill": "machine learning",           "freq":  89},
         {"skill": "REST API development",       "freq":  65}
       ]
     }
```

### 1.A.4. Worked Example — Automata Theory

```
   Course:   GICI3AT — Automata Theory (CS, Semester 2)
   CLOs:     CLO1 (K): Analyze formal languages...
             CLO2 (S): Design computational models...
             CLO3 (A): Evaluate problem-solving approaches...

   Apply STEPS 1–8:

   STEP 1: T(c) = synopsis + CLO1 + CLO2 + CLO3 +
                  lesson topics

   STEP 4: Cosine similarities against top market skills:
     Python programming           → 0.31
     SQL querying                 → 0.18
     machine learning             → 0.49
     algorithm design             → 0.71   ◄── matches
     stakeholder communication    → 0.12
     mathematical modelling       → 0.66   ◄── matches
     cloud computing (AWS)        → 0.09
     ...

   STEP 5: matched(GICI3AT) = { algorithm design,
                                 mathematical modelling,
                                 ... (5–10 skills total) }

   STEP 6: coverage(GICI3AT) ≈ 0.18
                              (low — most demanded skills
                               are NOT taught here)

   STEP 7: label = UPDATE   (because 0.18 < 0.33)

   STEP 8: gaps = top-5 high-frequency skills NOT matched
                = [Python, SQL, communication, ML, REST API]

   INTERPRETATION:
   ────────────────
   Automata Theory is highly theoretical. It teaches
   formal-language analysis and algorithmic reasoning, but
   does not cover the applied programming, data, and
   communication skills that dominate the Cambodian job
   market. The "UPDATE" label is a CORRECT, MEANINGFUL
   recommendation — it tells the educator that this course
   is misaligned with current employer demand, and the
   top-5 gaps suggest what could be added.

   ⚠️ Important: "UPDATE" does NOT mean "the course is bad."
       Theoretical foundations have intrinsic value. The
       label only measures market alignment, which is one
       of several legitimate curricular goals.
```

### 1.A.5. Pipeline Function Signatures (target)

```python
# T2 — aggregate_skills.py
def aggregate_skills(
    stage2_paths: list[str],
    top_n: int | None = None,
    use_esco: bool = True,
) -> dict[str, int]:
    """Read stage2_result.json files, optionally ESCO-normalize,
       return {skill_label: frequency}."""


# T4 — matcher.py (upgraded)
def match_course_to_skills(
    course: dict,                     # parsed PDF output
    skills: dict[str, int],           # job_freq from T2
    embedder: JobBertEmbedder,
    tau_sim: float = 0.60,
    tau_high: float = 0.66,
    tau_low: float = 0.33,
    k_gaps: int = 5,
) -> dict:
    """Run the 8-step algorithm. Returns:
       {course_code, course_title, coverage, label,
        matched_skills, gaps}"""


# T4 — run_pipeline.py orchestrator
def run_pipeline(
    pdf_folder: str,
    stage2_path: str,
    output_path: str,
    **threshold_overrides,
) -> list[dict]:
    """Parse all PDFs, aggregate skills, run matching for each
       course, save report JSON + CSV."""
```

### 1.A.6. Edge Cases to Handle

```
   ⚠️ Edge case A: course has NO CLOs (rare but possible)
       → fallback: use synopsis + lesson topics
       → if also missing: skip course, log warning

   ⚠️ Edge case B: course matches ZERO skills above τ_sim
       → coverage = 0.0
       → label = UPDATE (correctly flags the course)

   ⚠️ Edge case C: course matches MANY skills (>50 % of S)
       → coverage = high → MAINTAIN
       → also flag in report as "broad-coverage course"

   ⚠️ Edge case D: empty job_freq (e.g., no Phase 3 output)
       → return error to user; pipeline cannot run

   ⚠️ Edge case E: skill embedding cache stale
       → embed lazily; cache per (skill, model_name) tuple

   ⚠️ Edge case F: PDF parser returns malformed CLOs
       → validate Pydantic schema before scoring
       → if malformed, fall back to raw text matching
```

### 1.A.7. Why This Architecture Works

```
   ✅ Two distinct inputs are KEPT SEPARATE
      → reuse skill aggregation across many curriculum runs
      → reuse course parsing across many job-market refreshes

   ✅ The same algorithm is applied to every course
      → consistent, reproducible recommendations

   ✅ Frequency weighting reflects real labor-market demand
      → a missing high-frequency skill costs more than
        a missing low-frequency one

   ✅ Three-class output is actionable for educators
      → MAINTAIN, REVISE, UPDATE are clear directives

   ✅ Gap list makes the recommendation concrete
      → "this course should add X, Y, Z" — not just a label

   ✅ All thresholds are explicit and tunable
      → enables sensitivity analysis (T7.6) and reviewer
        defense
```

---

## 1.B. Curriculum Document Sub-Pipeline

> **Goal of this section.** Detail the LEFT-side flow:
> raw PDF → parsed text → Course Learning Outcomes → Skills.
> This produces the "Skills TAUGHT" input that feeds the
> Section 1.A algorithm.

### 1.B.1. Four-stage pipeline overview

```
   📄 STAGE 1                   📜 STAGE 2                  🎯 STAGE 3                    🔧 STAGE 4
   ──────────                   ──────────                  ──────────                    ──────────

   Curriculum                   Raw text                    Learning                     Taught
   document         ───────▶    (markdown)        ───────▶  Outcomes        ───────▶     skills
   (PDF / xlsx /                                            (CLOs)                       per CLO
    docx)
                                pymupdf4llm                 RAG retrieval                qwen3:14b LLM
                                                            + qwen3:14b LLM              + ESCO normalize
                                                            + Pydantic schema
```

This sub-pipeline mirrors the job-side pipeline structurally:

```
   ┌───────────────────────────────┬────────────────────────────────┐
   │  JOB SIDE (already built)     │  CURRICULUM SIDE (this design) │
   ├───────────────────────────────┼────────────────────────────────┤
   │  HTML job posting             │  PDF curriculum document       │
   │   │                           │   │                            │
   │   ▼                           │   ▼                            │
   │  Clean text (trafilatura)     │  Markdown (pymupdf4llm)        │
   │   │                           │   │                            │
   │   ▼                           │   ▼                            │
   │  LLM extracts skills          │  LLM extracts CLOs             │
   │  (qwen3:14b)                  │  (qwen3:14b)                   │
   │   │                           │   │                            │
   │   │                           │   ▼                            │
   │   │                           │  LLM extracts skills           │
   │   │                           │  per CLO (qwen3:14b)           │
   │   ▼                           │   ▼                            │
   │  ESCO normalize               │  ESCO normalize                │
   │   │                           │   │                            │
   │   ▼                           │   ▼                            │
   │  Aggregate frequencies        │  Per-course skill set          │
   │                               │                                │
   │  → Skills SOUGHT inventory    │  → Skills TAUGHT per course    │
   └───────────────────────────────┴────────────────────────────────┘

   ⚠️ Symmetry matters: by reducing both sides to ESCO-normalized
       skill sets, the matching step in Section 1.A can be made
       interpretable (set-based) AND fall back to embedding
       similarity for residual variation.
```

### 1.B.2. Stage 1 — Document → Raw text

```
   Input  : PDF file (or xlsx / docx)
   Tool   : pymupdf4llm
   Output : Markdown-formatted plain text (preserves layout
            cues like tables, lists, headings)

   Existing code: pdf_parser.py::extract_markdown(path)

   Behavior:
   ─────────
   • Loads PDF via PyMuPDF
   • Converts to LLM-friendly markdown (headings, tables,
     paragraphs)
   • Removes images, footers, page numbers
   • Returns a single string ready for chunking

   Quality notes:
   ──────────────
   • Tables are kept as markdown-table cells (important for
     fields 15, 16, 17, 18 of the ITC template)
   • Long horizontal forms (24-field ITC layout) survive
     reasonably well in markdown form
   • Khmer characters are preserved (UTF-8) — useful later
     when multilingual support is added
```

### 1.B.3. Stage 2 — Raw text → Structured CLOs

```
   Input  : Markdown text from Stage 1
   Tool   : RAG retrieval + qwen3:14b + Pydantic schema
   Output : JSON object with course metadata + CLO list +
            lesson plan

   Existing code: pdf_parser.py::parse_curriculum(text)

   Behavior:
   ─────────
   1. CHUNK    the markdown via RecursiveCharacterTextSplitter
               (1200 chars, 150 overlap)
   2. EMBED    each chunk with all-MiniLM-L6-v2
   3. RETRIEVE the top-K chunks for each target query:
                 • "Course title and code"
                 • "Course Learning Outcomes (CLOs)"
                 • "Weekly lesson plan"
                 • "Credits and programme"
   4. PROMPT   qwen3:14b with retrieved chunks +
               Pydantic schema definition
   5. PARSE    LLM JSON response into the structured object

   Output shape (already used by matcher.py):
   ─────────────────────────────────────────
     {
       "course": {
         "code": "GICI3AT",
         "title": "Automata Theory",
         "credits": 2,
         "programme": "Computer Sciences",
         "semester": 2
       },
       "synopsis": "...",
       "learning_outcomes": [
         {"id": "CLO1", "domain": "K",
          "text": "Analyze formal languages..."},
         {"id": "CLO2", "domain": "S",
          "text": "Design computational models..."},
         {"id": "CLO3", "domain": "A",
          "text": "Evaluate problem-solving approaches..."}
       ],
       "lessons": [
         {"week": "1-2", "topic": "Introduction to Automata
                                   Theory, Languages..."},
         ...
       ]
     }
```

### 1.B.4. Stage 3 — CLOs → Skills

> **This is a NEW stage** that does not yet exist in the
> codebase. It mirrors the job-side LLM extraction in
> `stage2_skill_extraction.py`.

```
   Input  : Structured course object from Stage 2
   Tool   : qwen3:14b with a CLO-specific extraction prompt
   Output : Per-CLO skill list + course-level skill union

   Target file: phase5_curricula/clo_skill_extractor.py

   Why this stage exists:
   ──────────────────────
   A CLO is written in pedagogical language ("Analyze formal
   languages...") that does not directly correspond to a
   labour-market skill phrase ("formal language theory",
   "automata analysis").

   To match CLOs against the Skills SOUGHT inventory in
   Section 1.A, we need EXPLICIT skill phrases drawn from
   each CLO. This stage performs that translation.

   Stage 3 prompt outline (qwen3:14b, temperature 0):
   ──────────────────────────────────────────────────
   System: "You are a curriculum skill extractor."

   User instruction:
     Given the Course Learning Outcome below, list the
     specific labour-market-relevant skills the learner
     would acquire after mastering it.

     Rules:
       • Return 3 to 8 skill phrases per CLO.
       • Each phrase: 2–5 words, action-verb style or
         noun-phrase style.
       • Preserve named technologies verbatim (e.g.,
         "Python programming", not "programming language").
       • Distinguish technical (hard) and interpersonal
         (soft) skills.
       • No degrees, certifications, years of experience,
         or job titles.

     CLO: "{clo.text}"

     Return ONLY a JSON array of strings.

   Output shape:
   ─────────────
     {
       "course_code": "GICI3AT",
       "course_title": "Automata Theory",
       "clo_skills": {
         "CLO1": ["formal language analysis",
                  "finite automata reasoning",
                  "computational model analysis",
                  "Turing machine theory"],
         "CLO2": ["automata design",
                  "context-free grammar construction",
                  "regular expression design"],
         "CLO3": ["logical reasoning",
                  "mathematical proof",
                  "peer review"]
       },
       "skills_union": [
         "formal language analysis",
         "finite automata reasoning",
         "computational model analysis",
         "Turing machine theory",
         "automata design",
         "context-free grammar construction",
         "regular expression design",
         "logical reasoning",
         "mathematical proof",
         "peer review"
       ]
     }

   Bloom-domain awareness:
   ───────────────────────
   • Each CLO carries a (K) / (S) / (A) tag
       K = Knowledge   (analytical / conceptual)
       S = Skill       (design / construct)
       A = Attitude    (evaluate / collaborate)
   • The prompt can be tuned per domain (e.g., S-domain
     CLOs tend to produce more applied tech skills; A-domain
     CLOs tend to produce soft skills)
   • Optional: weight CLOs by their PLO contribution
     percentages (field 15 of the ITC template)
```

### 1.B.5. Stage 4 — ESCO normalization (shared with job side)

```
   Input  : skills_union list from Stage 3
   Tool   : JobBERT cosine to ESCO labels
            (port from phase3/evaluation/esco_evaluation.py)
   Output : ESCO-normalized skill set per course

   Target file: phase5_curricula/esco.py
                (shared by job-side aggregation + curriculum side)

   Why share:
   ──────────
   Both sides need the SAME normalization so that overlap
   measurement (Section 1.A, Step 5) is on a common label
   space. Without this, "Python programming" (from CLO) and
   "Python development" (from postings) would miss-match
   even though they refer to the same ESCO skill.

   Output (final shape that feeds the matcher):
   ────────────────────────────────────────────
     {
       "course_code": "GICI3AT",
       "skills_taught": [
         "Python programming",          ← ESCO canonical
         "formal language theory",      ← ESCO canonical
         ...
       ],
       "skills_taught_freq": null         (no frequency on
                                           curriculum side —
                                           skills_taught is
                                           a SET)
     }
```

### 1.B.6. End-to-end shape transformation

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │  Stage  │ Input                  │ Output                   │
   │ ────────┼────────────────────────┼─────────────────────── │
   │  1      │ PDF file               │ Markdown text            │
   │  2      │ Markdown text          │ Course object with CLOs  │
   │  3      │ Course object          │ CLO → skill phrases      │
   │  4      │ Skill phrases          │ ESCO-normalized labels   │
   │                                                            │
   │  ⮕ Final output: per-course ESCO-normalized skill set     │
   │                                                            │
   └────────────────────────────────────────────────────────────┘

   This per-course skill set is then passed to Section 1.A,
   Step 4 (cosine similarity matrix) along with the
   aggregated Skills SOUGHT inventory.
```

### 1.B.7. New task entries (added to the breakdown in Section 3)

```
   T2.6 — CLO → skills extractor
     [ ] T2.6.1 — Write phase5_curricula/clo_skill_extractor.py
                  with the Stage 3 prompt
     [ ] T2.6.2 — Add unit tests on Automata Theory CLOs
     [ ] T2.6.3 — Validate Pydantic schema for output
     [ ] T2.6.4 — Wire into run_pipeline.py orchestrator

   T3.3 (revised) — Apply ESCO normalization in:
     [ ] (a) Skills SOUGHT (job-side aggregation)
     [ ] (b) Skills TAUGHT (per-course union from Stage 3)
```

### 1.B.8. Worked example — Automata Theory through all four stages

```
   STAGE 1 (PDF → markdown):
   ────────────────────────
   Input:  automa_theory_course.pdf (11 pages)
   Output: ~9000-character markdown string

   STAGE 2 (markdown → structured CLOs):
   ────────────────────────────────────
   Output:
     course: { code: "GICI3AT", title: "Automata Theory",
               credits: 2, programme: "CS", semester: 2 }
     CLOs:
       CLO1 (K): "Analyze formal languages and computational
                  models, including finite automata, pushdown
                  automata, and Turing machines..."
       CLO2 (S): "Design computational models and formal
                  representations to solve language recognition
                  and computation problems..."
       CLO3 (A): "Evaluate problem-solving approaches in
                  automata theory with respect to logical
                  rigor, correctness, and professional
                  responsibility..."

   STAGE 3 (CLOs → skill phrases):
   ───────────────────────────────
   clo_skills:
     CLO1: ["formal language analysis",
            "finite automata reasoning",
            "Turing machine theory",
            "computational complexity analysis"]
     CLO2: ["automata design",
            "context-free grammar construction",
            "regular expression design",
            "DFA / NFA construction"]
     CLO3: ["logical reasoning",
            "mathematical proof",
            "peer review",
            "scientific writing"]

   skills_union: 12 unique skill phrases

   STAGE 4 (ESCO normalize):
   ────────────────────────
   skills_taught (ESCO-normalized): 12 canonical labels

   → This skill set is now ready for Section 1.A matching
     against the Skills SOUGHT inventory.
```

### 1.B.9. Failure modes to handle in Stage 3

```
   ⚠️ Failure A: CLO is too short / vague
       e.g., "Apply concepts correctly."
       → Stage 3 may return 1–2 generic skills or empty list
       → mitigation: require ≥ 3 skills via prompt; if LLM
         returns empty, fall back to the CLO text itself

   ⚠️ Failure B: CLO contains pedagogical jargon
       e.g., "Demonstrate Bloom's-Taxonomy C4 mastery..."
       → Stage 3 LLM may not know how to translate
       → mitigation: add few-shot examples for pedagogical
         CLOs in the prompt

   ⚠️ Failure C: Domain (K / S / A) tag missing
       → Treat as default "S" (skill domain)
       → log warning for later QA

   ⚠️ Failure D: LLM hallucinates skills not implied by CLO
       e.g., adds "machine learning" to an Automata CLO
       → mitigation: post-hoc verification — embed each
         output skill, check similarity to CLO ≥ 0.30,
         drop low-similarity skills
```

### 1.B.10. Why this design improves over raw-text matching

```
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  Original Section 1.A approach (raw-text matching):          │
   │     CLO text  ── JobBERT embed ──▶  vs skill embedding       │
   │                                                              │
   │  Improved approach with this Stage 3:                        │
   │     CLO text  ── LLM extract ──▶  skill phrases              │
   │                                ── ESCO normalize ──▶ labels  │
   │                                ── set / similarity match ──▶ │
   │                                                              │
   │  ADVANTAGES:                                                 │
   │                                                              │
   │  ✅ Symmetric pipeline — both sides reduced to skill sets    │
   │     before matching (cleaner Ahadi-style framing)            │
   │                                                              │
   │  ✅ Interpretable — every match can be traced to a specific  │
   │     extracted skill from a specific CLO                      │
   │                                                              │
   │  ✅ ESCO-aligned — both sides live in the same label space,  │
   │     reducing surface-form noise                              │
   │                                                              │
   │  ✅ Bloom-aware — K / S / A tags can re-weight matching      │
   │     (skill-domain CLOs likely produce more market-relevant   │
   │     skills than attitude-domain CLOs)                        │
   │                                                              │
   │  ✅ Easier expert validation — educators can review the      │
   │     EXTRACTED SKILLS per CLO, not opaque embeddings          │
   │                                                              │
   │  TRADE-OFF:                                                  │
   │  ⚠️ Adds 1 LLM call per CLO (~3 per course) → for a 20-     │
   │     course curriculum that's 60 extra LLM calls per run,    │
   │     ~5–10 min on local qwen3:14b. Acceptable.                │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
```

---

## 2. Current State

### ✅ Built and working
- `pdf_parser.py` — PDF → markdown via `pymupdf4llm` → RAG retrieval →
  qwen3:14b with Pydantic schema → returns `{course, lessons,
  learning_outcomes}`
- `matcher.py` — Atlas-style SBERT cosine matching; computes
  `coverage_pct` and `demand_weighted_score`; accepts optional
  `job_freq` for frequency weighting
- `embedder.py` — Singleton SBERT wrapper; lazy-loaded model cache
- Frontend `CurriculaTab.js` — 3-step wizard with MOCK_COURSES,
  MOCK_SKILLS, MOCK_RECOMMENDATIONS
- Frontend `IdentifyGapsTab.js` — Full Ahadi et al. 2022 7-step
  methodology with sample subjects + jobs (no backend calls)

### ❌ Still missing
- Real ITC 2025–2026 curriculum data (only `automa_theory_course.pdf`
  exists as test)
- Top-N skill aggregation script (`stage2_result.json` →
  `{skill: frequency}`)
- FastAPI endpoint mount under `/api/curricula/*`
- ESCO normalization in Phase 5 path (only in Phase 4 currently)
- Frontend Step 1 wiring to real `/api/curricula/upload` endpoint
- Threshold-based maintain / revise / update classification in
  `matcher.py`
- JobBERT swap in `matcher.py` (currently uses SBERT
  all-MiniLM-L6-v2)
- Expert validation protocol for evaluating recommendation quality

---

## 3. Task Breakdown

### T1. Data preparation
- [ ] T1.1 — Obtain ITC 2025–2026 curriculum PDFs (Computer Sciences
  programme as starting target, ~12–20 courses)
- [ ] T1.2 — Verify each PDF follows the standard 24-field ITC
  course-specification template
- [ ] T1.3 — Store PDFs in `school_program/itc_cs_2025_2026/`

### T2. Skill aggregation script
- [ ] T2.1 — Write `aggregate_skills.py` to read all
  `outputs/stage2/*/stage2_result.json` files
- [ ] T2.2 — Flatten extracted skills across all postings
- [ ] T2.3 — Normalize via ESCO (port from Phase 4
  `esco_evaluation.py`)
- [ ] T2.4 — Output `job_freq.json` as `{ESCO_label: frequency}`
- [ ] T2.5 — Add CLI flag for top-N filtering

### T3. ESCO integration in Phase 5
- [ ] T3.1 — Port ESCO loader + JobBERT mapper from
  `phase3_skill_extraction/evaluation/esco_evaluation.py`
- [ ] T3.2 — Add `esco_normalize(skill)` helper in
  `phase5_curricula/esco.py`
- [ ] T3.3 — Apply normalization in both:
  - The aggregated skill inventory (Stream 1)
  - The matcher output (downstream interpretability)

### T4. Matcher upgrades
- [ ] T4.1 — Switch `matcher.py` backbone from
  `all-MiniLM-L6-v2` → JobBERT
  (`jjzha/jobbert-base-cased`) for methodological consistency with
  Phase 4 evaluation
- [ ] T4.2 — Implement coverage formula:
  `coverage(c) = Σ_{s ∈ matched(c)} f(s) / Σ_{s ∈ S} f(s)`
- [ ] T4.3 — Implement 3-class threshold classification:
  - `coverage(c) ≥ 0.66` → MAINTAIN
  - `0.33 ≤ coverage(c) < 0.66` → REVISE
  - `coverage(c) < 0.33` → UPDATE
- [ ] T4.4 — Implement gap extraction:
  `gaps(c) = top-K skills ∈ (S \ matched(c)) sorted by f(s) desc`
- [ ] T4.5 — Add `run_pipeline.py` orchestrator that consumes
  `job_freq.json` + a folder of course PDFs and emits a
  per-course recommendation JSON

### T5. FastAPI endpoint mount
- [ ] T5.1 — `POST /api/curricula/upload` — multipart PDF →
  `pdf_parser` → courses + CLOs JSON
- [ ] T5.2 — `GET /api/curricula/skills?top_n=50` — return
  top-N aggregated skills + frequency (calls `aggregate_skills`)
- [ ] T5.3 — `POST /api/curricula/analyze` — courses + skills →
  matcher → coverage report
- [ ] T5.4 — Register routes in
  `backend/phase1_job_site_discovery/api.py` (or extract to its own
  `phase5_api.py` module)

### T6. Frontend wiring
- [ ] T6.1 — Replace `MOCK_COURSES` / `MOCK_SKILLS` /
  `MOCK_RECOMMENDATIONS` in `CurriculaTab.js`
- [ ] T6.2 — Wire Step 1 (upload) to
  `/api/curricula/upload`
- [ ] T6.3 — Wire Step 3 (recommendation table) to
  `/api/curricula/analyze`
- [ ] T6.4 — Add loading + error states for each step
- [ ] T6.5 — Add CSV/JSON export button for the recommendation
  report

### T7. Evaluation
- [ ] T7.1 — Run the full pipeline on ITC CS 2025–2026 (T1.1
  PDFs)
- [ ] T7.2 — Generate per-course coverage + label JSON +
  CSV
- [ ] T7.3 — Expert validation: ask 2–3 ITC educators to
  independently rate each course's recommendation (Likert
  scale) and identify disagreements
- [ ] T7.4 — Compute label agreement (Cohen's κ or majority
  vote)
- [ ] T7.5 — Identify systematic failure cases (e.g.,
  theoretical CS courses systematically labeled UPDATE)
- [ ] T7.6 — Sensitivity analysis: vary τ_sim, τ_high, τ_low
  and measure label stability

### T8. Documentation
- [ ] T8.1 — Update `PIPELINE_DESIGN.md` with final API shapes
  + algorithm details
- [ ] T8.2 — Write `Phase 5 README` with run instructions
  end-to-end
- [ ] T8.3 — Update root `CLAUDE.md` Phase 5 progress table

### T9. Journal extension write-up
- [ ] T9.1 — Section "Full Institutional Evaluation": ITC CS
  case study results (coverage histogram, label distribution,
  top-K gaps across programme)
- [ ] T9.2 — Section "Expert Validation": Cohen's κ + qualitative
  agreement analysis
- [ ] T9.3 — Section "Sensitivity Analysis": threshold robustness
- [ ] T9.4 — Update Section 2.4 with explicit formulas (move
  from prose to math)
- [ ] T9.5 — Add curriculum recommendation result tables (per-course
  + per-programme aggregates)

---

## 4. Dependencies and Critical Path

```
   T1 (data) ────┐
                 ├──▶ T7 (eval) ──▶ T9 (journal write-up)
   T2 (skill agg) ┤
                 │
   T3 (ESCO Phase 5) ──┐
                       ├──▶ T4 (matcher) ──▶ T5 (API)
   T6 (frontend) ──────┘                         │
                                                 │
                                                 ▼
                                         T8 (docs)
```

**Critical path:** T1 → T2 → T4 → T7 → T9.
T1 is the gating step — without real ITC PDFs no evaluation can run.

---

## 5. Suggested Timeline (post-May-29)

| Week | Tasks |
|------|-------|
| 1 (Jun 02–06) | T1 (gather ITC PDFs), T2 (skill aggregation script) |
| 2 (Jun 09–13) | T3 (ESCO in Phase 5), T4 (matcher upgrades) |
| 3 (Jun 16–20) | T5 (FastAPI endpoints), T6 (frontend wiring) |
| 4 (Jun 23–27) | T7.1–T7.2 (run pipeline + per-course report) |
| 5 (Jun 30–Jul 04) | T7.3–T7.5 (expert validation, agreement analysis) |
| 6 (Jul 07–11) | T7.6 (sensitivity), T8 (documentation) |
| 7–8 (Jul 14–25) | T9 (journal extension write-up + figures) |

---

## 6. Open Decisions

- **Embedding model:** JobBERT vs `all-mpnet-base-v2` for matcher
  (T4.1). JobBERT is methodologically consistent with Phase 4; mpnet
  may give better general-purpose semantics. Decide after a small
  pilot comparison.
- **Threshold values:** τ_sim = 0.60, τ_high = 0.66, τ_low = 0.33
  are initial guesses from Phase 4 tuning. Confirm or adjust during
  T7.6 sensitivity analysis.
- **Granularity of matching:** match per-CLO (current design) vs
  per-lesson-outcome (finer, from Field 18 of the course spec).
  Pilot both on Automata Theory and pick the cleaner one.
- **Expert pool:** 2 educators minimum, 3+ preferred for κ. Need to
  identify volunteers from ITC Computer Sciences department.
- **Programme scope:** start with Computer Sciences only, or also
  include Industrial / Civil / Electrical Engineering programmes?
  Recommendation: start with CS, expand if time permits.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| ITC PDFs unavailable in time | Medium | Reach out to academic affairs office in Week 1; use Automata Theory + a few publicly-available syllabi as a fallback |
| pdf_parser fails on real PDFs | Medium | Run smoke tests on 2–3 PDFs in Week 1, fix template-edge cases early |
| Theoretical courses all score UPDATE | High | Acknowledge in paper as a meaningful finding (theoretical CS courses are not aligned with applied job market) |
| Expert validation reveals systematic errors | Medium | Treat as a finding, not a failure — iterate on thresholds and present an "after tuning" version |
| Threshold values produce uninformative distribution (e.g., 90 % UPDATE) | Medium | Run T7.6 sensitivity analysis; tune thresholds to produce a balanced 3-class output |

---

## 8. References (for the journal extension)

- [Ahadi et al. 2022 (EDM)](https://educationaldatamining.org/edm2022/proceedings/2022.EDM-posters.56/) — Skills Taught vs Skills Sought (baseline framework)
- [Baral & Ghimire 2026 (PECJ)](https://doi.org/10.3126/pecj.v3i1.93534) — Nepal IT curriculum gap (closest twin paper)
- [Almaleh et al. 2019 (Sustainability)](https://doi.org/10.3390/su11092607) — Align My Curriculum
- [Frej et al. 2024 (SIGIR)](https://arxiv.org/abs/2404.10876) — Course Recommender Systems Need to Consider the Job Market
- [Sabet et al. 2024 (arXiv:2404.13163)](https://arxiv.org/abs/2404.13163) — Course-Skill Atlas (SBERT matching at scale)
- [UniSkill 2026 (arXiv:2603.03134)](https://arxiv.org/abs/2603.03134) — University Curricula to Professional Competencies (BERT + ESCO)
