# Curriculum Recommendation — Proposed Method v2 (Tier-Gated, Level-Aware Scoring)

> Supersedes/extends `propose_pipeline.md`. Designed 2026-06-12 from the diagnosis of the
> Intro-to-AI false verdict (38.8% / UPDATE despite Relevance 100%). All numbers in this
> document are **measured on real repo data** (real ITC syllabus × real LinkedIn AI field
> inventory × Qwen3-Embedding), not estimates.

---

## 1. Goal

Given **Input 1** (job postings → market demand) and **Input 2** (a syllabus PDF), decide
whether the course/curriculum is aligned with labour-market demand —
**MAINTAIN / REVISE / FOCUSED / UPDATE / FOUNDATIONAL** — and when gaps exist, recommend
*which* skills to add, with evidence.

Design principle: **every missing percentage point must be nameable.** The score is only
trustworthy if each uncovered point corresponds to a real, explainable curriculum decision
(e.g. "deep learning is missing"), never to denominator noise (e.g. "kubernetes is missing
from an intro course").

---

## 2. Requirements (agreed spec)

- Input 1: LinkedIn 1.35M postings (+ Cambodian crawl), field-scoped by job title.
- Input 2: one ITC course-specification PDF per course (template-agnostic parsing).
- **Course gate (add-on requirement):** classify every uploaded course into
  `FOUNDATIONAL / APPLIED / PROFESSIONAL`. FOUNDATIONAL courses are **not analyzed**
  (no skill words for them exist in job postings) — output "not market-facing by design".
- Two scores per analyzed course: Completeness (job→course) and Relevance (course→job).
- Per-course label + covered skills with evidence + suggested additions + next-course topics.
- All LLM calls: local qwen3:14b (Ollama), schema-locked JSON, `num_ctx=16384`.
- Everything cacheable and auditable (JSON caches, human-editable).

---

## 3. Pipeline overview

```
INPUT 2: Syllabus PDF (1 course)                INPUT 1: Job postings (LinkedIn 1.35M + KH crawl)
        │                                                       │
        ▼                                                       │
┌─────────────────────────┐                                     │
│ PARSE (qwen3)           │                                     │
│ course, description,    │                                     │
│ CLOs, lessons,          │                                     │
│ skills + evidence       │                                     │
│ + tools from §Resources │ ← narrow tool inference (langs/libs)│
└───────────┬─────────────┘                                     │
            ▼                                                   │
┌───────────────────────────────────┐                           │
│ STEP 4 · COURSE GATE (1 qwen3 call)│                          │
│ course_type + field + role_patterns│                          │
└──────┬──────────┬──────────┬──────┘                           │
       │          │          │       role_patterns filter job titles
  FOUNDATIONAL  APPLIED  PROFESSIONAL                           │
       │          │          │                                  ▼
       ▼          │          │            ┌──────────────────────────────────┐
┌──────────────┐  │          │            │ FIELD INVENTORY (cached per field)│
│ STOP.        │  │          │            │ skill → demand                    │
│ "Not market- │  │          │            │ Imp = demand × IDF                │
│ facing by    │  │          │            │ − soft skills − off-domain        │
│ design"      │  │          │            │ − credentials ("computer science")│
│ (no score)   │  │          │            └────────────┬─────────────────────┘
└──────────────┘  │          │                         ▼
                  │          │            ┌──────────────────────────────────┐
                  │          │            │ TOP-25 CORE, tier-tagged (cached) │
                  │          │            │  applied: ml, python, statistics… │
                  │          │            │  professional: pytorch, aws, k8s… │
                  │          │            └────────────┬─────────────────────┘
                  ▼          ▼                         │
            ┌─────────────────────────┐                │
            │ DENOMINATOR SELECT      │◄───────────────┘
            │ APPLIED → applied tier  │   professional tier ──→ "NEXT-COURSE
            │ PROFESSIONAL → full core│                          TOPICS" list
            └───────────┬─────────────┘                          (not gaps!)
                        ▼
            ┌──────────────────────────────────────────┐
            │ STEP 5 · MATCH (Qwen3-Embedding cosine)   │
            │ + BAND-BASED RETRIEVE-THEN-VERIFY         │
            └───────────┬──────────────────────────────┘
                        ▼
            ┌──────────────────────────────────────────┐
            │ STEP 6 · TWO SCORES                       │
            │ Completeness = Σ Imp(covered core@tier)   │
            │                / Σ Imp(core@tier)         │
            │ Relevance    = Σ Imp(taught ∩ demand)     │
            │                / Σ Imp(taught)            │
            └───────────┬──────────────────────────────┘
                        ▼
            ┌──────────────────────────────────────────┐
            │ STEP 7 · TWO-AXIS LABEL                   │
            │ MAINTAIN / REVISE / FOCUSED / UPDATE      │
            └───────────┬──────────────────────────────┘
                        ▼
            ┌──────────────────────────────────────────┐
            │ STEP 8 · OUTPUT (explainable)             │
            │ ✓ covered core (+ taught-skill evidence)  │
            │ ✗ add these: LLM-curated, demand-ranked   │
            │ → next-course topics (above tier)         │
            │ ⚑ description↔content mismatch flag       │
            └──────────────────────────────────────────┘
```

---

## 4. Step 4 — Course classification gate (★ new)

### Classes (judged by CONTENT, never by title)

| Class | Definition | Test | Examples |
|---|---|---|---|
| **FOUNDATIONAL** | Theory/formalisms; concepts never appear as skill words in postings | "Would any posting list these skills verbatim?" → No | Automata, Calculus, Discrete Math |
| **APPLIED** | Transferable technical concepts/skills, hands-on, intro–intermediate | Overlaps demand at concept level (ml, sql) | Intro to AI, Databases, Programming |
| **PROFESSIONAL** | Industry practice: tooling, infra, workflows; terminal/specialization | Overlap includes tooling tier (aws, docker) | Capstone, DevOps, MLOps, Cloud |

### Method choice (research mode)

| Method | Verdict |
|---|---|
| Rule-based (Bloom K/S/A + title keywords) | Fails on own data: Automata CLO2 is "(S) Design…" — a Skill verb on pure theory. Titles lie. Keep as sanity-check feature only. |
| **Zero-shot LLM on course content** ← chosen | Extends the existing `classify_field` call (same call already returns field + role_patterns). Schema-locked, local, free, explainable via `reason`. |
| Embedding centroids | Needs labeled reference syllabi (we have ~3). Future work once override logs accumulate. |

**Final design = LLM-primary with guardrails:** `{course_type, confidence, reason}`;
low confidence or Bloom-rule disagreement → human-review flag; UI shows the result as an
**editable chip** (same pattern as the existing "detected: Artificial Intelligence" chip) —
a misclassification never dead-ends; overrides become future training labels.

Edge rules: mixed course (theory taught through tooling) → APPLIED; thin syllabus →
default APPLIED + flag (never default FOUNDATIONAL — silently gating out is the worse error).

**Evidence the gate is required:** Automata Theory scores 0% across *every* configuration
(2 corpora × 2 methods × all thresholds × symmetric/asymmetric ESCO). No related work
(Baral 2026, Almaleh 2019, Gromov 2020, Ahadi 2022, eDoer, Frej 2024, MDPI 2024) has this
gate — all score every course → misleading verdicts on theory courses. **Novelty claim.**

---

## 5. Tier tagging + denominator select (★ new)

The **market core** (top-25 by importance) is divided once per field; courses are never divided —
each course gets ONE type, the market skills get tiers:

```
PDF side:  whole COURSE → one label (gate)      Job side:  top-25 core → 2 tiers (LLM, cached)
                  └──────────────┬──────────────────────────────┘
                                 ▼
                 course type SELECTS which tier slice is the denominator
```

- One batched qwen3 call per field: *"lowest course type expected to teach this skill?"*
  → `{field}_field_levels.json` (e.g. `machine learning: applied`, `pytorch: professional`).
- Built **once per field**, reused by every course; plain JSON → **human-correctable**
  (print in thesis appendix; experts can review line by line).
- Selection at runtime is a one-line filter (no LLM):

```python
TIER_RANK = {"applied": 1, "professional": 2}
course_rank = 1 if course_type == "APPLIED" else 2
denominator        = [s for s in core if TIER_RANK[levels[s.skill]] <= course_rank]
next_course_topics = [s for s in core if TIER_RANK[levels[s.skill]] >  course_rank]
```

| Course type | Denominator | Professional-tier skills |
|---|---|---|
| APPLIED | applied tier only (~12 skills) | → "next-course topics" (never gaps) |
| PROFESSIONAL | full core (25) | counted normally |

---

## 6. Step 5 — Matching: band-based retrieve-then-verify (★ new band)

```
            cosine(market skill ↔ best taught skill)
  0 ───────────────── 0.55 ─────────────────── 0.70 ─────────────── 1.0
  │   NOT COVERED      │      GRAY ZONE          │   COVERED…         │
  │   (auto-reject)    │  one batched qwen3 call │   …unless verifier │
  │                    │  adjudicates each pair  │   VETOES a false   │
  │                    │                         │   positive         │
```

- Embedder = Qwen3-Embedding-0.6B (best jargon recall; compressed range ⇒ τ=0.70).
- ≥ 0.70: auto-covered, but the verifier can **veto** ("Supervised Learning ≠ deep learning").
- 0.55–0.70: LLM **rescues** true coverage ("Linear Regression + Cross-Validation IS applied
  statistics" at 0.68). Only ~9 pairs/course → one cheap batched call.
- < 0.55: rejected.
- **Why a band instead of a single τ:** the threshold is model- and vocab-specific
  (mpnet≈0.55 ≈ Qwen3≈0.70); a hard τ is brittle. The band lets the hard threshold decide
  only the clear cases, the LLM decides the gray zone. UniSkill 2026's macro-F1 0.87
  ceiling for pure matchers justifies verification.
- Plus **narrow tool inference** on the curriculum side: programming languages/libraries
  named in §Resources/Tools count as taught (fixes the `python` false gap) — languages/libs
  ONLY, no tutorial-URL noise.

---

## 7. Step 6 — Scoring formulas

```
Importance(s) = demand(s) × IDF(s)              (cross-category IDF; RCA-style weighting)

Completeness  = Σ Imp(covered core skills, tier ≤ course tier)
                ───────────────────────────────────────────────
                Σ Imp(core skills, tier ≤ course tier)

Relevance     = Σ Imp(taught skills in demand) / Σ Imp(all taught skills)
```

Denominator hygiene (extends existing soft-skill/off-domain exclusion): also exclude
**credentials/degree labels** (`computer science`, `bachelor's degree`) — they are
fields of study, not teachable skills.

### Measured ladder — Intro to AI (real data, each lever isolated)

| Lever | Fixes | Score | Label |
|---|---|---|---|
| L0 whole-IT denominator (V1) | — | ~5% | UPDATE (absurd) |
| L1 field-scoped top-25 + Imp=demand×IDF | scope + weighting | 32.9% (5/25) | UPDATE (misleading) |
| L2 **+ tier filter** | wrong-level skills | 43.9% (4/13) | REVISE |
| L3 + verifier veto (DL, TF false positives) | honesty (score *drops*) | 31.5% (3/13) | — |
| L4 + python tool inference | curriculum-side false gap | 46.5% | REVISE |
| L5 + gray-zone rescue (statistics .68, AI .67) | threshold brittleness | 56.2% (6/13) | REVISE |
| L6 + credential exclusion (computer science) | demand-table noise | **58.3% (6/12)** | **REVISE** |

Relevance = **100%** at every rung. L3's drop is a feature: it proves the verifier is not a
score-inflation device — each rung is one flag in the thesis ablation table.

---

## 8. Step 7 — Two-axis decision

```
                        Relevance (is what it teaches in demand?)
                        low ◄──────────── 0.70 ────────────► high
                   ┌─────────────────────────┬─────────────────────────┐
        ≥0.70      │  (rare)                 │  MAINTAIN 🟢            │
   Completeness    ├─────────────────────────┼─────────────────────────┤
      0.40–0.70    │  REVISE 🟡              │  REVISE 🟡  ★Intro-AI   │
                   ├─────────────────────────┼─────────────────────────┤
        <0.40      │  UPDATE 🔴 teaches what │  FOCUSED 🔵 narrow but  │
                   │  market doesn't want    │  in demand → extend,    │
                   │  (the real red flag)    │  don't rewrite          │
                   └─────────────────────────┴─────────────────────────┘
                  FOUNDATIONAL courses never enter this matrix (gated at Step 4)
```

Defensible improvement over Baral 2026's single Alignment Index: one number cannot
separate "narrow" from "bad"; two axes + the gate can.

**Program-level view:** per-course labels as above; program completeness = union of all
courses' covered skills over the FULL field core (no tier filter — the *program* should
reach the professional tier by graduation). All-FOCUSED courses with a poor union ⇒ the
program needs new courses even though no single course is bad.

---

## 9. Step 8 — Suggested additions

Uncovered core skills **within the course's tier**, ranked by Importance, then one qwen3
curation call with three constraints (existing `curate_suggestions`, now tier-aware):

1. **Field fit** — drop credentials, soft skills, off-domain.
2. **Tier fit** — intro extension (deep learning, NLP basics), not kubernetes.
3. **Builds on what's taught** — dedup variants (`nlp`/`natural language processing`) and
   concept-vs-tool stacking (suggest "deep learning", not also tensorflow AND pytorch).

Grounding guarantee: the LLM can only keep skills that exist **verbatim** in the demand
table — it selects and orders, never invents. Every suggestion carries its posting count.

---

## 10. Worked trace — Intro to AI (AIECS031), real numbers

```
PARSE  → 13 skills (EDA, K-Means, GMM, Supervised Learning, Linear/Logistic Regression,
         Decision Trees, Cross-Validation, Classification, KNN, Gradient Descent,
         Gradient Boosting, Model Evaluation) + tools: python (§VIII)
GATE   → APPLIED · field=ai                → denominator = 12 applied-tier core (Σ Imp 17,856)

MATCH  market skill        ← best taught          cos    verdict
       machine learning    ← Supervised Learning  0.82   ✓ auto
       data science        ← EDA                  0.74   ✓ auto
       data visualization  ← EDA                  0.73   ✓ auto
       deep learning       ← Supervised Learning  0.73   ✗ VETOED (not actually taught)
       statistics          ← Linear Regression    0.68   ✓ RESCUED (CV/eval IS applied stats)
       artificial intell.  ← Classification       0.67   ✓ RESCUED
       python              ← §VIII Tools           —     ✓ tool inference
       sql .51 / r .63 / NLP .63 / java .55 / CV .67    ✗ gray zone, LLM says no

SCORE  Completeness = (2941+2787+2101+1061+782+731) / 17,856 = 10,403/17,856 = 58.3% (6/12)
       Relevance    = 100%
LABEL  → REVISE 🟡
OUTPUT ✗ add: deep learning (721 postings) ⚑ description PROMISES neural networks/deep
              learning — lessons never deliver it (system rediscovers the human-visible gap)
              · natural language processing · computer vision
       → next-course topics: pytorch, tensorflow, aws, spark, docker, kubernetes, …
```

**Why 58.3% REVISE is the correct verdict:** before (L1) the gap list was noise
(pytorch, aws, kubernetes…); after (L6) every gap is a real curriculum decision
(deep learning, NLP, CV) — exactly what a curriculum committee should hear.
An automatic MAINTAIN would be flattery; UPDATE was slander; REVISE with nameable
gaps is the truth.

---

## 11. Implementation mapping (surgical changes to existing code)

All in `backend/phase5_curricula/curriculum_recommendation/main.py` unless noted:

| Change | Where |
|---|---|
| `course_type` (3-way) + `confidence` + `reason` in gate schema/prompt | `_FIELD_SCHEMA`, `_FIELD_SYS`, `classify_field()` |
| Tier-tag core skills, cached `{field}_field_levels.json` | new `get_field_levels()` (batched `_chat` call) |
| Tier-filtered denominator + `next_course_topics` | `compute_coverage()` |
| Gray-zone rescue band (0.55–0.70) | `verify_coverage()` (currently only checks pairs ≥ τ) |
| Credential/degree-label exclusion | `_FIELD_EXCLUDE` (+ new `_CREDENTIALS` set) |
| Two-axis label + FOUNDATIONAL/FOCUSED | `_label()` → `_label2(completeness, relevance)` |
| Narrow tool inference (languages/libs from §Resources only) | `baseline_analyze/parse_pdf/main.py` |
| Foundational gate + pass-through of course_type | `recommend_from_skills()`, `api.py` `/api/curricula/cr-recommend` (line ~1789) |
| Editable course-type chip; FOCUSED/FOUNDATIONAL label colors; next-course-topics panel | `frontend RecommendationV1Tab.js` / `RecommendationMatch.js` |

Runtime LLM cost per course: gate (1 call, extends existing) + verify/rescue (1 batched call)
+ curation (1 call, existing) = ~3 calls. Tier tags and field inventory are cached per field.

---

## 12. Validation plan

1. **Anchors:** Automata → FOUNDATIONAL (gated); Intro-AI → APPLIED, 58.3% REVISE,
   relevance 100%; Cyber-Crime → APPLIED/PROFESSIONAL.
2. **Gate study:** instructors label 20–30 courses (3 classes) → accuracy + Cohen's κ for
   LLM vs Bloom-rule baseline. (UniSkill's human–human κ≈0.45 sets expectations.)
3. **Ablation table:** the L0→L6 ladder (each lever one flag).
4. **Stability:** classify + score each course 3× → agreement rate.
5. **Expert verdict study:** 2–3 ITC instructors rate ~10 course verdicts
   (protocol per ArXiv 2512.13658: 79–83% agreement vs humans is the published bar).
6. **External comparison:** Baral 2026 Alignment Index.

---

## 13. References (verified — see memory `phase5_related_papers.md`)

- Ahadi et al., EDM 2022 — Skills Taught vs Skills Sought (framework adapted; RCA weighting)
- Baral & Ghimire, 2026 — Alignment Index, High/Med/Low (DOI 10.3126/pecj.v3i1.93534)
- Gromov et al., EDM 2020 — curriculum profile
- Xu et al., 2025 (ArXiv 2505.02324) — RAG/LLM extraction for curricular analytics
- Sabet et al., 2024 — Course-Skill Atlas (SBERT matching at scale)
- UniSkill, 2026 (ArXiv 2603.03134) — match-classification benchmark (macro-F1 0.87)
- Education Sciences 2026 (DOI 10.3390/educsci16040606) — symmetric ESCO curriculum/postings
- ArXiv 2512.13658, 2026 — embedding-based outcome alignment, expert-validation protocol
- SkillSpan (Zhang et al. 2022) — extraction benchmark (our qwen3: Hard F1 50.1 / Sem F1 78.0)
- Frej et al., SIGIR 2024 (ArXiv 2404.10876); eDoer 2022 — job-market-driven recommendation

### Novelty claims (vs the 7 closest works)
1. Course-type **gate** (FOUNDATIONAL/APPLIED/PROFESSIONAL) — none of the related systems
   has one; all mis-score theory courses.
2. **Tier-filtered denominator** (level-aware completeness) + next-course-topics output.
3. **Band-based retrieve-then-verify** matching (LLM adjudicates the gray zone).
4. Two-axis label with FOCUSED (separates "narrow" from "misaligned").
5. End-to-end Cambodian-HE deployment with web UI and live job-discovery pipeline.
