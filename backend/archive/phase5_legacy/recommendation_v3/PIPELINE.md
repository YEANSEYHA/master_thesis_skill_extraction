# Recommendation V3 — Full Pipeline (finalized design)

Course-to-labour-market alignment via **career-domain routing** +
**competency consolidation against a canonical market vocabulary**.

This is the authoritative spec for the V3 path. `propose.md` is the original
routing-first conceptual seed; this document supersedes it with the full,
agreed-upon 7-stage process (incl. the canonicalization stage).

---

## End-to-end process

```
Course PDF
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 1. SKILL EXTRACTION (qwen3)                                │
└──────────────────────────────────────────────────────────┘
   │
   ▼
Raw Skills
   K-Means · Decision Trees · Logistic Regression · KNN ·
   Gradient Boosting · Cross-Validation · EDA · Data Preparation ·
   AI Sub-Fields · …
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 2. CAREER-DOMAIN ROUTING   (route_hybrid on RAW skills)    │
└──────────────────────────────────────────────────────────┘
   Artificial Intelligence   0.81
   Data Science              0.79
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 3. BUILD CANONICAL MARKET VOCABULARY                       │
│    (one-time per domain, cached)                           │
└──────────────────────────────────────────────────────────┘
   AI Market Inventory  (23,999 postings · 1000 skills)
        │
        ▼   Importance ranking
            importance(skill) = demand × distinctiveness (IDF)
        │
        ▼   Top 60 important skills
            python · machine learning · data analysis · sql ·
            data engineering · statistics · cloud computing ·
            data visualization · …
        │
        ▼   Semantic de-duplication
            data analytics / business analytics → data analysis
            ml / machine-learning               → machine learning
            data mining                         → machine learning   (if semantic overlap > τ)
        │
        ▼
   Canonical Vocabulary (~50 concepts)
        python · machine learning · data analysis · sql ·
        data engineering · statistics · cloud computing ·
        model evaluation · data visualization · … + "other"

   (cache: domain → canonical_vocab.json)
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 4. COMPETENCY CONSOLIDATION   (1 cached qwen3 call/course) │
└──────────────────────────────────────────────────────────┘
   Prompt:
     "Group curriculum skills into competencies.
      Prefer a concept from the canonical vocabulary.
      If no suitable concept exists, create a curriculum-specific
      competency and mark it as SPECIALIZED.
      Keep evidence."

   Market Competencies
     machine learning  ◀── K-Means, Decision Trees, Logistic Regression,
                           KNN, Gradient Boosting, Gaussian Mixture
     data analysis     ◀── EDA, Data Preparation
     model evaluation  ◀── Cross-Validation

   Specialized Competencies
     AI Foundations    ◀── AI Sub-Fields

   (cache: course_code → competencies.json)
   │
   ├──────────► MARKET       (scored)
   └──────────► SPECIALIZED  (display only)
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 5. DOMAIN-SCOPED COVERAGE                                  │
└──────────────────────────────────────────────────────────┘
   Domain Core = Top 30 important skills  (⊂ canonical vocabulary)
        python · machine learning · sql · data analysis ·
        data engineering · statistics · cloud computing · …

   Coverage = Σ demand(covered core) / Σ demand(all core)   (demand-weighted)

   Only MARKET competencies participate.
   SPECIALIZED competencies are excluded.
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 6. GAP ANALYSIS & RECOMMENDATION                           │
└──────────────────────────────────────────────────────────┘
   Critical gaps  (uncovered core, ranked by importance = demand × distinctiveness)
        sql · data visualization · aws · statistics

   Recommendation rules (by critical-gap COUNT, not raw %):
        0–1 critical → MAINTAIN
        2–3 critical → REVISE
        > 3 critical → UPDATE
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 7. USER INTERFACE                                          │
└──────────────────────────────────────────────────────────┘
   Aligned Competencies
     ✓ Machine Learning · 6 techniques
         ▸ K-Means · Decision Trees · Logistic Regression ·
           KNN · Gradient Boosting · Gaussian Mixture
     ✓ Data Analysis · 2 techniques   ▸ EDA · Data Preparation
     ✓ Model Evaluation · 1 technique ▸ Cross-Validation

   Market Gaps
     ✗ SQL · ✗ Data Visualization · ✗ AWS · ✗ Statistics

   Specialized Topics
     • AI Foundations  ▸ AI Sub-Fields

   Recommendation
     Introduce SQL for AI data processing.
     Add Data Visualization (Matplotlib / Seaborn).
     Introduce basic cloud deployment concepts.
```

---

## Stage notes & design decisions

### 2. Routing runs on RAW skills (before canonicalization)
Routing must come first — the domain decides *which* inventory to canonicalize
against, so it cannot depend on the canonical vocab (would be circular). Raw
routing is already accurate (AI 0.81 / Data Science 0.79), so this ordering is
both necessary and sufficient.

### 3. Canonical vocabulary = the market's own top-demand terms (de-duped)
- **Not** all 1000 market skills (too many → the LLM classifies inconsistently
  course-to-course), and **not** a hand-invented taxonomy (un-defensible).
- Built from the routed domain's inventory: rank by
  `importance = demand × distinctiveness(IDF)`, take **top 60**, then
  **semantically de-duplicate** near-synonyms so each concept has *one* sensible
  target (e.g. `data analytics → data analysis`; `data mining → machine learning`
  when overlap > τ). Result ≈ **50 canonical concepts + `"other"`**.
- De-dup is a **one-time, cached, human-reviewable** pass per domain (~6 lists) —
  not a blind threshold merge (a pure cosine cutoff can mis-merge).
- Cache: `domain → canonical_vocab.json`.

### 4. Competency consolidation = LLM-as-classifier (Option B + escape hatch)
- One cached qwen3 call per course (temperature 0). The LLM **classifies** each
  raw skill **into** the canonical vocab — it does **not** invent the vocabulary.
- Output is constrained to **exact vocab strings** (enum-style, as in
  `route.py`'s `_ROUTE_SCHEMA`) so the labels match the core strings exactly.
- Anything with no market fit becomes a **SPECIALIZED** competency (LLM-named) —
  the readable replacement for a flat `"other"` bucket. This handles off-market /
  foundational content (e.g. *Quantum Computing*, *AI Foundations*) gracefully.
- Original curriculum terms are retained as **evidence** under each competency
  (explainability / traceability).
- Cache: `course_code → competencies.json` (runs once per syllabus, not per
  request).

### 5. Coverage = EXACT canonical-label match (no cosine threshold)
Because course competencies and the domain core are now drawn from the **same**
canonical vocabulary, coverage is a **set intersection on canonical labels**:

```
coverage = Σ demand(core ∩ course_market_competencies) / Σ demand(core)
```

This **retires the cosine τ=0.60** from the coverage step — matching is by concept
identity, which is deterministic and more defensible. Cosine remains only in
routing (stage 2) and the de-dup overlap (stage 3). SPECIALIZED competencies are
**excluded** from coverage (correct: off-market by design).

### 6. Recommendation = gap COUNT, not raw %
The verdict follows the **critical-gap count**, the way a curriculum committee
reasons — a covered high-demand skill is not a gap, so a well-aligned course is
not punished by the senior-stack denominator.

```
critical = uncovered core skill with ≥ 5% of core demand share
0–1 critical → MAINTAIN   ·   2–3 → REVISE   ·   > 3 → UPDATE
```

### 7. UI — Ahadi 3-set, canonical groups + evidence
- **Aligned competencies** — canonical groups with expandable evidence
  (`Machine Learning · 6 techniques ▸ …`).
- **Market gaps** — uncovered core, importance-ranked.
- **Specialized topics** — SPECIALIZED competencies (display only, not scored).
- **Coverage %** shown small/secondary (the gap list drives the verdict).
- Recommendation lines are prescriptive suggestions (*"Introduce SQL …"*), not a
  categorical verdict label.

---

## Caching summary

| Cache                          | Scope            | When built            |
| ------------------------------ | ---------------- | --------------------- |
| `domain_inventory.json`        | all domains      | offline, once         |
| `canonical_vocab.json`         | per domain (~6)  | one-time, reviewable  |
| `competencies.json`            | per course_code  | once per syllabus     |

---

## Honest scope (what this changes vs not)

- ✅ Collapses the noisy "mismatch from the course" set into a few clean canonical
  groups with evidence drill-down.
- ✅ Makes routing stable and the matching deterministic (exact label match, no τ).
- ✅ Defensible methodology: *canonical vocabulary derived from labour-market
  inventories (demand × distinctiveness, de-duped); the LLM is used only as a
  classifier mapping curriculum skills to an existing market concept or
  SPECIALIZED; original terms retained as evidence; coverage scored against the
  domain's top-30 in-demand core.*
- ⚠️ **Coverage % stays ~the same** — real gaps (sql, aws, r, tableau) remain in
  the denominator. The win is UI cleanliness, evidence, stable routing, and a
  deterministic metric — **not** a higher score.

---

## Build order

```
① build_canon_vocab(domain)   → top-60 → de-dup → cache        (stage 3)
② consolidate(raw, vocab)      → market + specialized, cached    (stage 4)
③ wire into analyze            → exact-match coverage            (stage 5–6)
④ UI grouping + evidence                                         (stage 7)
```
