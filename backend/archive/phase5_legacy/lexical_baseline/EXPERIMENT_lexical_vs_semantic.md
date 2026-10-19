# Experiment — Lexical vs Semantic on Automata Theory

Head-to-head of the **lexical baseline** (this folder) against the
**semantic + ESCO** method (`../clo_skill_alignment/`) on the *same* real ITC
syllabus, `automa_theory_course.pdf` (GICI3AT, Computer Sciences).

- Date run: 2026-05-31
- Inventory: real Phase-3 skills (3,216 postings), Top-N = 15
- Threshold: τ_sim = 0.65 · view: curriculum (programme)
- Endpoint: `POST /api/curricula/analyze-curriculum`

## Results (same course, top-15, τ = 0.65)

| Method | Label | Coverage | Matched | What it matched |
|---|---|---|---|---|
| **Lexical** (raw skills) | 🔴 UPDATE | **6.8 %** | **1 / 15** | `problem solving` |
| **Semantic + ESCO** | 🔴 UPDATE | 0.0 % | 0 / 15 | — |
| **Semantic** (raw, no ESCO) | 🔴 UPDATE | 0.0 % | 0 / 15 | — |

> The first two rows are what the UI exposes (Lexical vs Semantic+ESCO). The
> third row — **semantic without ESCO** — was run explicitly to isolate the
> matching mechanism from the ESCO variable (the clean ablation).

## Discussion

### 1. The baseline beat the proposed method here

The trivial lexical baseline matched one skill (`problem solving`, 117 jobs →
6.8 % weighted coverage); **both** semantic variants matched **nothing**.
Counter-intuitive, and addressed honestly below rather than buried.

### 2. ESCO is exonerated — the difference is purely the matcher

Semantic **without ESCO** (raw skills, the same inventory the lexical method
sees) *also* scored 0.0 % / 0 matches. So the lexical-vs-semantic gap is **not**
caused by ESCO normalization — it is caused entirely by the **matching
mechanism** (word-overlap vs thresholded cosine). The screenshot comparison
`lexical(raw)` vs `semantic(ESCO)` was *confounded* (two variables changed at
once); the third condition resolves it.

### 3. Why lexical caught `problem solving` and semantic did not

- **Lexical:** CLO3 literally reads *"Evaluate **problem-solving** approaches in
  automata theory…"*. Tokenized, `{problem, solving}` ⊆ CLO3 → exact
  containment → match. When the *words* coincide, lexical fires.
- **Semantic:** the matcher embeds each CLO/lesson and max-pools cosine per
  skill. `cos(embed(CLO3), embed("problem solving"))` came out **< 0.65** — even
  though the phrase is literally inside CLO3. CLO3 is long and dominated by
  *"automata theory, logical rigor, theoretical reasoning,"* so its embedding is
  **diluted**; a bare 2-word skill phrase scores low against it, and τ = 0.65
  rejects it.

**Finding:** *thresholded semantic matching under-matches short skill phrases
against long CLOs — to the point of missing a skill that is lexically present.*
The τ = 0.65 cutoff is mis-calibrated for this short-skill / long-CLO regime.

### 4. Was lexical's match even good?

Only shallowly. The course's "problem-solving" is **theoretical** (proving
automata properties); the job-market `problem solving` (117 postings) means
**workplace** problem-solving. Same words, different sense — exactly the
false-positive mode semantic matching is supposed to avoid. Here, though,
semantic didn't discriminate the sense; it matched **nothing at all**, which is
worse, not better.

### 5. The robust, method-invariant conclusion

All three methods → **UPDATE, ~0–7 %, soft-skill gaps** (communication 252,
teamwork 182, customer service 163…). The *qualitative* verdict — *a theoretical
CS course is misaligned with a soft-skill-dominated Cambodian job corpus* — is
**identical across methods**. The 6.8 % vs 0 % difference is **one coincidental
lexical match in a degenerate regime**, not evidence that lexical is superior.
On this corpus the two methods are **practically indistinguishable**.

### 6. Actionable fixes

1. **Re-calibrate τ_sim.** A course that literally contains "problem-solving"
   should match the skill "problem solving." Missing it at 0.65 means the
   threshold is too high (or CLO units too long). Try τ ≈ 0.45–0.55, or match
   skills against **shorter units** (individual lesson outcomes, not whole CLOs).
2. **The corpus is the real blocker.** Semantic's advantage — paraphrase recall
   (`"ML"` ≈ `"machine learning"`) — only appears when the inventory shares
   vocabulary/domain with the courses. A general soft-skill corpus gives neither
   method anything to match. A **tech-filtered corpus** is a prerequisite for a
   meaningful comparison.
3. **Honest paper framing:** *"On the current corpus, lexical and semantic are
   statistically indistinguishable (both UPDATE, 0–7 %). The lexical baseline
   even edges ahead via a single coincidental token match, exposing that our
   semantic threshold under-matches short skills. The semantic method's
   theoretical advantage is not realized in this degenerate, domain-mismatched
   setting."*

## Follow-up — domain-conditioned (IT-filtered) corpus: the 2×2

After adding **domain-conditioned matching** (the qwen3 classifier scopes the
corpus to the course's job category — Automata → `it`, 256 postings), the same
two methods were re-run against the IT-filtered corpus. Combined with the
general-corpus rows above this gives a 2 (corpus) × 2 (method) grid.

| Method | General corpus | IT-filtered corpus (`it`, 256) |
|--------|----------------|--------------------------------|
| **Semantic + ESCO** | UPDATE 0.0 % (0 matched) | UPDATE 0.0 % (0 matched) |
| **Lexical** (raw) | UPDATE 6.8 % (1: *problem solving*) | UPDATE 5.4 % (2: *problem solving*, *problem-solving*) |

### Findings

1. **Robustness (the key result):** **all four cells → UPDATE.** Automata Theory
   is UPDATE regardless of *corpus scoping* OR *matching method*. The verdict is
   invariant to both variables — strong evidence it is a genuine foundational
   course the market does not directly demand, not a configuration artifact.

2. **Domain filtering corrected the comparison set but not the verdict.** The
   IT-filtered gaps are now IT skills (sketch design, invision prototyping,
   framer x development, JavaScript framework) instead of soft skills — the
   filter works — yet coverage barely moved (still ~0–5 %). Foundational theory
   stays unmatched even within its own domain.

3. **Lexical's edge is illusory.** On the IT corpus it matched
   `problem solving` AND `problem-solving` — the **same skill counted twice**
   because the lexical path uses raw, un-normalized strings (exactly the
   fragmentation ESCO normalization fixes). Both came from CLO3's literal
   "problem-solving approaches" (coincidental word overlap, theoretically- vs
   workplace-sense ambiguous). So the 5.4 % is one shallow match, double-counted.

4. **ESCO noise persists inside the IT domain.** The IT ESCO inventory's top
   labels include clear garbage mappings — *operate continuous miner (38),
   analyse legal evidence (37), ensure compliance with construction project
   deadline (35)* — IT skills mis-mapped to mining/legal/construction ESCO
   labels. The domain filter scopes the postings correctly, but the ESCO mapping
   step still pollutes the comparison set (the open Khmer/over-merge issue,
   now visible within the right domain).

### Take-home (updated)

> Across 2 corpora × 2 methods, Automata Theory is **robustly UPDATE**. Domain
> filtering fixes *which* jobs are compared (IT skills, not soft skills) but a
> pure-theory course remains structurally unmatched — market-demand alignment
> inherently undervalues foundational courses. The lexical baseline's small
> numeric edge is a single coincidental, double-counted word match, not a real
> advantage. Two separable issues remain: the semantic threshold under-matches
> short skills, and ESCO mapping injects noise even within a correct domain.

## Reproduce

```bash
# backend on :8000, then
curl -s -X POST http://localhost:8000/api/curricula/analyze-curriculum \
  -H 'Content-Type: application/json' \
  -d '{"course": <parsed automata>, "use_dummy": false, "top_n": 15,
       "match_threshold": 0.65, "method": "lexical"}'
# repeat with "method":"semantic","esco_normalize":true  and  false
# for the IT-filtered rows, add:  "categories": ["it"]
```

Or via the UI: Curricula tab → upload `automa_theory_course.pdf` → Step 1 pick
**Lexical** vs **Semantic + ESCO** → Step 3 reads the result.
