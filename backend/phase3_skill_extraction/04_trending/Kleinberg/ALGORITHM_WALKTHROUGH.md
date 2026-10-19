# Kleinberg Burst Detection — Step-by-Step Walkthrough

> A complete trace of the algorithm using "PyTorch" from our job-postings dataset
> as a worked example. 8 steps, 4 key formulas.

---

## Reference

- **Algorithm**: Kleinberg (2002), *Bursty and Hierarchical Structure in Streams* (KDD)
- **Applied to skill bursts**: Börner et al. (PNAS 2018), *Skill discrepancies between research, education, and jobs*
- **Production defaults**: `s = 2`, `γ = 1` (Kleinberg's original paper)

---

## The 4 Formulas to Memorize

```
              n − 1
⭐ FORMULA 1   λ₀ = ────────                "base rate"
                  total_days


⭐ FORMULA 2   λ₁ = s × λ₀                  "burst rate"


⭐ FORMULA 3   tax = γ × log(n)             "switching penalty"


⭐ FORMULA 4   cost(state, gap) = −log(λ) + λ × gap
                                  ↑           ↑
                              "rarity"   "improbability"
```

---

## Worked Example — Inputs

```
Skill         : PyTorch
n             : 50 events
First event   : 2024-09-14
Last event    : 2026-04-03
Total span    : 565 days
s (slider)    : 4
γ (slider)    : 1
```

---

## Pipeline — 8 Steps

```
Step 1   Collect timestamps per skill
Step 2   Compute gaps between consecutive events
Step 3   Compute rates (λ₀, λ₁) and switching tax
Step 4   Score each gap (cost for state 0 and state 1)
Step 5   Viterbi — find cheapest path through all gaps
Step 6   Backtrace the optimal state sequence
Step 7   Extract burst intervals from state runs
Step 8   Format the headline burst output
```

---

## Step 1 — Collect timestamps per skill

**Purpose:** group all jobs by skill, so each unique skill gets its own timestamp list.

**Transformation:**

```
INPUT (jobs):                          OUTPUT (per-skill streams):
─────────────                          ──────────────────────────
job_001 2024-09-14 [PyTorch, ...]      PyTorch  → [Sep 14 '24, Oct 22 '24, ..., Apr 03 '26]
job_002 2024-10-22 [PyTorch, ...]      Python   → [Sep 14 '24, ...]
job_003 2026-02-04 [PyTorch, ...]      Docker   → [Sep 14 '24, ...]
...                                    ...
```

**Two rules:**
1. Dedupe within a job (`set(job["skills"])`) — count "Python" once even if it appears twice.
2. Skip jobs with missing/unparseable `posted_date`.

**Output:** `streams = { skill: [list of timestamps] }`

After this step, each skill is processed **independently**.

---

## Step 2 — Compute gaps between consecutive events

**Purpose:** measure the days between each consecutive pair of events.

**Formula:**
```
gap_i = t_{i+1} − t_i      (in days)
```

If a skill has `n` events, we get `n − 1` gaps.

**For PyTorch (50 events → 49 gaps):**

```
event:  ●─38d─●─39d─●─5d─●─49d─●  ...  ●─2d─●─1d─●─2d─●─1d─●─3d─●

         └─── early period (sparse) ──┘     └── late period (dense) ──┘
              LONG gaps                          SHORT gaps
```

**Output:** `gaps = [38, 39, 5, 49, 28, ..., 3, 2, 1, 1, 3, 1, 2]` — 49 numbers.

> The algorithm cares only about gap sizes from this point on. Original timestamps are set aside until Step 7.

---

## Step 3 — Compute rates and switching tax

**Purpose:** set up the 3 "rules of the game" — natural pace, burst pace, and the cost of switching.

### λ₀ — base rate

```
        n − 1            49
λ₀  =  ────────  =  ──────────  =  0.087  events/day   ≈  1 event every 11 days
       total_days       565
```

### λ₁ — burst rate

```
λ₁  =  s × λ₀  =  4 × 0.087  =  0.347  events/day   ≈  1 event every 2.9 days
```

### tax — switching penalty

```
tax  =  γ × log(n)  =  1 × log(50)  =  3.91   units of cost per state flip
```

**Visual:**

```
λ₀ (boring rhythm):    ●──11d──●──11d──●──11d──●──11d──●──11d──●

λ₁ (bursty rhythm):    ●─3d─●─3d─●─3d─●─3d─●─3d─●─3d─●─3d─●─3d─●

tax = 3.91 paid every time the algorithm flips between the two
```

**Output (3 constants):** `λ₀ = 0.087`, `λ₁ = 0.347`, `tax = 3.91`

---

## Step 4 — Score each gap

**Purpose:** for every gap, compute the cost under state 0 (normal) and state 1 (burst).

**Formula 4:**
```
cost(state, gap) = −log(λ) + λ × gap

   where  λ = λ₀  if state = 0
          λ = λ₁  if state = 1
```

**Apply to PyTorch:**

| gap (days) | cost(state 0) | cost(state 1) | winner |
|---|---|---|---|
| 38 | −log(0.087) + 0.087·38 = **5.74** | −log(0.347) + 0.347·38 = 14.25 | s0 ✅ |
| 39 | 5.83 | 14.59 | s0 |
| 5  | 2.88 | 2.79 | s1 (barely) |
| 49 | 6.70 | 18.07 | s0 |
| ... | ... | ... | ... |
| 2  | 2.61 | **1.75** | s1 ✅ |
| 1  | 2.53 | **1.41** | s1 ✅ |

**Output:** a 49 × 2 cost table.

> ⚠️ Don't pick the per-gap winner yet — that would ignore the switching tax. Step 5 (Viterbi) finds the best **whole path**.

---

## Step 5 — Viterbi: find the cheapest path

**Purpose:** find the single sequence of states (one per gap) that minimizes the total cost — including transition tax.

**Mental model:** every gap is a "cell" with two doors (state 0 / state 1). Picking a door costs `cost(state, gap)`. Switching doors between consecutive cells costs `tax = 3.91`. Find the cheapest path from the first cell to the last.

```
       gap 1     gap 2     gap 3     ...     gap 49
       ─────     ─────     ─────             ─────
       door 0    door 0    door 0    ...     door 0    ← state 0 path
         │  ╲      │  ╲      │
         │   tax   │   tax   │
         │   ╳     │   ╳     │
         │   tax   │   tax   │
         │  ╱      │  ╱      │
       door 1    door 1    door 1    ...     door 1    ← state 1 path
```

**Dynamic programming:**

For each gap `i` and state `s`, compute the minimum cost to reach it:

```
dp[i+1][s] = min over prev_state of (
    dp[i][prev]
    + (transition_cost if prev != s else 0)
    + cost(s, gap[i])
)
```

The DP table grows step by step; at each gap, both states track the cheapest accumulated cost so far.

**For PyTorch:** the path that wins is:

```
gap index:  1   2   3   ...   37   38   39   40   ...   48   49
states:     0   0   0   ...    0    1    1    1   ...    1    1
                                    ↑
                            transition: pay tax of 3.91,
                            but save much more on the
                            short gaps that follow
```

**Output:** the table `dp[i][s]` for every gap and state, plus a back-pointer telling us which previous state gave the minimum at each cell.

---

## Step 6 — Backtrace the optimal state sequence

**Purpose:** read the back-pointers backwards to recover the chosen state at every gap.

**For PyTorch:**

```
states = [
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    ← gaps 1-10 (sparse early)
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    ← gaps 11-20
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    ← gaps 21-30
   0, 0, 0, 0, 0, 0, 0, 1, 1, 1,    ← gaps 31-40 (transition at gap 38)
   1, 1, 1, 1, 1, 1, 1, 1, 1        ← gaps 41-49 (sustained burst)
]
                                ▲
                        state-1 run = the burst!
```

**Output:** a list of 49 zeros and ones — the optimal state assignment.

---

## Step 7 — Extract burst intervals

**Purpose:** find continuous runs of state 1 → those are the bursts.

```
states: [..., 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                       ▲                                ▲
                  burst starts                      stream ends
                  at event 39                       (still in state 1)
```

**For PyTorch:**

```
burst = {
   start_event: 20         ← bs+1 = (gap 38's right endpoint)
   end_event:   50         ← last event
   start_ts:    2026-02-04
   end_ts:      2026-04-03
   n_events:    31         ← 50 − 19
   ongoing:     True       ← state stayed at 1 until end of stream
}
```

**Output:** a list of bursts, each with start/end timestamps and event count.

---

## Step 8 — Format the headline output

**Purpose:** pick the most prominent burst and format it for display.

**Headline selection:** prefer ongoing bursts; tiebreak by event count.

**Display formula:**
```
"PyTorch  31ev · 58d ●"
          ↑    ↑    ↑
          │    │    └── ● = ongoing
          │    └── (end_ts − start_ts) in days = 58
          └── n_events = 31
```

**Final output (one row per skill):**
```
1.  ● PyTorch         31ev · 58d
2.  ● LLM             30ev · 65d
3.  ● Figma           29ev · 70d
4.  ● LangChain       29ev · 138d
...
```

---

## End-to-end summary

```
┌───────────────────────────────────────────────────────────────┐
│  STEP   INPUT          → OUTPUT                                │
├───────────────────────────────────────────────────────────────┤
│   1     all jobs       → per-skill timestamp lists             │
│   2     timestamps     → 49 gaps in days                       │
│   3     n, total_days  → λ₀, λ₁, tax  (the "rules")            │
│   4     gaps + λ's     → 49 × 2 cost table                     │
│   5     cost table     → cheapest path (Viterbi DP)            │
│   6     back-pointers  → state sequence (49 0s and 1s)         │
│   7     state sequence → burst intervals (start/end/n_events)  │
│   8     burst data     → "31ev · 58d ●"                         │
└───────────────────────────────────────────────────────────────┘
```

---

## Production parameters

```
DEFAULT_S      = 2     (Kleinberg 2002)
DEFAULT_GAMMA  = 1     (Kleinberg 2002)
MIN_OCCURRENCES = 10   (skip skills with too few events)
TOP_N           = 15
```

These are the values used by `pybursts`, R `bursts`, Sci2 Tool, and Börner et al. (PNAS 2018). Lock them as constants — don't expose to end users.

---

## Defense paragraph

> For trending skill detection we apply Kleinberg's burst detection (Kleinberg 2002),
> the academic gold standard for identifying state transitions in event streams. The
> algorithm models each skill's appearance pattern as a 2-state HMM (normal rate λ₀,
> burst rate λ₁ = s · λ₀) and uses Viterbi dynamic programming to find the optimal
> state sequence given the canonical parameters s = 2, γ = 1 (Kleinberg 2002; Börner
> et al., PNAS 2018). Bursts are extracted as continuous runs of state 1; for each
> skill, the headline burst's start date, duration, and event count are reported.

---

## Prompt for regenerating as PDF

> Paste this section into Claude to convert the walkthrough into a polished PDF.

```
You are a technical writer. Convert the following Markdown into a formal
LaTeX/PDF document with:

- Title page with authors and date
- Table of contents
- Numbered sections matching the 8 steps
- Math formulas rendered with LaTeX (use $$ for display, $ for inline)
- Code/calculation blocks in monospace
- Diagrams kept as ASCII art in verbatim blocks
- A references section citing Kleinberg (2002), Börner et al. (2018),
  Shewhart (1931), and Aggarwal (2007)

Use IEEE thesis-style formatting. Keep all numerical examples exactly
as written — do not modify the PyTorch worked example.

[paste this whole document below]
```
