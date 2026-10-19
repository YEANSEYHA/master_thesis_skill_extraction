# Playground — the thesis method on data you can read whole

The real pipeline runs 1,085 market skills × 130 CLOs (141,050 cells).
This tutorial runs the **same frozen instrument** (same encoder, same
thresholds, even the same `greedy_skill_selection` function, imported
from `../03_engine/pipeline.py`) on **10 market skills × 8 CLOs** — small enough
to check every number by eye.

## The whole method in one line per step

```
texts ──encode──▶ vectors ──matmul──▶ sim matrix ──row max──▶ coverage
        (1)          (1)      (2)        (2)          (3)        (3)
coverage ──thresholds──▶ status     demand × (1−coverage) ──▶ gap, KL
               (3)                              (4)
non-covered skills ──greedy facility-location──▶ ordered add-plan
                              (5)
```

## Workbook format — you write the code

Steps 2–5 are **workbooks**: setup is given at the top, then numbered
`PROBLEM` blocks with the key syntax inline. Type the code yourself
under each `# Your code below:`, complete problems **in order** (later
ones use your earlier variables), run, and paste the output for review.
(Step 1 is the original guided lesson, already completed.)

| workbook | teaches | the one idea |
|---|---|---|
| `step1_embedding.py` | embeddings | meaning → direction; dot product = similarity |
| `step2_similarity_matrix.py` | THE matrix | `skill_emb @ clo_emb.T`; everything else derives from it |
| `step3_coverage_status.py` | coverage + verdicts | row max, thresholds 0.73/0.69/0.66, soft-skill filter |
| `step4_gap_kl.py` | program numbers | demand × (1−coverage) = gap; KL = shape mismatch |
| `step5_greedy_selection.py` | the plan | submodular greedy, spillover, 80% stopping rule |

```bash
cd backend/phase5_curriculum_alignment
python3 tutorial/step2_similarity_matrix.py
```

`tiny_data.py` holds the shared miniature dataset (12 market skills ×
8 CLO skills) — edit it for the bonus experiments.

## Deliberately planted lessons

- **sql × "SQL Injection Prevention"** — argmax finds a plausible-looking
  host that the cross-encoder later demolished (0.69 cos → 0.03 rerank).
- **communication** — caught by `is_soft_skill`, never competes with
  technical skills.
- **aws** — its best CLO sits below `host_min`: argmax below the noise
  floor means *no host*, not a weak host.
- **spillover** — watch step 5's last column: adding one skill lifts its
  semantic neighbors for free.

Nothing here touches production files: no cache, no report_data.js,
no thresholds — read-only imports from `pipeline.py`.
