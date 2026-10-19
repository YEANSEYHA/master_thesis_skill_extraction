# Skill Range Comparison: qwen3:14b on SkillSpan Benchmark

**Date:** 2026-03-31
**Model:** qwen3:14b
**Similarity Model:** jjzha/jobbert-base-cased
**Dataset:** SkillSpan (263 job postings, all splits)
**Comparison:** Skills 5-10 vs Skills 5-15 vs Skills 5-20

---

## Executive Summary

| Metric | 5-10 | 5-15 | 5-20 | Winner |
|--------|------|------|------|--------|
| **Hard F1 @0.60** | 47.1% | 50.1% | **50.7%** | 5-20 |
| **Semantic F1 (Micro)** | 77.4% | **78.0%** | **78.0%** | 5-15 / 5-20 |
| **Semantic F1 (Macro)** | 75.7% | **76.8%** | **76.8%** | 5-15 / 5-20 |
| **Precision @0.60** | **96.4%** | 95.8% | 96.0% | 5-10 |
| **Recall @0.60** | 31.1% | 33.9% | **34.4%** | 5-20 |

**Winner: 5-15 skill range (recommended)** - Semantic F1 plateaus at 5-15; 5-20 adds minimal gain (+0.6% Hard F1) with negligible extra extractions (avg 12.5 vs 12.3 skills/job).

---

## Results: Skills 5-10

### Extraction Statistics
| Metric | Value |
|--------|-------|
| Jobs Evaluated | 263 |
| Total LLM Skills | 2,957 |
| Total Human Skills | 9,166 |
| Avg LLM Skills/Job | 11.2 |

### Hard F1 (Multi-Threshold)

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|-----------|----|----|----|-----------:|-------:|----:|
| @ 0.60 | 2,852 | 105 | 6,314 | 96.4% | 31.1% | 47.1% |
| @ 0.65 | 2,740 | 217 | 6,426 | 92.7% | 29.9% | 45.2% |
| @ 0.70 | 2,520 | 437 | 6,646 | 85.2% | 27.5% | 41.6% |
| @ 0.75 | 2,311 | 646 | 6,855 | 78.2% | 25.2% | 38.1% |
| @ 0.80 | 2,043 | 914 | 7,123 | 69.1% | 22.3% | 33.7% |

### Semantic F1

| Metric | Value |
|--------|------:|
| Micro Sem-Precision | 86.3% |
| Micro Sem-Recall | 70.2% |
| **Micro Sem-F1** | **77.4%** |
| **Macro Sem-F1** | **75.7%** |

### Precision@K (threshold=0.65)

| K | Matched | Total | Micro P@K | Macro P@K |
|---|--------:|------:|----------:|----------:|
| 5 | 1,232 | 1,264 | 97.5% | 94.0% |
| 10 | 2,338 | 2,436 | 96.0% | 92.5% |
| 15 | 2,834 | 2,957 | 95.8% | 92.1% |

---

## Results: Skills 5-20

### Extraction Statistics
| Metric | Value |
|--------|-------|
| Jobs Evaluated | 263 |
| Total LLM Skills | 3,287 |
| Total Human Skills | 9,166 |
| Avg LLM Skills/Job | 12.5 |

### Hard F1 (Multi-Threshold)

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|-----------|----|----|----|-----------:|-------:|----:|
| @ 0.60 | 3,155 | 132 | 6,011 | 96.0% | 34.4% | 50.7% |
| @ 0.65 | 3,003 | 284 | 6,163 | 91.4% | 32.8% | 48.2% |
| @ 0.70 | 2,747 | 540 | 6,419 | 83.6% | 30.0% | 44.1% |
| @ 0.75 | 2,480 | 807 | 6,686 | 75.4% | 27.1% | 39.8% |
| @ 0.80 | 2,173 | 1,114 | 6,993 | 66.1% | 23.7% | 34.9% |

### Semantic F1

| Metric | Value |
|--------|------:|
| Micro Sem-Precision | 85.9% |
| Micro Sem-Recall | 71.5% |
| **Micro Sem-F1** | **78.0%** |
| **Macro Sem-F1** | **76.8%** |

### Precision@K (threshold=0.65)

| K | Matched | Total | Micro P@K | Macro P@K |
|---|--------:|------:|----------:|----------:|
| 5 | 1,243 | 1,277 | 97.3% | 95.0% |
| 10 | 2,373 | 2,476 | 95.8% | 93.4% |
| 15 | 3,131 | 3,287 | 95.3% | 92.6% |

---

## Results: Skills 5-15

### Extraction Statistics
| Metric | Value |
|--------|-------|
| Jobs Evaluated | 263 |
| Total LLM Skills | 3,243 |
| Total Human Skills | 9,166 |
| Avg LLM Skills/Job | 12.3 |

### Hard F1 (Multi-Threshold)

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|-----------|----|----|----|-----------:|-------:|----:|
| @ 0.60 | 3,107 | 136 | 6,059 | 95.8% | 33.9% | 50.1% |
| @ 0.65 | 2,964 | 279 | 6,202 | 91.4% | 32.3% | 47.8% |
| @ 0.70 | 2,710 | 533 | 6,456 | 83.6% | 29.6% | 43.7% |
| @ 0.75 | 2,451 | 792 | 6,715 | 75.6% | 26.7% | 39.5% |
| @ 0.80 | 2,158 | 1,085 | 7,008 | 66.5% | 23.5% | 34.8% |

### Semantic F1

| Metric | Value |
|--------|------:|
| Micro Sem-Precision | 85.9% |
| Micro Sem-Recall | 71.4% |
| **Micro Sem-F1** | **78.0%** |
| **Macro Sem-F1** | **76.8%** |

### Precision@K (threshold=0.65)

| K | Matched | Total | Micro P@K | Macro P@K |
|---|--------:|------:|----------:|----------:|
| 5 | 1,241 | 1,277 | 97.2% | 94.7% |
| 10 | 2,363 | 2,474 | 95.5% | 92.9% |
| 15 | 3,085 | 3,243 | 95.1% | 92.4% |

---

## Side-by-Side Comparison

### Hard F1 Comparison

| Threshold | 5-10 F1 | 5-15 F1 | 5-20 F1 | Winner |
|-----------|--------:|--------:|--------:|--------|
| @ 0.60 | 47.1% | 50.1% | **50.7%** | 5-20 |
| @ 0.65 | 45.2% | 47.8% | **48.2%** | 5-20 |
| @ 0.70 | 41.6% | 43.7% | **44.1%** | 5-20 |
| @ 0.75 | 38.1% | 39.5% | **39.8%** | 5-20 |
| @ 0.80 | 33.7% | 34.8% | **34.9%** | 5-20 |

### Semantic F1 Comparison

| Metric | 5-10 | 5-15 | 5-20 | Winner |
|--------|-----:|-----:|-----:|--------|
| Micro Sem-Precision | **86.3%** | 85.9% | 85.9% | 5-10 |
| Micro Sem-Recall | 70.2% | 71.4% | **71.5%** | 5-20 |
| Micro Sem-F1 | 77.4% | **78.0%** | **78.0%** | 5-15 / 5-20 |
| Macro Sem-F1 | 75.7% | **76.8%** | **76.8%** | 5-15 / 5-20 |

### Precision@K Comparison (threshold=0.65)

| K | 5-10 Micro | 5-15 Micro | 5-20 Micro | Winner |
|---|----------:|----------:|----------:|--------|
| 5 | **97.5%** | 97.2% | 97.3% | 5-10 |
| 10 | **96.0%** | 95.5% | 95.8% | 5-10 |
| 15 | **95.8%** | 95.1% | 95.3% | 5-10 |

### Extraction Volume Comparison

| Metric | 5-10 | 5-15 | 5-20 |
|--------|-----:|-----:|-----:|
| Total LLM Skills | 2,957 | 3,243 | 3,287 |
| Avg Skills/Job | 11.2 | 12.3 | 12.5 |
| True Positives @0.60 | 2,852 | 3,107 | 3,155 |

---

## Discussion

### Key Findings

1. **5-20 achieves the highest Hard F1 across all thresholds**
   - Hard F1 improves from 47.1% (5-10) → 50.1% (5-15) → 50.7% (5-20)
   - The gain from 5-15 to 5-20 is only +0.6% — diminishing returns

2. **Semantic F1 plateaus at 5-15**
   - 5-15 and 5-20 are identical: Micro F1 = 78.0%, Macro F1 = 76.8%
   - Increasing the range beyond 15 yields no semantic quality gain

3. **The model self-limits around 12-13 skills/job**
   - 5-15 avg: 12.3 skills/job; 5-20 avg: 12.5 skills/job
   - Model ignores the upper bound — it stops when skills are exhausted
   - This explains the negligible difference between 5-15 and 5-20

4. **5-10 maintains the highest Precision@K**
   - Fewer extractions = fewer borderline skills = higher P@K
   - All three ranges remain >95% Micro P@K — all are high quality

5. **Recall is the bottleneck across all ranges**
   - Even 5-20 achieves only 34.4% recall vs ~35 human skills/job
   - Annotation style mismatch (fragments vs phrases) caps recall regardless of range

### Limitations

1. **Low absolute recall**: All ranges achieve ~31-34% recall because SkillSpan annotations are very fine-grained (avg 35 skills/job) while LLM extracts ~12 skills/job.

2. **Annotation style mismatch**: SkillSpan marks skill *mentions* (fragments) while LLM extracts complete skill *phrases*. This inherently limits recall.

3. **Model self-limiting**: The model caps itself at ~12-13 skills regardless of upper bound, meaning ranges above 5-15 have minimal practical effect.

### Recommendation

**Use 5-15 skill range for production** because:
- Semantic F1 fully plateaus at 5-15 (no gain from 5-20)
- Hard F1 gain from 5-15 → 5-20 is negligible (+0.6%)
- The model rarely extracts beyond 13 skills anyway
- Slightly lower P@K than 5-10 but significantly better recall

If your use case prioritizes precision over recall (e.g., strict matching), consider 5-10.

---

## Reproducibility Verification (Round 2)

To ensure result consistency, both configurations were run twice with identical settings.

### Round 1 vs Round 2 Comparison

#### Skills 5-10

| Metric | Round 1 | Round 2 | Match |
|--------|---------|---------|-------|
| Hard F1 @0.60 | 47.1% | 47.1% | 100% |
| True Positives | 2,852 | 2,852 | 100% |
| False Positives | 105 | 105 | 100% |
| Semantic F1 (Micro) | 77.4% | 77.4% | 100% |
| Semantic F1 (Macro) | 75.7% | 75.7% | 100% |
| Total LLM Skills | 2,957 | 2,957 | 100% |
| P@5 | 97.5% | 97.5% | 100% |
| P@10 | 96.0% | 96.0% | 100% |
| P@15 | 95.8% | 95.8% | 100% |

#### Skills 5-15

| Metric | Round 1 | Round 2 | Match |
|--------|---------|---------|-------|
| Hard F1 @0.60 | 50.1% | 50.1% | 100% |
| True Positives | 3,107 | 3,107 | 100% |
| False Positives | 136 | 136 | 100% |
| Semantic F1 (Micro) | 78.0% | 78.0% | 100% |
| Semantic F1 (Macro) | 76.8% | 76.8% | 100% |
| Total LLM Skills | 3,243 | 3,243 | 100% |
| P@5 | 97.2% | 97.2% | 100% |
| P@10 | 95.5% | 95.5% | 100% |
| P@15 | 95.1% | 95.1% | 100% |

### Consistency Summary

| Config | Round 1 F1 | Round 2 F1 | Status |
|--------|------------|------------|--------|
| 5-10 | 47.1% | 47.1% | **100% Reproducible** |
| 5-15 | 50.1% | 50.1% | **100% Reproducible** |

### Why Results Are Identical

1. **Temperature = 0**: Deterministic output from the LLM
2. **Same model weights**: qwen3:14b loaded identically
3. **Same prompt**: No variation in extraction instructions
4. **Same dataset**: SkillSpan benchmark is fixed

**Conclusion:** The extraction pipeline produces **stable, reproducible results** suitable for production deployment.

---

## File References

| File | Description |
|------|-------------|
| `skill5-10.json` | Benchmark results with 5-10 skill range (Round 1) |
| `skill5-15.json` | Benchmark results with 5-15 skill range (Round 1) |
| `skill5-20.json` | Benchmark results with 5-20 skill range |
| `skill5-10_round2.json` | Benchmark results with 5-10 skill range (Round 2) |
| `skill5-15_round2.json` | Benchmark results with 5-15 skill range (Round 2) |
| `skill_range_comparison_qwen3_14b.md` | This comparison report |

---

## Appendix: Configuration

```
Model: qwen3:14b
Similarity Model: jjzha/jobbert-base-cased
Dataset: SkillSpan (jjzha/skillspan)
Split: all (train + validation + test)
Jobs: 263
Temperature: 0
Context Window: 32768
```
