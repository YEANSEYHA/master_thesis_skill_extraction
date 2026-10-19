# Skill Range Comparison: 5-10 vs 10-15

**Model:** llama3.1:8b
**Dataset:** SkillSpan (263 jobs, all splits)
**Similarity Model:** jjzha/jobbert-base-cased
**Date:** 2026-03-30

---

## Metric 1: Hard F1 (Multi-Threshold, Micro-Averaged)

| Threshold | Precision (5-10) | Precision (10-15) | Recall (5-10) | Recall (10-15) | F1 (5-10) | F1 (10-15) |
|-----------|------------------|--------------------|---------------|----------------|-----------|------------|
| @ 0.60    | 98.1%            | 97.8%              | 18.9%         | 19.7%          | 31.7%     | **32.8%**  |
| @ 0.65    | 95.4%            | 95.1%              | 18.4%         | 19.1%          | 30.8%     | **31.8%**  |
| @ 0.70    | 89.8%            | 89.3%              | 17.3%         | 17.9%          | 29.0%     | **29.9%**  |
| @ 0.75    | 82.0%            | 82.0%              | 15.8%         | 16.5%          | 26.5%     | **27.5%**  |
| @ 0.80    | 74.3%            | 73.6%              | 14.3%         | 14.8%          | 24.0%     | **24.7%**  |

## Metric 2: Semantic F1

| Metric              | 5-10   | 10-15  |
|---------------------|--------|--------|
| Total LLM skills    | 1764   | 1843   |
| Total Human skills  | 9166   | 9166   |
| Micro Sem-Precision | **87.7%** | 87.4%  |
| Micro Sem-Recall    | 65.6%  | **65.7%** |
| Micro Sem-F1        | **75.1%** | 75.0%  |
| Macro Sem-F1        | **72.2%** | 71.8%  |

## Metric 3: Precision@K (threshold=0.65)

| K  | Micro P@K (5-10) | Micro P@K (10-15) | Macro P@K (5-10) | Macro P@K (10-15) |
|----|-------------------|--------------------|-------------------|--------------------|
| 5  | **97.7%**         | 97.1%              | **92.9%**         | 91.6%              |
| 10 | **97.5%**         | 96.8%              | **92.5%**         | 91.1%              |
| 15 | **97.6%**         | 96.9%              | **92.5%**         | 91.2%              |

---

## Key Takeaways

1. **Hard F1:** 10-15 wins across all thresholds (+0.7~1.0% F1) due to higher recall from extracting more skills (1843 vs 1764), while precision stays nearly identical.
2. **Semantic F1:** Nearly identical. 5-10 has a marginal edge in precision and macro F1, but the difference is negligible (~0.3%).
3. **Precision@K:** 5-10 is slightly better (~0.7-1.3% higher), meaning the smaller range produces slightly more precise top-K predictions.
4. **Overall:** The two ranges perform very similarly. 10-15 trades a tiny precision drop for better recall. For SkillSpan (avg ~35 human skills per job), both ranges extract far fewer skills than the ground truth, so recall remains the bottleneck regardless.

---

## Run Details

| Run       | Skills Range | Output JSON                                                        |
|-----------|--------------|--------------------------------------------------------------------|
| Run 1     | 5-10         | `outputs/skillspan_benchmark/20260330_142620/skillspan_benchmark.json` |
| Run 2     | 10-15        | `outputs/skillspan_benchmark/20260330_150858/skillspan_benchmark.json` |
