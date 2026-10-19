# Sensitivity of the benchmark to the JobBERT match threshold

Referee `jjzha/jobbert-base-cased`; predictions unchanged, only the match rule varies. Greedy one-to-one matching, top-15 predictions per posting.


| Threshold | Qwen3-14B F1 | Qwen3 prec. | Qwen3 recall | KeyBERT F1 | KeyBERT prec. | KeyBERT recall | Margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 51.9% | 99.4% | 35.2% | 57.1% | 95.0% | 40.9% | -5.2 pts |
| 0.50 | 51.8% | 99.0% | 35.0% | 55.1% | 91.5% | 39.4% | -3.3 pts |
| 0.55 | 51.2% | 97.9% | 34.6% | 49.1% | 81.6% | 35.1% | +2.1 pts |
| 0.60 **(adopted)** | 50.1% | 95.8% | 33.9% | 39.2% | 65.2% | 28.1% | +10.8 pts |
| 0.65 | 47.8% | 91.4% | 32.3% | 28.4% | 47.2% | 20.3% | +19.3 pts |
| 0.70 | 43.7% | 83.6% | 29.6% | 19.5% | 32.4% | 14.0% | +24.2 pts |
| 0.75 | 39.5% | 75.6% | 26.7% | 12.6% | 21.0% | 9.0% | +26.9 pts |
| 0.80 | 34.8% | 66.5% | 23.5% | 8.2% | 13.7% | 5.9% | +26.5 pts |
