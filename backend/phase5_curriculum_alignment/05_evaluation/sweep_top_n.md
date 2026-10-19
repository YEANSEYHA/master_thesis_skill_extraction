# Sensitivity of the results to the top-N comparison set

Encoder `BAAI/bge-base-en-v1.5` · corpus `stage2_result.json` (254 technology postings, 574 recurrent technical skills) · thresholds covered 0.7368 / borderline 0.6903.

Only the cut-off rank N varies; the encoder, market, curriculum, thresholds and selection rule are those of the locked runs. The N=30 slice reproduces all four locked runs exactly (checked at runtime).


## Per-configuration results by N

| Cohort | N | Covered | Coverage rate | Demand-weighted coverage | Half-credit | Demand mass in top-N | Postings at rank N | Recommended additions | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y3 | 10 | 5/10 | 50.0% | 57.4% | 61.7% | 12.2% | 27 | 4 | 0.3810 |
| Y3 | 20 | 7/20 | 35.0% | 43.3% | 49.6% | 19.8% | 18 | 9 | 0.3810 |
| Y3 | 30 | 12/30 | 40.0% | 44.9% | 52.2% | 25.9% | 16 | 12 | 0.3810 |
| Y3 | 40 | 15/40 | 37.5% | 42.4% | 49.3% | 31.0% | 13 | 17 | 0.3810 |
| Y3 | 50 | 18/50 | 36.0% | 40.9% | 48.9% | 35.3% | 12 | 21 | 0.3810 |
| Y4 | 10 | 4/10 | 40.0% | 39.7% | 53.1% | 12.2% | 27 | 4 | 0.3771 |
| Y4 | 20 | 11/20 | 55.0% | 51.9% | 62.3% | 19.8% | 18 | 6 | 0.3771 |
| Y4 | 30 | 16/30 | 53.3% | 51.4% | 62.8% | 25.9% | 16 | 10 | 0.3771 |
| Y4 | 40 | 20/40 | 50.0% | 49.7% | 60.8% | 31.0% | 13 | 13 | 0.3771 |
| Y4 | 50 | 25/50 | 50.0% | 49.9% | 60.8% | 35.3% | 12 | 16 | 0.3771 |
| Y5 | 10 | 5/10 | 50.0% | 52.9% | 61.6% | 12.2% | 27 | 4 | 0.3768 |
| Y5 | 20 | 11/20 | 55.0% | 55.5% | 63.2% | 19.8% | 18 | 6 | 0.3768 |
| Y5 | 30 | 17/30 | 56.7% | 56.3% | 64.6% | 25.9% | 16 | 9 | 0.3768 |
| Y5 | 40 | 22/40 | 55.0% | 55.4% | 63.9% | 31.0% | 13 | 12 | 0.3768 |
| Y5 | 50 | 27/50 | 54.0% | 54.7% | 64.0% | 35.3% | 12 | 15 | 0.3768 |
| Y3+Y4+Y5 | 10 | 8/10 | 80.0% | 84.6% | 92.3% | 12.2% | 27 | 2 | 0.3689 |
| Y3+Y4+Y5 | 20 | 17/20 | 85.0% | 87.1% | 91.9% | 19.8% | 18 | 3 | 0.3689 |
| Y3+Y4+Y5 | 30 | 25/30 | 83.3% | 85.3% | 88.9% | 25.9% | 16 | 4 | 0.3689 |
| Y3+Y4+Y5 | 40 | 30/40 | 75.0% | 79.7% | 84.4% | 31.0% | 13 | 7 | 0.3689 |
| Y3+Y4+Y5 | 50 | 38/50 | 76.0% | 79.8% | 84.5% | 35.3% | 12 | 8 | 0.3689 |

## What the comparison set buys

| N | Share of technical demand in the top-N | Added vs previous N | Postings supporting the skill at rank N |
|---:|---:|---:|---:|
| 10 | 12.2% | +12.2% | 27 |
| 20 | 19.8% | +7.6% | 18 |
| 30 | 25.9% | +6.1% | 16 |
| 40 | 31.0% | +5.0% | 13 |
| 50 | 35.3% | +4.3% | 12 |

## Stability of the conclusions

- KL divergence is identical at every N (`True`) — it is computed over the full technical segment, not over the top-N, so it cannot be used to argue for any particular N.
- Cohort ordering (Y3 <= Y4 <= Y5 <= programme):

| N | Demand-weighted coverage | Unweighted coverage rate | Half-credit |
|---:|---|---|---|
| 10 | False | False | False |
| 20 | True | True | True |
| 30 | True | True | True |
| 40 | True | True | True |
| 50 | True | True | True |

- The ordering holds on all three measures at N = [20, 30, 40, 50].
- Demand-weighted coverage varies across N by Y3 16.6 pt, Y4 12.2 pt, Y5 3.4 pt, Y3+Y4+Y5 7.4 pt.

## Text for the manuscript

The comparison set was fixed at the thirty most frequently demanded
technical skills. To establish that the reported findings are not an
artefact of that choice, the complete analysis was repeated with the
comparison set truncated at 10, 20, 30, 40 and 50 skills, holding the
encoder, the labour-market segment, the curriculum inventories, the
human-calibrated thresholds and the recommendation rule constant.

Three observations follow. First, the ranking of the curriculum
configurations is stable: Year 3 < Year 4 < Year 5 < complete programme
holds on unweighted coverage, demand-weighted coverage and the
half-credit measure for every comparison set of twenty skills or more,
and programme-level demand-weighted coverage remains between 79.7% and
87.1% throughout. Second, the ranking does not hold at ten skills, where
Year 3 (57.4%) exceeds Year 4 (39.7%). At that width the comparison set
consists mainly of foundational competencies already taught in Year 3,
and a single skill accounts for ten percentage points of the unweighted
rate, so the measure reflects the composition of a very small set rather
than curriculum breadth. Third, widening the set yields diminishing
evidential return: the top ten skills account for 12.2% of recurrent
technical demand and the top thirty for 25.9%, but each further increment
adds less demand while admitting progressively rarer skills, from sixteen
supporting postings at rank thirty to twelve at rank fifty.

Thirty was therefore retained: it lies inside the range over which the
conclusions are invariant, every skill in the set is supported by at
least sixteen postings (6.3% of the 254 technology-related
advertisements), and it provides a larger candidate pool for the
recommendation stage than a narrower set. Kullback-Leibler divergence is
unchanged at every value of N, since it is computed over all 574
recurrent technical skills rather than over the comparison set.

