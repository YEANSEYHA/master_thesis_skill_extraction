"""Shape-matched external benchmark: SkillMatch-1K (Techwolf, MIT).

1,000 related + 1,000 unrelated SHORT skill pairs (mean 3.2 words/pair)
mined from job ads -- the same text SHAPE as our skill x CLO pairs, but
a WEAKER relation ("related", e.g. grasping~pinching) than our
"covered/taught" (near-identity).

Purpose (diagnostic, claim scope limited):
  a) test the frozen instrument's behaviour on short x short pairs
  b) locate the distribution's own split point t* (fitted on this set
     directly -- a diagnostic, NOT a performance claim)
  c) third external data point for "cutoffs are task+distribution
     specific; the frozen 0.73 is a near-zero-false-positive operating
     point everywhere tested"

Usage: python3 05_evaluation/bench_skillmatch.py
Output: printed report + 05_evaluation/bench_skillmatch_results.json
Data: 05_evaluation/skillmatch_data/test.csv (vendored, MIT license)
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "03_engine"))   # engine module lives there

from pipeline import MODEL_NAME, get_thresholds, load_model  # noqa: E402

DATA = HERE / "skillmatch_data" / "test.csv"
RESULTS_FILE = HERE / "bench_skillmatch_results.json"
GRID = np.arange(0.30, 0.951, 0.005)


def grade(scores, gold, t):
    pred = scores >= t
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    tn = int((~pred & ~gold).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    return {"threshold": round(float(t), 3), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "macro_f1": round((f1 + f1n) / 2, 3)}


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open(DATA)))
    a = [r["skill_a"] for r in rows]
    b = [r["skill_b"] for r in rows]
    gold = np.array([r["related"] == "True" for r in rows])
    words = np.array([len((x + " " + y).split()) for x, y in zip(a, b)])
    print(f"SkillMatch-1K: {len(rows)} pairs "
          f"({int(gold.sum())} related / {int((~gold).sum())} not), "
          f"mean {words.mean():.1f} words/pair")

    thr = get_thresholds()
    model = load_model()
    A = model.encode(a, normalize_embeddings=True, batch_size=128,
                     show_progress_bar=False)
    B = model.encode(b, normalize_embeddings=True, batch_size=128,
                     show_progress_bar=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cos = (A * B).sum(axis=1)

    frozen = grade(cos, gold, thr["covered"])
    print(f"\nfrozen covered={thr['covered']} ({MODEL_NAME}):")
    print(f"  P={frozen['precision']:.2f} R={frozen['recall']:.2f} "
          f"macroF1={frozen['macro_f1']:.2f} "
          f"[tp={frozen['tp']} fp={frozen['fp']} fn={frozen['fn']} "
          f"tn={frozen['tn']}]")

    t_star = max(GRID, key=lambda t: grade(cos, gold, t)["macro_f1"])
    fitted = grade(cos, gold, t_star)
    print(f"\ndiagnostic split point (fitted on this set -- not a "
          f"performance claim):")
    print(f"  t*={t_star:.3f}  macroF1={fitted['macro_f1']:.2f} "
          f"P={fitted['precision']:.2f} R={fitted['recall']:.2f}")
    print(f"  medians: related={np.median(cos[gold]):.2f}  "
          f"unrelated={np.median(cos[~gold]):.2f}")

    results = {
        "dataset": "SkillMatch-1K (Techwolf, MIT), test.csv vendored",
        "model": MODEL_NAME,
        "n_pairs": len(rows), "mean_words_per_pair": round(
            float(words.mean()), 1),
        "frozen": frozen, "fitted_diagnostic": fitted,
        "median_related": round(float(np.median(cos[gold])), 3),
        "median_unrelated": round(float(np.median(cos[~gold])), 3),
        "claim_scope": ("Relation is 'related', weaker than our "
                        "'covered'; split point fitted on the full set "
                        "is diagnostic only. Shows frozen 0.73 stays "
                        "near-zero-false-positive on short x short "
                        "pairs; supports task+distribution-specific "
                        "cutoffs."),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=1))
    print(f"\nDone in {time.time() - t0:.0f}s -> {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
