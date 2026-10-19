"""Score the 200-pair human annotation sheet with OUR method (the instrument).

For every pair on the printed sheet (frozen in
compare/annotation/human_sheet_200_master.json) compute the instrument's
verdict so it can be compared against the teacher's 0-10 scores:

    cosine(skill, clo)  ->  Yes    (>= covered threshold)
                            Partly (>= borderline threshold)
                            No     (<  borderline threshold)

Thresholds are ENCODER-SPECIFIC and come from phase5_curriculum_alignment/03_engine/pipeline.py
(BGE: covered 0.73 / borderline 0.69; MiniLM: 0.50 / 0.42).

Usage:
    python3 score_method.py                              # default BAAI/bge-base-en-v1.5
    python3 score_method.py --model all-MiniLM-L6-v2     # re-run with another model
    python3 score_method.py --model X --covered 0.6 --borderline 0.5   # custom thresholds

Outputs (next to this script):
    method_scores_<model>.json   per-row verdicts + summary
    method_scores_<model>.csv    row,skill,clo,course,cos,verdict  (join with teacher sheet by row #)
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE6 = HERE.parents[1] / "phase5_curriculum_alignment"
MASTER = PHASE6 / "05_evaluation" / "annotation" / "human_sheet_200_master.json"

sys.path.insert(0, str(PHASE6 / "03_engine"))
from pipeline import MODEL_NAME, THRESHOLDS, load_model  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--covered", type=float, default=None)
    ap.add_argument("--borderline", type=float, default=None)
    args = ap.parse_args()

    th = THRESHOLDS.get(args.model, {})
    covered = args.covered if args.covered is not None else th.get("covered")
    borderline = args.borderline if args.borderline is not None else th.get("borderline")
    if covered is None or borderline is None:
        ap.error(f"no calibrated thresholds for '{args.model}' - pass --covered and --borderline")

    pairs = json.load(open(MASTER))["pairs"]
    print(f"model={args.model}  covered>={covered}  borderline>={borderline}  pairs={len(pairs)}")

    model = load_model(args.model)
    skills = sorted({p["skill"] for p in pairs})
    clos = sorted({p["clo"] for p in pairs})
    s_emb = model.encode(skills, normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(clos, normalize_embeddings=True, show_progress_bar=False)
    s_ix = {s: i for i, s in enumerate(skills)}
    c_ix = {c: i for i, c in enumerate(clos)}

    rows = []
    for p in pairs:
        cos = float(s_emb[s_ix[p["skill"]]] @ c_emb[c_ix[p["clo"]]])
        verdict = "Yes" if cos >= covered else "Partly" if cos >= borderline else "No"
        rows.append({
            "row": p["row"], "skill": p["skill"], "clo": p["clo"], "course": p["course"],
            "provenance": p["provenance"], "band": p.get("band"),
            "cos": round(cos, 4), "verdict": verdict,
            "master_cos": p.get("cos"),  # sanity: stored value from sample_150 (same encoder)
        })

    # sanity check against the cosines frozen in the master (sample_150 rows)
    drift = [r for r in rows
             if r["master_cos"] is not None and abs(r["cos"] - r["master_cos"]) > 0.01]
    verdicts = Counter(r["verdict"] for r in rows)
    by_band = {}
    for r in rows:
        by_band.setdefault(r["band"] or r["provenance"], Counter())[r["verdict"]] += 1

    summary = {
        "model": args.model,
        "thresholds": {"covered": covered, "borderline": borderline},
        "n_pairs": len(rows),
        "verdicts": dict(verdicts),
        "verdict_pct": {k: round(100 * v / len(rows), 1) for k, v in verdicts.items()},
        "mean_cos": round(sum(r["cos"] for r in rows) / len(rows), 4),
        "sanity_drift_vs_master": len(drift),
        "by_band": {b: dict(c) for b, c in sorted(by_band.items())},
    }

    out_json = HERE / f"method_scores_{slug(args.model)}.json"
    out_csv = HERE / f"method_scores_{slug(args.model)}.csv"
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "skill", "clo", "course", "cos", "verdict"])
        for r in rows:
            w.writerow([r["row"], r["skill"], r["clo"], r["course"], r["cos"], r["verdict"]])

    print(f"\nwrote {out_json.name} + {out_csv.name}\n")
    print(f"verdicts : {dict(verdicts)}  ({summary['verdict_pct']})")
    print(f"mean cos : {summary['mean_cos']}")
    print(f"sanity   : {len(drift)} rows drift >0.01 vs master cosines"
          + ("" if not drift else f"  e.g. {drift[:3]}"))
    print("\nper band / provenance:")
    for b, c in sorted(by_band.items()):
        print(f"  {b:18} {dict(c)}")


if __name__ == "__main__":
    main()
