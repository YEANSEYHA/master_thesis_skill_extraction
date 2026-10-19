"""
Trim the full category_inventory.json to top-N skills per category.

The full inventory keeps the entire long tail (~588K skills in the IT bucket)
for reference; the matcher only ever uses the demand HEAD (top-N by frequency).
This writes a tiny top-N cache the API can load cheaply per request — the full
file is left untouched on disk.

Usage:
  python trim_inventory.py                       # top-500, default paths
  python trim_inventory.py --top-n 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compute_idf(categories: dict) -> dict[str, float]:
    """Cross-category IDF: down-weight skills that appear in many categories.

    df(skill) = number of categories the skill appears in (full inventory).
    idf(skill) = ln((1 + N) / (1 + df)) + 1   (sklearn-style smoothing — never 0).

    A skill in ALL categories (e.g. "communication") → idf ≈ 1 (suppressed);
    a skill in 1-2 categories (e.g. "python") → idf high (boosted). Following the
    TF-IDF / skill-specificity literature (Skills-First; Shi et al. 2020).
    """
    n = len(categories)
    df: dict[str, int] = {}
    for skills in categories.values():
        for s in skills:
            df[s] = df.get(s, 0) + 1
    return {s: round(math.log((1 + n) / (1 + d)) + 1.0, 4) for s, d in df.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input",  default="../linkedin_job_posting/category_inventory.json")
    p.add_argument("--output", default="../linkedin_job_posting/category_inventory_topN.json")
    p.add_argument("--top-n", type=int, default=500,
                   help="skills kept per category (default 500)")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    full = json.loads(in_path.read_text(encoding="utf-8"))

    # `categories[cat]` is a dict already ordered by frequency desc
    # (built with Counter.most_common), so the first N items ARE the top-N.
    trimmed = {
        cat: dict(list(skills.items())[:args.top_n])
        for cat, skills in full["categories"].items()
    }

    # Cross-category IDF over the TRIMMED inventory: df = number of categories
    # where the skill is PROMINENT (in the top-N). Computing df on the full
    # inventory saturates (every skill appears somewhere in 1.3M postings), so
    # IDF must be based on prominence, not raw presence.
    idf_trim = compute_idf(trimmed)

    out = {
        "categories": trimmed,
        "idf": idf_trim,
        "category_job_counts": full.get("category_job_counts", {}),
        "total_jobs": full.get("total_jobs"),
        "meta": {**full.get("meta", {}), "top_n_per_category": args.top_n,
                 "idf": "cross-category smoothed: ln((1+N)/(1+df))+1"},
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # report
    full_mb = in_path.stat().st_size / 1e6
    trim_kb = out_path.stat().st_size / 1e3
    print(f"full inventory : {full_mb:,.0f} MB  ({in_path.name})")
    print(f"trimmed (top-{args.top_n}) : {trim_kb:,.0f} KB  ({out_path.name})")
    print(f"categories: {len(trimmed)} · total_jobs: {full.get('total_jobs'):,}")
    if "it" in trimmed:
        it = trimmed["it"]
        print(f"\nIT kept {len(it)} skills")
        print(f"{'RAW frequency':<34}{'TF-IDF (freq × idf)':<34}")
        raw_top = list(it.items())[:10]
        tfidf_top = sorted(it.items(), key=lambda kv: kv[1] * idf_trim.get(kv[0], 1.0),
                           reverse=True)[:10]
        for (rs, rn), (ts, tn) in zip(raw_top, tfidf_top):
            w = int(tn * idf_trim.get(ts, 1.0))
            print(f"  {rn:>6,}  {rs:<24}  {w:>9,}  {ts}")


if __name__ == "__main__":
    main()
