"""Build category_vocab.json — per-CATEGORY skill demand from the raw LinkedIn corpus.

The 1.35M-posting corpus (job_postings_categorized.jsonl) tags every posting with
one of 17 coarse job_category buckets (it, health, finance, …). This script
streams the file once and aggregates raw skill frequencies per category:

    {category: {"n_postings": N, "demand": {skill_lower: freq}}}   (top TOP_N skills)

The output feeds:
  - GET /api/curricula/v3/categories  → the Compare-against chip list in the UI
  - recommend._core()                 → top-K market core when a selected name is
                                        a category (fallback from canonical_vocab)

Run:  python build_category_vocab.py [path/to/job_postings_categorized.jsonl]
"""

import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_SRC = _HERE.parent.parent / "playground" / "data" / "job_postings_categorized.jsonl"
_OUT = _HERE / "category_vocab.json"

TOP_N = 3000          # skills kept per category (freq-ranked)
MAX_SKILL_LEN = 60    # drop extraction garbage (sentences masquerading as skills)


def build(src: Path = _DEFAULT_SRC, out: Path = _OUT) -> dict:
    counts: dict[str, Counter] = {}
    postings: Counter = Counter()
    total = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            cat = (row.get("job_category") or "").strip()
            if not cat:
                continue
            postings[cat] += 1
            c = counts.setdefault(cat, Counter())
            for s in row.get("skills") or []:
                s = str(s).strip().lower()
                if s and len(s) <= MAX_SKILL_LEN:
                    c[s] += 1
            if total % 200_000 == 0:
                print(f"  … {total:,} postings", flush=True)

    vocab = {
        cat: {
            "n_postings": postings[cat],
            "demand": dict(counts[cat].most_common(TOP_N)),
        }
        for cat in sorted(counts, key=lambda k: -postings[k])
    }
    out.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"{total:,} postings → {len(vocab)} categories → {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return vocab


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SRC
    build(src)
