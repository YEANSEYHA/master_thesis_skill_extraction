"""
Skill aggregation — produces the "Skills SOUGHT" inventory.

Reads all Phase-3 stage2_result.json outputs (LLM-extracted skills from
Cambodian job postings) and produces a frequency-weighted inventory:

    { "skill_label": frequency, ... }

This is INPUT 1 to the Phase-5 matcher.

CLI usage:
    python aggregate_skills.py                    # latest run, top-50
    python aggregate_skills.py --top_n 100        # different cutoff
    python aggregate_skills.py --all_runs         # merge ALL stage2 runs
    python aggregate_skills.py --output job_freq.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


# Paths relative to repo root (…/phase5_curricula/clo_skill_alignment → repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_STAGE2_ROOT = _REPO_ROOT / "backend" / "phase3_skill_extraction" / "outputs" / "stage2"


# ---------- I/O helpers ----------

def find_stage2_files(use_all_runs: bool = False) -> list[Path]:
    """Locate stage2_result.json files. By default returns latest run only."""
    all_files = sorted(_STAGE2_ROOT.glob("*/stage2_result.json"))
    if not all_files:
        raise FileNotFoundError(
            f"No stage2_result.json files found under {_STAGE2_ROOT}"
        )
    return all_files if use_all_runs else [all_files[-1]]


def load_records(path: Path) -> list[dict]:
    """Load a stage2_result.json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- skill normalization ----------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_skill(s: str) -> str:
    """Lowercase, strip whitespace, remove duplicate spaces."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def is_valid_skill(s: str) -> bool:
    """Filter obvious garbage: empty, too short, single non-letter char."""
    if not s:
        return False
    if len(s) < 2:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return True


# ---------- aggregation ----------

def aggregate(records_iter: Iterable[dict]) -> Counter:
    """Count skill frequencies across all postings."""
    counter: Counter = Counter()
    for rec in records_iter:
        out = rec.get("output", {}) or {}
        skills = out.get("skills", []) or []
        for raw in skills:
            norm = normalize_skill(raw)
            if is_valid_skill(norm):
                counter[norm] += 1
    return counter


def to_sorted_dict(counter: Counter, top_n: int | None) -> dict[str, int]:
    """Return a dict ordered by frequency desc (insertion order)."""
    items = counter.most_common(top_n)
    return {skill: freq for skill, freq in items}


# ---------- summary printer ----------

def print_summary(counter: Counter, top_n: int) -> None:
    total_skills = sum(counter.values())
    unique_skills = len(counter)
    top = counter.most_common(top_n)

    print()
    print("=" * 70)
    print(f"AGGREGATION SUMMARY")
    print("=" * 70)
    print(f"  Total skill mentions  : {total_skills}")
    print(f"  Unique skills         : {unique_skills}")
    print(f"  Average / skill       : {total_skills / unique_skills:.1f}")
    print()
    print(f"TOP {len(top)} SKILLS")
    print("-" * 70)
    for i, (skill, freq) in enumerate(top, 1):
        bar_len = int(50 * freq / top[0][1])
        bar = "█" * bar_len
        print(f"  {i:>3}. {skill:<40} {freq:>5}  {bar}")


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--top_n", type=int, default=50,
                   help="Keep top-N skills (default 50). Use 0 for all.")
    p.add_argument("--all_runs", action="store_true",
                   help="Merge ALL stage2 runs (default: latest run only)")
    p.add_argument("--output", default="job_freq.json",
                   help="Output JSON path (default: job_freq.json in CWD)")
    p.add_argument("--verbose", action="store_true",
                   help="Show the full top-N table on stdout")
    args = p.parse_args()

    print(f"[aggregate] looking for stage2 files under {_STAGE2_ROOT}")
    files = find_stage2_files(use_all_runs=args.all_runs)
    print(f"[aggregate] using {len(files)} file(s):")
    for f in files:
        print(f"  • {f.relative_to(_REPO_ROOT)}")

    all_records: list[dict] = []
    for f in files:
        records = load_records(f)
        print(f"  → {len(records)} records from {f.parent.name}")
        all_records.extend(records)

    print(f"[aggregate] total records: {len(all_records)}")
    counter = aggregate(all_records)

    if args.verbose or args.top_n <= 30:
        print_summary(counter, top_n=min(args.top_n or 30, 30))

    top_n = None if args.top_n == 0 else args.top_n
    result = to_sorted_dict(counter, top_n)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    out_path = Path(args.output).resolve()
    print()
    print(f"✅ Wrote {len(result)} skills to {out_path}")


if __name__ == "__main__":
    main()
