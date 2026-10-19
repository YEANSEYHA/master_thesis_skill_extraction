"""
Kleinberg Burst Detection — reference implementation in Python.

Reference:
    Kleinberg, J. (2002). Bursty and Hierarchical Structure in Streams.
    KDD '02. https://www.cs.cornell.edu/home/kleinber/bhs.pdf

Applied to job-skill streams (Börner et al., PNAS 2018):
    Skill discrepancies between research, education, and jobs reveal
    the critical need to supply soft skills for the data economy.

Algorithm:
    For a stream of timestamped events (one per skill mention), model
    the gaps between consecutive events as a 2-state HMM:
        State 0 — base rate λ₀ (events arrive at the long-run mean rate)
        State 1 — burst rate s·λ₀ (events arrive `s` times faster)
    Switching states costs γ·log(n) (penalty for noisy flips).
    Viterbi finds the globally optimal state sequence.

This implementation matches the JS port in
    frontend/src/app/components/TrendingCompare.js
so the two stay numerically aligned.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ---------- core algorithm ----------

def kleinberg_burst(
    timestamps: Iterable[float],
    s: float = 2.0,
    gamma: float = 1.0,
    min_events: int = 5,
) -> list[dict]:
    """Run 2-state Kleinberg burst detection on one timestamp stream.

    Args:
        timestamps:  iterable of seconds-since-epoch values for each event.
        s:           burst-state rate multiplier (state 1 rate = s · base rate).
        gamma:       state-transition cost coefficient.
        min_events:  skip streams shorter than this.

    Returns:
        List of bursts: [{"start_ts", "end_ts", "n_events", "ongoing"}, ...]
    """
    sorted_ts = sorted(timestamps)
    n = len(sorted_ts)
    if n < min_events:
        return []

    # gaps between consecutive events, in days
    gaps = []
    for i in range(1, n):
        dt = (sorted_ts[i] - sorted_ts[i - 1]) / 86400
        gaps.append(max(dt, 0.001))  # avoid log(0)

    total_days = (sorted_ts[-1] - sorted_ts[0]) / 86400
    if total_days <= 0:
        return []

    lambda0   = (n - 1) / total_days   # base rate
    lambda1   = lambda0 * s            # burst rate
    trans_cost = gamma * math.log(n)

    def emit(state: int, gap: float) -> float:
        # -log(λ * exp(-λg)) = -log(λ) + λg
        lam = lambda0 if state == 0 else lambda1
        return -math.log(lam) + lam * gap

    # Viterbi DP
    INF = float("inf")
    dp   = [[0.0, INF]]
    back = [[None, None]]
    for g in gaps:
        cur = [INF, INF]
        bk  = [None, None]
        for st in (0, 1):
            for prev in (0, 1):
                c = dp[-1][prev] + (0 if prev == st else trans_cost) + emit(st, g)
                if c < cur[st]:
                    cur[st] = c
                    bk[st]  = prev
        dp.append(cur)
        back.append(bk)

    # backtrace
    last = 0 if dp[-1][0] < dp[-1][1] else 1
    states = [0] * len(gaps)
    cur = last
    for i in range(len(gaps), 0, -1):
        states[i - 1] = cur
        cur = back[i][cur]

    # collect burst intervals
    bursts: list[dict] = []
    bs: int | None = None
    for i, st in enumerate(states):
        if st == 1 and bs is None:
            bs = i + 1
        elif st == 0 and bs is not None:
            bursts.append({
                "start_ts": sorted_ts[bs],
                "end_ts":   sorted_ts[i],
                "n_events": i - bs + 1,
                "ongoing":  False,
            })
            bs = None
    if bs is not None:
        bursts.append({
            "start_ts": sorted_ts[bs],
            "end_ts":   sorted_ts[-1],
            "n_events": n - bs,
            "ongoing":  True,
        })
    return bursts


# ---------- date parsing (handles multiple formats) ----------

_DATE_FORMATS = (
    "%Y-%m-%d",        # 2026-02-20
    "%d-%m-%Y",        # 20-02-2026
    "%d-%b-%Y",        # 20-Feb-2026
    "%d-%B-%Y",        # 20-February-2026
    "%Y/%m/%d",        # 2026/02/20
    "%d/%m/%Y",        # 20/02/2026
)

def _parse_date(value) -> float | None:
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


# ---------- driver ----------

def detect_trending_skills(
    jobs: list[dict],
    top_n: int = 15,
    min_occurrences: int = 10,
    s: float = 2.0,
    gamma: float = 1.0,
) -> list[dict]:
    """Apply Kleinberg burst detection to per-skill timestamp streams.

    Args:
        jobs: list of {"skills": [...], "posted_date": "YYYY-MM-DD"}.
        top_n: keep this many most-bursty skills.
        min_occurrences: skip skills with fewer events than this.
    """
    streams: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for job in jobs:
        ts = _parse_date(job.get("posted_date"))
        if ts is None:
            skipped += 1
            continue
        for skill in set(job.get("skills", [])):
            streams[skill].append(ts)
    if skipped:
        print(f"[kleinberg] skipped {skipped}/{len(jobs)} jobs without parseable posted_date")

    results = []
    for skill, ts_list in streams.items():
        if len(ts_list) < min_occurrences:
            continue
        bursts = kleinberg_burst(ts_list, s=s, gamma=gamma)
        if not bursts:
            continue

        # headline burst = highest event count, weighted by ongoing
        headline = max(
            bursts,
            key=lambda b: b["n_events"] * (1.5 if b["ongoing"] else 1.0),
        )
        span_days = (headline["end_ts"] - headline["start_ts"]) / 86400

        results.append({
            "skill":           skill,
            "total_events":    len(ts_list),
            "bursts":          bursts,
            "headline_start":  datetime.fromtimestamp(headline["start_ts"]).strftime("%Y-%m-%d"),
            "headline_end":    datetime.fromtimestamp(headline["end_ts"]).strftime("%Y-%m-%d"),
            "headline_days":   round(span_days),
            "headline_events": headline["n_events"],
            "ongoing":         headline["ongoing"],
            "intensity":       round(headline["n_events"] / span_days, 2) if span_days > 0 else headline["n_events"],
        })

    # rank: ongoing first, then by event count
    results.sort(key=lambda r: (not r["ongoing"], -r["headline_events"]))
    return results[:top_n]


# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python kleinberg_burst.py <stage2_result.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    jobs = json.loads(path.read_text())
    if isinstance(jobs, list) and jobs and "output" in jobs[0]:
        # stage2_result.json has wrapper objects: [{ "input": ..., "output": {...} }, ...]
        jobs = [j["output"] for j in jobs if "output" in j]

    results = detect_trending_skills(jobs, top_n=15, min_occurrences=10)

    print(f"Detected {len(results)} bursty skills (top 15)\n")
    for i, r in enumerate(results, 1):
        marker = "●" if r["ongoing"] else " "
        print(f"  {i:2d}. {marker} {r['skill']:<35} "
              f"{r['headline_events']:>3} events over {r['headline_days']:>3} days  "
              f"[{r['headline_start']} → {r['headline_end']}]")
