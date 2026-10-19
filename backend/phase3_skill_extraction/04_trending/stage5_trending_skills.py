"""
Stage 5: Trending Skill Detection using Growth Rate / Velocity
==============================================================
Method: Time-series frequency analysis with growth rate classification
Reference: LinkedIn Economic Graph (2019), Dawson et al. (2021)

Input:  job_cards[] with output.skills[] and output.posted_date
Output: Ranked skills by trend status (trending, emerging, declining, stable)
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime


# ============================================================
# STEP 1: Load job cards
# ============================================================

def load_job_cards(file_path):
    """Load job cards from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# STEP 2: Parse posted_date → time bucket (month or quarter)
# ============================================================

def parse_date(date_str):
    """
    Parse date string to datetime. Handles many LLM output formats:
    - DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY
    - "05 February, 2026", "Feb-04-2026", "07 Feb '26"
    - "05th March 2026", "Jan 28, 2026"
    - Skips unparseable like "7 hours ago"
    """
    if not date_str or date_str == "null":
        return None

    import re

    s = date_str.strip()

    # Skip relative dates like "7 hours ago", "2 days ago"
    if re.search(r'\b(ago|today|yesterday|just now)\b', s, re.IGNORECASE):
        return None

    # Remove ordinal suffixes: "05th" → "05", "1st" → "1"
    s = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', s)

    # Try many explicit formats
    formats = [
        "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d",
        "%d %B, %Y", "%d %B %Y", "%B %d, %Y", "%B %d %Y",
        "%d %b, %Y", "%d %b %Y", "%b %d, %Y", "%b %d %Y",
        "%b-%d-%Y", "%b-%d-%y",
        "%d %b '%y", "%d %B '%y",
        "%Y-%m-%d", "%m-%d-%Y",
    ]
    now = datetime.now()
    for fmt in formats:
        try:
            result = datetime.strptime(s, fmt)
            if result > now:  # Skip future dates (likely closing/expiry dates)
                continue
            return result
        except ValueError:
            continue

    # Last resort: dateutil (handles almost anything)
    try:
        from dateutil import parser as dateutil_parser
        result = dateutil_parser.parse(s, dayfirst=True)
    except Exception:
        return None

    # Filter out future dates (likely closing/expiry dates, not posted dates)
    if result and result > datetime.now():
        return None

    return result


def get_quarter(dt):
    """Convert datetime to 'YYYY-Q#' string."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def get_month(dt):
    """Convert datetime to 'YYYY-MM' string."""
    return dt.strftime("%Y-%m")


# ============================================================
# STEP 3: Normalize skills (embedding similarity clustering)
# ============================================================

def build_skill_normalizer(job_cards, similarity_threshold=0.93):
    """
    Auto-cluster similar skill names using embedding cosine similarity.

    1. Collect all unique skills from job cards
    2. Encode with sentence-transformers (JobBERT or all-MiniLM)
    3. Group skills where cosine similarity > threshold
    4. Pick most frequent variant as canonical name

    Returns:
        skill_map: dict mapping every skill variant → canonical name
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Collect all skills and their frequencies
    skill_freq = defaultdict(int)
    for card in job_cards:
        output = card.get("output", card)
        for skill in output.get("skills", []):
            s = skill.strip().lower()
            if s:
                skill_freq[s] += 1

    unique_skills = list(skill_freq.keys())
    print(f"  Found {len(unique_skills)} unique skill variants")

    if len(unique_skills) == 0:
        return {}

    # Encode all skills
    print("  Loading embedding model...")
    model = SentenceTransformer("jjzha/jobbert-base-cased")
    print("  Encoding skills...")
    embeddings = model.encode(unique_skills, show_progress_bar=False, normalize_embeddings=True)

    # Compute cosine similarity matrix (normalized embeddings → dot product)
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Greedy clustering: assign each skill to a cluster
    visited = set()
    skill_map = {}

    # Process skills by frequency (most frequent first → becomes canonical)
    sorted_indices = sorted(range(len(unique_skills)), key=lambda i: skill_freq[unique_skills[i]], reverse=True)

    for idx in sorted_indices:
        if idx in visited:
            continue

        canonical = unique_skills[idx]
        visited.add(idx)
        skill_map[canonical] = canonical

        # Find all similar skills not yet assigned
        for other_idx in range(len(unique_skills)):
            if other_idx in visited:
                continue
            if sim_matrix[idx][other_idx] >= similarity_threshold:
                skill_map[unique_skills[other_idx]] = canonical
                visited.add(other_idx)

    # Report clustering results
    clusters = defaultdict(list)
    for variant, canonical in skill_map.items():
        if variant != canonical:
            clusters[canonical].append(variant)

    merged_count = sum(len(v) for v in clusters.values())
    print(f"  Merged {merged_count} variants into {len(unique_skills) - merged_count} canonical skills")
    if clusters:
        # Show some example merges
        examples = list(clusters.items())[:5]
        for canonical, variants in examples:
            print(f"    '{canonical}' ← {variants}")

    return skill_map


def normalize_skill(skill, skill_map):
    """Normalize skill name using pre-built skill map."""
    s = skill.strip().lower()
    return skill_map.get(s, s)


# ============================================================
# STEP 4: Build skill frequency table per time period
# ============================================================

def build_frequency_table(job_cards, time_bucket="month", skill_map=None):
    """
    Build skill frequency table: {skill: {period: count}}

    Args:
        job_cards: list of job card dicts
        time_bucket: "month" or "quarter"
        skill_map: dict mapping skill variants → canonical names

    Returns:
        freq_table: {skill: {period: count}}
        all_periods: sorted list of all time periods
        skipped: count of cards without valid date
    """
    if skill_map is None:
        skill_map = {}

    bucket_fn = get_month if time_bucket == "month" else get_quarter
    freq_table = defaultdict(lambda: defaultdict(int))
    skipped = 0

    for card in job_cards:
        # Support both flat and nested formats
        if "output" in card:
            output = card["output"]
        else:
            output = card

        date_str = output.get("posted_date")
        skills = output.get("skills", [])

        dt = parse_date(date_str)
        if not dt:
            skipped += 1
            continue

        period = bucket_fn(dt)

        for skill in skills:
            normalized = normalize_skill(skill, skill_map)
            if normalized:  # skip empty
                freq_table[normalized][period] += 1

    # Get all unique periods, sorted
    all_periods = sorted(set(
        p for skill_periods in freq_table.values()
        for p in skill_periods
    ))

    # Filter out the current (incomplete) month — it will always look like a decline
    current_period = bucket_fn(datetime.now())
    if current_period in all_periods and len(all_periods) > 2:
        print(f"  Excluded current incomplete period: {current_period}")
        for skill in freq_table:
            freq_table[skill].pop(current_period, None)
        all_periods = [p for p in all_periods if p != current_period]

    # Filter out outlier periods with too few job cards (< 3 cards in a period)
    # This removes stray dates like a single 2022-03 card among 2025-2026 data
    if len(all_periods) > 2:
        period_card_count = defaultdict(int)
        for card in job_cards:
            output = card.get("output", card)
            dt = parse_date(output.get("posted_date"))
            if dt:
                period_card_count[bucket_fn(dt)] += 1

        min_cards_per_period = 3
        valid_periods = [p for p in all_periods if period_card_count[p] >= min_cards_per_period]

        if valid_periods and len(valid_periods) >= 2:
            removed = set(all_periods) - set(valid_periods)
            if removed:
                print(f"  Filtered out sparse periods: {removed} (< {min_cards_per_period} cards)")
                # Remove skills data for filtered periods
                for skill in freq_table:
                    for p in removed:
                        freq_table[skill].pop(p, None)
            all_periods = valid_periods

    # Remove skills that have 0 total mentions after period filtering
    empty_skills = [s for s, periods in freq_table.items()
                    if sum(periods.values()) == 0]
    for s in empty_skills:
        del freq_table[s]
    if empty_skills:
        print(f"  Removed {len(empty_skills)} skills with 0 mentions after filtering")

    return freq_table, all_periods, skipped


# ============================================================
# STEP 5: Compute growth rate and velocity
# ============================================================

def compute_growth_metrics(freq_table, all_periods):
    """
    Compute growth rate and velocity for each skill.

    growth_rate = (count_current - count_previous) / count_previous
    velocity    = growth_rate_current - growth_rate_previous  (acceleration)

    Returns list of dicts sorted by growth_rate descending.
    """
    results = []

    for skill, period_counts in freq_table.items():
        # Build counts array aligned to all_periods
        counts = [period_counts.get(p, 0) for p in all_periods]
        total_mentions = sum(counts)

        # Need at least 2 periods to compute growth
        if len(all_periods) < 2:
            results.append({
                "skill": skill,
                "total_mentions": total_mentions,
                "counts_per_period": dict(zip(all_periods, counts)),
                "growth_rate": None,
                "velocity": None,
                "trend": "insufficient_data",
            })
            continue

        # Growth rate: compare last period vs first period (overall)
        first_count = counts[0]
        last_count = counts[-1]

        if first_count > 0:
            overall_growth = (last_count - first_count) / first_count
        elif last_count > 0:
            overall_growth = float("inf")  # new skill (0 → something)
        else:
            overall_growth = 0.0

        # Period-over-period growth rates
        period_growths = []
        for i in range(1, len(counts)):
            if counts[i - 1] > 0:
                g = (counts[i] - counts[i - 1]) / counts[i - 1]
            elif counts[i] > 0:
                g = float("inf")
            else:
                g = 0.0
            period_growths.append(g)

        # Velocity: change in growth rate (last growth - previous growth)
        velocity = None
        if len(period_growths) >= 2:
            g_last = period_growths[-1] if period_growths[-1] != float("inf") else 10.0
            g_prev = period_growths[-2] if period_growths[-2] != float("inf") else 10.0
            velocity = g_last - g_prev

        # Average growth (use finite values only)
        finite_growths = [g for g in period_growths if g != float("inf")]
        avg_growth = sum(finite_growths) / len(finite_growths) if finite_growths else 0.0

        results.append({
            "skill": skill,
            "total_mentions": total_mentions,
            "counts_per_period": dict(zip(all_periods, counts)),
            "first_seen": next((all_periods[i] for i, c in enumerate(counts) if c > 0), None),
            "last_seen": next((all_periods[i] for i, c in enumerate(reversed(counts)) if c > 0), None),
            "growth_rate": overall_growth if overall_growth != float("inf") else 9.99,
            "avg_period_growth": avg_growth,
            "velocity": velocity,
            "trend": None,  # classified next
        })

    return results


# ============================================================
# STEP 6: Classify trend status
# ============================================================

def classify_trends(results, all_periods):
    """
    Classify each skill into trend categories:
    - emerging:   low total count + high growth (appeared recently, growing fast)
    - trending:   high count + positive growth
    - stable:     moderate/high count + near-zero growth
    - declining:  negative growth
    - new:        only appeared in the last period
    """
    if not results:
        return results

    # Compute median total mentions for relative thresholds
    totals = sorted(r["total_mentions"] for r in results)
    median_total = totals[len(totals) // 2] if totals else 1

    for r in results:
        if r["trend"] == "insufficient_data":
            continue

        growth = r["growth_rate"]
        total = r["total_mentions"]
        counts = r["counts_per_period"]

        # Count how many periods the skill appears in
        active_periods = sum(1 for p in all_periods if counts.get(p, 0) > 0)

        if growth is None:
            r["trend"] = "insufficient_data"
        elif active_periods == 1 and list(counts.values())[-1] > 0:
            r["trend"] = "new"
        elif growth >= 9.99:
            r["trend"] = "emerging"
        elif growth > 0.5 and total <= median_total:
            r["trend"] = "emerging"
        elif growth > 0.2:
            r["trend"] = "trending"
        elif growth >= -0.1:
            r["trend"] = "stable"
        else:
            r["trend"] = "declining"

    return results


# ============================================================
# STEP 7: Print results
# ============================================================

def print_results(results, all_periods, top_n=30):
    """Print trending skill analysis results."""

    print("=" * 70)
    print("STAGE 5: TRENDING SKILL ANALYSIS (Growth Rate / Velocity)")
    print("=" * 70)
    print(f"\nTime periods: {' → '.join(all_periods)}")
    print(f"Total unique skills: {len(results)}")

    # Group by trend
    trend_groups = defaultdict(list)
    for r in results:
        trend_groups[r["trend"]].append(r)

    # Print summary
    print(f"\n{'Trend':<15} {'Count':>6}")
    print("-" * 25)
    for trend in ["emerging", "trending", "new", "stable", "declining", "insufficient_data"]:
        count = len(trend_groups.get(trend, []))
        if count > 0:
            print(f"{trend:<15} {count:>6}")

    # Print top emerging/trending skills
    for category, label in [("emerging", "🚀 EMERGING SKILLS"), ("trending", "📈 TRENDING SKILLS"), ("declining", "📉 DECLINING SKILLS")]:
        group = trend_groups.get(category, [])
        if not group:
            continue

        # Sort by growth rate descending
        group.sort(key=lambda x: x.get("growth_rate", 0) or 0, reverse=True)

        print(f"\n{label}")
        print("-" * 70)
        print(f"{'Skill':<35} {'Total':>6} {'Growth':>10} {'Velocity':>10}")
        print("-" * 70)

        for r in group[:top_n]:
            growth_str = f"{r['growth_rate']:.0%}" if r["growth_rate"] is not None else "N/A"
            vel_str = f"{r['velocity']:.2f}" if r["velocity"] is not None else "N/A"
            print(f"{r['skill']:<35} {r['total_mentions']:>6} {growth_str:>10} {vel_str:>10}")

        # Show period breakdown for top 5
        print(f"\n  Period breakdown (top 5):")
        for r in group[:5]:
            period_str = " | ".join(f"{p}: {r['counts_per_period'].get(p, 0)}" for p in all_periods)
            print(f"    {r['skill']}: {period_str}")

    # Print top stable skills (most demanded)
    stable = trend_groups.get("stable", [])
    if stable:
        stable.sort(key=lambda x: x["total_mentions"], reverse=True)
        print(f"\n📊 TOP STABLE SKILLS (most demanded)")
        print("-" * 70)
        print(f"{'Skill':<35} {'Total':>6} {'Growth':>10}")
        print("-" * 70)
        for r in stable[:top_n]:
            growth_str = f"{r['growth_rate']:.0%}" if r["growth_rate"] is not None else "N/A"
            print(f"{r['skill']:<35} {r['total_mentions']:>6} {growth_str:>10}")


# ============================================================
# STEP 8: Save results to JSON
# ============================================================

def save_results(results, all_periods, output_path):
    """Save analysis results to JSON."""
    output = {
        "metadata": {
            "method": "Growth Rate / Velocity",
            "time_periods": all_periods,
            "total_skills_analyzed": len(results),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            trend: len([r for r in results if r["trend"] == trend])
            for trend in ["emerging", "trending", "new", "stable", "declining", "insufficient_data"]
        },
        "skills": sorted(results, key=lambda x: x.get("growth_rate", 0) or 0, reverse=True),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Usage:
        python stage5_trending_skills.py <job_cards.json> [month|quarter]

    Example:
        python stage5_trending_skills.py job_cards_all.json quarter
    """
    # Parse args
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "stage2_output.json")
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    time_bucket = sys.argv[2] if len(sys.argv) > 2 else "month"

    if time_bucket not in ("month", "quarter"):
        print(f"Invalid time bucket: {time_bucket}. Use 'month' or 'quarter'.")
        sys.exit(1)

    # Load data
    print(f"Loading job cards from: {input_path}")
    job_cards = load_job_cards(input_path)
    print(f"Loaded {len(job_cards)} job cards")

    # Normalize skills via embedding similarity
    print("\nBuilding skill normalizer (embedding similarity)...")
    skill_map = build_skill_normalizer(job_cards, similarity_threshold=0.93)

    # Build frequency table
    freq_table, all_periods, skipped = build_frequency_table(job_cards, time_bucket, skill_map)
    print(f"Time bucket: {time_bucket}")
    print(f"Periods found: {len(all_periods)}")
    print(f"Unique skills: {len(freq_table)}")
    if skipped:
        print(f"Skipped {skipped} cards (no valid posted_date)")

    if not all_periods:
        print("\nERROR: No valid dates found. Make sure job cards have 'posted_date' field.")
        sys.exit(1)

    # Compute metrics
    results = compute_growth_metrics(freq_table, all_periods)
    results = classify_trends(results, all_periods)

    # Print results
    print_results(results, all_periods)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"04_trending/outputs/stage5_trending_{time_bucket}_{timestamp}.json"
    save_results(results, all_periods, output_path)


if __name__ == "__main__":
    main()
