import json
import os
import sys
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt


def read_sample_data(file_path=None):
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "01_data_collection", "sample_data_detect_trending_skill", "sample_data.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {os.path.basename(file_path)}")
    return data


def filter_by_time_range(data, time_range=None):
    """
    Filter records within a time range from now.
    Input:  data — list of records, time_range — e.g. "24h", "7d", "3m", "1y", None (all)
    Output: filtered list of records
    """
    if time_range is None:
        return data

    from datetime import timedelta

    now = datetime.now()
    unit = time_range[-1]       # h, d, m, y
    value = int(time_range[:-1])  # 24, 7, 3, 1

    if unit == "h":
        cutoff = now - timedelta(hours=value)
    elif unit == "d":
        cutoff = now - timedelta(days=value)
    elif unit == "w":
        cutoff = now - timedelta(weeks=value)
    elif unit == "m":
        cutoff = now - timedelta(days=value * 30)
    elif unit == "y":
        cutoff = now - timedelta(days=value * 365)
    else:
        raise ValueError(f"Unknown time unit: {unit}. Use h/d/w/m/y")

    filtered = [r for r in data if datetime.fromisoformat(r["posted_date"]) >= cutoff]
    print(f"Filtered: {len(filtered)}/{len(data)} records (last {time_range})")
    return filtered


def auto_period(time_range=None):
    """
    Pick the best grouping granularity for a time range.
    Input:  time_range — e.g. "24h", "7d", "3m", "1y", None
    Output: period string — "hour", "day", "week", "month", "quarter"
    """
    if time_range is None:
        return "quarter"

    unit = time_range[-1]
    value = int(time_range[:-1])

    if unit == "h":
        return "hour"
    elif unit == "d" and value <= 14:
        return "day"
    elif unit == "d" or unit == "w" or (unit == "m" and value <= 3):
        return "week"
    elif unit == "m" and value <= 12:
        return "month"
    elif unit == "y" and value <= 2:
        return "month"
    else:
        return "quarter"


def group_by_period(data, period="quarter"):
    """
    Group records by time period.
    Input:  data — list of records, period — "hour", "day", "week", "month", "quarter", "year"
    Output: {period_key: [records]}
    """
    groups = defaultdict(list)
    for record in data:
        dt = datetime.fromisoformat(record["posted_date"])
        if period == "hour":
            key = dt.strftime("%Y-%m-%d %H:00")
        elif period == "day":
            key = dt.strftime("%Y-%m-%d")
        elif period == "week":
            key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        elif period == "quarter":
            q = (dt.month - 1) // 3 + 1
            key = f"{dt.year}-Q{q}"
        elif period == "month":
            key = f"{dt.year}-{dt.month:02d}"
        else:
            key = str(dt.year)
        groups[key].append(record)
    return dict(sorted(groups.items()))


def compute_skill_frequency(grouped_data, periods):
    skill_counts = {}
    for period in periods:
        counts = defaultdict(int)
        total_jobs = len(grouped_data[period])
        for record in grouped_data[period]:
            for skill in record.get("skills", []):
                counts[skill] += 1
        skill_counts[period] = {skill: c / total_jobs for skill, c in counts.items()} if total_jobs > 0 else {}
    return skill_counts


def select_top_skills(skill_counts, top_n=5):
    overall = defaultdict(float)
    for period, skills in skill_counts.items():
        for skill, freq in skills.items():
            overall[skill] += freq
    sorted_skills = sorted(overall.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills[:top_n]]


def normalize_to_100(skill_counts, periods, skills_to_track):
    series = {}
    for skill in skills_to_track:
        raw = [skill_counts[p].get(skill, 0) for p in periods]
        peak = max(raw) if raw else 0
        if peak > 0:
            series[skill] = [round((v / peak) * 100, 1) for v in raw]
        else:
            series[skill] = [0] * len(periods)

    return series


def compute_google_trends(grouped_data, skills_to_track=None, top_n=5):
    """
    Algorithm: Google Trends style scoring.
    Input:  grouped_data — {period: [records]}, each record has "skills" list
    Output: (periods, series) — periods is sorted list of period keys,
            series is {skill_name: [score_0_to_100, ...]} scaled so peak = 100
    """
    periods = sorted(grouped_data.keys())

    # Step 1: Count skill frequency per period (TF)
    skill_counts = compute_skill_frequency(grouped_data, periods)

    # Step 2: Auto-pick top skills if not specified
    if skills_to_track is None:
        skills_to_track = select_top_skills(skill_counts, top_n)

    # Step 3 & 4: Build time series and scale to 0-100
    series = normalize_to_100(skill_counts, periods, skills_to_track)

    return periods, series


def compute_growth_rate(grouped_data, recent_n=2, top_n=10):
    """
    Algorithm: Growth rate — detect truly TRENDING skills.
    Compares avg TF in recent periods vs earlier periods.

    Input:  grouped_data — {period: [records]}
            recent_n — how many recent periods = "recent window"
            top_n — return top N fastest growing skills
    Output: sorted list of (skill, growth_pct, tf_recent, tf_past)
            e.g. [("Rust", 400.0, 0.15, 0.03), ("Docker", 50.0, 0.18, 0.12), ...]
    """
    periods = sorted(grouped_data.keys())
    if len(periods) < recent_n + 1:
        print(f"Not enough periods ({len(periods)}) for growth rate (need >= {recent_n + 1})")
        return []

    # Step 1: Compute TF per period (reuse shared function)
    skill_tf = compute_skill_frequency(grouped_data, periods)

    # Step 2: Split into recent vs past
    recent_periods = periods[-recent_n:]
    past_periods = periods[:-recent_n]

    # Step 3: Avg TF per skill in each window
    def avg_tf(period_list):
        totals = defaultdict(float)
        for p in period_list:
            for skill, tf in skill_tf[p].items():
                totals[skill] += tf
        return {skill: val / len(period_list) for skill, val in totals.items()}

    tf_recent = avg_tf(recent_periods)
    tf_past = avg_tf(past_periods)

    # Step 4: Compute growth rate and classify each skill
    EMERGING_THRESHOLD = 0.05  # past TF below this = "didn't exist"
    all_skills = set(tf_recent.keys()) | set(tf_past.keys())
    results = []
    for skill in all_skills:
        recent = tf_recent.get(skill, 0)
        past = tf_past.get(skill, 0)
        if past > 0:
            growth = ((recent - past) / past) * 100
        elif recent > 0:
            growth = 9999.0  # new skill, cap growth for safe plotting
        else:
            growth = 0.0

        # Classify
        if past <= EMERGING_THRESHOLD and recent > EMERGING_THRESHOLD:
            category = "emerging"
        elif growth > 0 and past > EMERGING_THRESHOLD:
            category = "trending"
        elif growth < 0:
            category = "declining"
        else:
            category = "stable"

        results.append((skill, round(growth, 1), round(recent, 4), round(past, 4), category))

    # Step 5: Sort by growth rate descending, take top N
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def plot_growth_rate(results, save_path="growth_rate.png"):
    """Plot horizontal bar chart of growth rates."""
    skills = [r[0] for r in reversed(results)]
    growth = [r[1] for r in reversed(results)]
    colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in growth]

    plt.figure(figsize=(12, max(4, len(skills) * 0.5)))
    bars = plt.barh(skills, growth, color=colors)

    for bar, g in zip(bars, growth):
        label = f"+{g}%" if g > 0 else f"{g}%"
        if g >= 9999.0:
            label = "NEW"
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 label, va='center', fontsize=9)

    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.title("Skill Growth Rate (Recent vs Past Periods)")
    plt.xlabel("Growth Rate (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_google_trends(periods, series, save_path="google_trends_style.png"):
    """Plot the Google Trends style line chart."""
    plt.figure(figsize=(14, 6))
    for skill, values in series.items():
        plt.plot(periods, values, marker='o', markersize=3, label=skill)

    plt.title("Skill Trend (Google Trends Style, 0-100)")
    plt.xlabel("Period")
    plt.ylabel("Interest (0-100)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    # Usage: python google_trending.py [time_range]
    # Examples: 6h, 24h, 3d, 7d, 1w, 2w, 1m, 3m, 6m, 1y, 2y, (empty = all)
    time_range = sys.argv[1] if len(sys.argv) > 1 else None

    data = read_sample_data()
    filtered = filter_by_time_range(data, time_range)
    period = auto_period(time_range)
    grouped = group_by_period(filtered, period=period)

    print(f"Found {len(grouped)} periods ({period}): {list(grouped.keys())}")

    # 1. Popularity over time (Google Trends style)
    periods, series = compute_google_trends(grouped, top_n=5)
    plot_google_trends(periods, series)

    # 2. Actual trending detection (growth rate)
    trending = compute_growth_rate(grouped, recent_n=2, top_n=10)
    print("\n--- Skill Classification (by growth rate) ---")
    for skill, growth, tf_recent, tf_past, category in trending:
        icon = {"emerging": "🆕", "trending": "▲", "declining": "▼", "stable": "─"}[category]
        g_str = f"+{growth}%" if growth > 0 else f"{growth}%"
        if growth >= 9999.0:
            g_str = "NEW"
        print(f"  {icon} [{category:9s}] {skill:20s}  {g_str:>10s}   (recent={tf_recent:.4f}, past={tf_past:.4f})")
    plot_growth_rate(trending)
