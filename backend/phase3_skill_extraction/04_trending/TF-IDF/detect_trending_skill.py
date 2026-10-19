import json
import os
from collections import defaultdict
from datetime import datetime
import math


def read_sample_data(file_path=None):
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "01_data_collection", "sample_data_detect_trending_skill", "sample_data.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {os.path.basename(file_path)}")
    return data

def group_by_period(data, period="quarter"):
    """Group job postings by time period (quarter or month)."""
    groups = defaultdict(list)
    for record in data:
        dt = datetime.fromisoformat(record["posted_date"])
        if period == "quarter":
            q = (dt.month - 1) // 3 + 1
            key = f"{dt.year}-Q{q}"
        elif period == "month":
            key = f"{dt.year}-{dt.month:02d}"
        else:
            key = str(dt.year)
        groups[key].append(record)
    return dict(sorted(groups.items()))

def compute_tf(skill_list):
    # Example input: ["Python", "AI", "Python", "React"]
    # total = 4
    total = len(skill_list)
    if total == 0:
        return {}
    # counts = {"Python": 2, "AI": 1, "React": 1}
    counts = defaultdict(int)
    for skill in skill_list:
        counts[skill] += 1
    # return {"Python": 2/4=0.50, "AI": 1/4=0.25, "React": 1/4=0.25}
    return {skill: count / total for skill, count in counts.items()}




def compute_idf(period_skill_lists):
    # Example input: [["Python","AI","Python"], ["AI","React"], ["Python","React"]]
    # n_periods = 3
    n_periods = len(period_skill_lists)
    # Count how many periods each skill appears in (not how many times)
    # doc_freq = {"Python": 2, "AI": 2, "React": 2}
    doc_freq = defaultdict(int)
    for skills in period_skill_lists:
        unique_skills = set(skills)  # set removes duplicates within one period
        for skill in unique_skills:
            doc_freq[skill] += 1
    # return {"Python": log(3/2)=0.41, "AI": log(3/2)=0.41, "React": log(3/2)=0.41}
    # If a skill appears in ALL 3 periods: log(3/3) = 0 (too common)
    return {skill: math.log(n_periods / df) for skill, df in doc_freq.items()}


def compute_tfidf(grouped_data):
    period_skills = {}
    for period, records in grouped_data.items():
        skills = []
        for record in records:
            skills.extend(record.get("skills", []))
        period_skills[period] = skills

    idf = compute_idf(list(period_skills.values()))

    tfidf_per_period = {}
    for period, skills in period_skills.items():
        tf = compute_tf(skills)
        tfidf_per_period[period] = {
            skill: tf_val * idf.get(skill, 0)
            for skill, tf_val in tf.items()
        }

    # high TF-IDF score in the latest period, that skill is "hot" right now but wasn't common before — that's a trending skill.
    return tfidf_per_period

# trend_score = latest_tfidf - avg_earlier_tfidf
def detect_trending_skills(tfidf_per_period, top_n=10):
    # Step 1: Sort periods → ["2021-Q1", ..., "2026-Q1"]
    periods = sorted(tfidf_per_period.keys())
    if len(periods) < 2:
        print("Need at least 2 periods to detect trends.")
        return []

    # Step 2: Split into latest vs earlier
    latest = periods[-1]
    earlier = periods[:-1]

    # Step 3: Collect all unique skills
    all_skills = set()
    for scores in tfidf_per_period.values():
        all_skills.update(scores.keys())

    # Step 4: Average TF-IDF across earlier periods
    avg_earlier = {}
    for skill in all_skills:
        values = [tfidf_per_period[p].get(skill, 0) for p in earlier]
        avg_earlier[skill] = sum(values) / len(values)

    # Step 5: trend = latest - avg_earlier
    trends = []
    for skill in all_skills:
        latest_score = tfidf_per_period[latest].get(skill, 0)
        earlier_score = avg_earlier[skill]
        trend_score = latest_score - earlier_score
        trends.append({
            "skill": skill,
            "trend_score": round(trend_score, 4),
            "latest_tfidf": round(latest_score, 4),
            "avg_earlier_tfidf": round(earlier_score, 4),
            "direction": "rising" if trend_score > 0 else "declining" if trend_score < 0 else "stable"
        })

    trends.sort(key=lambda x: x["trend_score"], reverse=True)
    return trends[:top_n]


def print_tfidf_summary(tfidf_per_period, top_n=5):
    """Print top skills by TF-IDF for each period."""
    for period in sorted(tfidf_per_period.keys()):
        if not (period.startswith("2025") or period.startswith("2026")):
            continue
        scores = tfidf_per_period[period]
        sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n--- {period} (top {top_n} skills by TF-IDF) ---")
        for skill, score in sorted_skills:
            print(f"  {skill}: {score:.4f}")


def print_trends(trends):
    """Print trending skills summary."""
    print("\n=== TRENDING SKILLS (Rising) ===")
    rising = [t for t in trends if t["direction"] == "rising"]
    for t in rising:
        print(f"  {t['skill']}: trend={t['trend_score']:+.4f} "
              f"(latest={t['latest_tfidf']:.4f}, avg_earlier={t['avg_earlier_tfidf']:.4f})")

    declining = [t for t in trends if t["direction"] == "declining"]
    if declining:
        print("\n=== DECLINING SKILLS ===")
        for t in declining:
            print(f"  {t['skill']}: trend={t['trend_score']:+.4f} "
                  f"(latest={t['latest_tfidf']:.4f}, avg_earlier={t['avg_earlier_tfidf']:.4f})")


if __name__ == "__main__":
    data = read_sample_data()
    grouped = group_by_period(data, period="quarter")
    print(f"Found {len(grouped)} periods: {list(grouped.keys())}")
    tfidf = compute_tfidf(grouped)
    print_tfidf_summary(tfidf, top_n=5)
    trends = detect_trending_skills(tfidf, top_n=10)
    print_trends(trends)
