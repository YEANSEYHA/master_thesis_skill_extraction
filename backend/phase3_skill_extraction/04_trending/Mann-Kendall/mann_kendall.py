
# Static sample data: 7 days of job postings, each job has a list of skills
DATA = {
    "Day1": [["a", "b", "c"], ["b", "a", "d"], ["c", "e", "f"]],
    "Day2": [["a", "b", "c"], ["a", "c", "d"], ["e", "f", "b"]],
    "Day3": [["a", "c", "g"], ["a", "b", "c"], ["g", "c", "d"]],
    "Day4": [["a", "c", "g"], ["c", "g", "a"], ["b", "c", "g"]],
    "Day5": [["c", "g", "h"], ["a", "c", "g"], ["g", "c", "h"]],
    "Day6": [["c", "g", "h"], ["g", "h", "c"], ["c", "g", "h"]],
    "Day7": [["g", "h", "c"], ["h", "g", "c"], ["g", "h", "c"]],
}


# Step 1: Count skill frequency per day
def count_skill_per_period(data):
    skill_counts = {}
    for period, jobs in data.items():
        counts = {}
        for job in jobs:
            for skill in job:
                counts[skill] = counts.get(skill, 0) + 1
        skill_counts[period] = counts
    return skill_counts


# Step 2: Build time series per skill
def build_time_series(skill_counts):
    periods = sorted(skill_counts.keys())
    all_skills = set()
    for counts in skill_counts.values():
        all_skills.update(counts.keys())

    # { "a": [2, 2, 2, 2, 1, 0, 0], "g": [0, 0, 2, 2, 2, 2, 3], ... }
    time_series = {}
    for skill in all_skills:
        time_series[skill] = [skill_counts[p].get(skill, 0) for p in periods]

    return periods, time_series


# Step 3: Mann-Kendall test
def mann_kendall(series):
    n = len(series)
    s = 0
    # Compare all pairs (later vs earlier)
    for i in range(n):
        for j in range(i + 1, n):
            if series[j] > series[i]:
                s += 1
            elif series[j] < series[i]:
                s -= 1
            # equal → 0, no change

    # Tau = S / total possible pairs
    total_pairs = n * (n - 1) / 2
    tau = s / total_pairs if total_pairs > 0 else 0

    return s, round(tau, 4)


# Step 4: Detect & classify all skills
def detect_trending(time_series):
    results = []
    for skill, series in time_series.items():
        s, tau = mann_kendall(series)
        if tau > 0:
            direction = "trending"
        elif tau < 0:
            direction = "declining"
        else:
            direction = "stable"

        results.append({
            "skill": skill,
            "series": series,
            "S": s,
            "Tau": tau,
            "direction": direction,
        })

    # Sort by Tau descending (most trending first)
    results.sort(key=lambda x: x["Tau"], reverse=True)
    return results


# Step 5: Print results
def print_results(periods, results):
    print(f"\nPeriods: {periods}\n")
    print(f"{'Skill':<8} {'Series':<28} {'S':>4}  {'Tau':>6}  {'Direction'}")
    print("-" * 65)
    for r in results:
        print(f"{r['skill']:<8} {str(r['series']):<28} {r['S']:>+4}  {r['Tau']:>+6.2f}  {r['direction']}")


if __name__ == "__main__":
    # Step 1
    skill_counts = count_skill_per_period(DATA)
    # Step 2
    periods, time_series = build_time_series(skill_counts)
    # Step 3 + 4
    results = detect_trending(time_series)
    # Step 5
    print_results(periods, results)
