# Task Breakdown - TF-IDF Trending Skill Detection

## Data Flow Pipeline

```
Task 1: READ DATA
├── Input:  sample_data.json (1000 records)
└── Output: [{skills: [...], posted_date: "..."}, ...]

Task 2: GROUP BY PERIOD
├── Input:  1000 flat records
├── Logic:  Parse date → assign quarter key
└── Output: 21 groups (2021-Q1 → 2026-Q1)

Task 3: COMPUTE TF (per period)
├── Input:  All skills in one period ["A","B","A"]
├── Logic:  count(skill) / total_skills
└── Output: {A: 0.66, B: 0.33}

Task 4: COMPUTE IDF (across all periods)
├── Input:  21 period skill lists
├── Logic:  log(21 / periods_containing_skill)
│           Skill in all 21 periods → IDF = 0 (common)
│           Skill in 1 period → IDF = 3.04 (rare)
└── Output: {Skill A: 0.0, Skill X: 0.05, ...}

Task 5: TF-IDF = TF × IDF
├── Input:  TF per period + IDF global
└── Output: Score per skill per period
```

## Task 5: TF-IDF = TF × IDF (Visual Example)

### Input: 3 periods of grouped job postings

```
"2023-Q1": [{skills:["Python","AI"]}, {skills:["Python","React"]}]
"2023-Q2": [{skills:["AI","LLM"]}, {skills:["AI","React"]}]
"2023-Q3": [{skills:["LLM","LLM","Python"]}]
```

### Step 1: Flatten skills per period

```
"2023-Q1": ["Python","AI","Python","React"]
"2023-Q2": ["AI","LLM","AI","React"]
"2023-Q3": ["LLM","LLM","Python"]
```

### Step 2: Compute TF (per period)

```
Q1: Python=0.50, AI=0.25, React=0.25
Q2: AI=0.50, LLM=0.25, React=0.25
Q3: LLM=0.67, Python=0.33
```

### Step 3: Compute IDF (across all periods)

```
Python: log(3/2) = 0.41   (in 2/3 periods)
AI:     log(3/2) = 0.41   (in 2/3 periods)
React:  log(3/2) = 0.41   (in 2/3 periods)
LLM:    log(3/1) = 1.10   (in 1/3 periods — rare!)
```

### Step 4: Multiply TF × IDF

```
Q1: Python = 0.50 × 0.41 = 0.205
    AI     = 0.25 × 0.41 = 0.103
    React  = 0.25 × 0.41 = 0.103

Q2: AI     = 0.50 × 0.41 = 0.205
    LLM    = 0.25 × 1.10 = 0.275  ← high! (frequent + rare)
    React  = 0.25 × 0.41 = 0.103

Q3: LLM    = 0.67 × 1.10 = 0.737  ← highest! (very frequent + rare)
    Python = 0.33 × 0.41 = 0.135
```

### Key Insight
LLM gets highest TF-IDF: frequent in its periods (high TF) + doesn't appear everywhere (high IDF) = trending skill.

```

Task 6: DETECT TRENDS
├── Input:  TF-IDF scores across 21 periods
├── Logic:  trend = latest_score - avg(earlier_scores)
│           positive → rising
│           negative → declining
└── Output: Ranked list of trending skills

Task 7: DISPLAY RESULTS
├── Per-period top skills table
└── Rising / Declining summary
```

## Task 6: Detect Trending Skills (Visual Example)

Compare latest period vs average of all earlier periods.

### Step 1: Sort periods chronologically

```
["2021-Q1", "2021-Q2", ..., "2025-Q4", "2026-Q1"]
                                         ↑
              earlier (20 periods)       latest (1 period)
```

### Step 2: Get all unique skills

```
{"Python", "AI", "React", "LLM", ...}
```

### Step 3: For each skill, compute average TF-IDF across earlier periods

```
"LLM" avg_earlier = (0 + 0 + 0 + 0.275 + 0.737) / 20 = 0.051
```

### Step 4: For each skill, compute trend

```
trend = latest_tfidf - avg_earlier_tfidf

"LLM":    latest=0.800 - avg=0.051 = +0.749  (rising!)
"React":  latest=0.000 - avg=0.103 = -0.103  (declining!)
"Python": latest=0.135 - avg=0.170 = -0.035  (declining slightly)
```

### Step 5: Sort by trend_score (highest first), return top N

```
1. LLM:    +0.749  → rising
2. Python: -0.035  → declining
3. React:  -0.103  → declining
```

### Key Insight
- Positive trend = skill is more important now than before = **rising**
- Negative trend = skill was more important before = **declining**
- Zero = **stable**

## Function Mapping

| Task | Function | Input → Output |
|------|----------|----------------|
| 1 | `read_sample_data()` | file path → `[{skills, date}]` |
| 2 | `group_by_period()` | records → `{"2023-Q1": [records]}` |
| 3 | `compute_tf()` | `["Python","AI","AI"]` → `{AI: 0.66, Python: 0.33}` |
| 4 | `compute_idf()` | all period skill lists → `{skill: idf_score}` |
| 5 | `compute_tfidf()` | grouped data → `{period: {skill: score}}` |
| 6 | `detect_trending_skills()` | tfidf scores → `[{skill, trend, direction}]` |
| 7 | `print_tfidf_summary()` / `print_trends()` | tfidf/trends → printed output |
