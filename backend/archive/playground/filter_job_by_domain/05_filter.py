"""A2 Step 5 — filter_by_domain(postings, domain): the requirement fulfilled."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
IT_SLICE = HERE.parent / "domain_routing" / "output" / "it_slice.json"


# INPUT 1 — cohort.json (Step 4)        INPUT 2 — it_slice.json (all 16,069)
# ──────────────────────────────        ─────────────────────────────────────
# "Data Science": [                     ("senior data engineer",   [aws, spark…])
#   "data analyst",                     ("network engineer",       [cisco, lan…])
#   "data engineer",                    ("data analyst",           [sql, excel…])
#   "senior data engineer",             ("restaurant manager",     [service…])
#   ... 40 titles                       ("data analyst",           [python, bi…])
# ]                                     ("office manager",         [email…])
#             │                                   │
#             ▼                                   │
#    ┌────────────────────────┐                   │
#    │ titles = set(cohort)   │                   │
#    │ (set → O(1) lookup)    │                   │
#    └───────────┬────────────┘                   │
#                └──────────────┬─────────────────┘
#                               ▼
#             ┌───────────────────────────────────┐
#             │  for each posting:                │
#             │    title in cohort set ?          │
#             │                                   │
#             │  "senior data engineer"  ✅ keep  │
#             │  "network engineer"      ❌ skip  │  (wrong domain)
#             │  "data analyst"          ✅ keep  │
#             │  "restaurant manager"    ❌ skip  │  (no domain)
#             │  "data analyst"          ✅ keep  │  ← same title again:
#             │  "office manager"        ❌ skip  │    EVERY posting kept
#             └───────────────┬───────────────────┘    (fan-out: 40 titles
#                             ▼                         → 139 postings)
# OUTPUT — filtered_postings.json["Data Science"]
# ───────────────────────────────────────────────
# [ ("senior data engineer", [aws, spark, …]),
#   ("data analyst",         [sql, excel, …]),
#   ("data analyst",         [python, bi, …]),
#   ... 139 postings — ONLY data-science jobs ]
#                             │
#                             ▼
#         Step 6: pool their skills → market side vs course


def filter_by_domain(postings, domain, cohort):
    """ONLY the postings whose title is in the domain's verified cohort."""
    titles = set(cohort.get(domain, []))          # set → O(1) per posting
    return [p for p in postings if p[0].lower().strip() in titles]


if __name__ == "__main__":
    posts = json.loads(IT_SLICE.read_text())["posts"]
    cohort = json.loads((OUT / "cohort.json").read_text())

    filtered = {}
    for domain in cohort:
        hits = filter_by_domain(posts, domain, cohort)
        filtered[domain] = hits
        print(f"■ {domain}: {len(cohort[domain])} cohort titles "
              f"→ {len(hits)} postings")
        for title, skills in hits[:3]:
            print(f"   • {title}")
            print(f"       {', '.join(skills[:6])} ...")
        print()

    json.dump(filtered, open(OUT / "filtered_postings.json", "w"))
    print("saved output/filtered_postings.json")
