import json
import re
from collections import Counter
from pathlib import Path

# all generated results live in ./output
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

CORPUS = (Path(__file__).resolve().parents[2]
          / "phase5_legacy" / "linkedin_job_posting"
          / "job_postings_categorized.jsonl")

# Raise this later to use more of the 1.35M-posting corpus.
LIMIT = 200_000

# The one manual input
# job_category bucket -> the career domains inside it. That's it.

CATEGORY_DOMAINS = {
    "it": [
        "artificial_intelligence", "data_science", "software_engineering",
        "cybersecurity", "cloud_devops", "networking", "data_engineering",
        "data_analytics", "qa_testing", "business_analysis", "web_frontend",
        "mobile_development", "embedded_systems", "database_administration",
        "it_operations", "it_support", "project_product_management",
        "blockchain", "ui_ux_design", "game_development",
    ],
}

# Generic soft skills + credentials show up in EVERY domain, so they blur the
# profiles together. We drop them from every posting.
EXCLUDE = {
    "communication", "teamwork", "collaboration", "leadership",
    "problem solving", "time management", "critical thinking",
    "attention to detail", "project management", "english",
    "bachelor's degree", "master's degree", "computer science", "degree",
}

def load_it_postings():
    """One pass over the corpus -> (posts, title_count) for the 'it' slice.
    posts = [(title_lower, frozenset(skills))].  title_count = how often each
    unique title appears."""
    posts = []
    title_count = Counter()
    scanned = 0
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned > LIMIT:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("job_category") != "it":     # keep only IT postings
                continue
            title = re.sub(r"\s+", " ", (rec.get("job_title", "") or "").strip().lower())
            if not title:
                continue
            skills = frozenset(
                s.strip().lower() for s in rec.get("skills", []) if s.strip()
            ) - EXCLUDE
            posts.append((title, skills))
            title_count[title] += 1
    print(f"scanned {scanned:,} lines -> {len(posts):,} IT postings, "
          f"{len(title_count):,} unique titles")
    return posts, title_count





if __name__ == "__main__":
    posts, title_count = load_it_postings()
    print("\nDomains we will build (from CATEGORY_DOMAINS['it']):")
    print("  " + ", ".join(CATEGORY_DOMAINS["it"]))
    print("\n5 sample postings (title -> a few skills):")
    for title, skills in posts[:5]:
        print(f"  {title!r:45} {sorted(skills)[:4]}")
    print("\n8 most common IT job titles:")
    for t, n in title_count.most_common(8):
        print(f"  {n:4}  {t}")
    
        # save the slice so the next stages don't rescan the 924MB corpus
    out = {"posts": [[t, sorted(s)] for t, s in posts],
           "titles": list(title_count)}
    (OUT / "it_slice.json").write_text(json.dumps(out))
    print(f"\nsaved output/it_slice.json ({len(posts):,} posts, {len(out['titles']):,} titles)")
