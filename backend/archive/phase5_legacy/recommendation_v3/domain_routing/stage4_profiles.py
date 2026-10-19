"""Stage 4 — AGGREGATE + BUILD: cohort postings' skills → domain_profiles.json.

The build CLI that ties stages 1–3 together, per category:

    stage1  load the category's corpus slice
    stage2  anchor-bootstrap the title cohorts (retrieve)
    stage3  qwen3 audits the cohorts (verify)
    stage4  aggregate the surviving titles' postings → top-N skills = PROFILE

The build MERGES into domain_profiles.json, leaving other categories' profiles
untouched — rebuilding one faculty's taxonomy never re-verifies the others.

Output schema (stage5_route.py / build_inventory.py read it unchanged):
    { domain: { category, anchor, n_postings, n_titles, n_rejected,
                rejected_sample, top_titles, profile, skill_freq, cohort } }

`cohort` = the full LLM-verified title list — build_inventory.py's cohort route
filters postings with it (exact title membership), replacing the manual seeds.

Run:  python stage4_profiles.py                 # all categories in CATEGORY_DOMAINS
      python stage4_profiles.py it              # rebuild only the IT taxonomy
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_corpus as s1                       # noqa: E402
import stage2_retrieve as s2                     # noqa: E402
import stage3_verify as s3                       # noqa: E402

# How many top skills define a profile.
_TOP_N = 30


def _build_category(category: str, domains: list[str], posts: list,
                    title_count: Counter, model) -> dict:
    """Anchor-bootstrap + verify + aggregate ONE category's domain taxonomy."""
    titles = list(title_count)

    # PASS 1 — retrieve (stage 2), PASS 1.5 — verify (stage 3)
    ordered = s2.retrieve_cohorts(category, domains, titles, model)
    rejected = s3.verify_cohorts(ordered)
    cohort: dict[str, set] = {d: set(ordered[d]) - rejected[d] for d in domains}

    # PASS 2 — aggregate cohort postings' skills → profile
    title2dom = {t: d for d, ts in cohort.items() for t in ts}
    counts: dict[str, Counter] = {d: Counter() for d in domains}
    n_post: dict[str, int] = {d: 0 for d in domains}
    top_titles: dict[str, Counter] = {d: Counter() for d in domains}
    for title, skills in posts:
        d = title2dom.get(title)
        if d is None:
            continue
        n_post[d] += 1
        top_titles[d][title] += 1
        counts[d].update(skills)

    profiles = {}
    for d in domains:
        top = counts[d].most_common(_TOP_N)
        profiles[d] = {
            "category": category,
            "anchor": d.replace("_", " "),
            "n_postings": n_post[d],
            "n_titles": len(cohort[d]),
            "n_rejected": len(rejected[d]),
            "rejected_sample": sorted(rejected[d])[:10],
            "top_titles": [t for t, _ in top_titles[d].most_common(10)],
            "profile": [s for s, _ in top],
            "skill_freq": {s: c for s, c in top},
            "cohort": sorted(cohort[d]),
        }
    return profiles


def build(categories: list[str] | None = None) -> dict:
    """Build the given categories' taxonomies (default: all in CATEGORY_DOMAINS)
    and MERGE them into domain_profiles.json — other categories are preserved."""
    cats = list(categories or s1.CATEGORY_DOMAINS)
    unknown = [c for c in cats if c not in s1.CATEGORY_DOMAINS]
    if unknown:
        raise SystemExit(f"no taxonomy for {unknown} — add it to CATEGORY_DOMAINS "
                         f"(stage1_corpus.py)")

    corpus = s1.load_corpus(set(cats))
    model = s2.embedder()

    # merge over the existing file so single-category rebuilds are cheap
    try:
        merged = json.loads(s1.PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        merged = {}
    rebuilt = set(cats)
    merged = {d: v for d, v in merged.items()
              if d != "_meta" and isinstance(v, dict)
              and v.get("category", "it") not in rebuilt}

    for cat in cats:
        posts, title_count = corpus[cat]
        if not posts:
            print(f"[{cat}] no postings in corpus — skipped", file=sys.stderr)
            continue
        merged.update(_build_category(cat, s1.CATEGORY_DOMAINS[cat], posts,
                                      title_count, model))

    merged["_meta"] = {
        "method": "anchor_bootstrap_verify_v2", "model": s2._MODEL, "llm": s3._LLM,
        "floor": s2._FLOOR, "margin": s2._MARGIN, "max_titles": s2._MAX_TITLES,
        "categories": sorted({v.get("category", "it") for d, v in merged.items()
                              if d != "_meta"}),
    }
    s1.PROFILES_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return merged


if __name__ == "__main__":
    profiles = build(sys.argv[1:] or None)
    for d, p in profiles.items():
        if d == "_meta":
            continue
        print(f"■ {d} [{p.get('category')}]  ({p['n_postings']:,} cohort postings)")
        print("   titles: " + ", ".join(p["top_titles"][:4]))
        print("   skills: " + ", ".join(p["profile"][:12]) + " …")
        print()
