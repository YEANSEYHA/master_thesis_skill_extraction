import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IT_SLICE = HERE.parent / "data" / "job_postings_categorized.jsonl" 

def load_data(path=IT_SLICE, limit=None):
    """Load postings → [(job_title, [skills]), ...].

    Handles both:
      .jsonl  — one posting dict per line   (e.g. job_postings_categorized.jsonl)
      .json   — one object {"posts": [...]} (e.g. it_slice.json)
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        posts = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)                      # ONE object per line
                posts.append((r.get("job_title") or "",
                              [s.strip().lower() for s in r.get("skills", [])]))
                if limit and len(posts) >= limit:         # stop early on big files
                    break
        return posts
    return json.loads(p.read_text())["posts"]


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT — filter job postings by career domain
#
# DATA:
#   postings: list[dict], each posting = {
#       "job_title": str,          # e.g. "Restaurant Manager"
#       "skills":    list[str],    # e.g. ["leadership", "customer service", ...]
#   }
#
# INPUT:
#   domain: str — the career domain the user asks for, e.g.
#       "Network" | "Data Science" | "Software Engineering"
#   (case-insensitive: "network" and "Network" behave the same)
#
# OUTPUT:
#   list[dict] — ONLY the postings that belong to the requested domain.
#       filter_by_domain(postings, "Data Science")
#           -> [ postings whose role is a data-science job ]
#   Postings of any other domain are excluded.
#   If nothing matches -> return [] (empty list, not an error).
#
# RULE (how "belongs to the domain" is decided):
#   Title routing — a posting belongs to the domain iff its JOB TITLE
#   contains one of that domain's role titles as a whole word
#   (regex \b...\b), e.g. domain "Data Science" ⇐ titles containing
#   "data scientist", "data science intern", ...
#   Skills are NOT used for the decision (they leak: a Store Manager
#   listing "analytical skills" must NOT enter Data Science).
#
# EDGE CASES:
#   - a posting may belong to MORE than one domain (multi-label) —
#     it is returned for each domain that asks for it
#   - unknown domain name -> [] (or raise ValueError, choose one)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    posts = load_data()
    print(f"loaded {len(posts):,} postings\n")
    for title, skills in posts[:5]:
        print(json.dumps({"job_title": title, "skills": skills[:8]},
                         indent=2, ensure_ascii=False))
