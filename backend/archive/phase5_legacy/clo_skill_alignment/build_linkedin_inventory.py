"""
Build a domain-conditioned skill inventory from the LinkedIn job-posting jsonl.

The downloaded LinkedIn dataset has `job_category = null`. This script derives a
coarse `job_category` from the clean `job_title` using a HYBRID classifier —

    title --> keyword rule match?  --yes-->  category        (precise, free)
                    |no
                    v
              JobBERT embedding nearest-category (seed-title centroids)
                    |  cos < threshold
                    v
                  "other"

— then tallies a `{category: {skill: frequency}}` inventory the curriculum
matcher can use. Run ONCE; the API reads the small cached inventory (it never
re-scans the 5.7 GB file).

Outputs:
  - <out_inventory> : {category: {skill: freq}} + per-category posting counts
  - <out_jsonl>     : the records with job_category filled (slim: no description),
                      unless --no-labeled

Usage (see bottom of file / the README for the exact command):
  python build_linkedin_inventory.py --input ../linkedin_job_posting/job_postings_full.jsonl
  python build_linkedin_inventory.py --sample 100000        # quick ~1-2 min run
  python build_linkedin_inventory.py --no-embed             # keyword-only (no model)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

_JOBBERT = "jjzha/jobbert-base-cased"
_EMBED_THRESHOLD = 0.45     # min cosine for an embedding category assignment
_BATCH = 256


# ── Category schema: ordered keyword rules + seed titles for embedding ──────────
# Order matters: the FIRST matching rule wins. Tech-specific patterns are listed
# before generic "engineer" so "Data Engineer" -> it, not physical engineering.
# Bare "Engineer" has NO keyword (it's ambiguous) -> handed to the embedder.

CATEGORIES: list[dict] = [
    {"name": "it",
     "kw": r"software|web developer|\bdeveloper|programmer|data (scientist|analyst|engineer)|"
           r"data engineering|machine learning|\bml engineer|\bai engineer|devops|\bsre\b|"
           r"full.?stack|back.?end|front.?end|cloud (engineer|architect)|cyber|"
           r"information technology|\bit (support|manager|specialist|technician)|"
           r"system(s)? admin|network (engineer|administrator)|database admin|\bqa engineer|"
           r"data engineer|web design|android|ios developer|mobile developer",
     "seeds": ["Software Engineer", "Data Analyst", "DevOps Engineer", "Web Developer",
               "Backend Developer", "Machine Learning Engineer", "IT Support Specialist",
               "Data Scientist"]},
    {"name": "health",
     "kw": r"nurse|\brn\b|physician|\bdoctor\b|medical|clinical|pharmac|therapist|"
           r"healthcare|dental|surgeon|caregiver|paramedic|radiolog",
     "seeds": ["Registered Nurse", "Physician", "Medical Assistant", "Pharmacist",
               "Physical Therapist"]},
    {"name": "finance",
     "kw": r"finance|financial analyst|investment|treasury|account(ant|ing)|audit|"
           r"bookkeep|\btax\b|underwrit|actuar|credit analyst",
     "seeds": ["Financial Analyst", "Accountant", "Auditor", "Finance Manager",
               "Investment Banker"]},
    {"name": "sales",
     "kw": r"\bsales|account executive|business development|sales rep|account manager",
     "seeds": ["Sales Representative", "Account Executive", "Business Development Manager",
               "Sales Manager"]},
    {"name": "marketing",
     "kw": r"marketing|\bseo\b|content (writer|marketing|strateg)|brand|social media|"
           r"digital marketing|copywriter|growth",
     "seeds": ["Marketing Manager", "Digital Marketing Specialist", "Content Writer",
               "SEO Specialist", "Social Media Manager"]},
    {"name": "design",
     "kw": r"(graphic|ux|ui|product|visual|interior) design|\bdesigner|creative director",
     "seeds": ["Graphic Designer", "UX Designer", "Product Designer", "UI Designer"]},
    {"name": "engineering",
     "kw": r"mechanical|civil|electrical|structural|chemical|aerospace|industrial engineer|"
           r"manufacturing engineer|process engineer|hardware engineer",
     "seeds": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineer",
               "Manufacturing Engineer"]},
    {"name": "education",
     "kw": r"teacher|professor|tutor|lecturer|instructor|faculty|educat|principal|"
           r"teaching assistant",
     "seeds": ["Teacher", "Professor", "Tutor", "Instructor", "Teaching Assistant"]},
    {"name": "hr",
     "kw": r"human resources|\bhr\b|recruit|talent acquisition|people operations|"
           r"payroll specialist",
     "seeds": ["Human Resources Manager", "Recruiter", "Talent Acquisition Specialist"]},
    {"name": "legal",
     "kw": r"legal|attorney|lawyer|paralegal|\bcounsel|compliance officer",
     "seeds": ["Attorney", "Paralegal", "Legal Counsel", "Compliance Officer"]},
    {"name": "construction",
     "kw": r"construction|contractor|carpenter|electrician|plumber|welder|mason|"
           r"site supervisor",
     "seeds": ["Construction Manager", "Carpenter", "Electrician", "Site Supervisor"]},
    {"name": "logistics",
     "kw": r"logistics|supply chain|warehouse|procurement|inventory|freight|"
           r"\bdriver\b|dispatch|fleet",
     "seeds": ["Logistics Coordinator", "Supply Chain Analyst", "Warehouse Manager",
               "Truck Driver"]},
    {"name": "hospitality",
     "kw": r"restaurant|\bhotel|\bchef\b|\bcook\b|barista|\bserver\b|hospitality|"
           r"food service|bartender|housekeep",
     "seeds": ["Restaurant Manager", "Chef", "Barista", "Hotel Manager", "Server"]},
    {"name": "retail",
     "kw": r"retail|store (manager|associate|clerk)|cashier|merchandis|sales associate",
     "seeds": ["Store Manager", "Retail Associate", "Cashier", "Merchandiser"]},
    {"name": "customer_service",
     "kw": r"customer (service|support|success|experience)|call center|help desk|"
           r"client service|technical support",
     "seeds": ["Customer Service Representative", "Customer Success Manager",
               "Call Center Agent", "Help Desk Technician"]},
    {"name": "admin",
     "kw": r"administrative|admin assistant|executive assistant|receptionist|"
           r"office (manager|administrator)|secretary|\bclerk\b|data entry",
     "seeds": ["Administrative Assistant", "Office Manager", "Receptionist",
               "Executive Assistant"]},
]

# Pre-compile keyword regexes (case-insensitive).
for _c in CATEGORIES:
    _c["re"] = re.compile(_c["kw"], re.IGNORECASE)


def keyword_category(title: str) -> str | None:
    """First-match-wins keyword rule. Returns a category name or None."""
    if not title:
        return None
    for c in CATEGORIES:
        if c["re"].search(title):
            return c["name"]
    return None


# ── Embedding fallback (JobBERT seed-title centroids) ───────────────────────────

def build_prototypes(model) -> tuple[list[str], np.ndarray]:
    """Embed each category's seed titles and average -> one prototype per category."""
    names, vecs = [], []
    for c in CATEGORIES:
        emb = model.encode(c["seeds"], normalize_embeddings=True, show_progress_bar=False)
        centroid = emb.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        names.append(c["name"])
        vecs.append(centroid.astype(np.float32))
    return names, np.vstack(vecs)


def embed_classify(titles: list[str], model, proto_names, proto_vecs,
                   threshold: float) -> dict[str, str]:
    """Assign each title to its nearest prototype (or 'other' if below threshold)."""
    if not titles:
        return {}
    out: dict[str, str] = {}
    total = len(titles)
    t0 = time.time()
    for start in range(0, total, _BATCH):
        batch = titles[start:start + _BATCH]
        emb = model.encode(batch, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=_BATCH).astype(np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = emb @ proto_vecs.T            # (B, n_categories)
        best = sim.argmax(axis=1)
        for i, t in enumerate(batch):
            score = float(sim[i, best[i]])
            out[t] = proto_names[int(best[i])] if score >= threshold else "other"
        done = min(start + _BATCH, total)
        if (start // _BATCH) % 20 == 0 or done == total:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (total - done) / max(rate, 1e-6)
            print(f"\r[classify]   embedded {done:,}/{total:,}  "
                  f"({rate:,.0f}/s · ETA {eta:.0f}s)", end="", flush=True)
    print()
    return out


# ── I/O ─────────────────────────────────────────────────────────────────────────

def iter_records(path: Path, limit: int | None):
    """Yield parsed records from the jsonl, skipping malformed lines."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
            n += 1
            if limit and n >= limit:
                return


def norm_skill(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", default="../linkedin_job_posting/job_postings_full.jsonl",
                   help="path to the LinkedIn jsonl")
    p.add_argument("--out-inventory", default="../linkedin_job_posting/category_inventory.json")
    p.add_argument("--out-jsonl", default="../linkedin_job_posting/job_postings_categorized.jsonl")
    p.add_argument("--sample", type=int, default=0,
                   help="process only the first N records (0 = all)")
    p.add_argument("--threshold", type=float, default=_EMBED_THRESHOLD)
    p.add_argument("--no-embed", action="store_true",
                   help="keyword rules only; unmatched -> 'other' (no model load)")
    p.add_argument("--no-labeled", action="store_true",
                   help="skip writing the per-record categorized jsonl (inventory only)")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    limit = args.sample or None
    t0 = time.time()

    # ---- Pass 1: collect unique titles ----
    print(f"[pass 1] scanning titles in {in_path.name} …")
    title_counts: Counter = Counter()
    n_records = 0
    for rec in iter_records(in_path, limit):
        n_records += 1
        title_counts[(rec.get("job_title") or "").strip()] += 1
    unique_titles = [t for t in title_counts if t]
    print(f"         {n_records:,} records · {len(unique_titles):,} unique titles "
          f"· {time.time() - t0:.0f}s")

    # ---- Classify unique titles: keyword first, embedding for the tail ----
    title2cat: dict[str, str] = {}
    unmatched: list[str] = []
    for t in unique_titles:
        c = keyword_category(t)
        if c:
            title2cat[t] = c
        else:
            unmatched.append(t)
    kw_hits = len(title2cat)
    print(f"[classify] keyword-matched {kw_hits:,}/{len(unique_titles):,} titles "
          f"({100*kw_hits/max(len(unique_titles),1):.0f}%) · {len(unmatched):,} to embed")

    if unmatched and not args.no_embed:
        print(f"[classify] loading {_JOBBERT} + embedding the tail …")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_JOBBERT)
        proto_names, proto_vecs = build_prototypes(model)
        title2cat.update(embed_classify(unmatched, model, proto_names, proto_vecs,
                                        args.threshold))
    else:
        for t in unmatched:
            title2cat[t] = "other"

    # ---- Pass 2: tally inventory (+ optionally write labeled records) ----
    print(f"[pass 2] tallying skills per category …")
    inventory: dict[str, Counter] = {}
    cat_jobs: Counter = Counter()
    writer = None
    if not args.no_labeled:
        writer = open(Path(args.out_jsonl).resolve(), "w", encoding="utf-8")

    for rec in iter_records(in_path, limit):
        title = (rec.get("job_title") or "").strip()
        cat = title2cat.get(title, "other")
        cat_jobs[cat] += 1
        inv = inventory.setdefault(cat, Counter())
        for s in rec.get("skills", []) or []:
            ns = norm_skill(s)
            if ns:
                inv[ns] += 1
        if writer:
            writer.write(json.dumps({
                "job_title": rec.get("job_title"),
                "skills": rec.get("skills", []),
                "job_category": cat,
                "location": rec.get("location"),
                "company_name": rec.get("company_name"),
                "full_url": rec.get("full_url"),
            }, ensure_ascii=False) + "\n")
    if writer:
        writer.close()

    # ---- Write inventory cache ----
    out = {
        "categories": {c: dict(inv.most_common()) for c, inv in inventory.items()},
        "category_job_counts": dict(cat_jobs.most_common()),
        "total_jobs": n_records,
        "meta": {"source": in_path.name, "embed": not args.no_embed,
                 "threshold": args.threshold, "sample": args.sample},
    }
    inv_path = Path(args.out_inventory).resolve()
    inv_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # ---- Summary ----
    print()
    print("=" * 64)
    print(f"DONE in {time.time() - t0:.0f}s  ·  {n_records:,} postings")
    print("=" * 64)
    print("postings per category:")
    for c, n in cat_jobs.most_common():
        print(f"  {n:>8,}  {c}")
    if "it" in inventory:
        print("\ntop 12 IT skills:")
        for s, n in inventory["it"].most_common(12):
            print(f"  {n:>6,}  {s}")
    print(f"\n→ inventory: {inv_path}")
    if writer is not None:
        print(f"→ labeled : {Path(args.out_jsonl).resolve()}")


if __name__ == "__main__":
    main()
