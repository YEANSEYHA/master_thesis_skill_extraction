# Segment-matched curriculum–market gap analysis: the shared pipeline
# behind dashboard.py.
#
#   market side:  clean_jobs -> classify_jobs -> normalize_skills
#   school side:  load_curriculum (all syllabus PDFs in input1/)
#   measurement:  semantic coverage (max cosine) + KL divergence
#
# Superseded exploratory scripts (naive top-30 baseline, CLI reports)
# live in legacy/.

import json
import os
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# primary encoder: BGE won the stability comparison qualitatively and
# leads on published benchmarks; MiniLM kept as the baseline reference
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BASELINE_MODEL = "all-MiniLM-L6-v2"

# Similarity thresholds are ENCODER-SPECIFIC (proven by the stability
# experiment: BGE shifts the low end up by ~+0.25).
#
# BGE values are CALIBRATED against the 250-pair human reference study
# (05_evaluation/annotation/, sheet_hash 78dfd3fe…, 8 blind raters,
# ICC(2,k)=0.808, AUC=0.812). Each threshold is the max-Youden-J point
# against the gold cut matching the claim its band makes:
#
#   covered    0.7368  gold "teaches"   (MOS>=7)  95% CI [0.717, 0.787]
#   borderline 0.6903  gold "addresses" (MOS>=5)  95% CI [0.663, 0.737]
#   host_min   0.6903  same boundary as borderline — the human data
#                      resolves only TWO boundaries, not three
#
# Superseded PROVISIONAL values (linearly mapped from MiniLM, used by
# every run up to and including the locked 2026-07-27 thesis set):
#   covered 0.73 · borderline 0.69 · host_min 0.66
#
# CAVEATS to carry into any write-up: 8 complete raters against a
# pre-registered adequacy floor of 10, so these remain provisional; only
# one lecturer passed screening; and covered sits near the LOW end of its
# interval (precision 0.566 against "teaches"), so a stricter variant is
# equally compatible with the data. MiniLM values stay MiniLM-calibrated.
THRESHOLDS = {
    "all-MiniLM-L6-v2":      {"covered": 0.50, "borderline": 0.42,
                              "host_min": 0.35},
    "BAAI/bge-base-en-v1.5": {"covered": 0.7368, "borderline": 0.6903,
                              "host_min": 0.6903},
}


def get_thresholds(model_name: str = None) -> dict:
    return THRESHOLDS[model_name or MODEL_NAME]


def best_device():
    """Apple-Silicon GPU (MPS) when available — ~2.4x faster for
    base-size encoders on M-series; falls back to default otherwise."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=best_device())

# phase5 root — this file lives in 03_engine/, so every data path below stays
# anchored on the project root exactly as it was before the folder reorg
BASE_DIR = Path(__file__).resolve().parent.parent
CURRICULUM_DIR = BASE_DIR / "02_input_curriculum"  # ALL syllabus PDFs

# The job-market side can come from more than one corpus. Select with the
# JOB_SOURCE env var (default "linkedin"). `load_raw_postings` already reads
# both the flat LinkedIn array and the {input, output} Phase-3 scraper format,
# so the Cambodia stage2 output plugs in unchanged.
DEFAULT_SOURCE = "linkedin"
JOB_SOURCES = {
    # 140,393 pre-cleaned US LinkedIn postings — the validation baseline
    "linkedin": BASE_DIR / "01_input_jobmarket" / "it_job_linkedin.json",
    # Phase-3 full-corpus API extraction of the Cambodian crawl (4,876 postings)
    "cambodia": (BASE_DIR.parent / "phase3_skill_extraction"
                 / "outputs" / "stage2" / "api_full" / "stage2_result.json"),
}


def resolve_job_source(source: str | None = None) -> tuple[str, Path]:
    """(source_key, dataset_path) for the requested / env-selected corpus."""
    src = (source or os.environ.get("JOB_SOURCE") or DEFAULT_SOURCE).lower()
    if src not in JOB_SOURCES:
        raise ValueError(f"unknown JOB_SOURCE {src!r}; "
                         f"choose one of {list(JOB_SOURCES)}")
    return src, JOB_SOURCES[src]


JOB_SOURCE, RAW_JOBS = resolve_job_source()

# How the IT/AI sub-market is selected from the cleaned postings:
#   "classifier" — zero-shot English anchor classification.
#   "category"   — keep postings whose extracted job_category is
#                  IT-related (title as fallback). More reliable on
#                  corpora that carry a real category field (the
#                  Cambodian stage2 extraction, ~271 IT); NOT for the
#                  flat LinkedIn set, which has no job_category.
# The default follows the corpus (cambodia → category, else classifier)
# so an API-launched run without the env var lands on the validated
# config for that source; SEGMENT_MODE still overrides explicitly.
SEGMENT_MODE = (os.environ.get("SEGMENT_MODE")
                or ("category" if JOB_SOURCE == "cambodia"
                    else "classifier")).lower()

MIN_SKILL_FREQ = 2    # skill must appear in >= 2 segment jobs

# Cross-encoder host verification (retrieve-then-rerank, Nogueira & Cho
# 2019) — computed inline per run by gen_report_data against that run's
# own curriculum, replacing the 05_evaluation/rerank_results.json sidecar that
# went stale whenever the curriculum changed. Reranker scores are a
# THIRD scale — never compare them with bi-encoder cosines. Bands are
# PROVISIONAL until the task-#1 annotation calibrates them.
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 3          # CLO candidates per skill to re-read jointly
RERANK_SUPPORTED_AT = 0.50
RERANK_WEAK_AT = 0.10

_reranker = None


def rerank_pair_scores(pairs: list) -> np.ndarray:
    """Sigmoid relevance for (skill, CLO-text) pairs from the
    cross-encoder; the model loads lazily once per process (~2.3 GB,
    cached on disk after the first download)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_NAME, device=best_device(),
                                 max_length=256)
    scores = np.asarray(_reranker.predict(pairs, batch_size=32,
                                          show_progress_bar=False),
                        dtype=float)
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-scores))   # raw logits -> sigmoid
    return scores


def rerank_verdict(score: float) -> str:
    if score >= RERANK_SUPPORTED_AT:
        return "SUPPORTED"
    if score >= RERANK_WEAK_AT:
        return "WEAK"
    return "NOT SUPPORTED"
EPS = 1e-6            # smoothing for KL divergence
SKILL_MERGE_THRESHOLD = 0.80  # cosine sim to merge skill string variants

NON_JOB_CATEGORIES = ("call for proposals", "invitation for bids",
                      "tender", "expression of interest", "auction")

AI_JOB_ANCHORS = [
    "machine learning engineer building and deploying AI models",
    "data scientist doing statistical analysis and predictive modeling",
    "artificial intelligence engineer working on deep learning and NLP",
    "data analyst analyzing datasets and building reports and dashboards",
    "data engineer building data pipelines, databases and SQL queries",
]
IT_JOB_ANCHORS = [
    "software developer programming web and mobile applications",
    "IT support officer and system administrator managing networks",
    "devops engineer managing cloud infrastructure and deployment",
    "cybersecurity analyst protecting information systems",
    "database administrator maintaining database servers",
]
NON_IT_JOB_ANCHORS = [
    "sales and marketing officer promoting products and services",
    "accountant and finance officer preparing financial reports",
    "NGO program officer for community development projects",
    "teacher, trainer and education program coordinator",
    "customer service and front desk staff assisting clients",
    "human resources and administration officer",
    "logistics, procurement and supply chain coordinator",
    "civil engineer and construction site manager",
    "nurse, doctor and healthcare staff",
    "laboratory technician performing biological and medical lab tests",
    "banking and microfinance credit officer",
]

KHMER_RE = re.compile(r"[ក-៿]")
NON_LATIN_RE = re.compile(r"[Ѐ-ӿ一-鿿฀-๿]")
ENGLISH_HINTS = {"the", "and", "of", "to", "in", "for", "with", "a", "is",
                 "or", "on", "as", "will", "be", "work", "experience",
                 "skills", "team", "management", "support"}

# classroom rules, grading policy and exam logistics are not taught
# content — leaving them in creates false coverage for soft skills
# ("communication" matching "Pay attention in the classroom ...")
NON_CONTENT_RE = re.compile(
    r"come to class|pay attention|participate|do not litter|punctual|"
    r"pack up your bags|sidebars|be respectful|treat everyone|"
    r"leave (during )?class|exam session|plagiarism|cheating|turnitin|"
    r"absen[ct]|late (assessment|submission)|due date|deferred|"
    r"passing score|board of examiners|disciplinary|student webmail|"
    r"assessment extension|marks? (allocated|of zero)|working days",
    re.IGNORECASE)


# ------------------------------------------------------------ market side

def looks_english(text: str) -> bool:
    """Cheap language gate: the embedding model is English-only."""
    if NON_LATIN_RE.search(text):
        return False
    tokens = re.findall(r"[a-zA-Z']+", text.lower())[:80]
    if len(tokens) < 5:
        return False
    hits = sum(1 for t in tokens if t in ENGLISH_HINTS)
    return hits / len(tokens) >= 0.05


def as_text(value) -> str:
    """Flatten str / list-of-str / list-of-{text,...} fields to one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def as_category_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [c.strip() for c in value.split(",") if c.strip()]
    if isinstance(value, list):
        return [str(c).strip() for c in value if str(c).strip()]
    return []


# entries in `skills` that are job-ad boilerplate, not skills
# ("Full time", "Equal employment", "Bachelor's Degree", shift names, ...)
SKILL_NOISE_RE = re.compile(
    r"degree|diploma|full.?time|part.?time|\bshift\b|^day$|^night$|"
    r"equal employment|discriminat|harass|accommodat|workplace|"
    r"\bconduct\b|environment$|justice|commitment|year.? experience|"
    r"experience (with|in)\b",
    re.IGNORECASE)


def load_raw_postings(raw_path: Path) -> list[dict]:
    """Support all dataset formats seen so far:
    - .jsonl: one flat posting object per line (it_postings set)
    - .json : array of {input, output} scraper records (bongthom set)
              OR array of flat posting objects (it_job_linkedin set)"""
    if raw_path.suffix == ".jsonl":
        records = []
        with raw_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    return [rec["output"] if isinstance(rec.get("output"), dict) else rec
            for rec in json.loads(raw_path.read_text())]


def clean_jobs(raw_path: Path) -> tuple[list[dict], dict]:
    raw = load_raw_postings(raw_path)
    jobs, stats = [], Counter()
    seen: set[tuple] = set()
    for out in raw:
        cats = as_category_list(out.get("job_category"))
        if any(nj in c.lower() for c in cats for nj in NON_JOB_CATEGORIES):
            stats["dropped_non_job"] += 1
            continue
        title = as_text(out.get("job_title")).strip()
        desc = as_text(out.get("job_description")).strip()
        req = as_text(out.get("job_requirement")).strip()
        skills = []
        for s in out.get("skills") or []:
            s = str(s).lower().strip()
            # drop boilerplate pseudo-skills and requirement fragments
            if s and len(s) <= 50 and not SKILL_NOISE_RE.search(s):
                skills.append(s)
        # no description in the flat format: classify on title + skills
        body = desc or ", ".join(skills[:12])
        text = " | ".join(p for p in (title, ", ".join(cats), body) if p)
        if not text:
            stats["dropped_empty"] += 1
            continue
        # the English MiniLM model cannot score Khmer-dominant postings
        khmer_ratio = len(KHMER_RE.findall(text)) / max(len(text), 1)
        if khmer_ratio > 0.15:
            stats["skipped_khmer"] += 1
            continue
        # the stopword-ratio test only makes sense on prose; records
        # without a description are keyword lists ("python, sql, aws")
        # where near-zero stopwords is normal — gate those on script only
        if desc:
            if not looks_english(text):
                stats["skipped_non_english"] += 1
                continue
        elif NON_LATIN_RE.search(text):
            stats["skipped_non_english"] += 1
            continue
        # scraper repeats the same ad across pages: dedup on content
        key = (title.lower(), tuple(sorted(skills)))
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        stats["kept"] += 1
        jobs.append({
            "title": title or "(untitled)",
            "categories": cats,
            "text": text[:1500],
            "requirement": req,
            "skills": skills,
            # provenance passthrough (Phase-3 records only; empty on the flat
            # LinkedIn set) — used by 05_evaluation/dedup_sensitivity.py to
            # key cross-site duplicates on employer + site, never by the
            # scoring path
            "company": as_text(out.get("company_name")).strip(),
            "site": as_text(out.get("source_file")).split("/")[0],
            "url": as_text(out.get("full_url")),
            "posted_date": as_text(out.get("posted_date")),
        })
    return jobs, dict(stats)


def classify_jobs(model, jobs: list[dict]) -> None:
    """Assign each job a bucket: ai_data / other_it / non_it
    (zero-shot anchor classification)."""
    job_emb = model.encode([j["text"] for j in jobs],
                           normalize_embeddings=True, batch_size=128,
                           show_progress_bar=False)
    groups = {
        "ai_data": model.encode(AI_JOB_ANCHORS, normalize_embeddings=True),
        "other_it": model.encode(IT_JOB_ANCHORS, normalize_embeddings=True),
        "non_it": model.encode(NON_IT_JOB_ANCHORS, normalize_embeddings=True),
    }
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = {name: (job_emb @ emb.T).max(axis=1)
                  for name, emb in groups.items()}
    names = list(groups)
    stacked = np.stack([scores[n] for n in names])   # (3, jobs)
    best = stacked.argmax(axis=0)
    for i, job in enumerate(jobs):
        job["bucket"] = names[int(best[i])]
        job["bucket_scores"] = {n: round(float(scores[n][i]), 3)
                                for n in names}


# Category-first IT segmentation (SEGMENT_MODE="category"). Stage 1 keeps a
# posting when its extracted job_category is IT-related; stage 2 rescues the
# few IT jobs whose category was mislabeled (e.g. "DevOps Engineer" tagged
# "engineering") via a strict tech-title match. Mirrors phase-3 it_filter.py.
CATEGORY_IT_RE = re.compile(
    r"\b(it|ict|information technology|software|web|programming|developer|"
    r"network|database|system (admin|development)|cyber|data (science|analytics|"
    r"engineering)|artificial intelligence|machine learning|computer|"
    r"quality assurance|devops|telecommunication|digital)\b")
TITLE_IT_RE = re.compile(
    r"\b(software|developer|programmer|full.?stack|front.?end|back.?end|"
    r"web design|\.net|java(script)?|python|php|golang|node\.?js|react|"
    r"it officer|it support|it manager|it staff|it admin|it programmer|"
    r"system admin|sysadmin|network (admin|engineer|specialist)|"
    r"database admin|dba|data analyst|data engineer|data scientist|devops|"
    r"cyber ?security|qa engineer|software tester|mobile developer|"
    r"android developer|ios developer|ui/ux|ux/ui|ai engineer|"
    r"machine learning engineer|blockchain)\b", re.I)


def is_it_by_category(job: dict) -> bool:
    """IT membership from the extracted job_category (title as fallback) —
    the reliable filter for corpora with a real category field."""
    cat = " , ".join(job.get("categories") or []).lower()
    if CATEGORY_IT_RE.search(cat):
        return True
    return bool(TITLE_IT_RE.search(job.get("title") or ""))


MAX_SKILLS_TO_NORMALIZE = 8000  # greedy clustering is O(n*k); the tail
                                # of one-off strings can't reach the
                                # heatmap anyway (MIN_SKILL_FREQ)


def normalize_skills(model, skill_counter: Counter) -> dict[str, str]:
    """Map each raw skill string to a canonical one (greedy embedding
    clustering; the most frequent string in a cluster becomes canonical)."""
    raw = [s for s, _ in
           skill_counter.most_common(MAX_SKILLS_TO_NORMALIZE)]
    emb = model.encode(raw, normalize_embeddings=True,
                       show_progress_bar=False)
    canonical_idx: list[int] = []
    mapping: dict[str, str] = {}
    for i, s in enumerate(raw):
        if canonical_idx:
            with np.errstate(divide="ignore", over="ignore",
                             invalid="ignore"):
                sims = emb[canonical_idx] @ emb[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= SKILL_MERGE_THRESHOLD:
                mapping[s] = raw[canonical_idx[j]]
                continue
        canonical_idx.append(i)
        mapping[s] = s
    return mapping


# ------------------------------------------------------------ school side

def extract_curriculum_chunks(pdf_path: Path) -> list[str]:
    """Pull topic-bearing text chunks out of a syllabus PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    chunks = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" •-\t")
        # keep lines long enough to carry a topic, skip page furniture
        if len(line) < 15:
            continue
        if re.match(r"^(Slide|LCD|Laptop|Writing|Attendances|Take note|"
                    r"Discussion|Presentation|Ask question|Report|Research|"
                    r"Tutorial|Group activities|Self-paced|Assignment|Quiz|"
                    r"http|www\.)", line, re.IGNORECASE):
            continue
        if NON_CONTENT_RE.search(line):
            continue
        chunks.append(line)
    return chunks


def load_curriculum(courses_dir: Path) -> tuple[list[str], list[str]]:
    """Extract topic chunks from every syllabus PDF, tagged with the
    course it came from, so coverage can say WHICH course covers a skill.
    Course titles are not used as chunks on their own — a course named
    'Business Information Technology' that actually teaches calculus
    must not create false coverage for IT skills."""
    chunks, courses = [], []
    for pdf in sorted(courses_dir.glob("*.pdf")):
        for chunk in extract_curriculum_chunks(pdf):
            chunks.append(chunk)
            courses.append(pdf.stem)
    return chunks, courses


# soft-skill / non-technical strings to exclude from the market side
# before curriculum comparison — they are addressed by teaching methods,
# not course content, so they only clutter the technical gap ranking.
# (word patterns kept specific to avoid false positives: "creativity"
# but not "creative cloud"; "analytical skills" but not "analytics")
SOFT_SKILL_RE = re.compile(
    r"communicat|team.?work|team player|collaborat|interpersonal|"
    r"leadership|problem.?solving|critical thinking|"
    r"analytical (skills?|thinking|aptitude)|attention to detail|"
    r"detail.?orient|time management|organizational skills?|"
    r"multi.?task|adaptab|flexib|creativity|creative thinking|"
    r"work ethic|self.?motivat|self.?start|negotiat|decision.?making|"
    r"presentation skills?|public speaking|customer service|"
    r"customer support|client relation|relationship building|"
    r"verbal|written communication|listening|empath|"
    r"positive attitude|professionalism|work independently|"
    r"fast.?paced|under pressure|prioriti[zs]|"
    r"mentor|coaching|conflict resolution",
    re.IGNORECASE)


def is_soft_skill(skill: str) -> bool:
    """True when the skill string is a soft/behavioral skill that should
    not compete with technical skills in curriculum-content comparison."""
    return bool(SOFT_SKILL_RE.search(skill))


# ------------------------------------------------------------ measurement

# NOTE: use get_thresholds() — these module constants are the ACTIVE
# model's values, kept for import convenience.
COVERED_AT = THRESHOLDS[MODEL_NAME]["covered"]
BORDERLINE_AT = THRESHOLDS[MODEL_NAME]["borderline"]


def kl_divergence(q: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """D_KL(q || p) with smoothing; also returns per-skill contributions."""
    qs = (q + EPS) / (q + EPS).sum()
    ps = (p + EPS) / (p + EPS).sum()
    contrib = qs * np.log(qs / ps)
    return float(contrib.sum()), contrib


# ------------------------------------------------------------ market build

CACHE_VERSION = "v3"  # bump when market-build logic/params change


def market_cache_path(model_name: str, source: str = None,
                      segment_mode: str = None) -> Path:
    slug = model_name.replace("/", "_")
    # the default source + classifier mode keep the historical filename (no
    # cache invalidation); other corpora / modes get a suffix so all coexist
    src = source or JOB_SOURCE
    mode = segment_mode or SEGMENT_MODE
    suffix = "" if src == DEFAULT_SOURCE else f"_{src}"
    if mode != "classifier":
        suffix += f"_{mode}"
    return BASE_DIR / "cache" / f"market_cache_{slug}{suffix}.npz"


def build_market(model, model_name: str = MODEL_NAME) -> dict:
    """Clean + classify + count the job-market side; cached per dataset
    AND model version because embedding 100k+ postings takes minutes."""
    cache_file = market_cache_path(model_name)
    stat = RAW_JOBS.stat()
    cache_key = (f"{CACHE_VERSION}-{model_name}-{RAW_JOBS.name}-{SEGMENT_MODE}-"
                 f"{stat.st_size}-{int(stat.st_mtime)}")
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        if str(data["key"]) == cache_key:
            meta = json.loads(str(data["meta"]))
            meta["counts"] = Counter(meta["counts"])
            meta["demand_dist"] = data["demand_dist"]
            meta["skill_emb"] = data["skill_emb"]
            print("  (market loaded from cache)")
            return meta

    jobs, stats = clean_jobs(RAW_JOBS)
    classify_jobs(model, jobs)
    if SEGMENT_MODE == "category":
        # segment on the extracted job_category (title fallback) — reliable on
        # corpora that carry a real category field (Cambodian stage2)
        segment = [j for j in jobs if is_it_by_category(j) and j["skills"]]
    else:
        segment = [j for j in jobs if j["bucket"] in ("ai_data", "other_it")
                   and j["skills"]]
    raw = Counter(s for j in segment for s in set(j["skills"]))
    mapping = normalize_skills(model, raw)
    counts = Counter()
    for j in segment:
        counts.update({mapping.get(s, s) for s in j["skills"]})
    # skill-base bar scales with corpus size: >=2 of 119 jobs was fine,
    # but >=2 of 22k admits noise — require ~0.2% of segment jobs
    min_freq = max(MIN_SKILL_FREQ, len(segment) // 500)
    skill_base = sorted(s for s, c in counts.items() if c >= min_freq)
    demand = np.array([counts[s] for s in skill_base], dtype=float)
    bucket_counts = Counter(j["bucket"] for j in jobs)
    samples = [{"title": j["title"],
                "category": ", ".join(j["categories"][:2]) or "—",
                "bucket": j["bucket"],
                "skills": ", ".join(j["skills"][:4])}
               for j in segment[:5]]
    market = {
        "stats": stats,
        "n_raw": sum(stats.values()),
        "bucket_counts": dict(bucket_counts),
        "samples": samples,
        "n_jobs": len(segment),
        "skill_base": skill_base,
        "counts": counts,
        "demand_dist": demand / demand.sum(),
        "skill_emb": model.encode(skill_base, normalize_embeddings=True,
                                  show_progress_bar=False),
    }
    meta = {k: market[k] for k in ("stats", "n_raw", "bucket_counts",
                                   "samples", "n_jobs", "skill_base")}
    meta["counts"] = dict(counts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)   # cache/ (see reorg)
    np.savez_compressed(cache_file, key=cache_key,
                        meta=json.dumps(meta),
                        demand_dist=market["demand_dist"],
                        skill_emb=market["skill_emb"])
    return market


def greedy_skill_selection(demand_w, cov0, cand_names, cand_sim):
    """Prescriptive layer: WHICH candidate skills should the program add?
    Facility-location objective (monotone submodular):
        f(S) = sum_i demand_w[i] * max(cov0[i], max_{s in S} sim(s, i))
    over the whole demand distribution. Greedy maximization is
    (1 - 1/e)-near-optimal (Nemhauser, Wolsey & Fisher 1978) and its
    marginal gains are non-increasing, so a gain cutoff is a prefix of
    the returned order. "Adding skill s" is modeled as a new CLO whose
    embedding is s itself; similar market skills gain coverage through
    the same max-rule as everywhere else (spillover).
    Returns greedy-ordered steps: {skill, gain, profile-after}."""
    profile, steps = cov0.copy(), []
    remaining = list(range(len(cand_names)))
    while remaining:
        best_gain, best_c = -1.0, remaining[0]
        for c in remaining:
            gain = float((demand_w * np.clip(cand_sim[c] - profile,
                                             0.0, None)).sum())
            if gain > best_gain:
                best_gain, best_c = gain, c
        profile = np.maximum(profile, cand_sim[best_c])
        steps.append({"skill": cand_names[best_c], "gain": best_gain,
                      "profile": profile.copy()})
        remaining.remove(best_c)
    return steps
