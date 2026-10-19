"""ADD suggestions — which market skills should THIS course teach? (V7 panel)

Single-output pipeline (2026-07-12 design: only the ADD tier is produced;
next-course / program-level / not-a-skill groups were removed by design):

    1. CLEAN   two-pass mining over the IT posting cohort with
               deterministic variant merging (space-squash + 'x skills'→x,
               pooled demand recounted per posting — no double counts),
               then a frozen HARD-SKILL filter: soft-skill lexicon +
               credential/boilerplate regex are excluded up front.
    2. RANK    course-conditioned demand: lift(s | course) =
               P(s | posting demands a taught bucket) / P(s);
               rank = P(s|course) × lift. Embedding adjacency reported
               as evidence (esco_match.embed, same encoder as the card).
    3. JUDGE   qwen3 committee, ONE binary question per candidate:
               teach it in THIS course at its level — add / skip (+reason).
               Only 'add' survives into the payload.

Every suggestion carries: share of course-related postings demanding it,
lift vs whole-market baseline, nearest taught unit, committee reason.
Frozen constants below are the audit artifacts (thesis appendix).
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent / "recommendation_v3"
for _p in (str(_HERE), str(_V3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import esco_match as _match                 # noqa: E402  embed()

POSTINGS = _HERE.parent.parent / "playground" / "coverage_score" / "it_postings.jsonl"
_CACHE_FILE = _HERE / ".suggest_cache.json"

VERIFY_MODEL = "qwen3.5:9b"   # DEV speed model — swap back to "qwen3:14b" for frozen/thesis runs
VOCAB_N = 200
LLM_TOP = 25           # hard-skill candidates reviewed by the committee
LIFT_MIN = 1.15        # must be demanded WITH what the course teaches

# ---- frozen exclusion lexicon (audit artifact) ----------------------------
# Soft/professional skills: programme-level concerns, never course ADDs.
SOFT_SKILLS = {
    "communication", "teamwork", "collaboration", "leadership",
    "problem solving", "problemsolving", "critical thinking",
    "time management", "attention to detail", "customer service",
    "interpersonal", "adaptability", "creativity", "work ethic",
    "organization", "organizational", "multitasking", "decision making",
    "decisionmaking", "presentation", "negotiation", "mentoring",
    "coaching", "self motivated", "detail oriented", "analytical",
    "analytical skills", "verbal communication", "written communication",
    "flexibility", "initiative", "prioritization", "team player",
    "documentation",
}
# Credentials / job-ad boilerplate / bare field names — not teachable skills.
NOT_SKILL_RE = re.compile(
    r"(degree|bachelor|master|phd|diploma|certif|gpa"
    r"|years? of experience|experience (with|in))")
FIELD_NAMES = {
    "computer science", "data science", "information technology",
    "engineering", "software engineering", "software development",
    "technology",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _matches(anchor: str, skill: str) -> bool:
    """Whole-phrase match: 'python' hits 'python programming', not 'micropython'."""
    return anchor == skill or f" {anchor} " in f" {skill} "


def _is_hard_skill(skill: str) -> bool:
    if skill in SOFT_SKILLS or skill in FIELD_NAMES:
        return False
    if NOT_SKILL_RE.search(skill):
        return False
    # 'x skills'/'x ability' variants whose head is soft (analytical skills…)
    head = re.sub(r" (skills?|abilit\w+)$", "", skill)
    if head != skill and (head in SOFT_SKILLS or head in FIELD_NAMES):
        return False
    return True


def _cache_read() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except Exception:
        return {}


def _cache_write(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache))


def _canonical_map(vocab: list[tuple[str, int]]) -> dict[str, str]:
    """variant -> canonical (highest-df member). Deterministic rules only:
    A space-squash collisions · B 'x skills'/'x skill' -> bare head."""
    dfd = dict(vocab)
    names = [v[0] for v in vocab]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        keep, drop = (ra, rb) if dfd[ra] >= dfd[rb] else (rb, ra)
        parent[drop] = keep

    squash = {}
    for n in names:
        squash.setdefault(n.replace(" ", ""), []).append(n)
    for group in squash.values():
        for other in group[1:]:
            union(group[0], other)
    for n in names:
        for suf in (" skills", " skill"):
            if n.endswith(suf) and n[: -len(suf)] in dfd:
                union(n, n[: -len(suf)])
    return {n: find(n) for n in names if find(n) != n}


def _mine(anchors: list[str], vocab_n: int) -> dict:
    """Two-pass corpus mining with variant merging (posting counted once
    per canonical skill). Cached by anchors + corpus mtime."""
    key = "mine:" + hashlib.sha1(json.dumps(
        ["v3", sorted(anchors), vocab_n, int(POSTINGS.stat().st_mtime)]
    ).encode()).hexdigest()
    cache = _cache_read()
    if key in cache:
        return cache[key]

    df_raw = Counter()
    n_posts = 0
    with POSTINGS.open(encoding="utf-8") as f:
        for line in f:
            skills = {_norm(s) for s in json.loads(line).get("skills", [])}
            skills.discard("")
            n_posts += 1
            df_raw.update(skills)
    canon = _canonical_map(df_raw.most_common(vocab_n * 2))
    anchors_c = sorted({canon.get(a, a) for a in anchors})

    n_anchored = 0
    df, df_anchored = Counter(), Counter()
    with POSTINGS.open(encoding="utf-8") as f:
        for line in f:
            skills = {canon.get(s, s) for s in
                      (_norm(x) for x in json.loads(line).get("skills", []))}
            skills.discard("")
            if any(_matches(a, sk) for a in anchors_c for sk in skills):
                n_anchored += 1
                df_anchored.update(skills)
            df.update(skills)
    mined = {"n_posts": n_posts, "n_anchored": n_anchored,
             "vocab": df.most_common(vocab_n),
             "df_anchored": dict(df_anchored),
             "canon_map": canon, "anchors_canonical": anchors_c,
             "n_merged": len(canon)}
    cache = _cache_read()
    cache[key] = mined
    _cache_write(cache)
    return mined


def _candidates(mined: dict, taught: list[str], anchors: list[str],
                course_title: str) -> tuple[list[dict], int]:
    """Hard-skill candidates with lift stats. Returns (candidates,
    n_excluded_nonhard) — soft skills/credentials never become candidates."""
    n, n_a = mined["n_posts"], mined["n_anchored"]
    canon = mined.get("canon_map", {})
    have = {canon.get(_norm(t), _norm(t)) for t in taught} \
        | {canon.get(_norm(a), _norm(a)) for a in anchors} \
        | {_norm(course_title)}
    out, excluded = [], 0
    for skill, n_s in mined["vocab"]:
        if any(_matches(h, skill) or _matches(skill, h) for h in have if h):
            continue
        if not _is_hard_skill(skill):
            excluded += 1
            continue
        n_sa = mined["df_anchored"].get(skill, 0)
        p_s, p_sa = n_s / n, (n_sa / n_a) if n_a else 0.0
        lift = (p_sa / p_s) if p_s else 0.0
        if lift < LIFT_MIN:
            continue          # not demanded together with what the course teaches
        out.append({"skill": skill, "demand": n_s, "co_support": n_sa,
                    "p_s": round(p_s, 4),
                    "p_s_given_course": round(p_sa, 4),
                    "lift": round(lift, 2),
                    "score": round(p_sa * lift * 100, 3)})
    return sorted(out, key=lambda c: -c["score"]), excluded


def _adjacency(cands: list[dict], taught: list[str]) -> None:
    if not cands or not taught:
        return
    import numpy as np
    # errstate: Apple Accelerate BLAS raises spurious FP flags in matmul
    # (values verified bit-exact vs manual dot product).
    with np.errstate(all="ignore"):
        sims = _match.embed([c["skill"] for c in cands]) @ _match.embed(taught).T
    for c, row in zip(cands, sims):
        c["adjacency"] = round(float(row.max()), 2)
        c["nearest_taught"] = taught[int(row.argmax())]


_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["add", "skip"]},
                       "reason": {"type": "string"}},
        "required": ["index", "verdict", "reason"]}}},
    "required": ["decisions"],
}


def _llm_verdicts(cands: list[dict], taught: list[str],
                  course_title: str) -> None:
    """One binary committee question per candidate: teach it in THIS course?"""
    import ollama

    numbered = "\n".join(
        f"{i}. {c['skill']} — demanded in {c['p_s_given_course']:.0%} of "
        f"related postings (lift {c['lift']}x vs the whole IT market)"
        for i, c in enumerate(cands))
    response = ollama.chat(
        model=VERIFY_MODEL,
        think=False,
        options={"temperature": 0, "num_ctx": 8192},
        format=_SCHEMA,
        messages=[
            {"role": "system", "content": (
                "You are a university curriculum committee member. For each "
                "candidate skill decide ONE thing: should THIS specific "
                "course add it to its syllabus? 'add' only when it is a "
                "teachable technical skill that naturally extends what the "
                "course already teaches, at this course's level — something "
                "the lecturer could realistically weave into the existing "
                "lessons. 'skip' when it is too advanced for the course's "
                "level, belongs to a different course or specialization, or "
                "is broader than a syllabus line. Short reason each.")},
            {"role": "user", "content": (
                f"Course: {course_title}\n"
                f"Currently teaches: {', '.join(taught)}\n\n"
                f"CANDIDATE ADDITIONS (with labour-market evidence):\n{numbered}")},
        ],
    )
    decisions = json.loads(response.message.content).get("decisions", [])
    by_index = {int(d["index"]): d for d in decisions
                if isinstance(d, dict) and isinstance(d.get("index"), int)}
    for i, c in enumerate(cands):
        d = by_index.get(i, {})
        c["llm_verdict"] = d.get("verdict", "n/a")
        c["llm_reason"] = str(d.get("reason", "")).strip()


def suggest_additions(taught: list[str], anchors: list[str],
                      course_title: str = "", vocab_n: int = VOCAB_N,
                      llm: bool = True, top: int = LLM_TOP,
                      force: bool = False) -> dict:
    """CLEAN → RANK → JUDGE. Payload contains ONLY committee-approved ADDs
    (plus the reviewed-but-skipped list for the audit trail)."""
    key = "result:" + hashlib.sha1(json.dumps(
        ["v3", sorted(taught), sorted(anchors), course_title, vocab_n,
         llm, top, VERIFY_MODEL if llm else None,
         int(POSTINGS.stat().st_mtime)]).encode()).hexdigest()
    cache = _cache_read()
    if not force and key in cache:
        return {**cache[key], "cached": True}

    mined = _mine(anchors, vocab_n)
    cands, n_nonhard = _candidates(mined, taught, anchors, course_title)
    cands = cands[:top]
    _adjacency(cands, taught)
    if llm and cands:
        _llm_verdicts(cands, taught, course_title)
        additions = [c for c in cands if c.get("llm_verdict") == "add"]
        reviewed_skipped = [
            {"skill": c["skill"], "reason": c.get("llm_reason", "")}
            for c in cands if c.get("llm_verdict") != "add"]
    else:
        additions, reviewed_skipped = cands, []

    result = {
        "course_title": course_title,
        "anchors": anchors,
        "anchors_canonical": mined.get("anchors_canonical", anchors),
        "corpus": {"file": POSTINGS.name, "n_posts": mined["n_posts"],
                   "n_anchored": mined["n_anchored"]},
        "params": {"vocab_n": vocab_n, "lift_min": LIFT_MIN,
                   "model": VERIFY_MODEL if llm else None, "llm_top": top,
                   "n_variants_merged": mined.get("n_merged", 0),
                   "n_nonhard_excluded": n_nonhard},
        "additions": additions,
        "reviewed_skipped": reviewed_skipped,
        "method": "suggest_add_v3_add_only",
    }
    cache = _cache_read()
    cache[key] = result
    _cache_write(cache)
    return result
