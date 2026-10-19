"""Course-conditioned ADD suggestions — which market skills should this course add?

Replaces the raw demand-ranked ADD list with a three-layer recommender:

    Layer 1  ML   association mining over the IT posting cohort
                  lift(s | course) = P(s | posting demands a taught skill) / P(s)
                  -> demotes generic skills (communication: demanded everywhere,
                     lift ~1) and promotes skills demanded TOGETHER with what the
                     course already teaches (sql/aws with python+ML).
    Layer 2  DL   MiniLM embedding adjacency: max cosine(candidate, taught unit)
                  -> reported as supporting evidence (is it semantically near the
                     course's territory?).
    Layer 3  AI   qwen3 curriculum-committee verdict on the top candidates:
                  add / skip + one-line reason. Rejects credentials and skills
                  that do not belong in THIS course.

Demand and lift are computed from the SAME corpus (it_postings.jsonl, one pass),
so the numbers are internally consistent. Anchors default to the verified
taught buckets of the Intro-AI action-card run (V7 cache, it lens).

Run:  python suggest_add.py            (full: stats + embeddings + qwen3)
      python suggest_add.py --no-llm   (stats + embeddings only)
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from main import _embed  # playground MiniLM helper (lazy encoder)

HERE = Path(__file__).resolve().parent
POSTINGS = HERE / "it_postings.jsonl"
CACHE = HERE / ".suggest_cache.json"
RESULT = HERE / "suggest_result.json"

VERIFY_MODEL = "qwen3:14b"
VOCAB_N = 200          # candidate pool = top-N market skills by posting frequency
LLM_TOP = 40           # only the best-scored candidates reach the LLM
LIFT_ADD = 1.15        # course-conditioned demand
LIFT_GENERIC = 0.80    # below ADD, above this = cross-cutting/program-level

# Intro-AI (AIECS031) — the 18 taught units of the cached V7 it-lens run.
TAUGHT = [
    "machine learning", "K-Nearest Neighbor", "statistics",
    "Python (computer programming)", "Clustering", "Gaussian Mixture",
    "Linear Regression", "Decision Trees", "Cross-Validation",
    "utilise machine learning", "Gradient Descent", "AI Application Development",
    "adapt evaluation methodology", "Data Preparation",
    "Algorithm Selection for Datasets", "AI Sub-Fields", "pandas", "jupyter",
]
# Verified taught buckets (taught_in_demand.sought_as + core_covered.skill).
ANCHORS = ["python", "machine learning", "statistics",
           "data analysis", "analytical skills"]
COURSE_TITLE = "Introduction to Artificial Intelligence (AIECS031)"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _matches(anchor: str, skill: str) -> bool:
    """Whole-phrase match: 'python' hits 'python programming', not 'micropython'."""
    return anchor == skill or f" {anchor} " in f" {skill} "


def _mine(postings_path: Path, anchors: list[str], vocab_n: int) -> dict:
    """One pass over the corpus: vocab frequencies + anchor co-occurrence."""
    n_posts = 0
    n_anchored = 0
    df = Counter()          # skill -> posting frequency
    df_anchored = Counter() # skill -> frequency within anchored postings
    with postings_path.open(encoding="utf-8") as f:
        for line in f:
            skills = {_norm(s) for s in json.loads(line).get("skills", [])}
            skills.discard("")
            n_posts += 1
            anchored = any(
                _matches(a, sk) for a in anchors for sk in skills)
            if anchored:
                n_anchored += 1
                df_anchored.update(skills)
            df.update(skills)
    vocab = df.most_common(vocab_n)
    return {"n_posts": n_posts, "n_anchored": n_anchored,
            "vocab": vocab, "df_anchored": dict(df_anchored)}


def _candidates(mined: dict, taught: list[str], anchors: list[str]) -> list[dict]:
    """Vocab minus what the course already has, with lift statistics."""
    n, n_a = mined["n_posts"], mined["n_anchored"]
    have = {_norm(t) for t in taught} | set(anchors) | {_norm(COURSE_TITLE)}
    out = []
    for skill, n_s in mined["vocab"]:
        if any(_matches(h, skill) or _matches(skill, h) for h in have):
            continue
        n_sa = mined["df_anchored"].get(skill, 0)
        p_s, p_sa = n_s / n, (n_sa / n_a) if n_a else 0.0
        lift = (p_sa / p_s) if p_s else 0.0
        out.append({
            "skill": skill, "demand": n_s, "co_support": n_sa,
            "p_s": round(p_s, 4), "p_s_given_course": round(p_sa, 4),
            "lift": round(lift, 2),
            # rank: common together with the course AND distinctive to it
            "score": round(p_sa * lift * 100, 3),
        })
    return sorted(out, key=lambda c: -c["score"])


def _adjacency(cands: list[dict], taught: list[str]) -> None:
    """DL layer: max cosine of each candidate to any taught unit (MiniLM)."""
    sims = _embed([c["skill"] for c in cands]) @ _embed(taught).T
    for c, row in zip(cands, sims):
        c["adjacency"] = round(float(row.max()), 2)
        c["nearest_taught"] = taught[int(row.argmax())]


_VERDICTS = ["add", "next_course", "program_level", "not_a_skill", "off_stack"]
_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"},
                       "verdict": {"type": "string", "enum": _VERDICTS},
                       "reason": {"type": "string"}},
        "required": ["index", "verdict", "reason"]}}},
    "required": ["decisions"],
}


def _llm_verdicts(cands: list[dict], taught: list[str]) -> None:
    """AI layer: qwen3 as curriculum committee — typed verdict per candidate."""
    import ollama

    numbered = "\n".join(
        f"{i}. {c['skill']} — demanded in {c['p_s_given_course']:.0%} of "
        f"related postings (lift {c['lift']}x vs the whole IT market)"
        for i, c in enumerate(cands))
    print(f"  asking {VERIFY_MODEL} for verdicts on {len(cands)} candidates",
          flush=True)
    response = ollama.chat(
        model=VERIFY_MODEL,
        think=False,
        options={"temperature": 0, "num_ctx": 8192},
        format=_SCHEMA,
        messages=[
            {"role": "system", "content": (
                "You are a university curriculum committee member deciding what "
                "to do with each candidate skill for THIS specific course. "
                "Verdicts:\n"
                "add — teachable technical skill that naturally extends what "
                "this course already teaches, at this course's level.\n"
                "next_course — real technical skill in the same career track, "
                "but belongs in a later/advanced or sibling course, not this "
                "one (too advanced, or a specialized tool/platform beyond this "
                "course's scope).\n"
                "program_level — generic professional/soft skill every course "
                "builds implicitly (communication, teamwork, leadership); a "
                "programme concern, not a syllabus line.\n"
                "not_a_skill — credential, degree requirement, field name, or "
                "job-ad boilerplate, not a teachable skill.\n"
                "off_stack — technical skill from a different specialization "
                "with no natural place in this course's track.\n"
                "One decision per numbered candidate, with a short reason.")},
            {"role": "user", "content": (
                f"Course: {COURSE_TITLE}\n"
                f"Currently teaches: {', '.join(taught)}\n\n"
                f"CANDIDATE ADDITIONS (with labour-market evidence):\n{numbered}")},
        ],
    )
    decisions = json.loads(response.message.content).get("decisions", [])
    by_index = {int(d["index"]): d for d in decisions
                if isinstance(d, dict) and isinstance(d.get("index"), int)}
    for i, c in enumerate(cands):
        d = by_index.get(i, {})
        verdict = d.get("verdict")
        c["llm_verdict"] = verdict if verdict in _VERDICTS else "n/a"
        c["llm_reason"] = str(d.get("reason", "")).strip()


def _tier(c: dict) -> str:
    verdict = c.get("llm_verdict")
    if verdict == "add":
        return "ADD" if c["lift"] >= LIFT_ADD else "ADD (weak demand link)"
    if verdict == "next_course":
        return "NEXT-COURSE"
    if verdict == "program_level":
        return "PROGRAM-LEVEL"
    if verdict == "not_a_skill":
        return "NOT-A-SKILL"
    if verdict == "off_stack":
        return "OTHER-STACK"
    # no LLM review (below LLM_TOP or --no-llm): statistics only
    if c["lift"] >= LIFT_ADD:
        return "unreviewed (lift high)"
    if c["lift"] >= LIFT_GENERIC:
        return "unreviewed (generic)"
    return "unreviewed (low lift)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--vocab-n", type=int, default=VOCAB_N)
    args = ap.parse_args()

    key = hashlib.sha1(json.dumps(
        ["v1", ANCHORS, args.vocab_n, int(POSTINGS.stat().st_mtime)]
    ).encode()).hexdigest()
    try:
        cached = json.loads(CACHE.read_text())
    except Exception:
        cached = {}
    if key in cached:
        mined = cached[key]
        print(f"corpus stats cached ({mined['n_posts']:,} postings)", flush=True)
    else:
        print(f"mining {POSTINGS.name} …", flush=True)
        mined = _mine(POSTINGS, ANCHORS, args.vocab_n)
        cached[key] = mined
        CACHE.write_text(json.dumps(cached))
        print(f"  {mined['n_posts']:,} postings, "
              f"{mined['n_anchored']:,} anchored "
              f"({mined['n_anchored'] / mined['n_posts']:.0%})", flush=True)

    cands = _candidates(mined, TAUGHT, ANCHORS)
    _adjacency(cands, TAUGHT)
    if not args.no_llm:
        _llm_verdicts(cands[:LLM_TOP], TAUGHT)

    for c in cands:
        c["tier"] = _tier(c)

    raw_rank = {c["skill"]: i + 1
                for i, c in enumerate(sorted(cands, key=lambda c: -c["demand"]))}
    print(f"\n{COURSE_TITLE}")
    print(f"anchors (verified taught buckets): {', '.join(ANCHORS)}")
    print(f"{'':2}{'skill':<24}{'demand':>7}{'P(s|A)':>8}{'lift':>6}"
          f"{'adj':>5}{'raw#':>5}  tier / llm")
    for i, c in enumerate(cands[:40], 1):
        shift = raw_rank[c["skill"]] - i
        arrow = f"({'+' if shift > 0 else ''}{shift})" if shift else ""
        llm = f" · {c['llm_verdict']}: {c['llm_reason'][:60]}" \
            if c.get("llm_verdict") else ""
        print(f"{i:>2} {c['skill']:<24}{c['demand']:>7,}"
              f"{c['p_s_given_course']:>8.1%}{c['lift']:>6.2f}"
              f"{c['adjacency']:>5.2f}{raw_rank[c['skill']]:>4}{arrow:<6}"
              f"{c['tier']}{llm}")

    RESULT.write_text(json.dumps({
        "course": COURSE_TITLE, "taught": TAUGHT, "anchors": ANCHORS,
        "corpus": {"file": POSTINGS.name, "n_posts": mined["n_posts"],
                   "n_anchored": mined["n_anchored"]},
        "params": {"vocab_n": args.vocab_n, "lift_add": LIFT_ADD,
                   "lift_generic": LIFT_GENERIC, "llm_top": LLM_TOP,
                   "model": None if args.no_llm else VERIFY_MODEL},
        "suggestions": cands,
    }, indent=1))
    print(f"\nsaved -> {RESULT.name}")


if __name__ == "__main__":
    main()
