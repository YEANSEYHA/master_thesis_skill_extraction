# LLM judges for the frozen 250-pair V8 annotation sheet (0-10 scale).
#
# Same protocol as llm_judge_itc.py (the completed 150-pair categorical
# study), upgraded to the human study's 0-10 scale so LLM and human
# scores are directly comparable on the SAME items:
#   - judges read annotation/sheet_250_master.json in its frozen row
#     order but see ONLY skill / outcome / course(+year) — never cosine,
#     band, set or jobs;
#   - identical frozen instruction for every judge and every pair;
#   - two independent model families = two annotators;
#   - model IDs are recorded on every cached line and in the results
#     meta, so the thesis can cite exactly what judged what, and a
#     cache line from a different model is never silently reused.
#
# Usage:
#   export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
#   python3 05_evaluation/judge_250.py --judge both
#   python3 05_evaluation/judge_250.py --judge gpt --gpt-model gpt-5.6-terra
#   python3 05_evaluation/judge_250.py --score
#
# Calls are cached per judge+model in annotation/judge250_<name>.jsonl;
# safe to re-run after a crash -- only missing pairs are sent.
# NOTE: claude-sonnet-5 rejects the temperature parameter and GPT-5.x
# fixes it internally -- determinism caveat disclosed in the write-up.

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANNO = HERE / "annotation"
MASTER = ANNO / "sheet_250_master.json"
RESULTS_FILE = HERE / "judge250_results.json"

CLAUDE_MODEL_DEFAULT = "claude-sonnet-5"
GPT_MODEL_DEFAULT = "gpt-5.6-terra"

YEAR_NAMES = {"Y3": "Year 3", "Y4": "Year 4", "Y5": "Year 5"}

INSTRUCTION = (
    "You are annotating curriculum coverage for a university program. "
    "Each item pairs a skill demanded in Cambodian IT job postings with "
    "one learning outcome from an ITC (Institute of Technology of "
    "Cambodia) course. Judge from the text alone: how well does this "
    "course outcome teach this skill? Score 0-10: 0 = this skill is not "
    "taught here; 5 = related knowledge or foundation, but not this "
    "skill itself; 10 = fully teaches this exact skill. Use the whole "
    "scale; intermediate values express partial teaching. Reply ONLY "
    'with JSON: {"score": <integer 0-10>, "note": "<one short sentence>"}'
)


def load_pairs():
    m = json.loads(MASTER.read_text())
    out = []
    for p in m["pairs"]:
        out.append({"id": f"R{p['row']:03d}",
                    "skill": p["skill"], "clo": p["clo"],
                    "course": f"{p['course']} ({YEAR_NAMES[p['cohort']]})",
                    "cos": p["cos"], "band": p["band"], "set": p["set"],
                    "anchor": p["anchor"]})
    return m, out


def user_msg(p):
    return (f"Job-market skill: {p['skill']}\n"
            f"Course learning outcome: {p['clo']}\n"
            f"Course: {p['course']}")


# ---------------- API plumbing (same as llm_judge_itc.py) ----------------

def http_json(url, headers, body, tries=5):
    data = json.dumps(body).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]}") from e
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def ask_claude(model, msg):
    key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit(
        "ANTHROPIC_API_KEY not set")
    r = http_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": model, "max_tokens": 200,
         "system": INSTRUCTION,
         "messages": [{"role": "user", "content": msg}]})
    return r["content"][0]["text"]


def ask_gpt(model, msg):
    key = os.environ.get("OPENAI_API_KEY") or sys.exit(
        "OPENAI_API_KEY not set")
    body = {"model": model,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": msg}],
            "max_completion_tokens": 2000}
    # scoring task is trivial; keep reasoning spend minimal where supported
    try:
        return _gpt_call(key, dict(body, reasoning_effort="low"))
    except RuntimeError as e:
        if "reasoning_effort" not in str(e):
            raise
        return _gpt_call(key, body)


def _gpt_call(key, body):
    r = http_json("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {key}",
                   "content-type": "application/json"}, body)
    return r["choices"][0]["message"]["content"]


def parse_answer(text):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group())
        s = int(d.get("score"))
        if 0 <= s <= 10:
            return s, str(d.get("note", ""))[:200]
    except Exception:
        pass
    m = re.search(r"\b([0-9]|10)\b", text)
    if m:
        return int(m.group(1)), text.strip()[:200]
    return None, text.strip()[:200]


# ---------------- judging loop ----------------

def cache_path(name):
    return ANNO / f"judge250_{name}.jsonl"


def load_cache(name, model):
    done = {}
    f = cache_path(name)
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("model") == model:      # never reuse another model's line
                done[d["id"]] = d
    return done


def judge(name, model, ask, pairs):
    done = load_cache(name, model)
    todo = [p for p in pairs if p["id"] not in done]
    print(f"[{name}:{model}] {len(done)} cached, {len(todo)} to score")
    with cache_path(name).open("a") as fh:
        for k, p in enumerate(todo, 1):
            score, note = parse_answer(ask(model, user_msg(p)))
            if score is None:
                print(f"  {p['id']}: UNPARSED -> retry once")
                score, note = parse_answer(ask(model, user_msg(p)))
            rec = {"id": p["id"], "score": score, "note": note,
                   "model": model}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} scored")
    print(f"[{name}:{model}] complete")


# ---------------- scoring ----------------

def cat(score):
    return "No" if score <= 3 else ("Partly" if score <= 6 else "Yes")


def kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(claude_model, gpt_model):
    import numpy as np
    meta, pairs = load_pairs()
    by_id = {p["id"]: p for p in pairs}
    judges = {}
    for name, model in (("claude", claude_model), ("gpt", gpt_model)):
        done = load_cache(name, model)
        missing = [p["id"] for p in pairs if p["id"] not in done]
        if missing:
            sys.exit(f"{name}:{model} missing {len(missing)} pairs "
                     f"(e.g. {missing[:5]}) -- run --judge first")
        judges[name] = {i: d["score"] for i, d in done.items()}

    ids = [p["id"] for p in pairs]
    c = np.array([judges["claude"][i] for i in ids], float)
    g = np.array([judges["gpt"][i] for i in ids], float)
    cos = np.array([by_id[i]["cos"] for i in ids], float)
    consensus = (c + g) / 2

    out = {
        "sheet_hash": meta["sheet_hash"],
        "locked_run": meta["locked_run"],
        "models": {"claude": claude_model, "gpt": gpt_model},
        "n_scored": len(ids),
        "score_histogram": {
            "claude": {str(v): int((c == v).sum()) for v in range(11)},
            "gpt": {str(v): int((g == v).sum()) for v in range(11)}},
        "inter_judge": {
            "pearson_r": round(float(np.corrcoef(c, g)[0, 1]), 3),
            "mean_abs_diff": round(float(np.abs(c - g).mean()), 3),
            "within_1": round(float((np.abs(c - g) <= 1).mean()), 3),
            "within_2": round(float((np.abs(c - g) <= 2).mean()), 3),
            "categorical_agreement": round(float(np.mean(
                [cat(x) == cat(y) for x, y in zip(c, g)])), 3),
            "categorical_kappa": round(kappa(
                [cat(x) for x in c], [cat(y) for y in g],
                ("Yes", "Partly", "No")), 3)},
    }

    # cosine threshold that best reproduces judge-consensus "covered"
    # (consensus >= 7 = Yes) -- silver preview of the human calibration
    y = consensus >= 7
    sweep = {}
    best = None
    for t in [round(0.60 + 0.01 * k, 2) for k in range(31)]:
        pred = cos >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        sweep[str(t)] = {"precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3)}
        if best is None or f1 > best[1]:
            best = (t, f1)
    out["silver_threshold_fit"] = {
        "definition": "consensus mean score >= 7 treated as covered",
        "n_covered_consensus": int(y.sum()),
        "best_threshold_by_f1": best[0], "best_f1": round(best[1], 3),
        "sweep": sweep}

    # per-band consensus profile
    per_band = {}
    for i, m in zip(ids, consensus):
        per_band.setdefault(by_id[i]["band"], []).append(float(m))
    out["per_band_consensus_mean"] = {
        b: {"n": len(v), "mean": round(sum(v) / len(v), 2)}
        for b, v in sorted(per_band.items())}

    # control sanity: clear negatives should score low for both judges
    ctrl = [i for i in ids if by_id[i]["band"].startswith("H")]
    out["clear_negative_check"] = {
        "n": len(ctrl),
        "consensus_mean": round(float(np.mean(
            [consensus[ids.index(i)] for i in ctrl])), 2),
        "max": round(float(max(consensus[ids.index(i)] for i in ctrl)), 1)}

    # disagreements >= 4 points, flagged for the human study, never guessed
    out["large_disagreements"] = [
        {"id": i, "skill": by_id[i]["skill"], "clo": by_id[i]["clo"],
         "course": by_id[i]["course"], "claude": int(judges["claude"][i]),
         "gpt": int(judges["gpt"][i]), "cos": round(by_id[i]["cos"], 3)}
        for i in ids if abs(judges["claude"][i] - judges["gpt"][i]) >= 4]

    RESULTS_FILE.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("silver_threshold_fit",
                                   "large_disagreements",
                                   "score_histogram")}, indent=1))
    print(f"\nbest silver threshold (F1): {best[0]} (F1={best[1]:.3f})")
    print(f"large disagreements (>=4): {len(out['large_disagreements'])}")
    print(f"wrote {RESULTS_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=("claude", "gpt", "both"))
    ap.add_argument("--claude-model", default=CLAUDE_MODEL_DEFAULT)
    ap.add_argument("--gpt-model", default=GPT_MODEL_DEFAULT)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: judge only the first N pairs")
    args = ap.parse_args()

    _, pairs = load_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    if args.judge in ("claude", "both"):
        judge("claude", args.claude_model, ask_claude, pairs)
    if args.judge in ("gpt", "both"):
        judge("gpt", args.gpt_model, ask_gpt, pairs)
    if args.score:
        score(args.claude_model, args.gpt_model)
    if not args.judge and not args.score:
        ap.print_help()


if __name__ == "__main__":
    main()
