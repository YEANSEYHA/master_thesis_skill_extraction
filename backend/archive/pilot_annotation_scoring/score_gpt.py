"""Score the 200-pair annotation sheet with a GPT judge (0-10, teacher protocol).

Second annotator for the evaluation: GPT reads the same blind pairs the
teacher gets (skill + CLO + course, no cosine/band) and gives the same
0-10 score with the same anchors printed on the PDF:

    0 = not taught here
    5 = related / foundation, but not this skill itself
    10 = fully teaches this exact skill

The 0-10 is then mapped to classes exactly like the teacher sheet footer
(0-3 = No, 4-6 = Partly, 7-10 = Yes) and compared against the instrument
verdicts from score_method.py (agreement, per-class P/R/F1, cosine
correlation).

Usage:
    python3 score_gpt.py                       # default model gpt-5.5
    python3 score_gpt.py --model gpt-5         # any OpenAI chat model
    python3 score_gpt.py --limit 5             # smoke test on first 5 rows

Requires OPENAI_API_KEY (read from backend/.env or the environment).
Answers are cached per row in gpt_cache_<model>.jsonl - safe to re-run.

Outputs:
    gpt_scores_<model>.json   per-row scores + comparison summary
    gpt_scores_<model>.csv    row,skill,clo,course,gpt_score,gpt_verdict
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTER = HERE.parents[1] / "phase5_curriculum_alignment" / "05_evaluation" / "annotation" / "human_sheet_200_master.json"
ENV = HERE.parent / ".env"

INSTRUCTION = (
    "You are annotating curriculum coverage for a university program. "
    "Each item pairs a skill demanded in job postings with one learning "
    "outcome (CLO) from an ITC (Institute of Technology of Cambodia) "
    "course. Judge from the text alone: how well does this course outcome "
    "teach this skill? Give an integer score 0-10: 0 = not taught here, "
    "5 = related/foundation but not this skill itself, 10 = fully teaches "
    "this exact skill. Reply ONLY with JSON: "
    '{"score": <0-10>, "note": "<one short sentence>"}'
)


def load_env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            m = re.match(rf"\s*{name}\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


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


def ask_gpt(key, model, user_msg):
    body = {"model": model,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": user_msg}]}
    if model.startswith(("gpt-4", "gpt-3")):
        body["temperature"] = 0
        body["max_tokens"] = 150
    else:                       # gpt-5 family / o-series: fixed temperature
        body["max_completion_tokens"] = 2000
    r = http_json("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {key}",
                   "content-type": "application/json"}, body)
    return r["choices"][0]["message"]["content"]


def parse_score(text):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group())
        s = int(d.get("score"))
        if 0 <= s <= 10:
            return s, str(d.get("note", ""))[:200]
    except Exception:
        pass
    m = re.search(r"\b(10|[0-9])\b", text)
    return (int(m.group(1)) if m else None), text[:200]


def to_verdict(score: int) -> str:
    return "No" if score <= 3 else "Partly" if score <= 6 else "Yes"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):                      # average ranks for ties
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    args = ap.parse_args()

    key = load_env_key("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY not set (env or backend/.env)")
    pairs = json.load(open(MASTER))["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]

    cache_path = HERE / f"gpt_cache_{slug(args.model)}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            d = json.loads(line)
            cache[d["row"]] = d
    todo = [p for p in pairs if p["row"] not in cache]
    print(f"model={args.model}  pairs={len(pairs)}  cached={len(pairs) - len(todo)}  to ask={len(todo)}")

    with open(cache_path, "a") as cf:
        for i, p in enumerate(todo, 1):
            msg = (f"Market skill: {p['skill']}\n"
                   f"Course: {p['course']}\n"
                   f"Course learning outcome: {p['clo']}")
            score, note = parse_score(ask_gpt(key, args.model, msg))
            d = {"row": p["row"], "score": score, "note": note}
            cache[p["row"]] = d
            cf.write(json.dumps(d) + "\n")
            cf.flush()
            print(f"  [{i}/{len(todo)}] row {p['row']:>3}  {p['skill'][:30]:30} -> {score}")

    rows = []
    for p in pairs:
        d = cache[p["row"]]
        rows.append({
            "row": p["row"], "skill": p["skill"], "clo": p["clo"], "course": p["course"],
            "gpt_score": d["score"],
            "gpt_verdict": to_verdict(d["score"]) if d["score"] is not None else None,
            "note": d["note"],
        })
    unparsed = [r["row"] for r in rows if r["gpt_score"] is None]

    # ---- compare with the instrument (score_method.py output) ----
    comparison = None
    inst_path = HERE / "method_scores_baai_bge_base_en_v1_5.json"
    if inst_path.exists():
        inst = {r["row"]: r for r in json.load(open(inst_path))["rows"]}
        both = [(inst[r["row"]], r) for r in rows
                if r["gpt_verdict"] and r["row"] in inst]
        agree = sum(1 for a, b in both if a["verdict"] == b["gpt_verdict"])
        prf = {}
        for cls in ("Yes", "Partly", "No"):
            tp = sum(1 for a, b in both if a["verdict"] == cls and b["gpt_verdict"] == cls)
            fp = sum(1 for a, b in both if a["verdict"] == cls and b["gpt_verdict"] != cls)
            fn = sum(1 for a, b in both if a["verdict"] != cls and b["gpt_verdict"] == cls)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            prf[cls] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)}
        cos = [a["cos"] for a, _ in both]
        gsc = [b["gpt_score"] for _, b in both]
        comparison = {
            "n": len(both),
            "agreement": round(agree / len(both), 3) if both else None,
            "instrument_vs_gpt_prf": prf,
            "pearson_cos_vs_score": round(pearson(cos, gsc), 3),
            "spearman_cos_vs_score": round(spearman(cos, gsc), 3),
        }

    summary = {
        "model": args.model,
        "n_pairs": len(rows),
        "unparsed_rows": unparsed,
        "gpt_verdicts": dict(Counter(r["gpt_verdict"] for r in rows if r["gpt_verdict"])),
        "gpt_score_mean": round(sum(r["gpt_score"] for r in rows if r["gpt_score"] is not None)
                                / max(1, len(rows) - len(unparsed)), 2),
        "vs_instrument": comparison,
    }

    out_json = HERE / f"gpt_scores_{slug(args.model)}.json"
    out_csv = HERE / f"gpt_scores_{slug(args.model)}.csv"
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "skill", "clo", "course", "gpt_score", "gpt_verdict"])
        for r in rows:
            w.writerow([r["row"], r["skill"], r["clo"], r["course"], r["gpt_score"], r["gpt_verdict"]])

    print(f"\nwrote {out_json.name} + {out_csv.name}\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
