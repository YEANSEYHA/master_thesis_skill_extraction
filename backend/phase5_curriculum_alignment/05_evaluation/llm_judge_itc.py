"""LLM-as-annotator for the 150-pair ITC annotation sample (tasks #1/#2).

The SAME frozen sheet is answered by three kinds of annotator:
  - GPT and Claude now (this script; machine silver labels)
  - optionally humans later (the generated CSV sheet, unchanged)

Design follows the validated-annotator literature: judges never see
cosine scores or bands; identical frozen instruction for every
annotator; two independent model families = two annotators -> Cohen's
kappa; consensus labels become silver gold; disagreements are flagged,
never guessed. (Pangakis et al. 2023 arXiv:2306.00176; Calderon et al.
ACL 2025 alt-test; Xu et al. 2026 JLA.)

Usage:
  python3 05_evaluation/llm_judge_itc.py --make-sheets       # CSVs for humans
  export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
  python3 05_evaluation/llm_judge_itc.py --judge both        # ~300 calls, <$1
  python3 05_evaluation/llm_judge_itc.py --score             # after judges ran

Calls are cached per judge in annotation/judge_<name>.jsonl -- safe to
re-run after any crash; only missing pairs are sent.
NOTE: both current flagship models fix temperature internally
(claude-sonnet-5 rejects the parameter; GPT-5 ignores it) --
determinism caveat disclosed in the write-up.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "03_engine"))   # engine module lives there

ANNO = HERE / "annotation"
SAMPLE = ANNO / "sample_150.json"
RESULTS_FILE = HERE / "llm_judge_itc_results.json"
SHEET_SEED = 42
LABELS = ("Yes", "Partly", "No")

COURSE_NAMES = {
    "1_introduction_ai": "Introduction to AI",
    "2_system_admin": "System Administration",
    "3_cybersecurity": "Cybersecurity",
    "4_oop": "Object-Oriented Programming",
    "5_programming_c++": "C++ Programming",
}

INSTRUCTION = (
    "You are annotating curriculum coverage for a university program. "
    "Each item pairs a skill demanded in job postings with one learning "
    "outcome (CLO) from an ITC (Institute of Technology of Cambodia) "
    "course. Judge from the text alone: does this course outcome teach "
    "this skill? Answer Yes (this skill is taught), Partly "
    "(related/foundation but not this skill itself), or No (not taught "
    "here). Reply ONLY with JSON: "
    '{"answer": "Yes"|"Partly"|"No", "note": "<one short sentence>"}'
)


def load_pairs():
    """Sample rows in frozen sheet order (shuffled once, seed 42)."""
    rows = json.loads(SAMPLE.read_text())
    idx = list(range(len(rows)))
    random.Random(SHEET_SEED).shuffle(idx)
    out = []
    for n, i in enumerate(idx, 1):
        r = rows[i]
        out.append({"id": f"P{n:03d}", "skill": r["skill"], "clo": r["clo"],
                    "course": COURSE_NAMES[r["course"]], "cos": r["cos"],
                    "band": r["band"], "set": r["set"]})
    return out


def make_sheets(pairs):
    ANNO.mkdir(exist_ok=True)
    with open(ANNO / "sheet_annotator.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "job_market_skill", "course_learning_outcome",
                    "course", "answer_yes_partly_no", "note"])
        for p in pairs:
            w.writerow([p["id"], p["skill"], p["clo"], p["course"], "", ""])
    with open(ANNO / "sheet_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "skill", "clo", "course", "cosine", "band", "set"])
        for p in pairs:
            w.writerow([p["id"], p["skill"], p["clo"], p["course"],
                        f"{p['cos']:.4f}", p["band"], p["set"]])
    print(f"wrote {ANNO / 'sheet_annotator.csv'} (score-free, give to "
          f"humans)\nwrote {ANNO / 'sheet_key.csv'} (KEEP HIDDEN)")


# ---------------- API judges ----------------

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


def ask_claude(model, user_msg):
    key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit(
        "ANTHROPIC_API_KEY not set")
    r = http_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": model, "max_tokens": 150,
         "system": INSTRUCTION,
         "messages": [{"role": "user", "content": user_msg}]})
    return r["content"][0]["text"]


def ask_gpt(model, user_msg):
    key = os.environ.get("OPENAI_API_KEY") or sys.exit(
        "OPENAI_API_KEY not set")
    body = {"model": model,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": user_msg}]}
    if model.startswith(("gpt-4", "gpt-3")):
        body["temperature"] = 0
        body["max_tokens"] = 150
    else:                       # gpt-5 / o-series: fixed temperature
        body["max_completion_tokens"] = 2000
    r = http_json("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {key}",
                   "content-type": "application/json"}, body)
    return r["choices"][0]["message"]["content"]


def parse_answer(text):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group())
        ans = str(d.get("answer", "")).strip().capitalize()
        if ans in LABELS:
            return ans, str(d.get("note", ""))[:200]
    except Exception:
        pass
    m = re.search(r"\b(yes|partly|no)\b", text, re.I)
    return (m.group(1).capitalize() if m else None), text[:200]


def run_judge(name, model, pairs):
    cache = ANNO / f"judge_{name}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            done[d["id"]] = d
    todo = [p for p in pairs if p["id"] not in done]
    print(f"{name} ({model}): {len(done)} cached, {len(todo)} to ask")
    ask = ask_claude if name == "claude" else ask_gpt
    with open(cache, "a") as f:
        for k, p in enumerate(todo, 1):
            msg = (f'Job-market skill: "{p["skill"]}"\n'
                   f'Course learning outcome: "{p["clo"]}"\n'
                   f'Course: {p["course"]}')
            ans, note = parse_answer(ask(model, msg))
            if ans is None:
                print(f"  {p['id']}: UNPARSEABLE, skipping (rerun later)")
                continue
            f.write(json.dumps({"id": p["id"], "answer": ans,
                                "note": note}) + "\n")
            f.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} answered")
    return {d["id"]: d for line in cache.read_text().splitlines()
            for d in [json.loads(line)]}


# ---------------- scoring ----------------

def kappa(a, b):
    """Cohen's kappa, 3-class."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def prf(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(not p and g for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def score(pairs, judges):
    from pipeline import get_thresholds
    thr = get_thresholds()

    def instrument(p):
        if p["cos"] >= thr["covered"]:
            return "Yes"
        if p["cos"] >= thr["borderline"]:
            return "Partly"
        return "No"

    ids = [p["id"] for p in pairs
           if all(p["id"] in j for j in judges.values())]
    if len(ids) < len(pairs):
        print(f"WARNING: only {len(ids)}/{len(pairs)} pairs answered by "
              "all judges")
    by_id = {p["id"]: p for p in pairs}
    names = list(judges)
    lab = {n: [judges[n][i]["answer"] for i in ids] for n in names}
    inst = [instrument(by_id[i]) for i in ids]

    out = {"n_scored": len(ids), "thresholds": thr,
           "judge_label_counts": {n: {c: lab[n].count(c) for c in LABELS}
                                  for n in names},
           "instrument_label_counts": {c: inst.count(c) for c in LABELS}}

    print(f"\nscored pairs: {len(ids)}")
    for n in names:
        agree = sum(x == y for x, y in zip(lab[n], inst)) / len(ids)
        print(f"  {n} vs instrument 3-class agreement: {agree:.2f}")
        out[f"{n}_vs_instrument_agreement"] = round(agree, 3)

    if len(names) == 2:
        k = kappa(lab[names[0]], lab[names[1]])
        agree = sum(x == y for x, y in
                    zip(lab[names[0]], lab[names[1]])) / len(ids)
        print(f"  inter-judge: raw agreement {agree:.2f}, "
              f"Cohen's kappa {k:.2f}")
        out["inter_judge"] = {"agreement": round(agree, 3),
                              "kappa": round(k, 3)}
        cons, disag = {}, []
        for i, a, b in zip(ids, lab[names[0]], lab[names[1]]):
            if a == b:
                cons[i] = a
            else:
                disag.append({"id": i, names[0]: a, names[1]: b,
                              "skill": by_id[i]["skill"],
                              "clo": by_id[i]["clo"],
                              "cos": round(by_id[i]["cos"], 3)})
        print(f"  consensus on {len(cons)} pairs, "
              f"{len(disag)} disagreements (flagged for human audit)")
        out["disagreements"] = disag

        # frozen-instrument accuracy + local cutoff, both Partly readings
        for variant, pos in [("strict_yes_only", ("Yes",)),
                             ("lenient_yes_partly", ("Yes", "Partly"))]:
            cids = list(cons)
            gold = [cons[i] in pos for i in cids]
            pred = [by_id[i]["cos"] >= thr["covered"] for i in cids]
            m = prf(pred, gold)
            grid = [round(0.30 + 0.005 * k, 3) for k in range(131)]

            def mf1(t):
                pr = [by_id[i]["cos"] >= t for i in cids]
                a = prf(pr, gold)
                b = prf([not x for x in pr], [not g for g in gold])
                return (a["f1"] + b["f1"]) / 2
            t_star = max(grid, key=mf1)
            print(f"  [{variant}] frozen {thr['covered']}: "
                  f"P={m['precision']:.2f} R={m['recall']:.2f} "
                  f"F1={m['f1']:.2f} | fitted local cutoff "
                  f"t*={t_star:.3f} (macroF1 {mf1(t_star):.2f}, n={len(cids)})")
            out[variant] = {"frozen": m, "t_star": t_star,
                            "t_star_macro_f1": round(mf1(t_star), 3),
                            "n_consensus": len(cids)}

        # per-band consensus profile
        bands = {}
        for i in cons:
            bands.setdefault(by_id[i]["band"], []).append(cons[i])
        out["per_band"] = {b: {c: v.count(c) for c in LABELS}
                           for b, v in sorted(bands.items())}
        print("  per-band consensus (band: Yes/Partly/No):")
        for b, v in sorted(bands.items()):
            print(f"    {b:<14} {v.count('Yes')}/{v.count('Partly')}/"
                  f"{v.count('No')}  (n={len(v)})")

    RESULTS_FILE.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULTS_FILE.name}")
    print("Claim scope: silver labels from validated-annotator protocol; "
          "judge certification on UniSkill gold + small human audit "
          "still strengthen this for the defense.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sheets", action="store_true")
    ap.add_argument("--judge", choices=["claude", "gpt", "both"])
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--claude-model", default="claude-sonnet-5")
    ap.add_argument("--gpt-model", default="gpt-5")
    args = ap.parse_args()

    pairs = load_pairs()
    if args.make_sheets:
        make_sheets(pairs)
    judges = {}
    if args.judge in ("claude", "both"):
        judges["claude"] = run_judge("claude", args.claude_model, pairs)
    if args.judge in ("gpt", "both"):
        judges["gpt"] = run_judge("gpt", args.gpt_model, pairs)
    if args.score or judges:
        if not judges:  # --score alone: load caches
            for name in ("claude", "gpt"):
                c = ANNO / f"judge_{name}.jsonl"
                if c.exists():
                    judges[name] = {d["id"]: d for line in
                                    c.read_text().splitlines()
                                    for d in [json.loads(line)]}
        if judges:
            score(pairs, judges)
        else:
            print("no judge caches found -- run --judge first")


if __name__ == "__main__":
    main()
