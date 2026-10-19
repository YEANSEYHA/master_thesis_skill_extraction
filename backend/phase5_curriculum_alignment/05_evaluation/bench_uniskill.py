"""External benchmark: the frozen matcher graded on UniSkill.

UniSkill (arXiv 2603.03134, Apache-2.0) provides human-labeled binary
pairs of (course sentence / course title) x ESCO skill.  This script
makes our production instrument -- pipeline.load_model() (BGE) + the
"covered" cutoff from pipeline.get_thresholds() -- answer every test
pair, then grades the answers against the human labels.

Claim scope: a good score shows the matcher GENERALIZES to independent,
externally annotated course-skill data.  It does NOT prove ITC-specific
accuracy (that needs our own annotated pairs, task #1), and the
threshold sweep below is DIAGNOSTIC ONLY -- production thresholds are
calibrated on ITC annotations, never on UniSkill.

Usage:
    python3 05_evaluation/bench_uniskill.py                 # full run
    python3 05_evaluation/bench_uniskill.py --self-check-only  # no model needed
    python3 05_evaluation/bench_uniskill.py --skip-rerank   # bi-encoder only

Data: 05_evaluation/uniskill_data/*.xlsx (447 kB total; auto-downloaded from
Hugging Face if missing).  Output: printed tables + bench_uniskill_results.json.
"""
import json
import sys
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_engine"))   # engine module

DATA_DIR = HERE / "uniskill_data"
RESULTS_FILE = HERE / "bench_uniskill_results.json"
HF_BASE = "https://huggingface.co/datasets/nurlanm/UniSkill_dataset/resolve/main/"
FILES = ["evaldf_public_m.xlsx", "traindf_public_m.xlsx"]

RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_SUPPORTED = 0.50          # provisional bands, same as production
RERANK_WEAK = 0.10

# published bars from the UniSkill paper (their table; F1 = class average)
LADDER = {"fine-tuned BERT": 0.87, "domain models": "0.81-0.84",
          "human inter-annotator": 0.73}

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ---------------------------------------------------------------- data ----

def ensure_data():
    DATA_DIR.mkdir(exist_ok=True)
    for f in FILES:
        path = DATA_DIR / f
        if not path.exists():
            url = HF_BASE + f
            print(f"  downloading {f} ...")
            urllib.request.urlretrieve(url, path)
        print(f"  {f}: {path.stat().st_size:,} bytes")


def read_xlsx(path: Path) -> list[dict]:
    """Read sheet1 of an xlsx into row dicts keyed by header names."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{XLSX_NS}si"):
                shared.append("".join(t.text or ""
                                      for t in si.iter(f"{XLSX_NS}t")))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        raw = []
        for row in root.iter(f"{XLSX_NS}row"):
            cells = {}
            for c in row:
                col = "".join(ch for ch in c.get("r", "") if ch.isalpha())
                v = c.find(f"{XLSX_NS}v")
                if v is None:
                    cells[col] = ""
                elif c.get("t") == "s":
                    cells[col] = shared[int(v.text)]
                else:
                    cells[col] = v.text
            raw.append(cells)
    header, out = raw[0], []
    for cells in raw[1:]:
        out.append({header[col]: val for col, val in cells.items()
                    if col in header and header[col]})
    return out


def as_label(value):
    try:
        v = int(float(value))
        return v if v in (0, 1) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- grading ----

def grade(pred, gold) -> dict:
    """Four-box counts + precision/recall/F1 (positive class) + macro-F1."""
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    tn = int((~pred & ~gold).sum())

    def f1(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f = f1(tp, fp, fn)
    _, _, f_neg = f1(tn, fn, fp)          # F1 of the "not taught" class
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f,
            "f1_neg": f_neg, "macro_f1": (f + f_neg) / 2,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1)}


def print_grade(name, g):
    print(f"  {name:<18} P={g['precision']:.2f}  R={g['recall']:.2f}  "
          f"F1={g['f1']:.2f}  macroF1={g['macro_f1']:.2f}   "
          f"[hit={g['tp']} falarm={g['fp']} miss={g['fn']} "
          f"creject={g['tn']}]")


# ---------------------------------------------------------------- main ----

def main():
    self_check_only = "--self-check-only" in sys.argv
    skip_rerank = "--skip-rerank" in sys.argv
    t0 = time.time()
    results = {"benchmark": "UniSkill (arXiv 2603.03134)",
               "run_date": time.strftime("%Y-%m-%d %H:%M"),
               "ladder_published": LADDER}

    print("UniSkill external benchmark -- frozen instrument")
    print("=" * 64)
    ensure_data()

    rows = read_xlsx(DATA_DIR / "evaldf_public_m.xlsx")
    tit_rows = [r for r in rows
                if as_label(r.get("eval_all_tit")) is not None
                and r.get("course_title") and r.get("skill_label")]
    sen_rows = [r for r in rows
                if as_label(r.get("eval_all_sen")) is not None
                and r.get("matched_course_sentence") and r.get("skill_label")]
    uniq_tit = {(r["course_title"], r["skill_label"]) for r in tit_rows}
    uniq_sen = {(r["matched_course_sentence"], r["skill_label"])
                for r in sen_rows}
    print(f"\nTest sheet audit: {len(rows)} rows -> "
          f"{len(tit_rows)} title questions ({len(uniq_tit)} unique pairs), "
          f"{len(sen_rows)} sentence questions ({len(uniq_sen)} unique pairs)")
    for name, rs, col in [("title", tit_rows, "eval_all_tit"),
                          ("sentence", sen_rows, "eval_all_sen")]:
        pos = sum(as_label(r[col]) for r in rs)
        print(f"  {name}: gold YES={pos}  NO={len(rs) - pos}")
    results["audit"] = {
        "rows": len(rows), "title_q": len(tit_rows),
        "sentence_q": len(sen_rows), "unique_title_pairs": len(uniq_tit),
        "unique_sentence_pairs": len(uniq_sen)}

    # -- fairness self-check: grade THEIR model's answers with our marker --
    print("\nSelf-check -- grading THEIR published model with our marker")
    print("(their paper reports F1 0.87, class-average, on this test set)")
    self_check = {}
    both_pred, both_gold = [], []
    for name, rs, gold_col, pred_col in [
            ("title", tit_rows, "eval_all_tit", "preds_tit"),
            ("sentence", sen_rows, "eval_all_sen", "preds_sen")]:
        pairs = [(as_label(r.get(pred_col)), as_label(r[gold_col]))
                 for r in rs if as_label(r.get(pred_col)) is not None]
        pred, gold = zip(*pairs)
        g = grade(pred, gold)
        print_grade(f"their {name}", g)
        self_check[name] = g
        both_pred += list(pred)
        both_gold += list(gold)
    g = grade(both_pred, both_gold)
    print_grade("their ALL", g)
    self_check["all"] = g
    results["self_check_their_model"] = self_check

    if self_check_only:
        RESULTS_FILE.write_text(json.dumps(results, indent=1))
        print(f"\nself-check only -- wrote {RESULTS_FILE.name}")
        return

    # -- frozen instrument ------------------------------------------------
    from pipeline import MODEL_NAME, get_thresholds, load_model
    th = get_thresholds()
    cutoff = th["covered"]
    print(f"\nFrozen instrument: {MODEL_NAME}  "
          f"cutoffs covered={th['covered']:.2f} "
          f"borderline={th['borderline']:.2f} host={th['host_min']:.2f}")
    model = load_model()

    def embed(texts):
        uniq = sorted(set(texts))
        vecs = model.encode(uniq, normalize_embeddings=True, batch_size=128,
                            show_progress_bar=False)
        lut = dict(zip(uniq, vecs))
        return np.stack([lut[t] for t in texts])

    def cosines(course_texts, skill_texts):
        a, b = embed(course_texts), embed(skill_texts)
        with np.errstate(all="ignore"):
            return (a * b).sum(axis=1)

    views = {}
    for name, rs, text_col, gold_col in [
            ("title", tit_rows, "course_title", "eval_all_tit"),
            ("sentence", sen_rows, "matched_course_sentence",
             "eval_all_sen")]:
        cos = cosines([r[text_col] for r in rs],
                      [r["skill_label"] for r in rs])
        gold = np.array([as_label(r[gold_col]) for r in rs], bool)
        views[name] = {"cos": cos, "gold": gold, "rows": rs,
                       "text_col": text_col}

    print(f"\nFrozen result -- skill label only, cutoff {cutoff:.2f} "
          "(THE headline numbers)")
    frozen = {}
    for name, v in views.items():
        frozen[name] = grade(v["cos"] >= cutoff, v["gold"])
        print_grade(name, frozen[name])
    all_cos = np.concatenate([v["cos"] for v in views.values()])
    all_gold = np.concatenate([v["gold"] for v in views.values()])
    frozen["all"] = grade(all_cos >= cutoff, all_gold)
    print_grade("ALL", frozen["all"])
    results["frozen"] = frozen
    ours = frozen["all"]["macro_f1"]
    print(f"\n  ladder:  fine-tuned BERT 0.87 | domain 0.81-0.84 | "
          f"human 0.73 | OURS (zero-shot) {ours:.2f}")
    print("  (published bars are class-average F1 -> compare macroF1)")

    # -- diagnostics (do not change any claim) -----------------------------
    print("\nDiagnostics (no claims; production thresholds stay frozen)")
    diags = {}
    for label, cut in [("borderline", th["borderline"]),
                       ("host_min", th["host_min"])]:
        g = grade(all_cos >= cut, all_gold)
        diags[f"cutoff_{label}"] = g
        print_grade(f"all @ {cut:.2f}", g)

    for name, v in views.items():                # skill label + description
        v["cos_desc"] = cosines(
            [r[v["text_col"]] for r in v["rows"]],
            [f"{r['skill_label']}: {r.get('skill_description', '')}"
             for r in v["rows"]])
    desc_cos = np.concatenate([v["cos_desc"] for v in views.values()])
    g = grade(desc_cos >= cutoff, all_gold)
    diags["skill_plus_description"] = g
    print_grade("all +desc @ .73", g)

    # threshold sweep, fitted on their REAL train rows, applied once here
    train = [r for r in read_xlsx(DATA_DIR / "traindf_public_m.xlsx")
             if r.get("data_source") == "real"]
    tr_texts, tr_skills, tr_gold = [], [], []
    for r in train:
        for text_col, gold_col in [("course_title", "eval_all_tit"),
                                   ("matched_course_sentence",
                                    "eval_all_sen")]:
            lab = as_label(r.get(gold_col))
            if lab is not None and r.get(text_col) and r.get("skill_label"):
                tr_texts.append(r[text_col])
                tr_skills.append(r["skill_label"])
                tr_gold.append(lab)
    tr_cos = cosines(tr_texts, tr_skills)
    tr_gold = np.array(tr_gold, bool)
    grid = np.arange(0.30, 0.951, 0.005)
    tr_f1 = [grade(tr_cos >= t, tr_gold)["macro_f1"] for t in grid]
    t_star = float(grid[int(np.argmax(tr_f1))])
    g = grade(all_cos >= t_star, all_gold)
    diags["sweep"] = {"train_rows": len(tr_gold), "t_star": t_star,
                      "train_macro_f1": float(max(tr_f1)),
                      "test_at_t_star": g}
    print(f"  sweep: best cutoff on their {len(tr_gold):,} real train "
          f"questions = {t_star:.2f} (train macroF1 {max(tr_f1):.2f})")
    print_grade(f"all @ t*={t_star:.2f}", g)
    print("  DIAGNOSTIC ONLY -- our production cutoff stays 0.73 until "
          "ITC calibration (task #2)")
    results["diagnostics"] = diags

    RESULTS_FILE.write_text(json.dumps(results, indent=1, default=float))
    print(f"\n[bi-encoder results saved -- {time.time() - t0:.0f}s so far]")

    # -- cross-encoder pass (verification layer) ---------------------------
    if not skip_rerank:
        from pipeline import best_device
        from sentence_transformers import CrossEncoder
        print(f"\nCross-encoder pass -- {RERANKER} "
              f"(device={best_device() or 'cpu'}, provisional bands "
              f"{RERANK_SUPPORTED:.2f}/{RERANK_WEAK:.2f})")
        reranker = CrossEncoder(RERANKER, device=best_device(),
                                max_length=256)
        rr = {}
        all_scores, all_g = [], []
        for name, v in views.items():
            pairs = [(r["skill_label"], r[v["text_col"]])
                     for r in v["rows"]]
            scores = np.asarray(reranker.predict(pairs, batch_size=32,
                                                 show_progress_bar=False),
                                dtype=float)
            if scores.min() < 0.0 or scores.max() > 1.0:
                scores = 1.0 / (1.0 + np.exp(-scores))
            rr[name] = {"supported": grade(scores >= RERANK_SUPPORTED,
                                           v["gold"]),
                        "weak": grade(scores >= RERANK_WEAK, v["gold"])}
            print_grade(f"{name} @ .50", rr[name]["supported"])
            all_scores.append(scores)
            all_g.append(v["gold"])
        s, gg = np.concatenate(all_scores), np.concatenate(all_g)
        rr["all"] = {"supported": grade(s >= RERANK_SUPPORTED, gg),
                     "weak": grade(s >= RERANK_WEAK, gg)}
        print_grade("ALL @ .50", rr["all"]["supported"])
        print_grade("ALL @ .10", rr["all"]["weak"])
        results["reranker"] = {"model": RERANKER, "bands": rr}
        RESULTS_FILE.write_text(json.dumps(results, indent=1, default=float))

    print("\nClaim scope: external generalization of the matcher only --")
    print("not ITC-specific accuracy (task #1) and no threshold change.")
    print(f"Done in {time.time() - t0:.0f}s -> {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
