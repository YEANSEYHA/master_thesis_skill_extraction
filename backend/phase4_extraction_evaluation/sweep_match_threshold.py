# Sensitivity of the SkillSpan benchmark to the JobBERT match threshold.
#
#   python3 sweep_match_threshold.py        (from backend/phase4_extraction_evaluation)
#
# Reviewer question: "why is a match cosine >= 0.60 — why not 0.50?".
# The production benchmark reports hard F1 at 0.60 only; this script
# recomputes it across 0.45–0.80 for BOTH systems (Qwen3-14B and the
# KeyBERT baseline) from the stored prediction files, so the choice can
# be argued from evidence instead of convention.
#
# Nothing is re-extracted: the .json files hold each posting's predicted
# and human-annotated skills, so only the scoring rule changes. The
# greedy bipartite matching is copied from skill_evaluation_metric.py so
# the 0.60 column reproduces the published table exactly.

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
SIM_MODEL = "jjzha/jobbert-base-cased"      # the benchmark's referee
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SYSTEMS = {
    "Qwen3-14B": HERE / "outputs/skillspan_benchmark/skill5-15.json",
    "KeyBERT": (HERE.parent / "phase3_skill_extraction/outputs"
                / "skillspan_benchmark/20260522_141203/keybert.json"),
}
TOP_K = 15
OUT = HERE / "outputs/skillspan_benchmark/match_threshold_sweep.md"


def greedy_match(sim, t):
    """Verbatim from skill_evaluation_metric.py — highest-similarity pairs
    first, one-to-one, stopping once the threshold is crossed."""
    n_llm, n_gt = sim.shape
    pairs = sorted([(sim[i][j], i, j) for i in range(n_llm)
                    for j in range(n_gt)], reverse=True)
    m_llm, m_gt = set(), set()
    for score, i, j in pairs:
        if score < t:
            break
        if i not in m_llm and j not in m_gt:
            m_llm.add(i)
            m_gt.add(j)
    return m_llm, m_gt


def evaluate(records, model):
    totals = {t: {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS}
    for r in records:
        llm = r.get("llm_skills", [])[:TOP_K]
        gold = r.get("human_skills", [])
        if not llm or not gold:
            for t in THRESHOLDS:
                totals[t]["fp"] += len(llm)
                totals[t]["fn"] += len(gold)
            continue
        # macOS Accelerate emits spurious fp warnings on these matmuls
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = cosine_similarity(
                model.encode(llm, normalize_embeddings=True),
                model.encode(gold, normalize_embeddings=True))
        for t in THRESHOLDS:
            tp = len(greedy_match(sim, t)[0])
            totals[t]["tp"] += tp
            totals[t]["fp"] += len(llm) - tp
            totals[t]["fn"] += len(gold) - tp
    rows = []
    for t in THRESHOLDS:
        tp, fp, fn = (totals[t][k] for k in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        rows.append({"t": t, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": rec, "f1": f1})
    return rows


def main():
    print(f"referee: {SIM_MODEL}")
    model = SentenceTransformer(SIM_MODEL, device="cpu")
    results = {}
    for name, path in SYSTEMS.items():
        records = json.loads(path.read_text())
        print(f"\n{name}: {len(records)} postings from {path.name}")
        rows = evaluate(records, model)
        results[name] = rows
        print(f"  {'thr':>5} {'TP':>6} {'FP':>6} {'FN':>6} "
              f"{'prec':>7} {'recall':>7} {'F1':>7}")
        for r in rows:
            print(f"  {r['t']:>5.2f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                  f"{r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%}")

    # the claim that actually matters: does the winner change with the rule?
    q = {r["t"]: r["f1"] for r in results["Qwen3-14B"]}
    k = {r["t"]: r["f1"] for r in results["KeyBERT"]}
    margins = {t: q[t] - k[t] for t in THRESHOLDS}
    print("\nQwen3 − KeyBERT margin (F1 points):")
    for t in THRESHOLDS:
        print(f"  @{t:.2f}  {margins[t] * 100:+.1f}")
    print(f"\nQwen3 wins at every threshold: {all(v > 0 for v in margins.values())}")

    lines = ["# Sensitivity of the benchmark to the JobBERT match threshold\n",
             f"Referee `{SIM_MODEL}`; predictions unchanged, only the "
             f"match rule varies. Greedy one-to-one matching, top-{TOP_K} "
             f"predictions per posting.\n",
             "\n| Threshold | Qwen3-14B F1 | Qwen3 prec. | Qwen3 recall | "
             "KeyBERT F1 | KeyBERT prec. | KeyBERT recall | Margin |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, t in enumerate(THRESHOLDS):
        a, b = results["Qwen3-14B"][i], results["KeyBERT"][i]
        star = " **(adopted)**" if t == 0.60 else ""
        lines.append(
            f"| {t:.2f}{star} | {a['f1']:.1%} | {a['precision']:.1%} | "
            f"{a['recall']:.1%} | {b['f1']:.1%} | {b['precision']:.1%} | "
            f"{b['recall']:.1%} | {(a['f1'] - b['f1']) * 100:+.1f} pts |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
