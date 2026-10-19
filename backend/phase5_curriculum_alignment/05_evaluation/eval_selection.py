"""Machine-only certificates for the prescriptive selection layer.

Certifies the SELECTOR (pipeline.greedy_skill_selection + the 80%
stopping rule) with three experiments that need no human labels:

  EXP1  Exact optimality -- enumerate all 2^n candidate subsets and
        compare greedy prefixes to the true optimum at every budget k,
        plus the exact smallest subset reaching the 80% target
        (submodular-cover view; Wolsey 1982 / Nemhauser et al. 1978).
  EXP2  Bootstrap stability -- resample skill demand B times, rerun
        the production greedy, report per-skill selection frequency
        and order stability.  NOTE: multinomial resampling of skill
        mention counts approximates posting-level bootstrap (ignores
        within-posting skill correlation) -- disclosed design choice.
  EXP3  Sensitivity -- sweep the CUM_FRAC design parameter and the
        covered-status cutoff; report how the selected set changes.

The candidate set, coverage profile and demand weights are rebuilt
EXACTLY as gen_report_data.py builds them, then verified against the
shipped report_data.js before any experiment runs (self-check).

Usage: python3 05_evaluation/eval_selection.py
Output: printed report + 05_evaluation/eval_selection_results.json
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "03_engine"))   # engine module lives there

from pipeline import (COVERED_AT, MODEL_NAME, greedy_skill_selection,
                      is_soft_skill, load_model, market_cache_path)

RESULTS_FILE = HERE / "eval_selection_results.json"
TOP_N = 25          # mirrors gen_report_data.py
CUM_FRAC = 0.80
B_BOOT = 1000
SEED = 42


def f_of(subset_rows, cand_sim, profile, demand):
    """Facility-location objective f(S) = sum demand*max(profile, sim)."""
    if len(subset_rows) == 0:
        return float((demand * profile).sum())
    cov = np.maximum(profile, cand_sim[list(subset_rows)].max(axis=0))
    return float((demand * cov).sum())


def selected_prefix(steps, cum_frac):
    """Names selected under the production stopping rule."""
    total = sum(st["gain"] for st in steps)
    out, cum = [], 0.0
    for st in steps:
        if cum < cum_frac * total:
            out.append(st["skill"])
        cum += st["gain"]
    return out


def kendall_tau(a, b):
    """Kendall rank correlation of two orderings of the same items."""
    pos = {x: i for i, x in enumerate(b)}
    rb = [pos[x] for x in a]
    n, disc = len(rb), 0
    for i in range(n):
        for j in range(i + 1, n):
            disc += rb[i] > rb[j]
    pairs = n * (n - 1) / 2
    return 1.0 - 2.0 * disc / pairs


def main():
    t0 = time.time()
    print("Selection-layer certificates (machine-only)")
    print("=" * 64)

    # ---- rebuild the production inputs (mirror gen_report_data.py) ----
    z = np.load(market_cache_path(MODEL_NAME), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    base, counts = meta["skill_base"], meta["counts"]
    skill_emb, demand_dist = z["skill_emb"], z["demand_dist"]

    clo_texts = []
    for f in sorted((BASE / "02_input_curriculum" /
                     "clo_skills").glob("*.json")):
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else data.get("skills", [])
        clo_texts += [it["skill"] if isinstance(it, dict) else it
                      for it in items]
    model = load_model()
    clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                           show_progress_bar=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = skill_emb @ clo_emb.T

    hard_idx = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top = sorted(hard_idx, key=lambda i: -counts[base[i]])[:TOP_N]
    demand = demand_dist[hard_idx]
    demand = demand / demand.sum()
    profile = np.clip(sim[hard_idx].max(axis=1), 0.0, None)

    cand_glob = [i for i in top if float(sim[i].max()) < COVERED_AT]
    cand_names = [base[i] for i in cand_glob]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cand_sim = np.clip(skill_emb[cand_glob] @ skill_emb[hard_idx].T,
                           0.0, None)
    n = len(cand_names)
    print(f"candidates (non-covered of top-{TOP_N}): {n}")

    steps = greedy_skill_selection(demand, profile, cand_names, cand_sim)
    greedy_order = [st["skill"] for st in steps]
    greedy_sel = selected_prefix(steps, CUM_FRAC)

    # self-check vs the shipped report
    rep_txt = (BASE / "report_data.js").read_text()
    rep = json.loads(rep_txt[rep_txt.index("=") + 1:].rstrip().rstrip(";"))
    rep_order = [e["skill"] for e in rep["selection"]["plan"]]
    rep_sel = [e["skill"] for e in rep["selection"]["plan"] if e["selected"]] \
        if "selected" in rep["selection"]["plan"][0] else None
    ok = rep_order == greedy_order
    print(f"self-check vs report_data.js: order match = {ok}")
    if not ok:
        print("  REPORT:", rep_order)
        print("  HERE:  ", greedy_order)
        sys.exit("reproduction failed -- refusing to certify")

    results = {"n_candidates": n, "greedy_order": greedy_order,
               "greedy_selected": greedy_sel, "cum_frac": CUM_FRAC,
               "self_check_order_match": ok}

    # ---- EXP1: exact enumeration --------------------------------------
    print(f"\nEXP1 -- exact optimum vs greedy over all 2^{n} = "
          f"{2 ** n:,} subsets")
    idx = list(range(n))
    f0 = f_of([], cand_sim, profile, demand)
    f_all = f_of(idx, cand_sim, profile, demand)
    target = f0 + CUM_FRAC * (f_all - f0)

    exact = {}
    greedy_f = {}
    running = []
    for k, name in enumerate(greedy_order, 1):
        running.append(cand_names.index(name))
        greedy_f[k] = f_of(running, cand_sim, profile, demand)
    cover_exact = None
    for k in range(1, n + 1):
        best, best_S = -1.0, None
        for S in itertools.combinations(idx, k):
            v = f_of(S, cand_sim, profile, demand)
            if v > best:
                best, best_S = v, S
        exact[k] = {"opt": best,
                    "opt_set": sorted(cand_names[i] for i in best_S),
                    "greedy": greedy_f[k],
                    "ratio": greedy_f[k] / best if best else 1.0}
        if cover_exact is None and best >= target - 1e-12:
            cover_exact = k
    worst = min(e["ratio"] for e in exact.values())
    print(f"  budget k: greedy/optimal ratio "
          f"(1.0000 = greedy exactly optimal)")
    for k in range(1, n + 1):
        e = exact[k]
        mark = "" if e["ratio"] > 0.99999 else "  <-- gap"
        print(f"   k={k:>2}  ratio={e['ratio']:.5f}{mark}")
    print(f"  worst ratio over all budgets: {worst:.5f}")
    print(f"  80% target: greedy needs {len(greedy_sel)} skills, "
          f"exact minimum = {cover_exact}")
    results["exp1"] = {"worst_ratio": worst,
                       "per_k": {k: {"ratio": e["ratio"]} for k, e in
                                 exact.items()},
                       "cover_greedy": len(greedy_sel),
                       "cover_exact": cover_exact}

    # ---- EXP2: bootstrap stability ------------------------------------
    print(f"\nEXP2 -- bootstrap stability (B={B_BOOT:,}, multinomial over "
          "skill mention counts; approximates posting bootstrap)")
    counts_hard = np.array([counts[base[i]] for i in hard_idx], dtype=float)
    assert np.allclose(demand, counts_hard / counts_hard.sum(), atol=1e-9), \
        "demand_dist is not proportional to counts -- check cache"
    M = int(counts_hard.sum())
    rng = np.random.default_rng(SEED)
    sel_freq = {s: 0 for s in cand_names}
    n_sel_hist = {}
    taus = []
    for _ in range(B_BOOT):
        w = rng.multinomial(M, counts_hard / M).astype(float)
        d_b = w / w.sum()
        st_b = greedy_skill_selection(d_b, profile, cand_names, cand_sim)
        sel_b = selected_prefix(st_b, CUM_FRAC)
        for s in sel_b:
            sel_freq[s] += 1
        n_sel_hist[len(sel_b)] = n_sel_hist.get(len(sel_b), 0) + 1
        taus.append(kendall_tau([s["skill"] for s in st_b], greedy_order))
    print("  selection frequency (share of replicates each skill "
          "is selected):")
    for s in greedy_order:
        f = sel_freq[s] / B_BOOT
        tag = "SELECTED" if s in greedy_sel else "skipped "
        print(f"   {tag}  {s:<28} {sel_freq[s]:>4}/{B_BOOT}  ({f:.1%})")
    print(f"  plan size distribution: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(n_sel_hist.items())))
    print(f"  greedy order stability: Kendall tau mean={np.mean(taus):.3f} "
          f"min={np.min(taus):.3f}")
    results["exp2"] = {"B": B_BOOT, "selection_frequency":
                       {s: sel_freq[s] / B_BOOT for s in cand_names},
                       "n_selected_hist": n_sel_hist,
                       "kendall_tau_mean": float(np.mean(taus)),
                       "kendall_tau_min": float(np.min(taus))}

    # ---- EXP3: design-parameter sensitivity ----------------------------
    print("\nEXP3a -- CUM_FRAC sweep (stopping rule; order is unchanged)")
    sweep, prev = {}, []
    for cf in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        sel = selected_prefix(steps, cf)
        sweep[f"{cf:.2f}"] = sel
        added = [s for s in sel if s not in prev]
        mark = "  <-- production" if abs(cf - CUM_FRAC) < 1e-9 else ""
        extra = f"  (+ {', '.join(added)})" if added and prev else ""
        print(f"   {cf:.2f}: {len(sel):>2} skills{extra}{mark}")
        prev = sel
    results["exp3_cum_frac"] = sweep

    print("\nEXP3b -- covered-cutoff sensitivity (candidate set changes)")
    cutoff_runs = {}
    for cut in [COVERED_AT - 0.02, COVERED_AT, COVERED_AT + 0.02]:
        cg = [i for i in top if float(sim[i].max()) < cut]
        cn = [base[i] for i in cg]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cs = np.clip(skill_emb[cg] @ skill_emb[hard_idx].T, 0.0, None)
        st = greedy_skill_selection(demand, profile, cn, cs)
        sel = selected_prefix(st, CUM_FRAC)
        cutoff_runs[f"{cut:.2f}"] = {"n_candidates": len(cn),
                                     "selected": sel}
        mark = "  <-- production" if abs(cut - COVERED_AT) < 1e-9 else ""
        print(f"   cutoff {cut:.2f}: {len(cn)} candidates -> "
              f"{len(sel)} selected{mark}")
    prod = set(cutoff_runs[f"{COVERED_AT:.2f}"]["selected"])
    for cut, r in cutoff_runs.items():
        diff_in = sorted(set(r["selected"]) - prod)
        diff_out = sorted(prod - set(r["selected"]))
        if diff_in or diff_out:
            print(f"     {cut}: enters {diff_in or '-'}  leaves "
                  f"{diff_out or '-'}")
    results["exp3_cutoff"] = cutoff_runs

    RESULTS_FILE.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nDone in {time.time() - t0:.0f}s -> {RESULTS_FILE.name}")
    print("Claim scope: optimality/stability/sensitivity of the FORMAL "
          "objective only;\neducational validity needs instructors "
          "(task #3).")


if __name__ == "__main__":
    main()
