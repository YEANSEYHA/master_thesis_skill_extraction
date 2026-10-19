# Who the annotators were, and whether it matters — for the thesis revision.
#
#   python3 05_evaluation/annotation/rater_profile.py     (from the phase5_curriculum_alignment root)
#
# Examiner comment on §3.9.3: "The role of these annotators is very
# important in this study. They should have enough understanding about the
# curriculum of the programme. I recommend addressing who they are."
#
# The web study recorded, per rater, a self-declared role (lecturer /
# teaching assistant / student / industry) and familiarity with the ITC
# computer-science programme (teach / studied / general / none). This
# script reads the tracked, anonymised export and produces:
#
#   1. a per-rater profile table (codes only — names never leave the
#      off-repo map), with the pre-registered quality-gate statistics;
#   2. the panel composition;
#   3. a familiarity-subgroup analysis: do raters who STUDIED on the
#      programme judge the pairs differently from raters with only GENERAL
#      knowledge? — subgroup MOS correlation, per-subgroup ICC(2,k), label
#      agreement at the two human criteria (MOS ≥ 5 / ≥ 7), and the
#      Youden-J cosine thresholds refitted on each subgroup alone, with
#      bootstrap CIs, checked against the adopted 0.6903 / 0.7368.
#
# Inclusion follows EVALUATION_RESULTS.md (ADOPTED rule): every rater who
# completed all 250 pairs, K = 10. Nothing here refits the adopted
# thresholds; it asks whether the reference standard depends on who was
# in the panel.

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RATINGS = _HERE / "human_ratings_250_anon.json"
MASTER = _HERE / "sheet_250_master.json"
OUT_JSON = _HERE / "rater_profile.json"
OUT_MD = _HERE / "rater_profile.md"

ADOPTED = {"addresses": 0.6903, "teaches": 0.7368}   # pipeline.THRESHOLDS
CRITERIA = {"addresses": 5.0, "teaches": 7.0}         # MOS cut-offs (§3.9.3)
N_PAIRS = 250
B = 3000
SEED = 20260728

ROLE_LABEL = {"lecturer": "Lecturer / faculty member",
              "assistant": "Teaching assistant",
              "student": "Student (final-year or postgraduate)",
              "industry": "Industry practitioner"}
FAM_LABEL = {"teach": "teaches on the programme",
             "studied": "studied on the programme",
             "general": "knows it in general terms",
             "none": "not familiar"}


# ------------------------------------------------------------ statistics
def icc_2k(M: np.ndarray) -> tuple[float, float]:
    """ICC(2,k) and ICC(2,1): two-way random effects, absolute agreement
    (Shrout & Fleiss 1979). M is (targets, raters), complete."""
    n, k = M.shape
    grand = M.mean()
    row_m, col_m = M.mean(axis=1, keepdims=True), M.mean(axis=0, keepdims=True)
    ss_r = k * ((row_m - grand) ** 2).sum()
    ss_c = n * ((col_m - grand) ** 2).sum()
    ss_e = ((M - row_m - col_m + grand) ** 2).sum()
    ms_r, ms_c, ms_e = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    icc1 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icck = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    return float(icck), float(icc1)


def spearman(a, b) -> float:
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    return sums[inv] / cnt[inv]


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def youden(cos: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Cosine cut-off maximising TPR − FPR (predict positive if cos ≥ t);
    candidates are the observed cosines. Returns (t*, J*, AUC)."""
    pos, neg = label.astype(bool), ~label.astype(bool)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), float("nan")
    cands = np.unique(cos)
    best_t, best_j = float("nan"), -1.0
    for t in cands:
        pred = cos >= t
        j = (pred & pos).sum() / pos.sum() - (pred & neg).sum() / neg.sum()
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, best_j


def auc(cos: np.ndarray, label: np.ndarray) -> float:
    pos, neg = cos[label.astype(bool)], cos[~label.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def youden_ci(cos, label_fn, mos, rng) -> tuple[float, float]:
    ts = []
    for _ in range(B):
        idx = rng.integers(0, len(cos), len(cos))
        t, _ = youden(cos[idx], label_fn(mos[idx]))
        if not np.isnan(t):
            ts.append(t)
    return float(np.percentile(ts, 2.5)), float(np.percentile(ts, 97.5))


# ------------------------------------------------------------------ main
def main() -> None:
    data = json.loads(RATINGS.read_text())
    master = {r["row"]: r for r in json.loads(MASTER.read_text())["pairs"]}
    assert len(master) == N_PAIRS
    if data["sheet"]["sheet_hash"][:8] != "78dfd3fe":
        sys.exit("ABORT: not the frozen 250-pair sheet")

    by_rater = defaultdict(dict)
    for r in data["ratings"]:
        if not r["unable"] and r["score"] is not None:
            by_rater[r["rater_code"]][r["row"]] = r
    raters = {R["rater_code"]: R for R in data["raters"]}
    complete = sorted(c for c, rs in by_rater.items() if len(rs) == N_PAIRS)
    if len(complete) != 10:
        sys.exit(f"ABORT: expected 10 complete raters, found {len(complete)}")

    rows = sorted(master)
    M = np.array([[by_rater[c][row]["score"] for c in complete] for row in rows],
                 dtype=float)                                # (250, 10)
    cos = np.array([master[row]["cos"] for row in rows])
    anchor = np.array([master[row]["anchor"] for row in rows], dtype=bool)

    # 1. per-rater profile + pre-registered gate statistics
    profile = []
    for j, c in enumerate(complete):
        R = raters[c]
        ms = np.array([by_rater[c][row]["ms"] for row in rows
                       if isinstance(by_rater[c][row].get("ms"), (int, float))])
        others = np.delete(M[anchor], j, axis=1).mean(axis=1)
        rho = spearman(M[anchor, j], others)
        profile.append({
            "code": c, "role": R["role"], "familiarity": R["familiarity"],
            "pairs": int(len(by_rater[c])),
            "median_rt_s": round(float(np.median(ms)) / 1000, 2),
            "anchor_rho_loo": round(rho, 3),
            "mean_score": round(float(M[:, j].mean()), 2),
            "gate_rt": bool(np.median(ms) / 1000 >= 2.0),
            "gate_rho": bool(rho >= 0.20),
        })
    excluded = [{"code": c, "role": raters[c]["role"],
                 "familiarity": raters[c]["familiarity"],
                 "pairs": len(by_rater[c])}
                for c in sorted(by_rater) if c not in complete]

    composition = {
        "complete": len(complete),
        "by_role": {r: sum(p["role"] == r for p in profile)
                    for r in ("lecturer", "assistant", "student", "industry")},
        "by_familiarity": {f: sum(p["familiarity"] == f for p in profile)
                           for f in ("teach", "studied", "general", "none")},
        "prereg_rule5_lecturer_weighting_triggered":
            sum(p["role"] in ("lecturer", "assistant") for p in profile) >= 5,
    }

    # 2. subgroup analysis on familiarity
    groups = {
        "all": list(range(len(complete))),
        "studied": [j for j, c in enumerate(complete)
                    if raters[c]["familiarity"] == "studied"],
        "general": [j for j, c in enumerate(complete)
                    if raters[c]["familiarity"] == "general"],
    }
    rng = np.random.default_rng(SEED)
    mos = {g: M[:, idx].mean(axis=1) for g, idx in groups.items()}
    sub = {}
    for g, idx in groups.items():
        icck, icc1 = icc_2k(M[:, idx])
        fits = {}
        for crit, cut in CRITERIA.items():
            lab = (mos[g] >= cut).astype(int)
            t, jstar = youden(cos, lab)
            lo, hi = youden_ci(cos, lambda m, cut=cut: (m >= cut).astype(int),
                               mos[g], rng)
            fits[crit] = {
                "n_positive": int(lab.sum()), "t_star": round(t, 4),
                "youden_j": round(jstar, 3), "auc": round(auc(cos, lab), 3),
                "ci95": [round(lo, 4), round(hi, 4)],
                "adopted": ADOPTED[crit],
                "adopted_inside_ci": bool(lo <= ADOPTED[crit] <= hi),
            }
        sub[g] = {"k": len(idx), "icc_2k": round(icck, 3),
                  "icc_2_1": round(icc1, 3), "mean_mos": round(float(mos[g].mean()), 2),
                  "fits": fits}

    agree = {
        "spearman_mos_studied_vs_general": round(spearman(mos["studied"], mos["general"]), 3),
        "pearson_mos_studied_vs_general": round(float(np.corrcoef(mos["studied"], mos["general"])[0, 1]), 3),
        "spearman_mos_studied_vs_all": round(spearman(mos["studied"], mos["all"]), 3),
        "spearman_mos_general_vs_all": round(spearman(mos["general"], mos["all"]), 3),
        "mean_abs_diff_studied_general": round(float(np.abs(mos["studied"] - mos["general"]).mean()), 2),
    }
    for crit, cut in CRITERIA.items():
        a, b = (mos["studied"] >= cut).astype(int), (mos["general"] >= cut).astype(int)
        agree[f"label_agreement_{crit}"] = {
            "raw": round(float((a == b).mean()), 3),
            "kappa": round(cohen_kappa(a, b), 3),
            "n_pos_studied": int(a.sum()), "n_pos_general": int(b.sum()),
        }

    payload = {
        "generated_by": "05_evaluation/annotation/rater_profile.py",
        "sheet_hash": data["sheet"]["sheet_hash"],
        "exported_at": data["exported_at"],
        "inclusion": "all raters who completed all 250 pairs (EVALUATION_RESULTS.md, ADOPTED)",
        "profile": profile, "excluded_accounts": excluded,
        "composition": composition, "subgroups": sub, "subgroup_agreement": agree,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    OUT_MD.write_text(render_md(payload))
    print(OUT_MD.read_text())


def render_md(p: dict) -> str:
    L = ["# Annotator panel — profile and familiarity-subgroup check", "",
         f"Generated by `{p['generated_by']}` · sheet `{p['sheet_hash'][:8]}…` · "
         f"export {p['exported_at'][:10]} · inclusion: {p['inclusion']}", "",
         "## Per-rater profile (K = 10 complete raters; codes only)", "",
         "| code | role | familiarity with the ITC CS programme | pairs | median RT (s) | anchor ρ (LOO) | mean score | RT gate ≥ 2 s | ρ gate ≥ 0.20 |",
         "|---|---|---|---:|---:|---:|---:|:-:|:-:|"]
    for r in p["profile"]:
        L.append(f"| {r['code']} | {ROLE_LABEL[r['role']]} | {FAM_LABEL[r['familiarity']]} | "
                 f"{r['pairs']} | {r['median_rt_s']:.2f} | {r['anchor_rho_loo']:.3f} | "
                 f"{r['mean_score']:.2f} | {'✓' if r['gate_rt'] else '✗'} | {'✓' if r['gate_rho'] else '✗'} |")
    c = p["composition"]
    L += ["", "Accounts not included: "
          + "; ".join(f"{e['code']} ({ROLE_LABEL[e['role']]}, {FAM_LABEL[e['familiarity']]}, {e['pairs']} pair(s))"
                      for e in p["excluded_accounts"]), "",
          "## Panel composition", "",
          f"- role: " + ", ".join(f"{ROLE_LABEL[k]} {v}" for k, v in c["by_role"].items() if v),
          f"- familiarity: " + ", ".join(f"{FAM_LABEL[k]} {v}" for k, v in c["by_familiarity"].items() if v),
          f"- pre-registered rule 5 (lecturer/TA-weighted reference if ≥ 5 such raters): "
          f"{'triggered' if c['prereg_rule5_lecturer_weighting_triggered'] else 'NOT triggered — all-rater mean is the reference, as pre-declared'}",
          "", "## Familiarity subgroups", "",
          "| subgroup | k | ICC(2,k) | ICC(2,1) | mean MOS | addresses t* (MOS ≥ 5) | 95 % CI | 0.6903 inside | teaches t* (MOS ≥ 7) | 95 % CI | 0.7368 inside |",
          "|---|---:|---:|---:|---:|---:|---|:-:|---:|---|:-:|"]
    for g, s in p["subgroups"].items():
        a, t = s["fits"]["addresses"], s["fits"]["teaches"]
        L.append(f"| {g} | {s['k']} | {s['icc_2k']:.3f} | {s['icc_2_1']:.3f} | {s['mean_mos']:.2f} | "
                 f"{a['t_star']:.4f} (AUC {a['auc']:.2f}) | [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {'✓' if a['adopted_inside_ci'] else '✗'} | "
                 f"{t['t_star']:.4f} (AUC {t['auc']:.2f}) | [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] | {'✓' if t['adopted_inside_ci'] else '✗'} |")
    ag = p["subgroup_agreement"]
    L += ["", "### Studied-on-programme vs general-knowledge raters", "",
          f"- Spearman ρ between subgroup MOS over the 250 pairs: {ag['spearman_mos_studied_vs_general']:.3f} "
          f"(Pearson {ag['pearson_mos_studied_vs_general']:.3f}); mean |ΔMOS| = {ag['mean_abs_diff_studied_general']:.2f} points",
          f"- each subgroup vs the full panel: studied ρ = {ag['spearman_mos_studied_vs_all']:.3f}, general ρ = {ag['spearman_mos_general_vs_all']:.3f}"]
    for crit in CRITERIA:
        la = ag[f"label_agreement_{crit}"]
        L.append(f"- label agreement at the *{crit}* criterion: raw {la['raw']:.3f}, κ = {la['kappa']:.3f} "
                 f"(positives: studied {la['n_pos_studied']}, general {la['n_pos_general']})")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
