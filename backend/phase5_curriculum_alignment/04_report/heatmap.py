"""Render a skill x course coverage heatmap from a phase5 run's report.json.

The full 30 x N cosine matrix already lives in report.json under
skills[i]["per_course"] (course -> BGE cosine) — this script only *draws* it, so
any archived run can regenerate its figure without re-running the pipeline.

Cells are BGE cosine similarity (0-1), NOT percentages. The thresholds that turn a
cosine into covered/borderline/gap come from the run itself (report["thresholds"]).

Usage:
    python3 04_report/heatmap.py                       # latest run under outputs/
    python3 04_report/heatmap.py --run 20260724T034547Z-475eb3
    python3 04_report/heatmap.py --report path/to/report.json --top 15
    python3 04_report/heatmap.py --top 30 --csv         # also dump the matrix as CSV

Writes heatmap.png (and, with --csv, heatmap.csv) next to the report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"


def latest_report() -> Path:
    """Newest outputs/<run_id>/report.json by run-id name (ids are timestamped)."""
    runs = sorted(
        (p for p in OUTPUTS.iterdir() if p.is_dir() and (p / "report.json").exists()),
        key=lambda p: p.name,
    )
    if not runs:
        raise SystemExit(f"no run with a report.json under {OUTPUTS}")
    return runs[-1] / "report.json"


def resolve_report(args: argparse.Namespace) -> Path:
    if args.report:
        return Path(args.report).resolve()
    if args.run:
        p = OUTPUTS / args.run / "report.json"
        if not p.exists():
            raise SystemExit(f"no report.json for run {args.run} ({p})")
        return p
    return latest_report()


def build_matrix(report: dict, top: int):
    """Return (skills[top], courses, matrix[top x n], jobs[top]) sorted by demand."""
    courses = report["courses"]
    rows = sorted(report["skills"], key=lambda s: s["jobs"], reverse=True)[:top]
    skills = [r["skill"] for r in rows]
    jobs = [r["jobs"] for r in rows]
    mat = np.array(
        [[r["per_course"].get(c, 0.0) for c in courses] for r in rows], dtype=float
    )
    return skills, courses, mat, jobs


def draw(report: dict, top: int, out: Path) -> Path:
    skills, courses, mat, jobs = build_matrix(report, top)
    th = report["thresholds"]
    covered, borderline = th["covered"], th["borderline"]

    # White -> amber -> teal ramp; the covered band lands in the teal end.
    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#f7f7f7", "#fde68a", "#f59e0b", "#0d9488", "#0f766e"]
    )

    nrow, ncol = mat.shape
    fig, ax = plt.subplots(figsize=(0.9 * ncol + 4.5, 0.42 * nrow + 2.0))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.45, vmax=1.0)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels(courses, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{s}  ({j})" for s, j in zip(skills, jobs)], fontsize=8)
    ax.set_xlabel("Course", fontsize=9)
    ax.set_ylabel("Market skill  (jobs)", fontsize=9)

    # Annotate each cell; ring the row-max (the value that becomes coverage).
    rowmax = mat.max(axis=1)
    for i in range(nrow):
        for j in range(ncol):
            v = mat[i, j]
            is_max = v == rowmax[i]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if v >= borderline else "#444",
                fontweight="bold" if is_max else "normal",
            )
            if is_max:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="#111", linewidth=1.4)
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("BGE cosine similarity (not %)", fontsize=8)
    for y, lab in ((covered, f"covered ≥{covered}"), (borderline, f"borderline ≥{borderline}")):
        cbar.ax.axhline(y, color="#111", linewidth=0.8)
        cbar.ax.text(1.6, y, lab, va="center", fontsize=7, transform=cbar.ax.get_yaxis_transform())

    src = report.get("source", "?")
    coh = report.get("cohort") or "all"
    ax.set_title(
        f"Skill–course coverage — {src} · cohort {coh} · "
        f"KL {report['kl']:.4f} · covered {report['covered']}/{report['top_n']}\n"
        f"boxed cell = row max (the coverage value); cells are BGE cosine, not percentages",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def dump_csv(report: dict, top: int, out: Path) -> Path:
    skills, courses, mat, jobs = build_matrix(report, top)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["skill", "jobs", *courses, "row_max"])
        for i, (s, j) in enumerate(zip(skills, jobs)):
            w.writerow([s, j, *[f"{v:.4f}" for v in mat[i]], f"{mat[i].max():.4f}"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run id under outputs/ (default: latest)")
    ap.add_argument("--report", help="explicit path to a report.json")
    ap.add_argument("--top", type=int, default=15, help="top-K skills by demand (default 15)")
    ap.add_argument("--out", help="output PNG path (default: heatmap.png beside report.json)")
    ap.add_argument("--csv", action="store_true", help="also write heatmap.csv beside the PNG")
    args = ap.parse_args()

    report_path = resolve_report(args)
    report = json.loads(report_path.read_text())
    out = Path(args.out).resolve() if args.out else report_path.parent / "heatmap.png"

    draw(report, args.top, out)
    print(f"wrote {out}")
    if args.csv:
        csv_out = out.with_suffix(".csv")
        dump_csv(report, args.top, csv_out)
        print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
