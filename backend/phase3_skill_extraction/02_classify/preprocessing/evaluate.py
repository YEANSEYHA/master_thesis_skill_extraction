"""Standalone F1 evaluation script — reads stage3_output.json and scores instantly.
No LLM calls needed. Quickly test different TOP_K and MATCH_THRESHOLD values.

Usage:
  python evaluate.py                          # defaults: 100 jobs, top-10, threshold 0.55
  python evaluate.py --limit 500              # evaluate all 500
  python evaluate.py --top-k 8 --threshold 0.50
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="F1 Evaluation: LLM Skills vs ESCO Ground Truth")
    parser.add_argument("--input", default="stage3_output.json", help="Path to stage3 output")
    parser.add_argument("--limit", type=int, default=100, help="Number of jobs to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Cap both sides to top-K skills")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity match threshold")
    parser.add_argument("--verbose", action="store_true", help="Show match/mismatch details per job")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    all_results = all_results[:args.limit]
    console.print(f"Evaluating [bold]{len(all_results)}[/bold] jobs | TOP_K={args.top_k} | threshold={args.threshold}")

    # Load similarity model
    similarity_model = SentenceTransformer("jjzha/jobbert-base-cased")

    table = Table(title="Per-Job F1 Score")
    table.add_column("Job Title", style="white", max_width=40)
    table.add_column("LLM Skills", justify="right")
    table.add_column("ESCO GT", justify="right")
    table.add_column("TP", justify="right", style="green")
    table.add_column("FP", justify="right", style="red")
    table.add_column("FN", justify="right", style="yellow")
    table.add_column("Precision", justify="right", style="cyan")
    table.add_column("Recall", justify="right", style="cyan")
    table.add_column("F1", justify="right", style="bold cyan")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in all_results:
        llm_skills = result.get("skills", [])
        ground_truth = result.get("esco_ground_truth", [])
        title = (result.get("job_title") or "Unknown")[:40]

        # Cap both sides to top-K
        llm_skills = llm_skills[:args.top_k]
        ground_truth = ground_truth[:args.top_k]

        if not llm_skills or not ground_truth:
            tp, fp, fn = 0, len(llm_skills), len(ground_truth)
            if args.verbose:
                console.print(f"\n[bold]── {title} ──[/bold] (skipped: no LLM skills or no GT)")
        else:
            llm_embeddings = similarity_model.encode(llm_skills, normalize_embeddings=True)
            gt_embeddings = similarity_model.encode(ground_truth, normalize_embeddings=True)
            sim_matrix = cosine_similarity(llm_embeddings, gt_embeddings)

            matched_gt = set()
            matched_llm = set()
            matched_pairs = []
            pairs = []
            for i in range(len(llm_skills)):
                for j in range(len(ground_truth)):
                    pairs.append((sim_matrix[i][j], i, j))
            pairs.sort(reverse=True)

            for score, i, j in pairs:
                if score < args.threshold:
                    break
                if i not in matched_llm and j not in matched_gt:
                    matched_llm.add(i)
                    matched_gt.add(j)
                    matched_pairs.append((score, llm_skills[i], ground_truth[j]))

            tp = len(matched_llm)
            fp = len(llm_skills) - tp
            fn = len(ground_truth) - tp

            if args.verbose:
                console.print(f"\n[bold]── {title} ──[/bold]")
                if matched_pairs:
                    console.print("  [green]✓ Matches:[/green]")
                    for score, llm_s, gt_s in matched_pairs:
                        console.print(f"    [green]✓[/green] \"{llm_s}\" ↔ \"{gt_s}\"  [dim](cosine: {score:.3f})[/dim]")

                unmatched_llm = [llm_skills[i] for i in range(len(llm_skills)) if i not in matched_llm]
                if unmatched_llm:
                    console.print("  [red]✗ FP (extra LLM skills):[/red]")
                    for s in unmatched_llm:
                        idx = llm_skills.index(s)
                        best_j = int(np.argmax(sim_matrix[idx]))
                        best_score = sim_matrix[idx][best_j]
                        console.print(f"    [red]✗[/red] \"{s}\"  [dim](best GT: \"{ground_truth[best_j]}\" = {best_score:.3f})[/dim]")

                unmatched_gt = [ground_truth[j] for j in range(len(ground_truth)) if j not in matched_gt]
                if unmatched_gt:
                    console.print("  [yellow]? FN (missed GT skills):[/yellow]")
                    for s in unmatched_gt:
                        idx = ground_truth.index(s)
                        best_i = int(np.argmax(sim_matrix[:, idx]))
                        best_score = sim_matrix[best_i][idx]
                        console.print(f"    [yellow]?[/yellow] \"{s}\"  [dim](best LLM: \"{llm_skills[best_i]}\" = {best_score:.3f})[/dim]")

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if args.verbose:
            console.print(f"  [cyan]P={precision:.0%}  R={recall:.0%}  F1={f1:.0%}[/cyan]  (TP={tp} FP={fp} FN={fn})")

        table.add_row(
            title, str(len(llm_skills)), str(len(ground_truth)),
            str(tp), str(fp), str(fn),
            f"{precision:.2%}", f"{recall:.2%}", f"{f1:.2%}",
        )

    console.print(table)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    summary = Table(title="Overall Metrics (Micro-Averaged)")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Total LLM Skills", str(total_tp + total_fp))
    summary.add_row("Total ESCO Ground Truth", str(total_tp + total_fn))
    summary.add_row("True Positives (matched)", str(total_tp))
    summary.add_row("False Positives (extra LLM)", str(total_fp))
    summary.add_row("False Negatives (missed GT)", str(total_fn))
    summary.add_row("Micro Precision", f"{micro_p:.4f} ({micro_p*100:.1f}%)")
    summary.add_row("Micro Recall", f"{micro_r:.4f} ({micro_r*100:.1f}%)")
    summary.add_row("Micro F1 Score", f"{micro_f1:.4f} ({micro_f1*100:.1f}%)")
    summary.add_row("Match Threshold", f"{args.threshold}")
    summary.add_row("Top-K", f"{args.top_k}")
    console.print(summary)


if __name__ == "__main__":
    main()
