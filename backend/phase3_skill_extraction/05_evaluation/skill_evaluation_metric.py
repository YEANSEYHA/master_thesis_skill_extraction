"""
Stage 4: Evaluate LLM skills vs Human skills (SkillSpan)

Reads stage3_skillspan_output.json and compares:
  - llm_skills (from your LLM)
  - human_skills (from SkillSpan human experts)

Metrics:
  1. Hard F1 (multi-threshold: 0.60, 0.65, 0.70, 0.75, 0.80)
  2. Semantic F1 (arXiv:2509.21633)
  3. Precision@K (of top-K LLM skills, how many match a human skill?)

Usage:
  python stage4_metric_evaluation.py
  python stage4_metric_evaluation.py --input stage3_skillspan_output.json --verbose --inspect
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]


def greedy_match(sim_matrix, threshold):
    """Greedy bipartite matching: pick highest similarity pairs first, above threshold."""
    num_llm, num_gt = sim_matrix.shape
    pairs = sorted(
        [(sim_matrix[i][j], i, j) for i in range(num_llm) for j in range(num_gt)],
        reverse=True,
    )
    matched_llm, matched_gt, matched_pairs = set(), set(), []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i not in matched_llm and j not in matched_gt:
            matched_llm.add(i)
            matched_gt.add(j)
            matched_pairs.append((score, i, j))
    return matched_llm, matched_gt, matched_pairs


def evaluate_hard_f1(all_results, similarity_model, top_k, verbose):
    """Metric 1: Multi-Threshold Hard F1."""
    console.print("\n[bold cyan]--- Metric 1: Hard F1 (Multi-Threshold) ---[/bold cyan]")

    threshold_totals = {t: {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS}

    if verbose:
        detail_table = Table(title="Per-Job Hard F1")
        detail_table.add_column("Job Title", style="white", max_width=40)
        detail_table.add_column("LLM", justify="right")
        detail_table.add_column("Human", justify="right")
        for t in THRESHOLDS:
            detail_table.add_column(f"F1@{t}", justify="right", style="cyan")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:40]

        if not llm_skills or not human_skills:
            for t in THRESHOLDS:
                threshold_totals[t]["fp"] += len(llm_skills)
                threshold_totals[t]["fn"] += len(human_skills)
            if verbose:
                detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)),
                                     *["--" for _ in THRESHOLDS])
            continue

        llm_emb = similarity_model.encode(llm_skills, normalize_embeddings=True)
        gt_emb = similarity_model.encode(human_skills, normalize_embeddings=True)
        sim_matrix = cosine_similarity(llm_emb, gt_emb)

        job_f1s = []
        for t in THRESHOLDS:
            matched_llm, matched_gt, _ = greedy_match(sim_matrix, t)
            tp = len(matched_llm)
            fp = len(llm_skills) - tp
            fn = len(human_skills) - tp

            threshold_totals[t]["tp"] += tp
            threshold_totals[t]["fp"] += fp
            threshold_totals[t]["fn"] += fn

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            job_f1s.append(f"{f1:.0%}")

        if verbose:
            detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)), *job_f1s)

    if verbose:
        console.print(detail_table)

    # Summary
    summary = Table(title="Hard F1 Summary (Micro-Averaged)")
    summary.add_column("Threshold", style="bold")
    summary.add_column("TP", justify="right", style="green")
    summary.add_column("FP", justify="right", style="red")
    summary.add_column("FN", justify="right", style="yellow")
    summary.add_column("Precision", justify="right", style="cyan")
    summary.add_column("Recall", justify="right", style="cyan")
    summary.add_column("F1", justify="right", style="bold cyan")

    for t in THRESHOLDS:
        tp = threshold_totals[t]["tp"]
        fp = threshold_totals[t]["fp"]
        fn = threshold_totals[t]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        summary.add_row(f"@ {t:.2f}", str(tp), str(fp), str(fn),
                        f"{p*100:.1f}%", f"{r*100:.1f}%", f"{f1*100:.1f}%")

    console.print(summary)


def evaluate_semantic_f1(all_results, similarity_model, top_k, verbose):
    """Metric 2: Semantic F1 (arXiv:2509.21633)."""
    console.print("\n[bold cyan]--- Metric 2: Semantic F1 ---[/bold cyan]")

    job_f1s = []
    micro_p_sum, micro_r_sum = 0.0, 0.0
    total_llm, total_gt = 0, 0

    if verbose:
        detail_table = Table(title="Per-Job Semantic F1")
        detail_table.add_column("Job Title", style="white", max_width=40)
        detail_table.add_column("LLM", justify="right")
        detail_table.add_column("Human", justify="right")
        detail_table.add_column("Sem-P", justify="right", style="cyan")
        detail_table.add_column("Sem-R", justify="right", style="cyan")
        detail_table.add_column("Sem-F1", justify="right", style="bold cyan")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:40]

        if not llm_skills or not human_skills:
            job_f1s.append(0.0)
            total_llm += len(llm_skills)
            total_gt += len(human_skills)
            if verbose:
                detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)),
                                     "0.0%", "0.0%", "0.0%")
            continue

        llm_emb = similarity_model.encode(llm_skills, normalize_embeddings=True)
        gt_emb = similarity_model.encode(human_skills, normalize_embeddings=True)
        sim_matrix = cosine_similarity(llm_emb, gt_emb)

        sem_p = float(np.mean(np.max(sim_matrix, axis=1)))
        sem_r = float(np.mean(np.max(sim_matrix, axis=0)))
        sem_f1 = 2 * sem_p * sem_r / (sem_p + sem_r) if (sem_p + sem_r) > 0 else 0

        job_f1s.append(sem_f1)
        micro_p_sum += float(np.sum(np.max(sim_matrix, axis=1)))
        micro_r_sum += float(np.sum(np.max(sim_matrix, axis=0)))
        total_llm += len(llm_skills)
        total_gt += len(human_skills)

        if verbose:
            detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)),
                                 f"{sem_p*100:.1f}%", f"{sem_r*100:.1f}%", f"{sem_f1*100:.1f}%")

    if verbose:
        console.print(detail_table)

    # Summary
    micro_p = micro_p_sum / total_llm if total_llm > 0 else 0
    micro_r = micro_r_sum / total_gt if total_gt > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    macro_f1 = sum(job_f1s) / len(job_f1s) if job_f1s else 0

    summary = Table(title="Semantic F1 Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Jobs evaluated", str(len(job_f1s)))
    summary.add_row("Total LLM skills", str(total_llm))
    summary.add_row("Total Human skills", str(total_gt))
    summary.add_row("", "")
    summary.add_row("Micro Sem-Precision", f"{micro_p*100:.1f}%")
    summary.add_row("Micro Sem-Recall", f"{micro_r*100:.1f}%")
    summary.add_row("[bold]Micro Sem-F1[/bold]", f"[bold]{micro_f1*100:.1f}%[/bold]")
    summary.add_row("", "")
    summary.add_row("[bold]Macro Sem-F1[/bold]", f"[bold]{macro_f1*100:.1f}%[/bold]")

    console.print(summary)


def evaluate_precision_at_k(all_results, similarity_model, top_k, verbose, threshold=0.60):
    """Metric 3: Precision@K — of top-K LLM skills, how many match a human skill?"""
    K_VALUES = [5, 10, 15]
    console.print(f"\n[bold cyan]--- Metric 3: Precision@K (threshold={threshold}) ---[/bold cyan]")

    k_totals = {k: {"matched": 0, "total": 0} for k in K_VALUES}
    job_precisions = {k: [] for k in K_VALUES}

    if verbose:
        detail_table = Table(title=f"Per-Job Precision@K (threshold={threshold})")
        detail_table.add_column("Job Title", style="white", max_width=40)
        detail_table.add_column("LLM", justify="right")
        detail_table.add_column("Human", justify="right")
        for k in K_VALUES:
            detail_table.add_column(f"P@{k}", justify="right", style="cyan")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:40]

        if not llm_skills or not human_skills:
            for k in K_VALUES:
                job_precisions[k].append(0.0)
            if verbose:
                detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)),
                                     *["--" for _ in K_VALUES])
            continue

        llm_emb = similarity_model.encode(llm_skills, normalize_embeddings=True)
        gt_emb = similarity_model.encode(human_skills, normalize_embeddings=True)
        sim_matrix = cosine_similarity(llm_emb, gt_emb)

        row_p_at_k = []
        for k in K_VALUES:
            top_k_skills = llm_skills[:k]
            top_k_sim = sim_matrix[:k]
            # A prediction "matches" if its best similarity to any human skill >= threshold
            matched = sum(1 for i in range(len(top_k_skills)) if np.max(top_k_sim[i]) >= threshold)
            p_at_k = matched / len(top_k_skills) if top_k_skills else 0

            k_totals[k]["matched"] += matched
            k_totals[k]["total"] += len(top_k_skills)
            job_precisions[k].append(p_at_k)
            row_p_at_k.append(f"{p_at_k*100:.0f}%")

        if verbose:
            detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)), *row_p_at_k)

    if verbose:
        console.print(detail_table)

    # Summary
    summary = Table(title="Precision@K Summary")
    summary.add_column("K", style="bold")
    summary.add_column("Matched", justify="right", style="green")
    summary.add_column("Total", justify="right", style="white")
    summary.add_column("Micro P@K", justify="right", style="cyan")
    summary.add_column("Macro P@K", justify="right", style="bold cyan")

    for k in K_VALUES:
        matched = k_totals[k]["matched"]
        total = k_totals[k]["total"]
        micro = matched / total if total > 0 else 0
        macro = sum(job_precisions[k]) / len(job_precisions[k]) if job_precisions[k] else 0
        summary.add_row(str(k), str(matched), str(total),
                        f"{micro*100:.1f}%", f"{macro*100:.1f}%")

    console.print(summary)


def inspect_errors(all_results, similarity_model, top_k, threshold):
    """Side-by-side inspection: TP / FP / FN per job."""
    console.print(f"\n[bold magenta]--- Inspection Mode (threshold={threshold}) ---[/bold magenta]\n")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:60]

        if not llm_skills or not human_skills:
            continue

        llm_emb = similarity_model.encode(llm_skills, normalize_embeddings=True)
        gt_emb = similarity_model.encode(human_skills, normalize_embeddings=True)
        sim_matrix = cosine_similarity(llm_emb, gt_emb)

        matched_llm, matched_gt, matched_pairs = greedy_match(sim_matrix, threshold)

        tp = len(matched_llm)
        fp = len(llm_skills) - tp
        fn = len(human_skills) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        table = Table(
            title=f"[bold]{title}[/bold]  LLM={len(llm_skills)} Human={len(human_skills)} | P={p:.0%} R={r:.0%} F1={f1:.0%}",
            box=box.SIMPLE_HEAD, show_lines=True,
        )
        table.add_column("MATCHED (TP)", style="green", min_width=45)
        table.add_column("FALSE POSITIVE (LLM extra)", style="red", min_width=40)
        table.add_column("FALSE NEGATIVE (Human missed)", style="yellow", min_width=40)

        tp_rows = [
            f'"{llm_skills[i]}" <-> "{human_skills[j]}" ({score:.2f})'
            for score, i, j in matched_pairs
        ]
        fp_rows = [
            f'"{llm_skills[i]}" (best: {sim_matrix[i][int(np.argmax(sim_matrix[i]))]:.2f})'
            for i in range(len(llm_skills)) if i not in matched_llm
        ]
        fn_rows = [
            f'"{human_skills[j]}" (best: {sim_matrix[int(np.argmax(sim_matrix[:, j])), j]:.2f})'
            for j in range(len(human_skills)) if j not in matched_gt
        ]

        max_rows = max(len(tp_rows), len(fp_rows), len(fn_rows), 1)
        for row_idx in range(max_rows):
            table.add_row(
                tp_rows[row_idx] if row_idx < len(tp_rows) else "",
                fp_rows[row_idx] if row_idx < len(fp_rows) else "",
                fn_rows[row_idx] if row_idx < len(fn_rows) else "",
            )

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate LLM vs Human skills")
    parser.add_argument("--input", default="../outputs/skillspan_benchmark/20260309_154732/skillspan_benchmark.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-threshold", type=float, default=0.65)
    parser.add_argument("--model", default="jjzha/jobbert-base-cased")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    if args.limit is not None:
        all_results = all_results[:args.limit]
    console.print(f"[bold]Evaluating {len(all_results)} jobs | top_k={args.top_k} | model={args.model}[/bold]\n")
    console.print(f"[bold green]Ground truth: HUMAN (SkillSpan domain experts)[/bold green]")

    console.print(f"Loading similarity model: {args.model}...")
    similarity_model = SentenceTransformer(args.model)

    evaluate_hard_f1(all_results, similarity_model, args.top_k, args.verbose)
    evaluate_semantic_f1(all_results, similarity_model, args.top_k, args.verbose)
    evaluate_precision_at_k(all_results, similarity_model, args.top_k, args.verbose, args.inspect_threshold)

    if args.inspect:
        inspect_errors(all_results, similarity_model, args.top_k, args.inspect_threshold)


if __name__ == "__main__":
    main()
