"""
ESCO Taxonomy Evaluation: Compare LLM vs Human skills at ESCO level

Instead of comparing free-text skills directly (where "Python programming" != "computer programming"),
we map BOTH sides to the standardized ESCO taxonomy first, then compare at ESCO level.

Steps:
  1. Load ESCO taxonomy from local CSV (preferredLabel + altLabels for better coverage)
  2. Encode all ESCO labels with JobBERT → embeddings
  3. Map LLM skills → nearest ESCO skill (via cosine similarity)
  4. Map Human skills → nearest ESCO skill (same process)
  5. Compare ESCO code sets → Precision, Recall, F1

Usage:
  python esco_evaluation.py
  python esco_evaluation.py --input ../outputs/skillspan_benchmark/20260309_154732/skillspan_benchmark.json
  python esco_evaluation.py --verbose
  python esco_evaluation.py --inspect
"""

import json
import csv
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ── Step 1: Load ESCO Skills from local CSV ──

ESCO_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data_collection", "esco_skills_en.csv")


def load_esco_skills(csv_path=ESCO_CSV_PATH):
    """
    Load ESCO skills from local CSV with preferredLabel + altLabels.

    Returns:
        label_to_preferred: dict mapping every label (preferred + alt) → preferredLabel
        all_labels: list of all unique labels (for encoding)
    """
    console.print(f"[bold]Loading ESCO taxonomy from {csv_path}...[/bold]")

    label_to_preferred = {}  # "manage music staff" → "manage musical staff"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preferred = row["preferredLabel"].strip()
            if not preferred:
                continue

            # Add preferred label → maps to itself
            label_to_preferred[preferred.lower()] = preferred

            # Add alt labels → map to preferred
            alt_labels = row.get("altLabels", "")
            if alt_labels:
                for alt in alt_labels.split("\n"):
                    alt = alt.strip()
                    if alt:
                        label_to_preferred[alt.lower()] = preferred

    all_labels = sorted(label_to_preferred.keys())
    num_preferred = len(set(label_to_preferred.values()))
    console.print(f"[green]Loaded {num_preferred} ESCO skills with {len(all_labels)} total labels (preferred + alt)[/green]")
    return label_to_preferred, all_labels


# ── Step 2: Encode ESCO Labels ──

def encode_esco_labels(all_labels, model):
    """Encode all ESCO labels (preferred + alt) into embeddings using JobBERT."""
    console.print(f"[bold]Encoding {len(all_labels)} ESCO labels with JobBERT...[/bold]")
    esco_embeddings = model.encode(all_labels, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    console.print("[green]ESCO embeddings ready[/green]")
    return esco_embeddings


# ── Step 3 & 4: Map free-text skills → nearest ESCO skill ──

def map_to_esco(skills, label_to_preferred, all_labels, esco_embeddings, model):
    """
    Map a list of free-text skills to their nearest ESCO preferred label.
    Matches against all labels (preferred + alt), then resolves to the preferred label.
    Returns list of dicts: {original, matched_label, esco_skill (preferred), similarity}
    """
    if not skills:
        return []

    skill_embeddings = model.encode(skills, normalize_embeddings=True)
    sim_matrix = cosine_similarity(skill_embeddings, esco_embeddings)

    mappings = []
    for i, skill in enumerate(skills):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        matched_label = all_labels[best_idx]
        preferred = label_to_preferred[matched_label]
        mappings.append({
            "original": skill,
            "matched_label": matched_label,
            "esco_skill": preferred,
            "similarity": best_score,
        })

    return mappings


# ── Step 5: Compare at ESCO level ──

def evaluate_esco_f1(all_results, model, label_to_preferred, all_labels, esco_embeddings, top_k, verbose):
    """Compare LLM vs Human after both are mapped to ESCO. Exact set match on ESCO labels."""
    console.print("\n[bold cyan]--- ESCO Taxonomy F1 (Exact Match on ESCO Labels) ---[/bold cyan]")

    total_tp, total_fp, total_fn = 0, 0, 0
    job_f1s = []

    if verbose:
        detail_table = Table(title="Per-Job ESCO F1")
        detail_table.add_column("Job Title", style="white", max_width=40)
        detail_table.add_column("LLM→ESCO", justify="right")
        detail_table.add_column("Human→ESCO", justify="right")
        detail_table.add_column("Matched", justify="right", style="green")
        detail_table.add_column("Precision", justify="right", style="cyan")
        detail_table.add_column("Recall", justify="right", style="cyan")
        detail_table.add_column("F1", justify="right", style="bold cyan")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:40]

        if not llm_skills or not human_skills:
            job_f1s.append(0.0)
            if verbose:
                detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)),
                                     "0", "0.0%", "0.0%", "0.0%")
            continue

        # Map both sides to ESCO
        llm_mappings = map_to_esco(llm_skills, label_to_preferred, all_labels, esco_embeddings, model)
        human_mappings = map_to_esco(human_skills, label_to_preferred, all_labels, esco_embeddings, model)

        # Get unique ESCO labels for each side
        llm_esco_set = set(m["esco_skill"] for m in llm_mappings)
        human_esco_set = set(m["esco_skill"] for m in human_mappings)

        # Set comparison
        tp = len(llm_esco_set & human_esco_set)
        fp = len(llm_esco_set - human_esco_set)
        fn = len(human_esco_set - llm_esco_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        job_f1s.append(f1)

        if verbose:
            detail_table.add_row(title, str(len(llm_esco_set)), str(len(human_esco_set)),
                                 str(tp), f"{p:.0%}", f"{r:.0%}", f"{f1:.0%}")

    if verbose:
        console.print(detail_table)

    # Summary
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    macro_f1 = sum(job_f1s) / len(job_f1s) if job_f1s else 0

    summary = Table(title="ESCO Taxonomy F1 Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Jobs evaluated", str(len(job_f1s)))
    summary.add_row("", "")
    summary.add_row("Micro Precision", f"{micro_p * 100:.1f}%")
    summary.add_row("Micro Recall", f"{micro_r * 100:.1f}%")
    summary.add_row("[bold]Micro F1[/bold]", f"[bold]{micro_f1 * 100:.1f}%[/bold]")
    summary.add_row("", "")
    summary.add_row("[bold]Macro F1[/bold]", f"[bold]{macro_f1 * 100:.1f}%[/bold]")
    summary.add_row("", "")
    summary.add_row("Total TP", str(total_tp))
    summary.add_row("Total FP", str(total_fp))
    summary.add_row("Total FN", str(total_fn))

    console.print(summary)


def evaluate_esco_precision_at_k(all_results, model, label_to_preferred, all_labels, esco_embeddings, top_k, verbose):
    """Precision@K: of top-K LLM skills mapped to ESCO, how many match a human ESCO skill?"""
    K_VALUES = [5, 10, 15]
    console.print(f"\n[bold cyan]--- ESCO Precision@K ---[/bold cyan]")

    k_totals = {k: {"matched": 0, "total": 0} for k in K_VALUES}
    job_precisions = {k: [] for k in K_VALUES}

    if verbose:
        detail_table = Table(title="Per-Job ESCO Precision@K")
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

        # Map human skills to ESCO set
        human_mappings = map_to_esco(human_skills, label_to_preferred, all_labels, esco_embeddings, model)
        human_esco_set = set(m["esco_skill"] for m in human_mappings)

        # Map LLM skills to ESCO (need per-skill mapping for top-K slicing)
        llm_mappings = map_to_esco(llm_skills, label_to_preferred, all_labels, esco_embeddings, model)

        row_p_at_k = []
        for k in K_VALUES:
            top_k_mappings = llm_mappings[:k]
            top_k_esco = set(m["esco_skill"] for m in top_k_mappings)
            matched = len(top_k_esco & human_esco_set)
            p_at_k = matched / len(top_k_esco) if top_k_esco else 0

            k_totals[k]["matched"] += matched
            k_totals[k]["total"] += len(top_k_esco)
            job_precisions[k].append(p_at_k)
            row_p_at_k.append(f"{p_at_k*100:.0f}%")

        if verbose:
            detail_table.add_row(title, str(len(llm_skills)), str(len(human_skills)), *row_p_at_k)

    if verbose:
        console.print(detail_table)

    # Summary
    summary = Table(title="ESCO Precision@K Summary")
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


def inspect_esco_mappings(all_results, model, label_to_preferred, all_labels, esco_embeddings, top_k):
    """Show how each skill maps to ESCO for debugging."""
    console.print("\n[bold magenta]--- ESCO Mapping Inspection ---[/bold magenta]\n")

    for result in all_results:
        llm_skills = result.get("llm_skills", [])[:top_k]
        human_skills = result.get("human_skills", [])
        title = (result.get("job_title") or "Unknown")[:60]

        if not llm_skills or not human_skills:
            continue

        llm_mappings = map_to_esco(llm_skills, label_to_preferred, all_labels, esco_embeddings, model)
        human_mappings = map_to_esco(human_skills, label_to_preferred, all_labels, esco_embeddings, model)

        llm_esco_set = set(m["esco_skill"] for m in llm_mappings)
        human_esco_set = set(m["esco_skill"] for m in human_mappings)
        matched_esco = llm_esco_set & human_esco_set

        tp = len(matched_esco)
        fp = len(llm_esco_set - human_esco_set)
        fn = len(human_esco_set - llm_esco_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # LLM mappings table
        table = Table(
            title=f"[bold]{title}[/bold]  P={p:.0%} R={r:.0%} F1={f1:.0%}",
            box=box.SIMPLE_HEAD, show_lines=True,
        )
        table.add_column("Side", style="bold", min_width=6)
        table.add_column("Original Skill", style="white", min_width=30)
        table.add_column("→ ESCO Mapping", style="cyan", min_width=35)
        table.add_column("Sim", justify="right", min_width=5)
        table.add_column("Match?", justify="center", min_width=6)

        for m in llm_mappings:
            is_match = m["esco_skill"] in matched_esco
            match_str = "[green]YES[/green]" if is_match else "[red]NO[/red]"
            esco_display = m["esco_skill"]
            if m["matched_label"] != m["esco_skill"].lower():
                esco_display += f" (via: {m['matched_label']})"
            table.add_row("LLM", m["original"], esco_display, f"{m['similarity']:.2f}", match_str)

        table.add_section()

        for m in human_mappings:
            is_match = m["esco_skill"] in matched_esco
            match_str = "[green]YES[/green]" if is_match else "[yellow]MISSED[/yellow]"
            esco_display = m["esco_skill"]
            if m["matched_label"] != m["esco_skill"].lower():
                esco_display += f" (via: {m['matched_label']})"
            table.add_row("Human", m["original"], esco_display, f"{m['similarity']:.2f}", match_str)

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="ESCO Taxonomy Evaluation")
    parser.add_argument("--input", default="../outputs/skillspan_benchmark/20260309_154732/skillspan_benchmark.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--model", default="jjzha/jobbert-base-cased")
    args = parser.parse_args()

    # Load benchmark results (llm_skills + human_skills per job)
    with open(args.input, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    if args.limit is not None:
        all_results = all_results[:args.limit]

    console.print(f"[bold]ESCO Taxonomy Evaluation[/bold]")
    console.print(f"[bold]Evaluating {len(all_results)} jobs | top_k={args.top_k}[/bold]\n")

    # Step 1: Load ESCO taxonomy from local CSV
    label_to_preferred, all_labels = load_esco_skills()

    # Step 2: Load model and encode ESCO labels
    console.print(f"[bold]Loading similarity model: {args.model}...[/bold]")
    model = SentenceTransformer(args.model)
    esco_embeddings = encode_esco_labels(all_labels, model)

    # Step 5: Evaluate
    evaluate_esco_f1(all_results, model, label_to_preferred, all_labels, esco_embeddings, args.top_k, args.verbose)
    evaluate_esco_precision_at_k(all_results, model, label_to_preferred, all_labels, esco_embeddings, args.top_k, args.verbose)

    # Optional: inspect mappings
    if args.inspect:
        inspect_esco_mappings(all_results, model, label_to_preferred, all_labels, esco_embeddings, args.top_k)


if __name__ == "__main__":
    main()
