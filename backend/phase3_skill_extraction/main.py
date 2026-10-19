"""
Phase 3 — Classify & Extract pipeline.

Reads crawled HTML from phase2 output, then:
  Stage 1: Classify pages (single job detail vs non-job)
  Stage 2: Extract structured job data from job detail pages only

Data flow:
  phase2 outputs/<run_id>/<domain>/page_*.html
    → Stage 1 classify → outputs/classification/single_job_detail/
    → Stage 2 extract  → outputs/stage2/<timestamp>/stage2_result.json

Usage:
  python main.py --run-id 2026-03-11_13-35-15           # process a specific crawl run
  python main.py --run-id 2026-03-11_13-35-15 --limit 10
  python main.py --classify-only --run-id ...            # stage 1 only
  python main.py --extract-only                          # stage 2 only (assumes stage 1 already ran)
  python main.py --list-runs                             # show available crawl runs
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE2_OUTPUTS = os.path.join(PROJECT_ROOT, "..", "archive", "phase2_playwright_crawler", "outputs")
CLASSIFY_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "classification")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "02_classify"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_extract"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib"))

from stage1_classify import main as classify_main
from stage2_skill_extraction import main as extract_main


def list_runs():
    """Show available crawl runs from phase2."""
    outputs = Path(PHASE2_OUTPUTS)
    if not outputs.exists():
        print(f"No phase2 outputs found at {PHASE2_OUTPUTS}")
        return

    runs = sorted([d for d in outputs.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        print("No crawl runs found.")
        return

    print(f"{'Run ID':<28} {'Domains':<10} {'HTML Files':<12}")
    print("-" * 50)
    for run_dir in runs:
        domains = [d for d in run_dir.iterdir() if d.is_dir()]
        html_count = sum(1 for d in domains for _ in d.glob("page_*.html"))
        print(f"{run_dir.name:<28} {len(domains):<10} {html_count:<12}")


def run(args):
    if args.list_runs:
        list_runs()
        return

    # Resolve input directory from phase2
    if args.run_id:
        input_dir = os.path.join(PHASE2_OUTPUTS, args.run_id)
        if not os.path.exists(input_dir):
            print(f"Error: Run '{args.run_id}' not found at {input_dir}")
            print("Use --list-runs to see available runs.")
            return
    else:
        input_dir = None

    # Stage 1: Classify
    if not args.extract_only:
        print("=" * 60)
        print("STAGE 1: Classify HTML pages")
        if input_dir:
            print(f"  Input:  {input_dir}")
        print(f"  Output: {CLASSIFY_OUTPUT}")
        print("=" * 60)

        if input_dir:
            classify_main(
                limit=args.limit,
                input_dir=input_dir,
                output_dir=CLASSIFY_OUTPUT,
                nested=True,
            )
        else:
            classify_main(limit=args.limit, output_dir=CLASSIFY_OUTPUT)
        print()

    # Stage 2: Extract from classified job pages
    if not args.classify_only:
        job_pages_dir = os.path.join(CLASSIFY_OUTPUT, "single_job_detail")
        if not os.path.exists(job_pages_dir):
            print(f"Error: No classified job pages found at {job_pages_dir}")
            print("Run stage 1 first (or use --classify-only to check).")
            return

        print("=" * 60)
        print("STAGE 2: Extract job data from classified pages")
        print(f"  Input:  {job_pages_dir}")
        print("=" * 60)
        extract_main(input_dir=job_pages_dir, limit=args.limit)
        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Classify & Extract pipeline")
    parser.add_argument("--run-id", type=str, help="Phase 2 crawl run ID (folder name)")
    parser.add_argument("--list-runs", action="store_true", help="List available phase2 crawl runs")
    parser.add_argument("--classify-only", action="store_true", help="Run stage 1 only")
    parser.add_argument("--extract-only", action="store_true", help="Run stage 2 only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()
    run(args)
