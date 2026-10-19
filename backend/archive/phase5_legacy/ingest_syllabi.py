#!/usr/bin/env python3
"""Seed the syllabus document store from the syllabi already in the repo.

`backend/course_syllabus/` holds the ITC course specifications as git-tracked
`.docx` — the same files that would otherwise be hand-dropped into the V8 Step 1
UI one at a time. This walks that tree and pushes every file through the exact
API path the browser uses (`POST /api/curricula/documents`): sha256 → R2 archive
→ row → Markdown. Skill extraction stays manual; it is a ~1-2 min LLM call per
course and should be reviewed.

Re-runnable by construction: the endpoint is idempotent on the file's content,
so a second run reports everything as cached and costs nothing.

    python ingest_syllabi.py --dry-run          # list what would be sent
    python ingest_syllabi.py                    # ingest all of course_syllabus/
    python ingest_syllabi.py --dir "…/Year 5 Semester 1"
    python ingest_syllabi.py --api http://localhost:8000

The API server must be running (it owns the DB and R2 credentials).
"""

import argparse
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE.parents[1] / "course_syllabus"
SUPPORTED = {".docx", ".pdf"}


def find_syllabi(root: Path) -> list[Path]:
    """Every supported syllabus under `root`, sorted, skipping Office lock files
    (`~$name.docx`) and .xlsx timetables that neither converter can read."""
    return sorted(p for p in root.rglob("*")
                  if p.is_file()
                  and p.suffix.lower() in SUPPORTED
                  and not p.name.startswith("~$"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest repo syllabi into the curriculum document store.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"folder to walk (default: {DEFAULT_DIR})")
    ap.add_argument("--api", default="http://localhost:8000",
                    help="API base URL (default: http://localhost:8000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files, send nothing")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-file timeout in seconds (docling is slow)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        sys.exit(f"[ingest] folder not found: {root}")

    files = find_syllabi(root)
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"[ingest] {len(files)} syllabi ({total_mb:.1f} MB) under {root}")
    if not files:
        return

    if args.dry_run:
        for p in files:
            print("   ", p.relative_to(root))
        print("[ingest] dry-run — nothing sent")
        return

    stat = {"new": 0, "cached": 0, "failed": 0}
    url = args.api.rstrip("/") + "/api/curricula/documents"
    with httpx.Client(timeout=args.timeout) as client:
        for i, p in enumerate(files, 1):
            rel = p.relative_to(root)
            try:
                with p.open("rb") as fh:
                    r = client.post(url, files={"file": (p.name, fh)})
                if r.status_code != 200:
                    detail = r.json().get("detail", r.text) if r.text else r.status_code
                    raise RuntimeError(detail)
                doc = r.json()
            except Exception as e:                      # noqa: BLE001 — keep going
                stat["failed"] += 1
                print(f"[{i:>3}/{len(files)}] FAIL {rel}: {str(e)[:120]}")
                continue
            key = "cached" if doc.get("cached") else "new"
            stat[key] += 1
            print(f"[{i:>3}/{len(files)}] {'reused ' if doc.get('cached') else 'ingested'} "
                  f"id={doc['id']:<4} {doc.get('chars') or 0:>7,} chars  "
                  f"r2={doc.get('r2_status'):<7} {rel}")

    print(f"[ingest] DONE — {stat['new']} new, {stat['cached']} already stored, "
          f"{stat['failed']} failed")
    if stat["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
