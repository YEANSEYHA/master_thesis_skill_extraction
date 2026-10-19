"""Wipe extracted skills so every course can be re-extracted in one uniform pass.

Deletes all rows from curriculum_course_skills and resets each document's
status from 'extracted' back to 'converted'. Documents, markdown and R2
originals are untouched — re-dropping the same files in the V8 tab hits the
cached path (no docling) and only re-runs the LLM extraction.

Usage:
    python3 reset_extractions.py            # dry-run: show what would be removed
    python3 reset_extractions.py --yes      # actually delete
"""

import sys

from db.job_sites import get_connection


def main() -> None:
    do_it = "--yes" in sys.argv
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT course_code, course_title FROM curriculum_course_skills ORDER BY course_code")
        rows = cur.fetchall()
        cur.execute("SELECT count(*) FROM curriculum_documents WHERE status = 'extracted'")
        n_docs = cur.fetchone()[0]

        print(f"{len(rows)} extracted course(s), {n_docs} document(s) marked 'extracted':")
        for code, title in rows:
            print(f"  {code:12s} {title}")

        if not do_it:
            print("\nDry-run only. Re-run with --yes to delete.")
            return

        cur.execute("DELETE FROM curriculum_course_skills")
        deleted = cur.rowcount
        cur.execute("UPDATE curriculum_documents SET status = 'converted', updated_at = NOW() WHERE status = 'extracted'")
        reset = cur.rowcount
        conn.commit()
        print(f"\nDeleted {deleted} course-skill row(s); reset {reset} document(s) to 'converted'.")
        print("Markdown + R2 originals kept — re-drop the files and extraction re-runs on the frozen prompt.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
