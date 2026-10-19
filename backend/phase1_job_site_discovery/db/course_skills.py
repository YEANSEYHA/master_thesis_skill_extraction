"""Saved course-skill extractions (V8 Step 1) — snapshot store.

One row = one syllabus: course identity + the extracted skills JSON (exact
shape the extract-skills endpoint returned: [{name, source, evidence}]) + the
source markdown for provenance. The frontend pulls a saved row instead of
re-running the qwen3 extraction (no Ollama on replay).

Upsert key: (course_code, course_title) — re-saving the same course replaces
its skills instead of inserting a duplicate row.
"""

import json

from db.job_sites import get_connection

TABLE = "curriculum_course_skills"

_ensured = False


def create_course_skills_table():
    """Create curriculum_course_skills table + indexes if not exists."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id            SERIAL PRIMARY KEY,
            course_code   TEXT NOT NULL DEFAULT '',
            course_title  TEXT NOT NULL,
            category      TEXT,
            filename      TEXT,
            markdown      TEXT,
            skills        JSONB NOT NULL,
            skills_count  INT NOT NULL DEFAULT 0,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            updated_at    TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (course_code, course_title)
        );
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated_at
            ON {TABLE}(updated_at DESC);
        """)
        # Link to curriculum_documents (the uploaded file these skills came
        # from). Added here too — save_course_skills references the column, so
        # it must exist even when only this module has been ensured. The FK
        # constraint itself is added by db/curriculum_documents.py.
        cursor.execute(f"""
            ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS document_id INT;
        """)
        # Study year (1-6) within the programme — LLM-inferred at parse time
        # with a course-code/filename regex fallback (parse_pdf.infer_year).
        cursor.execute(f"""
            ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS year INT;
        """)
        conn.commit()
        print(f"✔ Table `{TABLE}` ready.")
    except Exception as e:
        print(f"❌ Error creating table {TABLE}: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _ensure_table():
    global _ensured
    if not _ensured:
        create_course_skills_table()
        _ensured = True


def save_course_skills(course_code: str, course_title: str, category: str | None,
                       filename: str | None, markdown: str | None,
                       skills: list, document_id: int | None = None,
                       year: int | None = None) -> dict:
    """Upsert one course's extracted skills by (course_code, course_title).

    `document_id` links back to the curriculum_documents row these skills were
    extracted from (added with the document store; None for older callers and
    for courses saved straight from a markdown paste).
    """
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {TABLE}
              (course_code, course_title, category, filename, markdown,
               skills, skills_count, document_id, year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (course_code, course_title) DO UPDATE SET
              category     = EXCLUDED.category,
              filename     = EXCLUDED.filename,
              markdown     = EXCLUDED.markdown,
              skills       = EXCLUDED.skills,
              skills_count = EXCLUDED.skills_count,
              document_id  = COALESCE(EXCLUDED.document_id, {TABLE}.document_id),
              year         = COALESCE(EXCLUDED.year, {TABLE}.year),
              updated_at   = NOW()
            RETURNING id, (created_at = updated_at) AS inserted, updated_at
        """, (course_code or "", course_title, category, filename, markdown,
              json.dumps(skills), len(skills), document_id, year))
        row_id, inserted, updated_at = cursor.fetchone()
        conn.commit()
        return {"id": row_id, "created": bool(inserted),
                "updated_at": str(updated_at)}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def list_course_skills(limit: int = 100) -> list[dict]:
    """Summaries, newest first — enough to render the saved-courses picker."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, course_code, course_title, category, filename,
                   skills_count, updated_at, year
            FROM {TABLE}
            ORDER BY year NULLS LAST, course_code, updated_at DESC LIMIT %s
        """, (limit,))
        return [{"id": r[0], "course_code": r[1], "course_title": r[2],
                 "category": r[3], "filename": r[4], "skills_count": r[5],
                 "updated_at": str(r[6]), "year": r[7]}
                for r in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_course_skills(row_id: int) -> dict | None:
    """Full row (skills JSON + markdown) for replay without re-extraction."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, course_code, course_title, category, filename,
                   markdown, skills, skills_count, updated_at
            FROM {TABLE} WHERE id = %s
        """, (row_id,))
        r = cursor.fetchone()
        if not r:
            return None
        return {"id": r[0], "course_code": r[1], "course_title": r[2],
                "category": r[3], "filename": r[4], "markdown": r[5],
                "skills": r[6], "skills_count": r[7], "updated_at": str(r[8])}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_course_skills(row_id: int) -> bool:
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE} WHERE id = %s", (row_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
