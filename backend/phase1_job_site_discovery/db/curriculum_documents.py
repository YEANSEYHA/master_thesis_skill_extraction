"""Uploaded syllabus documents (V8 Step 1) — the upload-once store.

One row = one *file*, identified by the sha256 of its bytes rather than its
name: ITC ships the same document as `..._2024-2025.docx` and `..._2024_2025.docx`,
so filename dedupe would miss. The row holds where the original is archived
(Cloudflare R2) and the Markdown extracted from it; skill extraction then reads
the Markdown from Postgres and never touches the file again.

Lifecycle, driven by `status`:

    uploaded  -> row exists, bytes archived, not converted yet
    converted -> markdown present (this is what the pipeline consumes)
    extracted -> curriculum_course_skills has a row pointing back here
    error     -> conversion failed; `error` holds the message

`r2_status` tracks the archive independently (pending|stored|failed|skipped) so
a failed put can be retried on a later upload of the same file without touching
the Markdown. See `find_by_sha256` — the caller checks each stage separately
instead of treating "row exists" as "nothing to do".
"""

from db.job_sites import get_connection

TABLE = "curriculum_documents"
SKILLS_TABLE = "curriculum_course_skills"

_ensured = False


def create_documents_table():
    """Create curriculum_documents + index, and link curriculum_course_skills."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id          SERIAL PRIMARY KEY,
            sha256      TEXT UNIQUE NOT NULL,
            filename    TEXT NOT NULL,
            ext         TEXT NOT NULL,
            size_bytes  INT  NOT NULL DEFAULT 0,
            r2_bucket   TEXT,
            r2_key      TEXT,
            r2_status   TEXT NOT NULL DEFAULT 'pending',
            markdown    TEXT,
            chars       INT,
            parser      TEXT,
            status      TEXT NOT NULL DEFAULT 'uploaded',
            error       TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_updated_at
            ON {TABLE}(updated_at DESC);
        """)
        # Skills trace back to the exact bytes they came from. Column and
        # constraint are added separately because course_skills.py also creates
        # the bare column (its INSERT references it) and whichever table is
        # ensured first must not leave the other half-built.
        cursor.execute(f"""
            ALTER TABLE {SKILLS_TABLE}
            ADD COLUMN IF NOT EXISTS document_id INT;
        """)
        # ON DELETE SET NULL: dropping a document from the queue must not
        # destroy the skills extracted from it.
        cursor.execute(f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_constraint
                          WHERE conname = 'fk_course_skills_document') THEN
            ALTER TABLE {SKILLS_TABLE}
              ADD CONSTRAINT fk_course_skills_document
              FOREIGN KEY (document_id) REFERENCES {TABLE}(id) ON DELETE SET NULL;
          END IF;
        END $$;
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
        create_documents_table()
        _ensured = True


def _row(r: tuple) -> dict:
    """SELECT tuple (without markdown) -> dict, the shape the API returns."""
    return {"id": r[0], "sha256": r[1], "filename": r[2], "ext": r[3],
            "size_bytes": r[4], "r2_bucket": r[5], "r2_key": r[6],
            "r2_status": r[7], "chars": r[8], "parser": r[9], "status": r[10],
            "error": r[11], "created_at": str(r[12]), "updated_at": str(r[13])}


_COLS = ("id, sha256, filename, ext, size_bytes, r2_bucket, r2_key, r2_status, "
         "chars, parser, status, error, created_at, updated_at")


def find_by_sha256(sha256: str) -> dict | None:
    """The resume gate's input: the row for these exact bytes, or None.

    Deliberately excludes `markdown` — callers only need to know *whether* it is
    there (`chars`), and the column can be ~100 KB.
    """
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT {_COLS} FROM {TABLE} WHERE sha256 = %s", (sha256,))
        r = cursor.fetchone()
        return _row(r) if r else None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_document(sha256: str, filename: str, ext: str, size_bytes: int,
                  r2_bucket: str | None, r2_key: str | None,
                  r2_status: str = "pending") -> dict:
    """Insert a new document row, or return the existing one for these bytes.

    ON CONFLICT keeps the first filename (same bytes, two names -> one row) but
    refreshes the archive columns, which is what lets a failed R2 put be retried
    on a later upload.
    """
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {TABLE}
              (sha256, filename, ext, size_bytes, r2_bucket, r2_key, r2_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sha256) DO UPDATE SET
              r2_bucket  = COALESCE(EXCLUDED.r2_bucket, {TABLE}.r2_bucket),
              r2_key     = COALESCE(EXCLUDED.r2_key, {TABLE}.r2_key),
              r2_status  = EXCLUDED.r2_status,
              updated_at = NOW()
            RETURNING {_COLS}
        """, (sha256, filename, ext, size_bytes, r2_bucket, r2_key, r2_status))
        row = _row(cursor.fetchone())
        conn.commit()
        return row
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def set_markdown(doc_id: int, markdown: str, parser: str) -> dict:
    """Store the conversion result and move the row to `converted`."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {TABLE}
               SET markdown = %s, chars = %s, parser = %s,
                   status = 'converted', error = NULL, updated_at = NOW()
             WHERE id = %s
            RETURNING {_COLS}
        """, (markdown, len(markdown), parser, doc_id))
        row = _row(cursor.fetchone())
        conn.commit()
        return row
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def set_status(doc_id: int, status: str, error: str | None = None) -> None:
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {TABLE} SET status = %s, error = %s, updated_at = NOW()
             WHERE id = %s
        """, (status, error, doc_id))
        conn.commit()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def set_r2(doc_id: int, r2_bucket: str | None, r2_key: str | None,
           r2_status: str) -> None:
    """Update only the archive columns (used when retrying a failed put)."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {TABLE}
               SET r2_bucket = %s, r2_key = %s, r2_status = %s, updated_at = NOW()
             WHERE id = %s
        """, (r2_bucket, r2_key, r2_status, doc_id))
        conn.commit()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def list_documents(limit: int = 200) -> list[dict]:
    """Queue rebuild — newest first, markdown excluded (fetch per row on demand).

    Carries the joined course identity so Step 1 can show what a document became
    without a second request per row.
    """
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {', '.join('d.' + c.strip() for c in _COLS.split(','))},
                   s.id, s.course_code, s.course_title, s.category, s.skills_count
              FROM {TABLE} d
              LEFT JOIN {SKILLS_TABLE} s ON s.document_id = d.id
             ORDER BY d.updated_at DESC
             LIMIT %s
        """, (limit,))
        out = []
        for r in cursor.fetchall():
            doc = _row(r[:14])
            doc["course"] = ({"id": r[14], "code": r[15], "title": r[16],
                              "category": r[17], "skills_count": r[18]}
                             if r[14] else None)
            out.append(doc)
        return out
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_document(doc_id: int, with_markdown: bool = True) -> dict | None:
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT {_COLS}, markdown FROM {TABLE} WHERE id = %s",
                       (doc_id,))
        r = cursor.fetchone()
        if not r:
            return None
        doc = _row(r[:14])
        if with_markdown:
            doc["markdown"] = r[14] or ""
        return doc
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_document(doc_id: int) -> bool:
    """Drop the row. The R2 object is intentionally kept — it is the archive."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE} WHERE id = %s", (doc_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
