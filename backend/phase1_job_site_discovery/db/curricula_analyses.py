"""Saved curriculum analyses (Phase 5 V6) — snapshot store.

One row = one completed analyze run: course identity + the settings that
produced it + the FULL endpoint result JSON. The frontend replays a snapshot
without recomputing (no Ollama, no re-verify).
"""

import hashlib
import json

from db.job_sites import get_connection

TABLE = "curricula_analyses"

_ensured = False


def create_curricula_analyses_table():
    """Create curricula_analyses table + indexes if not exists."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id             SERIAL PRIMARY KEY,
            course_code    TEXT,
            course_title   TEXT,
            domain         TEXT,
            coverage_pct   NUMERIC,
            recommendation TEXT,
            settings       JSONB,
            result         JSONB,
            result_hash    TEXT,
            created_at     TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_course_code
            ON {TABLE}(course_code);
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_created_at
            ON {TABLE}(created_at DESC);
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
        create_curricula_analyses_table()
        _ensured = True


def save_analysis(course_code: str, course_title: str, domain: str,
                  coverage_pct, recommendation: str,
                  settings: dict, result: dict) -> dict:
    """Insert a snapshot. Dedup: identical (course_code + settings + result)
    returns the existing row instead of inserting a duplicate."""
    _ensure_table()
    payload = json.dumps([course_code, settings, result], sort_keys=True)
    result_hash = hashlib.sha1(payload.encode()).hexdigest()

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, created_at FROM {TABLE} WHERE result_hash = %s LIMIT 1",
            (result_hash,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "created_at": str(row[1]), "duplicate": True}

        cursor.execute(f"""
            INSERT INTO {TABLE}
              (course_code, course_title, domain, coverage_pct, recommendation,
               settings, result, result_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
        """, (course_code, course_title, domain, coverage_pct, recommendation,
              json.dumps(settings), json.dumps(result), result_hash))
        new_id, created_at = cursor.fetchone()
        conn.commit()
        return {"id": new_id, "created_at": str(created_at), "duplicate": False}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def list_analyses(course_code: str | None = None, limit: int = 50) -> list[dict]:
    """Summaries, newest first — enough to render the saved-analyses list."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = f"""
            SELECT id, course_code, course_title, domain, coverage_pct,
                   recommendation, settings, created_at
            FROM {TABLE}
        """
        params: tuple = ()
        if course_code:
            sql += " WHERE course_code = %s"
            params = (course_code,)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params += (limit,)
        cursor.execute(sql, params)
        return [{"id": r[0], "course_code": r[1], "course_title": r[2],
                 "domain": r[3],
                 "coverage_pct": float(r[4]) if r[4] is not None else None,
                 "recommendation": r[5], "settings": r[6],
                 "created_at": str(r[7])}
                for r in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_analysis(analysis_id: int) -> dict | None:
    """Full snapshot (settings + result JSON) for replay."""
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, course_code, course_title, domain, coverage_pct,
                   recommendation, settings, result, created_at
            FROM {TABLE} WHERE id = %s
        """, (analysis_id,))
        r = cursor.fetchone()
        if not r:
            return None
        return {"id": r[0], "course_code": r[1], "course_title": r[2],
                "domain": r[3],
                "coverage_pct": float(r[4]) if r[4] is not None else None,
                "recommendation": r[5], "settings": r[6], "result": r[7],
                "created_at": str(r[8])}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_analysis(analysis_id: int) -> bool:
    _ensure_table()
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE} WHERE id = %s", (analysis_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
