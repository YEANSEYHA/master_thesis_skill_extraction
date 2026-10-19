"""
Crawl History DB — tracks which URLs have been crawled to avoid re-crawling.

Table: crawl_history
  - page_url (UNIQUE) — the URL that was crawled
  - site_domain — e.g. "bongthom.com"
  - crawled_at — timestamp
  - status — "success" or "failed"
  - run_id — which crawl run produced this
"""

import psycopg2
from psycopg2.extras import execute_values
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME


def get_connection():
    """Create PostgreSQL connection to Neon."""
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        sslmode="require"
    )


def create_crawl_history_table():
    """Create crawl_history table if not exists."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawl_history (
                id SERIAL PRIMARY KEY,
                page_url TEXT UNIQUE NOT NULL,
                site_domain TEXT NOT NULL,
                crawled_at TIMESTAMP DEFAULT NOW(),
                status TEXT NOT NULL DEFAULT 'success',
                run_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_crawl_history_domain
                ON crawl_history(site_domain);
            CREATE INDEX IF NOT EXISTS idx_crawl_history_run_id
                ON crawl_history(run_id);
        """)
        conn.commit()
        print("[DB] crawl_history table ready")
    except Exception as e:
        print(f"[DB] Error creating table: {e}")
    finally:
        if conn:
            conn.close()


def load_crawled_urls_for_domain(domain):
    """
    Load ALL previously crawled URLs for a domain into memory.
    Called once at crawl start to avoid repeated DB queries.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT page_url FROM crawl_history WHERE site_domain = %s AND status = 'success'",
            (domain,)
        )
        urls = {row[0] for row in cursor.fetchall()}
        print(f"[HISTORY] Loaded {len(urls)} previously-crawled URLs for {domain}")
        return urls
    except Exception as e:
        print(f"[DB] Error loading crawl history: {e}")
        return set()
    finally:
        if conn:
            conn.close()


def check_urls_crawled(urls):
    """
    Given a list of URLs, return the set that already exist
    in crawl_history with status='success'.
    """
    if not urls:
        return set()

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT page_url FROM crawl_history WHERE page_url = ANY(%s) AND status = 'success'",
            (list(urls),)
        )
        return {row[0] for row in cursor.fetchall()}
    except Exception as e:
        print(f"[DB] Error checking crawl history: {e}")
        return set()
    finally:
        if conn:
            conn.close()


def mark_urls_crawled(urls, domain, run_id, status="success"):
    """
    Insert crawled URLs into crawl_history.
    On conflict (re-crawl), update timestamp and status.
    """
    if not urls:
        return

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        values = [(url, domain, status, run_id) for url in urls]
        execute_values(
            cursor,
            """
            INSERT INTO crawl_history (page_url, site_domain, status, run_id)
            VALUES %s
            ON CONFLICT (page_url) DO UPDATE SET
                crawled_at = NOW(),
                status = EXCLUDED.status,
                run_id = EXCLUDED.run_id
            """,
            values
        )
        conn.commit()
    except Exception as e:
        print(f"[DB] Error marking URLs crawled: {e}")
    finally:
        if conn:
            conn.close()


def get_crawl_stats(domain=None):
    """Get crawl history stats, optionally filtered by domain."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        if domain:
            cursor.execute(
                "SELECT COUNT(*), MIN(crawled_at), MAX(crawled_at) FROM crawl_history WHERE site_domain = %s AND status = 'success'",
                (domain,)
            )
        else:
            cursor.execute(
                "SELECT COUNT(*), MIN(crawled_at), MAX(crawled_at) FROM crawl_history WHERE status = 'success'"
            )
        row = cursor.fetchone()
        return {"total_urls": row[0], "first_crawl": row[1], "last_crawl": row[2]}
    except Exception as e:
        print(f"[DB] Error getting stats: {e}")
        return {"total_urls": 0}
    finally:
        if conn:
            conn.close()
