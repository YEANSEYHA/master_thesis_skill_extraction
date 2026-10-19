import json
import psycopg2
from psycopg2.extras import execute_values

from config import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME, DB_SSLMODE, DB_TABLE


def get_connection():
    """Create PostgreSQL connection."""
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        sslmode=DB_SSLMODE
    )


def create_job_sites_table(table_name: str = DB_TABLE):
    """Create job_sites table if not exists."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            name TEXT,
            country TEXT,
            site_type TEXT,
            description TEXT,
            confidence REAL,
            reason TEXT,
            features TEXT,
            is_verified BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """

        cursor.execute(create_table_sql)

        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_country
            ON {table_name}(country);
        """)

        conn.commit()
        print(f"✔ Table `{table_name}` ready.")

    except Exception as e:
        print(f"❌ Error creating table: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_job_sites(
    sites: list,
    country: str,
    table_name: str = DB_TABLE
) -> int:
    """
    Insert verified job sites into PostgreSQL.

    Args:
        sites: List of verified site dicts with keys:
               - url, name, type, description, confidence, reason, features_found
        country: Country name (e.g., "Cambodia")
        table_name: Database table name

    Returns:
        Number of rows inserted
    """
    if not sites:
        print("⚠️ No sites to insert.")
        return 0

    conn = None
    cursor = None

    try:
        print(f"\n🔌 Connecting to PostgreSQL...")
        conn = get_connection()
        cursor = conn.cursor()
        print("✔ Connected to PostgreSQL")

        rows = []
        for site in sites:
            features = site.get("features_found", [])
            if isinstance(features, list):
                features = json.dumps(features)

            rows.append((
                site.get("url"),
                site.get("name"),
                country,
                site.get("type") or site.get("site_type"),
                site.get("description"),
                float(site.get("confidence", 0.0)),
                site.get("reason"),
                features,
                True  # is_verified
            ))

        insert_sql = f"""
        INSERT INTO {table_name}
            (url, name, country, site_type, description, confidence, reason, features, is_verified)
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            name = EXCLUDED.name,
            country = EXCLUDED.country,
            site_type = EXCLUDED.site_type,
            description = EXCLUDED.description,
            confidence = EXCLUDED.confidence,
            reason = EXCLUDED.reason,
            features = EXCLUDED.features,
            is_verified = EXCLUDED.is_verified,
            updated_at = NOW();
        """

        execute_values(cursor, insert_sql, rows)
        conn.commit()

        print(f"🎉 Successfully inserted/updated {len(rows)} job sites for {country}!")
        return len(rows)

    except Exception as e:
        print(f"❌ PostgreSQL error: {e}")
        return 0

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("🔌 PostgreSQL connection closed.")


def insert_single_job_site(
    url: str,
    name: str,
    country: str,
    site_type: str = None,
    description: str = None,
    confidence: float = 0.0,
    reason: str = None,
    features: list = None,
    table_name: str = DB_TABLE
) -> bool:
    """Insert a single job site into PostgreSQL."""
    site = {
        "url": url,
        "name": name,
        "type": site_type,
        "description": description,
        "confidence": confidence,
        "reason": reason,
        "features_found": features or []
    }

    result = insert_job_sites([site], country, table_name)
    return result > 0


def get_job_sites_by_country(country: str, table_name: str = DB_TABLE) -> list:
    """Get all job sites for a specific country."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, url, name, country, site_type, description,
                   confidence, reason, features, is_verified, created_at
            FROM {table_name}
            WHERE country = %s
            ORDER BY confidence DESC;
        """, (country,))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        print(f"❌ Error fetching sites: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_all_job_sites(table_name: str = DB_TABLE) -> list:
    """Get all job sites from database."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, url, name, country, site_type, description,
                   confidence, reason, features, is_verified, created_at
            FROM {table_name}
            ORDER BY country, confidence DESC;
        """)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        print(f"❌ Error fetching sites: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_countries_summary(table_name: str = DB_TABLE) -> list:
    """Get summary of job sites by country."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT country, COUNT(*) as site_count, AVG(confidence) as avg_confidence
            FROM {table_name}
            GROUP BY country
            ORDER BY site_count DESC;
        """)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        print(f"❌ Error fetching summary: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_job_site(url: str, table_name: str = DB_TABLE) -> bool:
    """Delete a job site by URL."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {table_name} WHERE url = %s;", (url,))
        conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            print(f"✔ Deleted: {url}")
        else:
            print(f"⚠️ URL not found: {url}")

        return deleted

    except Exception as e:
        print(f"❌ Error deleting site: {e}")
        return False

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_all_by_country(country: str, table_name: str = DB_TABLE) -> int:
    """Delete all job sites for a country."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {table_name} WHERE country = %s;", (country,))
        conn.commit()

        deleted_count = cursor.rowcount
        print(f"✔ Deleted {deleted_count} sites for {country}")

        return deleted_count

    except Exception as e:
        print(f"❌ Error deleting sites: {e}")
        return 0

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
