"""Cloudflare R2 archive for uploaded syllabus files (S3-compatible).

Same credentials and client shape the crawler already uses
(`archive/phase2_playwright_crawler/crawler/storage.py`, `phase2_crawling/upload_to_r2.py`):
``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY`` / ``R2_SECRET_KEY`` out of ``backend/.env``,
which ``config.py`` already loads for the whole API process.

Objects live under the ``syllabi/`` prefix of ``thesis-raw-htmls`` — that API
token is scoped to this one bucket (ListBuckets is denied and a separate
``thesis-syllabi`` 403s), so the prefix is what keeps course specs apart from
the crawled HTML. Point ``R2_BUCKET_NAME_SYLLABI`` at a dedicated bucket if one
is ever created with a token that can reach it.

**R2 is never on the critical path.** Every function here degrades to
False/None when boto3 or the credentials are missing, or when the call fails —
Postgres holds the Markdown the pipeline actually consumes, and R2 only holds
the original file for provenance. An upload must still succeed with the archive
down.
"""

import os
import sys
import threading

BUCKET = os.getenv("R2_BUCKET_NAME_SYLLABI",
                   os.getenv("R2_BUCKET_NAME", "thesis-raw-htmls"))

_CONTENT_TYPE = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
}

_client = None
_client_lock = threading.Lock()   # FastAPI serves uploads on a thread pool


def configured() -> bool:
    return all([os.getenv("R2_ACCOUNT_ID"), os.getenv("R2_ACCESS_KEY"),
                os.getenv("R2_SECRET_KEY")])


def _get_client():
    """Cached boto3 S3 client pointed at R2, or None if unusable."""
    global _client
    if _client is not None:
        return _client
    if not configured():
        return None
    with _client_lock:
        if _client is not None:
            return _client
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            print("[R2] boto3 not installed — archiving disabled "
                  "(pip install boto3)", file=sys.stderr)
            return None
        _client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
            region_name="auto",
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )
        return _client


def key_for(sha256: str, ext: str) -> str:
    """Content-addressed key — same bytes always land on the same object."""
    return f"syllabi/{sha256}.{ext.lower().lstrip('.')}"


def put_bytes(key: str, data: bytes, ext: str) -> bool:
    """Archive one file. Returns False (never raises) if R2 is unavailable."""
    s3 = _get_client()
    if s3 is None:
        return False
    try:
        s3.put_object(
            Bucket=BUCKET, Key=key, Body=data,
            ContentType=_CONTENT_TYPE.get(ext.lower().lstrip("."),
                                          "application/octet-stream"),
        )
        return True
    except Exception as e:                       # noqa: BLE001 — archive is best-effort
        print(f"[R2] put failed for {key}: {e}", file=sys.stderr)
        return False


def presign(key: str, ttl: int = 3600) -> str | None:
    """Time-limited download URL. R2 buckets are private, so a stored
    https://<acct>.r2.cloudflarestorage.com/... link would 401 — the URL has to
    be signed at request time."""
    s3 = _get_client()
    if s3 is None or not key:
        return None
    try:
        return s3.generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)
    except Exception as e:                       # noqa: BLE001
        print(f"[R2] presign failed for {key}: {e}", file=sys.stderr)
        return None
