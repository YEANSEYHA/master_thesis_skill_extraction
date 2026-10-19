from pathlib import Path
import json
import os
from datetime import datetime
from urllib.parse import urlparse
from config import ACTIVE_MODEL, R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET_NAME

BASE_OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _site_dir(run_id: str, start_url: str) -> Path:
    domain = urlparse(start_url).netloc
    return BASE_OUTPUT / run_id / domain


# --------------------------------------------------
# Index (filename -> URL mapping)
# --------------------------------------------------
def _load_index(site_dir: Path) -> dict:
    index_path = site_dir / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(site_dir: Path, index: dict):
    with open(site_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Public API
# --------------------------------------------------
def save_html(index: int, page_url: str, html: str, start_url: str, run_id: str):
    """
    Save HTML and update index.json:
    Output/<run_id>/<domain>/page_<n>.html
    Output/<run_id>/<domain>/index.json
    """
    site = _site_dir(run_id, start_url)
    site.mkdir(parents=True, exist_ok=True)

    filename = f"page_{index}.html"
    file = site / filename
    file.write_text(html, encoding="utf-8")

    # Update index.json
    page_index = _load_index(site)
    page_index[filename] = page_url
    _save_index(site, page_index)


def save_metadata(meta: dict, run_id: str):
    """
    Save metadata to:
    Output/<run_id>/<domain>/metadata.json
    """
    start_url = meta.get("start_url")
    if not start_url:
        raise ValueError("metadata must contain 'start_url'")

    site = _site_dir(run_id, start_url)
    site.mkdir(parents=True, exist_ok=True)

    meta.update({
        "model": ACTIVE_MODEL,
        "saved_at": datetime.now().isoformat()
    })

    with open(site / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------
# Cloudflare R2 Upload
# --------------------------------------------------
def upload_run_to_r2(run_id: str):
    """
    Upload entire run folder to Cloudflare R2.
    Mirrors local structure: outputs/<run_id>/... → R2: <run_id>/...
    """
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]):
        print("[R2] Skipping upload — R2 credentials not configured")
        return False

    try:
        import boto3
    except ImportError:
        print("[R2] Skipping upload — boto3 not installed (pip install boto3)")
        return False

    run_dir = BASE_OUTPUT / run_id
    if not run_dir.exists():
        print(f"[R2] Run folder not found: {run_dir}")
        return False

    # Connect to R2
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )

    # Walk the run folder and upload every file
    uploaded = 0
    failed = 0

    for file_path in run_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # R2 key mirrors local path: <run_id>/<domain>/page_0.html
        r2_key = str(file_path.relative_to(BASE_OUTPUT))

        # Set content type
        if file_path.suffix == ".html":
            content_type = "text/html"
        elif file_path.suffix == ".json":
            content_type = "application/json"
        elif file_path.suffix == ".txt":
            content_type = "text/plain"
        else:
            content_type = "application/octet-stream"

        try:
            s3.upload_file(
                str(file_path),
                R2_BUCKET_NAME,
                r2_key,
                ExtraArgs={"ContentType": content_type}
            )
            uploaded += 1
        except Exception as e:
            print(f"[R2] Failed to upload {r2_key}: {e}")
            failed += 1

    print(f"[R2] Upload complete: {uploaded} files uploaded, {failed} failed")
    return failed == 0
