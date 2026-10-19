#!/usr/bin/env python3
"""Upload crawled raw HTML to Cloudflare R2 (bucket ``thesis-raw-htmls``) as
archival evidence / cold storage for the thesis.

R2 is S3-compatible. Credentials are read from the phase-2 autonomous crawler's
``.env`` (``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY`` / ``R2_SECRET_KEY``) — the same
ones ``crawler/storage.py`` already uses — never hardcoded.

    python upload_to_r2.py                    # uploads outputs/2026-07-15/
    python upload_to_r2.py outputs/2026-07-15  # explicit folder
    python upload_to_r2.py --dry-run           # list what would upload, no creds needed to connect
    python upload_to_r2.py --workers 32        # more parallelism

R2 keys mirror the local tree: ``2026-07-15/<site>/page_N.html`` (sits next to
the existing ``2026-03-10_09-46-27/`` folder in the bucket).

Re-runnable: objects already in the bucket with a matching size are skipped, so
an interrupted upload resumes where it stopped (``--no-resume`` to force).
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "outputs" / "2026-07-15"
# single source of truth for all phases (R2 creds live here)
ENV_FILE = HERE.parent / ".env"

_CONTENT_TYPE = {"html": "text/html", "json": "application/json",
                 "txt": "text/plain"}


def _load_env(path: Path) -> None:
    """Populate os.environ from a KEY=VAL / ``export KEY="VAL"`` .env file
    without overriding variables already set in the real environment."""
    try:
        from dotenv import load_dotenv          # noqa: handles most formats
        load_dotenv(path)
    except Exception:
        pass
    if not path.exists():
        return
    for line in path.read_text().splitlines():   # also covers `export` + quotes
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _existing_objects(s3, bucket: str, prefix: str) -> dict:
    """{key: size} already in the bucket under ``prefix`` — used to resume."""
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            out[o["Key"]] = o["Size"]
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a crawl output folder to Cloudflare R2.")
    ap.add_argument("src", nargs="?", default=str(DEFAULT_SRC),
                    help="folder to upload (default: outputs/2026-07-15)")
    ap.add_argument("--prefix", default=None,
                    help="R2 key prefix (default: the source folder's name)")
    ap.add_argument("--workers", type=int, default=24, help="parallel uploads")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would upload, then exit")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-upload every file even if already present")
    args = ap.parse_args()

    _load_env(ENV_FILE)
    _load_env(HERE / ".env")                      # optional local override

    acct = os.getenv("R2_ACCOUNT_ID", "")
    ak = os.getenv("R2_ACCESS_KEY", "")
    sk = os.getenv("R2_SECRET_KEY", "")
    bucket = os.getenv("R2_BUCKET_NAME", "thesis-raw-htmls")

    src = Path(args.src).resolve()
    if not src.is_dir():
        sys.exit(f"[R2] source folder not found: {src}")
    prefix = (args.prefix or src.name).strip("/")

    files = [p for p in src.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"[R2] {len(files)} files ({total_mb:.1f} MB) "
          f"-> s3://{bucket}/{prefix}/")

    if args.dry_run:
        for p in files[:10]:
            print("   ", f"{prefix}/{p.relative_to(src).as_posix()}")
        if len(files) > 10:
            print(f"    … +{len(files) - 10} more")
        creds_ok = all([acct, ak, sk])
        print(f"[R2] credentials loaded: {'yes' if creds_ok else 'NO — check ' + str(ENV_FILE)}"
              + (f"  (account {acct[:6]}…)" if creds_ok else ""))
        print("[R2] dry-run — nothing uploaded")
        return

    if not all([acct, ak, sk]):
        sys.exit("[R2] missing R2_ACCOUNT_ID / R2_ACCESS_KEY / R2_SECRET_KEY "
                 f"(looked in {ENV_FILE})")

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit("[R2] boto3 not installed — run: pip install boto3")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"},
                      max_pool_connections=args.workers + 4),
    )

    have = {} if args.no_resume else _existing_objects(s3, bucket, prefix + "/")
    if have:
        print(f"[R2] {len(have)} objects already under {prefix}/ — matching sizes will be skipped")

    lock = threading.Lock()
    stat = {"up": 0, "skip": 0, "fail": 0}

    def put(p: Path) -> None:
        key = f"{prefix}/{p.relative_to(src).as_posix()}"
        if have.get(key) == p.stat().st_size:
            with lock:
                stat["skip"] += 1
            return
        ctype = _CONTENT_TYPE.get(p.suffix.lstrip("."), "application/octet-stream")
        try:
            s3.upload_file(str(p), bucket, key, ExtraArgs={"ContentType": ctype})
        except Exception as e:                    # noqa: report and keep going
            with lock:
                stat["fail"] += 1
            print(f"[R2] FAIL {key}: {e}")
            return
        with lock:
            stat["up"] += 1
            n = stat["up"] + stat["skip"]
        if n % 250 == 0:
            print(f"[R2] {n}/{len(files)}  (uploaded={stat['up']} skipped={stat['skip']})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(put, files))

    print(f"[R2] DONE — uploaded {stat['up']}, skipped {stat['skip']}, failed {stat['fail']}")
    print(f"[R2] bucket path: s3://{bucket}/{prefix}/")
    if stat["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
