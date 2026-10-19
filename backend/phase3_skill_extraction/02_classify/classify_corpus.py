#!/usr/bin/env python3
"""Classify the full crawl corpus via OpenRouter Llama-3.1-8B (Stage 1).

Reads every job page in
  phase2_crawling/outputs/2026-07-15/<site>/page_N.html
cleans it with Stage 1's cleaner, classifies single_job_detail vs non_job
CONCURRENTLY, and writes a manifest so Stage 2 extracts only the survivors.

  - concurrent (default 24 workers) -> full 6,607 corpus in ~5-10 min
  - resumable: re-running skips pages already in the manifest (--fresh to reset)
  - built-in retry on transient API errors

Outputs:
  outputs/classification_api/                         (manifest + provenance)
    manifest.json   { "<site>/page_N.html": {is_job, url, seconds, in, out, cost} }
    job_detail.txt  paths classified single_job_detail
    non_job.txt     paths classified non_job (filtered out)
    summary.txt
  outputs/classify/<site>/                            (Stage 2 input — point here)
    page_N.html     raw HTML of job-detail pages only
    index.json      per-site {filename: url} subset
  (--no-save-html skips the copy and leaves just the manifest/lists)

Then auto-uploads outputs/classify/ to R2 (bucket thesis-raw-htmls, prefix
'classify') by reusing phase2_crawling/upload_to_r2.py — concurrent + resumable.
(--no-upload skips it.)

Secrets from backend/.env: OPENROUTER_API_KEY + R2_* (never hardcoded).

Usage (thesis env python):
  python classify_corpus.py                          # all sites, all pages
  python classify_corpus.py --sites jobnet.com.kh    # one/few sites
  python classify_corpus.py --workers 30
  python classify_corpus.py --limit 50               # small trial
  python classify_corpus.py --fresh                  # ignore prior manifest
  python classify_corpus.py --no-save-html           # manifest only, no HTML copy
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # phase3 root (script now lives one level down)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this folder — sibling stage module
from stage1_classify import (          # noqa: E402
    clean_html_trafilatura, build_prompt, safe_parse_json,
)

CORPUS = HERE.parent / "phase2_crawling" / "outputs" / "2026-07-15"
ENV_FILE = HERE.parent / ".env"                         # single source: backend/.env
OUT_DIR = HERE / "outputs" / "classification_api"       # manifest + provenance
SAVE_DIR = HERE / "outputs" / "classify"                # job-detail HTML -> Stage 2 input

API_MODEL_DEFAULT = "meta-llama/llama-3.1-8b-instruct"
API_BASE_DEFAULT = "https://openrouter.ai/api/v1"
PRICE_IN, PRICE_OUT = 0.02 / 1e6, 0.03 / 1e6

_THINK = re.compile(r"<think>.*?</think>", re.S)


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _parse_verdict(text: str) -> bool:
    text = _THINK.sub("", text).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    return bool(safe_parse_json(text).get("is_single_job_detail", False))


def classify_api(prompt: str, model: str, base: str, key: str, retries: int = 4) -> dict:
    import requests
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a classifier. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "provider": {"sort": "throughput"},
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost", "X-Title": "skill-extraction-thesis"}
    url = f"{base}/chat/completions"
    t, last = time.time(), None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 400:
                for k in ("response_format", "provider", "usage"):
                    body.pop(k, None)
                r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code in (429, 500, 502, 503, 520, 524):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        except requests.RequestException as ex:
            last = str(ex)
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"api failed after {retries} retries: {last}")
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cost = usage.get("cost")
    if cost is None and pt is not None:
        cost = pt * PRICE_IN + ct * PRICE_OUT
    return {"is_job": _parse_verdict(msg), "seconds": round(time.time() - t, 2),
            "in": pt, "out": ct, "cost": cost or 0.0}


def gather_pages(sites: list | None, limit: int | None) -> list:
    """(key, url, path) for every job page; key = '<site>/page_N.html'."""
    pages = []
    for site_dir in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        if sites and site_dir.name not in sites:
            continue
        idx_path = site_dir / "index.json"
        if not idx_path.exists():
            continue
        for fn, url in json.loads(idx_path.read_text()).items():
            pages.append((f"{site_dir.name}/{fn}", url, site_dir / fn))
    pages.sort(key=lambda t: t[0])
    return pages[:limit] if limit else pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify the full corpus via OpenRouter Llama-3.1-8B.")
    ap.add_argument("--sites", default=None, help="comma list to limit to (default: all)")
    ap.add_argument("--workers", type=int, default=24, help="concurrent API calls")
    ap.add_argument("--limit", type=int, default=None, help="cap total pages (trial)")
    ap.add_argument("--fresh", action="store_true", help="ignore any prior manifest")
    ap.add_argument("--no-save-html", action="store_true",
                    help="skip copying job-detail HTML into outputs/classify/ (manifest only)")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip auto-uploading outputs/classify/ to R2")
    args = ap.parse_args()
    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
    save_html = not args.no_save_html
    upload = not args.no_upload

    load_env()
    key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("CLASSIFY_API_MODEL", API_MODEL_DEFAULT)
    base = os.getenv("API_BASE_URL", API_BASE_DEFAULT)
    if not key:
        sys.exit(f"[classify] OPENROUTER_API_KEY not set in {ENV_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists() and not args.fresh:
        manifest = json.loads(manifest_path.read_text())

    pages = gather_pages(sites, args.limit)
    todo = [p for p in pages if p[0] not in manifest]
    print(f"[classify] corpus={len(pages)} pages | already done={len(pages)-len(todo)} | "
          f"to do={len(todo)} | workers={args.workers}")
    print(f"[classify] model={model}  out={manifest_path}\n")
    if not todo:
        print("[classify] nothing to do — all pages already in manifest.")
        _finalize(manifest, pages, save_html, upload)
        return

    lock = threading.Lock()
    stat = {"done": 0, "err": 0, "job": 0, "nonjob": 0, "cost": 0.0}
    start = time.time()

    def work(item):
        key_, url, path = item
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_html_trafilatura(html)
            if not cleaned.strip():                      # empty/JS-only page -> non_job
                return key_, {"is_job": False, "url": url, "note": "no_content",
                              "seconds": 0.0, "in": 0, "out": 0, "cost": 0.0}
            out = classify_api(build_prompt(cleaned), model, base, key)
            out["url"] = url
            return key_, out
        except Exception as ex:                          # noqa: record error, keep going
            return key_, {"error": str(ex)[:200], "url": url}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for fut in as_completed(futs):
            key_, res = fut.result()
            with lock:
                manifest[key_] = res
                if "error" in res:
                    stat["err"] += 1
                else:
                    stat["job" if res["is_job"] else "nonjob"] += 1
                    stat["cost"] += res.get("cost", 0.0)
                stat["done"] += 1
                n = stat["done"]
                if n % 100 == 0 or n == len(todo):
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
                    rate = n / (time.time() - start)
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"[classify] {n}/{len(todo)}  job={stat['job']} non={stat['nonjob']} "
                          f"err={stat['err']}  ${stat['cost']:.4f}  ETA {eta:.1f}min")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[classify] finished {stat['done']} in {(time.time()-start)/60:.1f} min  "
          f"cost ${stat['cost']:.4f}")
    _finalize(manifest, pages, save_html, upload)


def _materialize(job_keys: list, manifest: dict) -> int:
    """Copy the raw HTML of every job-detail page into SAVE_DIR/<site>/, with a
    fresh per-site index.json — a self-contained corpus Stage 2 points at."""
    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)                          # rebuild clean each run
    per_site_index: dict[str, dict] = {}
    copied = 0
    for key in job_keys:
        site, fn = key.split("/", 1)
        src = CORPUS / site / fn
        if not src.exists():
            continue
        dst_dir = SAVE_DIR / site
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / fn)
        per_site_index.setdefault(site, {})[fn] = manifest.get(key, {}).get("url")
        copied += 1
    for site, idx in per_site_index.items():
        (SAVE_DIR / site / "index.json").write_text(
            json.dumps(idx, ensure_ascii=False, indent=2))
    return copied


def _upload_to_r2(folder: Path) -> None:
    """Upload the classified folder to R2 (prefix 'classify') by reusing the
    phase2 uploader with the same interpreter — concurrent + resumable."""
    uploader = HERE.parent / "phase2_crawling" / "upload_to_r2.py"
    if not uploader.exists():
        print(f"[r2] uploader not found ({uploader}); skipping upload")
        return
    if not os.getenv("R2_ACCOUNT_ID"):
        print("[r2] R2 creds not in backend/.env; skipping upload")
        return
    print(f"[r2] uploading {folder.name}/ -> R2 bucket (prefix '{folder.name}') ...")
    # src folder name is 'classify' -> keys become classify/<site>/page_N.html
    subprocess.run([sys.executable, str(uploader), str(folder)], check=False)


def _finalize(manifest: dict, pages: list, save_html: bool = True,
              upload: bool = True) -> None:
    keys = [k for k, _, _ in pages]
    job = [k for k in keys if manifest.get(k, {}).get("is_job")]
    non = [k for k in keys if k in manifest and not manifest[k].get("is_job")
           and "error" not in manifest[k]]
    err = [k for k in keys if manifest.get(k, {}).get("error")]
    (OUT_DIR / "job_detail.txt").write_text("\n".join(job) + "\n")
    (OUT_DIR / "non_job.txt").write_text("\n".join(non) + "\n")

    copied = _materialize(job, manifest) if save_html else 0
    if save_html and upload and copied:
        _upload_to_r2(SAVE_DIR)

    total_cost = sum(v.get("cost", 0.0) for v in manifest.values())
    summary = (
        f"CLASSIFY CORPUS (API) — {len(keys)} pages\n"
        f"  single_job_detail : {len(job)}\n"
        f"  non_job           : {len(non)}\n"
        f"  errors            : {len(err)}\n"
        f"  total cost        : ${total_cost:.4f}\n"
        + (f"  saved HTML        : {copied} pages -> {SAVE_DIR}\n"
           f"  -> Stage 2 input  : {SAVE_DIR} (per-site index.json included)\n"
           if save_html else
           f"  -> Stage 2 input  : job_detail.txt ({len(job)} pages)\n")
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print("\n" + summary)
    if err:
        print(f"  {len(err)} errors — re-run to retry just those (they stay out of the lists).")


if __name__ == "__main__":
    main()
