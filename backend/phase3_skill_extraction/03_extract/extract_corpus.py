#!/usr/bin/env python3
"""Full-corpus Stage-2 skill extraction via OpenRouter — concurrent + resumable.

Reads the Stage-1 job-detail corpus (outputs/classify/, ~4,884 pages across 8
Cambodian sites), cleans and prompts each page with the *production* Stage-2
pipeline, and extracts skills with the API model (qwen/qwen3-14b, thinking OFF)
concurrently. Everything except the host (local Ollama -> OpenRouter API) and the
concurrency is identical to 03_extract/stage2_skill_extraction.py:

    - clean_html_trafilatura   (imported, unchanged)
    - build_prompt             (imported, unchanged)
    - extract_url_from_html    (imported, unchanged)
    - output record schema     ({"input": cleaned, "output": {...9 fields...,
                                  "source_file", "full_url"}})  -> stage2_result.json

Why threads, not OS subprocesses: API extraction is network-bound, so a thread
pool of concurrent HTTP requests saturates the API far better than spawning
processes, and it keeps ONE coherent stage2_result.json (no shard-merge step).
Bump --workers for more throughput.

Resumable: rerun after a crash / Ctrl-C and it skips every page already present in
stage2_result.json (keyed by "<site>/<file>"); failed pages simply aren't written,
so a rerun retries them automatically.

Secrets from backend/.env (OPENROUTER_API_KEY) — never hardcoded.

Run with the thesis env python, from this directory:
  /opt/miniconda3/envs/thesis/bin/python extract_corpus.py                 # full run
  python extract_corpus.py --limit 20                                       # quick trial
  python extract_corpus.py --sites camhr.com,jobnet.com.kh                  # subset
  python extract_corpus.py --workers 32                                     # more concurrency
  python extract_corpus.py --fresh                                          # ignore prior output
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # phase3 root (script now lives one level down)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this folder — sibling stage module
from stage2_skill_extraction import (          # noqa: E402
    clean_html_trafilatura, build_prompt, extract_url_from_html,
)

CORPUS = HERE / "outputs" / "classify"                          # Stage-1 job-detail pages
ENV_FILE = HERE.parent / ".env"                                 # single source: backend/.env
OUT_DIR = HERE / "outputs" / "stage2" / "api_full"             # fixed dir so reruns resume
OUT_FILE = OUT_DIR / "stage2_result.json"

API_MODEL_DEFAULT = "qwen/qwen3-14b"
API_BASE_DEFAULT = "https://openrouter.ai/api/v1"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.24 / 1e6                    # OpenRouter qwen/qwen3-14b, $/token
DEFAULT_WORKERS = 24
FLUSH_EVERY = 25                                               # save the array every N completions

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


def iter_pages(sites: list | None) -> list:
    """(site, filename, raw_html, url) for every page in the classify corpus."""
    out = []
    for site_dir in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        if sites and site_dir.name not in sites:
            continue
        idx = {}
        idx_path = site_dir / "index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
        for fn in sorted(idx.keys()) if idx else sorted(p.name for p in site_dir.glob("page_*.html")):
            html = (site_dir / fn).read_text(encoding="utf-8", errors="ignore")
            out.append((site_dir.name, fn, html, idx.get(fn)))
    return out


def parse_json(text: str) -> dict:
    """Best-effort JSON out of an LLM message (strips think tags / fences)."""
    text = _THINK.sub("", text).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1:
        text = text[a:b + 1]
    return json.loads(text)


def run_api(prompt: str, model: str, base: str, key: str, retries: int = 4) -> dict:
    import requests
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a JSON data extraction tool. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},         # qwen3 hybrid: thinking OFF
        "provider": {"sort": "throughput"},      # route to the fastest host
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost", "X-Title": "skill-extraction-thesis"}
    url = f"{base}/chat/completions"
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=300)
            if r.status_code == 400:            # provider rejects optional knobs — retry bare
                for k in ("response_format", "reasoning", "provider", "usage"):
                    body.pop(k, None)
                r = requests.post(url, headers=headers, json=body, timeout=300)
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
        cost = pt * PRICE_IN + (ct or 0) * PRICE_OUT
    return {"parsed": parse_json(msg), "prompt_tokens": pt,
            "completion_tokens": ct, "cost": cost or 0.0}


def extract_one(page, model, base, key) -> dict:
    """Run the production Stage-2 pipeline on one page -> stage2_result record."""
    site, fn, html, url = page
    cleaned = clean_html_trafilatura(html)                      # production cleaning, unchanged
    prompt = build_prompt(cleaned)                             # production prompt, unchanged
    out = run_api(prompt, model, base, key)
    parsed = out["parsed"]
    parsed["source_file"] = f"{site}/{fn}"                     # site-qualified (filenames collide)
    parsed["full_url"] = url or extract_url_from_html(html)    # prefer index.json url
    return {"input": cleaned, "output": parsed,
            "_toks_in": out["prompt_tokens"], "_toks_out": out["completion_tokens"],
            "_cost": out["cost"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrent, resumable API Stage-2 extraction.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"concurrent requests (default {DEFAULT_WORKERS})")
    ap.add_argument("--limit", type=int, default=None, help="cap total pages (trial runs)")
    ap.add_argument("--sites", default="", help="comma list of site folders (default all)")
    ap.add_argument("--out", default=str(OUT_FILE), help="output stage2_result.json path")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing output (no resume)")
    args = ap.parse_args()

    load_env()
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit(f"[extract] OPENROUTER_API_KEY not set in {ENV_FILE}")
    model = os.getenv("API_MODEL", API_MODEL_DEFAULT)
    base = os.getenv("API_BASE_URL", API_BASE_DEFAULT)
    if not CORPUS.exists():
        sys.exit(f"[extract] corpus not found: {CORPUS}")

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()] or None

    # ---- resume: load prior results, skip pages already done ----
    results, done = [], set()
    if out_file.exists() and not args.fresh:
        try:
            results = json.loads(out_file.read_text())
            done = {r["output"]["source_file"] for r in results if r.get("output")}
        except Exception:
            results, done = [], set()

    pages = iter_pages(sites)
    todo = [p for p in pages if f"{p[0]}/{p[1]}" not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[extract] corpus={CORPUS.name}  model={model}  workers={args.workers}")
    print(f"[extract] total={len(pages)}  already_done={len(done)}  to_extract={len(todo)}")
    print(f"[extract] out={out_file}\n")
    if not todo:
        print("[extract] nothing to do — all pages already extracted.")
        return

    lock = threading.Lock()
    t0 = time.time()
    ok = err = 0
    tin = tout = 0
    cost = 0.0

    def flush():
        tmp = out_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        tmp.replace(out_file)                                  # atomic save

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, p, model, base, key): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            site, fn, _, _ = futs[fut]
            try:
                rec = fut.result()
                tin += rec.pop("_toks_in") or 0
                tout += rec.pop("_toks_out") or 0
                cost += rec.pop("_cost") or 0.0
                nsk = len(rec["output"].get("skills") or [])
                with lock:
                    results.append(rec)
                    ok += 1
                    if ok % FLUSH_EVERY == 0:
                        flush()
                tag = f"{nsk}sk"
            except Exception as e:                              # noqa: keep going
                with lock:
                    err += 1
                tag = f"ERR {str(e)[:60]}"
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60 if rate else 0
            print(f"[{i}/{len(todo)}] {site}/{fn} — {tag}  | {rate:.1f} pg/s  ETA {eta:.1f}m")

    flush()
    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE  ok={ok}  err={err}  in {dt/60:.1f} min ({ok/dt if dt else 0:.1f} pg/s)")
    print(f"total in file: {len(results)}  (of {len(pages)} corpus pages)")
    print(f"tokens: in={tin:,} out={tout:,}   cost this run: ${cost:.4f}")
    print(f"output: {out_file}")
    print("=" * 60)
    (out_file.parent / "run_meta.json").write_text(json.dumps({
        "finished": datetime.now().isoformat(timespec="seconds"),
        "model": model, "workers": args.workers,
        "ok": ok, "err": err, "total_in_file": len(results),
        "corpus_pages": len(pages), "tokens_in": tin, "tokens_out": tout,
        "cost_usd": round(cost, 4), "minutes": round(dt / 60, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
