#!/usr/bin/env python3
"""Local vs OpenRouter Llama-3.1-8B *classification* bake-off (Stage 1).

Samples N job pages per site from the consolidated crawl corpus
(phase2_crawling/outputs/2026-07-15), cleans each with Stage 1's own
``clean_html_trafilatura``, and runs Stage 1's *single_job_detail vs non_job*
prompt through two engines:

  - local : Ollama ``llama3.1:8b``                 (http://localhost:11434)
  - api   : OpenRouter ``meta-llama/llama-3.1-8b-instruct`` (same weights)

Both engines see the identical pages + prompt, so the comparison isolates the
host (local vs API), not the model. Reports per engine: speed, tokens, $ cost,
and the single/non-job split — plus local↔api label agreement (and every
disagreement listed for eyeballing), since there's no gold label here.

Secrets from backend/.env (OPENROUTER_API_KEY), never hardcoded.

Run with the thesis env python:
  /opt/miniconda3/envs/thesis/bin/python classify_bakeoff.py --n 10
  python classify_bakeoff.py --n 10 --engines local     # local only (no key)
  python classify_bakeoff.py --n 2  --engines api        # quick api smoke
"""
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # phase3 root (script now lives one level down)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this folder — sibling stage module
from stage1_classify import (          # noqa: E402
    clean_html_trafilatura, build_prompt, safe_parse_json,
)

CORPUS = HERE.parent / "phase2_crawling" / "outputs" / "2026-07-15"
ENV_FILE = HERE.parent / ".env"                         # single source: backend/.env

LOCAL_MODEL = "llama3.1:8b"
API_MODEL_DEFAULT = "meta-llama/llama-3.1-8b-instruct"
API_BASE_DEFAULT = "https://openrouter.ai/api/v1"
PRICE_IN, PRICE_OUT = 0.02 / 1e6, 0.03 / 1e6            # OpenRouter llama-3.1-8b
CORPUS_TOTAL = 6607


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


def sample_pages(n: int) -> list:
    """(site, filename, html) for n random pages per site — fixed seed."""
    rng = random.Random(42)
    out = []
    for site_dir in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        idx_path = site_dir / "index.json"
        if not idx_path.exists():
            continue
        files = sorted(json.loads(idx_path.read_text()).keys())
        pick = files if len(files) <= n else rng.sample(files, n)
        for fn in pick:
            html = (site_dir / fn).read_text(encoding="utf-8", errors="ignore")
            out.append((site_dir.name, fn, html))
    return out


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _parse_verdict(text: str) -> bool:
    text = _THINK.sub("", text).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    return bool(safe_parse_json(text).get("is_single_job_detail", False))


def run_local(prompt: str) -> dict:
    import requests
    t = time.time()
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": LOCAL_MODEL, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0}, "think": False,
    }, timeout=120)
    r.raise_for_status()
    d = r.json()
    return {"is_job": _parse_verdict(d.get("response", "")),
            "seconds": time.time() - t,
            "prompt_tokens": d.get("prompt_eval_count"),
            "completion_tokens": d.get("eval_count"), "cost": 0.0}


def run_api(prompt: str, model: str, base: str, key: str, retries: int = 3) -> dict:
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
    dt = time.time() - t
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cost = usage.get("cost")
    if cost is None and pt is not None:
        cost = pt * PRICE_IN + ct * PRICE_OUT
    return {"is_job": _parse_verdict(msg), "seconds": dt,
            "prompt_tokens": pt, "completion_tokens": ct, "cost": cost or 0.0}


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Local vs API Llama-3.1-8B classification bake-off.")
    ap.add_argument("--n", type=int, default=10, help="pages per site (default 10)")
    ap.add_argument("--engines", default="local,api", help="comma list: local,api")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    load_env()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_model = os.getenv("CLASSIFY_API_MODEL", API_MODEL_DEFAULT)
    api_base = os.getenv("API_BASE_URL", API_BASE_DEFAULT)
    if "api" in engines and not api_key:
        sys.exit(f"[classify] OPENROUTER_API_KEY not set in {ENV_FILE}")

    pages = sample_pages(args.n)
    run_dir = HERE / "outputs" / "classify_bakeoff" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    print(f"[classify] {len(pages)} pages ({args.n}/site) | engines: {engines}")
    print(f"[classify] model: local={LOCAL_MODEL}  api={api_model}")
    print(f"[classify] out: {results_path}\n")

    records = []
    stats = {e: {"ok": 0, "err": 0, "job": 0, "nonjob": 0, "sec": [],
                 "in": [], "out": [], "cost": 0.0} for e in engines}
    agree = both = 0
    disagreements = []

    for i, (site, fn, html) in enumerate(pages, 1):
        cleaned = clean_html_trafilatura(html)
        rec = {"site": site, "file": fn, "chars": len(cleaned), "engines": {}}
        if not cleaned.strip():                          # Stage 1 skips empty (JS-only) pages
            rec["note"] = "no_content_after_clean"
        prompt = build_prompt(cleaned)
        for e in engines:
            if not cleaned.strip():
                rec["engines"][e] = {"is_job": False, "note": "no_content"}
                stats[e]["nonjob"] += 1
                stats[e]["ok"] += 1
                continue
            try:
                out = run_local(prompt) if e == "local" else run_api(prompt, api_model, api_base, api_key)
                rec["engines"][e] = {"is_job": out["is_job"], "seconds": round(out["seconds"], 2),
                                     "prompt_tokens": out["prompt_tokens"],
                                     "completion_tokens": out["completion_tokens"], "cost": out["cost"]}
                s = stats[e]
                s["ok"] += 1
                s["job" if out["is_job"] else "nonjob"] += 1
                s["sec"].append(out["seconds"]); s["in"].append(out["prompt_tokens"])
                s["out"].append(out["completion_tokens"]); s["cost"] += out["cost"]
            except Exception as ex:                        # noqa: keep going
                stats[e]["err"] += 1
                rec["engines"][e] = {"error": str(ex)[:200]}

        le, ae = rec["engines"].get("local", {}), rec["engines"].get("api", {})
        if "is_job" in le and "is_job" in ae:
            both += 1
            if le["is_job"] == ae["is_job"]:
                agree += 1
            else:
                disagreements.append((site, fn, le["is_job"], ae["is_job"]))
        records.append(rec)
        results_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        tag = " | ".join(f"{e}:{'JOB' if rec['engines'][e].get('is_job') else ('non' if 'is_job' in rec['engines'][e] else 'ERR')}"
                         for e in engines)
        print(f"[{i}/{len(pages)}] {site}/{fn} — {tag}")

    # ---- summary ----
    L = ["", "=" * 66, f"CLASSIFY BAKE-OFF  ({len(pages)} pages, {args.n}/site)", "=" * 66]
    for e in engines:
        s = stats[e]
        tin, tout = sum(x for x in s["in"] if x), sum(x for x in s["out"] if x)
        proj = (s["cost"] / s["ok"] * CORPUS_TOTAL) if s["ok"] else 0.0
        L += [f"\n[{e}]  ok={s['ok']}  err={s['err']}",
              f"   single_job_detail : {s['job']}   non_job : {s['nonjob']}",
              f"   avg latency       : {_avg(s['sec']):.2f}s",
              (f"   tokens/cost       : in={tin:,} out={tout:,}  ${s['cost']:.4f}  "
               f"(full corpus ~${proj:.2f})") if e == "api" else "   cost              : $0 (local)"]
    if both:
        L += ["", f"local↔api agreement: {agree}/{both} = {agree/both*100:.1f}%"]
        if disagreements:
            L += ["disagreements (site, file, local, api):"]
            for site, fn, lj, aj in disagreements:
                L += [f"   {site}/{fn}: local={'JOB' if lj else 'non'}  api={'JOB' if aj else 'non'}"]
    L += ["", f"full results: {results_path}", "=" * 66]
    summary = "\n".join(L)
    print(summary)
    (run_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
