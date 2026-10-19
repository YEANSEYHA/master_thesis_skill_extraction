#!/usr/bin/env python3
"""Local vs OpenRouter Qwen3-14B extraction bake-off.

Samples N job pages per site from a chosen source (``--source``):
  classify (default) : the Stage-1 job-detail pages (outputs/classify, 4,884)
  raw                : every crawled page (phase2_crawling/outputs/2026-07-15)
cleans each with the *production* ``clean_html_trafilatura``, and runs the
*production* ``build_prompt`` through two engines:

  - local : Ollama ``qwen3:14b``          (http://localhost:11434)
  - api   : OpenRouter ``qwen/qwen3-14b``  (OpenAI-compatible, thinking OFF)

Both engines see the identical pages and prompt, so the comparison isolates the
host (local vs API), not the model — the same dense 14.8B weights on each side.

Reports per engine: jobs ok, avg skills/job, latency, tokens, and $ cost (+ a
projection to the full corpus); plus local-vs-api skill agreement (Jaccard over
normalized skill tokens) so you can eyeball whether the API matches local.

Secrets come from env / .env (never hardcoded):
  OPENROUTER_API_KEY   required for the `api` engine
  API_MODEL            default: qwen/qwen3-14b
  API_BASE_URL         default: https://openrouter.ai/api/v1

Run with the thesis env python, e.g.:
  /opt/miniconda3/envs/thesis/bin/python bakeoff.py --n 5           # 5/site, both engines, classified pages
  python bakeoff.py --n 5 --engines local       # local only (no key needed)
  python bakeoff.py --n 2 --engines local       # quick smoke test
  python bakeoff.py --n 5 --engines api          # api only
  python bakeoff.py --n 5 --source raw           # sample from all crawled pages instead
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
from stage2_skill_extraction import (          # noqa: E402
    clean_html_trafilatura, build_prompt, ask_llm, extract_url_from_html,
)

RAW_CORPUS = HERE.parent / "phase2_crawling" / "outputs" / "2026-07-15"   # all crawled pages
CLASSIFY_CORPUS = HERE / "outputs" / "classify"                          # Stage-1 job-detail only
# source name -> (corpus dir, full-corpus size for the cost projection)
SOURCES = {"classify": (CLASSIFY_CORPUS, 4884), "raw": (RAW_CORPUS, 6607)}
ENV_FILES = [HERE.parent / ".env"]          # single source of truth: backend/.env

LOCAL_MODEL = "qwen3:14b"
API_MODEL_DEFAULT = "qwen/qwen3-14b"
API_BASE_DEFAULT = "https://openrouter.ai/api/v1"
# OpenRouter list price for qwen/qwen3-14b, $ per token
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.24 / 1e6


def load_env() -> None:
    """Populate os.environ from the project .env files (no overrides)."""
    for f in ENV_FILES:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            line = line.removeprefix("export ").strip()
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sample_pages(n: int, corpus: Path) -> list:
    """(site, filename, html) for n pages per site — fixed seed, reproducible."""
    rng = random.Random(42)
    out = []
    for site_dir in sorted(p for p in corpus.iterdir() if p.is_dir()):
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


def parse_json(text: str) -> dict:
    """Best-effort JSON out of an LLM message (strips think tags / fences)."""
    text = _THINK.sub("", text).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1:
        text = text[a:b + 1]
    return json.loads(text)


def run_local(prompt: str) -> dict:
    t = time.time()
    resp = ask_llm(LOCAL_MODEL, prompt)          # Ollama: temp 0, think off, json
    return {"parsed": json.loads(resp), "seconds": time.time() - t,
            "prompt_tokens": None, "completion_tokens": None, "cost": 0.0}


def run_api(prompt: str, model: str, base: str, key: str, retries: int = 3) -> dict:
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
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost",
               "X-Title": "skill-extraction-thesis"}
    url = f"{base}/chat/completions"
    t = time.time()
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
        except requests.RequestException as ex:  # timeouts, premature-end, conn resets
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
    return {"parsed": parse_json(msg), "seconds": dt,
            "prompt_tokens": pt, "completion_tokens": ct, "cost": cost or 0.0}


_WORD = re.compile(r"[a-z0-9]+")


def _skill_tokens(skills) -> set:
    return {w for s in (skills or []) for w in _WORD.findall(str(s).lower())}


def jaccard(a, b) -> float:
    A, B = _skill_tokens(a), _skill_tokens(b)
    return (len(A & B) / len(A | B)) if (A | B) else 1.0


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Local vs API Qwen3-14B extraction bake-off.")
    ap.add_argument("--n", type=int, default=20, help="pages per site (default 20)")
    ap.add_argument("--engines", default="local,api",
                    help="comma list: local,api (default both)")
    ap.add_argument("--source", choices=list(SOURCES), default="classify",
                    help="classify = Stage-1 job-detail pages (default); raw = all crawled pages")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    corpus, corpus_total = SOURCES[args.source]
    if not corpus.exists():
        sys.exit(f"[bakeoff] source '{args.source}' not found: {corpus}")

    load_env()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_model = os.getenv("API_MODEL", API_MODEL_DEFAULT)
    api_base = os.getenv("API_BASE_URL", API_BASE_DEFAULT)
    if "api" in engines and not api_key:
        sys.exit("[bakeoff] OPENROUTER_API_KEY not set — add it to a .env "
                 "(phase3 or phase2_autonomous) or export it.")

    pages = sample_pages(args.n, corpus)
    run_dir = HERE / "outputs" / "bakeoff" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    print(f"[bakeoff] {len(pages)} pages ({args.n}/site) | source={args.source} "
          f"({corpus_total} full) | engines: {engines}")
    print(f"[bakeoff] out: {results_path}\n")

    records, stats = [], {e: {"ok": 0, "err": 0, "sec": [], "skills": [],
                              "in": [], "out": [], "cost": 0.0} for e in engines}
    agreements = []

    for i, (site, fn, html) in enumerate(pages, 1):
        cleaned = clean_html_trafilatura(html)
        prompt = build_prompt(cleaned)
        rec = {"site": site, "file": fn, "chars": len(cleaned),
               "url": extract_url_from_html(html), "engines": {}}
        for e in engines:
            try:
                out = (run_local(prompt) if e == "local"
                       else run_api(prompt, api_model, api_base, api_key))
                skills = out["parsed"].get("skills", []) or []
                rec["engines"][e] = {
                    "title": out["parsed"].get("job_title"),
                    "skills": skills, "n_skills": len(skills),
                    "seconds": round(out["seconds"], 2),
                    "prompt_tokens": out["prompt_tokens"],
                    "completion_tokens": out["completion_tokens"],
                    "cost": out["cost"],
                }
                s = stats[e]
                s["ok"] += 1
                s["sec"].append(out["seconds"])
                s["skills"].append(len(skills))
                s["in"].append(out["prompt_tokens"])
                s["out"].append(out["completion_tokens"])
                s["cost"] += out["cost"]
            except Exception as ex:                        # noqa: keep going
                stats[e]["err"] += 1
                rec["engines"][e] = {"error": str(ex)[:200]}

        if "local" in rec["engines"] and "api" in rec["engines"] \
                and "skills" in rec["engines"]["local"] and "skills" in rec["engines"]["api"]:
            j = jaccard(rec["engines"]["local"]["skills"], rec["engines"]["api"]["skills"])
            rec["skill_jaccard"] = round(j, 3)
            agreements.append(j)

        records.append(rec)
        results_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        tag = " | ".join(
            f"{e}:{rec['engines'][e].get('n_skills','ERR')}sk/"
            f"{rec['engines'][e].get('seconds','-')}s" for e in engines)
        print(f"[{i}/{len(pages)}] {site}/{fn} — {tag}")

    # ---- summary ----
    lines = ["", "=" * 64, f"BAKE-OFF SUMMARY  ({len(pages)} pages, {args.n}/site)", "=" * 64]
    for e in engines:
        s = stats[e]
        avg_sec, avg_sk = _avg(s["sec"]), _avg(s["skills"])
        tin, tout = sum(x for x in s["in"] if x), sum(x for x in s["out"] if x)
        proj = (s["cost"] / s["ok"] * corpus_total) if s["ok"] else 0.0
        lines += [
            f"\n[{e}]  ok={s['ok']}  err={s['err']}",
            f"   avg skills/job : {avg_sk:.1f}",
            f"   avg latency    : {avg_sec:.1f}s   (~{avg_sec*corpus_total/3600:.1f}h serial for {corpus_total})",
            f"   tokens         : in={tin:,} out={tout:,}" if tin else "   tokens         : (n/a — local)",
            f"   cost this run  : ${s['cost']:.4f}" if e == "api" else "   cost this run  : $0 (local)",
            f"   projected full : ${proj:.2f} for {corpus_total} postings" if e == "api" else "",
        ]
    if agreements:
        lines += ["", f"local↔api skill agreement (Jaccard): "
                      f"avg={_avg(agreements):.3f}  min={min(agreements):.3f}  "
                      f"(1.0 = identical skill tokens)"]
    lines += ["", f"full results: {results_path}", "=" * 64]
    summary = "\n".join(l for l in lines if l is not None)
    print(summary)
    (run_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
