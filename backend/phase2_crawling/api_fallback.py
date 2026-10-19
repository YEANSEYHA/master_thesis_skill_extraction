"""API-fallback harvester for SPA job boards (React/Next/Angular sites whose
job cards are click-handlers, not <a href> links — link crawling is blind there).

Dynamic across websites, same philosophy as the URL taxonomy:

  1. SNIFF    render the site's job-listing pages with network capture on and
              collect the JSON/XHR endpoints the page ITSELF calls.
  2. MAP      DeepSeek reads the endpoints + response samples and returns the
              job-list endpoint, pagination scheme, items/id fields, and the
              job-detail endpoint template. Learned at runtime, per site.
  3. HARVEST  page through the list API, fetch each job's detail JSON, render
              it to a simple HTML document, and save via SiteOutput — the same
              page_N.html + index.json layout Phase 3 reads.

No hardcoded site logic. Sites where the sniff finds no job API simply return
False and the caller falls back to deep crawling.
"""
import asyncio
import html as html_mod
import json
import re
from urllib.parse import urlparse

import requests
from crawl4ai import CacheMode, CrawlerRunConfig

from llm import ask_llm

# Third-party/analytics hosts never worth probing as job APIs.
_NOISE_HOSTS = ("google", "gstatic", "facebook", "mixpanel", "mxpnl", "sentry",
                "hotjar", "clarity", "cloudflare", "payway", "doubleclick",
                "googletagmanager", "analytics", "tiktok", "linkedin")
_LISTING_HINTS = ("job", "vacan", "career", "position", "recruit", "search",
                  "find", "announce")


def _is_candidate_request(ev: dict, site_root: str) -> bool:
    if ev.get("event_type") not in (None, "request"):
        return False
    url = ev.get("url", "")
    if not url.startswith("http"):
        return False
    host = urlparse(url).netloc.lower()
    if any(n in host for n in _NOISE_HOSTS):
        return False
    # Same site or a sibling subdomain (api.example.com / server.example.com)
    if site_root not in host:
        return False
    if ev.get("resource_type") not in ("xhr", "fetch", None):
        return False
    # Static assets sneak in as fetch on some frameworks
    if re.search(r"\.(js|css|png|jpe?g|svg|woff2?|ico|map)(\?|$)", url, re.I):
        return False
    return True


def _fetch_json(url: str, method: str = "GET", post_data=None, timeout: int = 20):
    """Re-fetch a sniffed endpoint and return parsed JSON, or None."""
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        if method == "POST":
            r = requests.post(url, headers=headers, timeout=timeout,
                              json=post_data if isinstance(post_data, (dict, list)) else None,
                              data=None if isinstance(post_data, (dict, list)) else post_data)
        else:
            r = requests.get(url, headers=headers, timeout=timeout)
        if "json" not in r.headers.get("content-type", "") and not r.text.lstrip().startswith(("{", "[")):
            return None
        return r.json()
    except Exception:
        return None


# --------------------------------------------------
# 1. SNIFF — capture the XHR calls the site's own pages make
# --------------------------------------------------
async def sniff_api_endpoints(crawler, start_url: str) -> list:
    """Render the homepage + its job-listing-ish pages with network capture on.
    Returns [{url, method, post_data, sample}] for endpoints that answer JSON."""
    site_root = urlparse(start_url).netloc.removeprefix("www.")
    # scan_full_page scrolls to the bottom: infinite-scroll job lists only
    # call their REAL paginated API on scroll — the initial load often uses
    # a small banner/featured feed.
    cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, magic=True,
                           wait_until="networkidle", page_timeout=30000,
                           scan_full_page=True, scroll_delay=0.3,
                           capture_network_requests=True, verbose=False)

    # Homepage first; then up to 3 of its links that look like job listings.
    pages = [start_url]
    res = await crawler.arun(start_url, config=cfg)
    events = list(res.network_requests or [])
    links = [l.get("href") for l in (res.links or {}).get("internal", [])
             if l.get("href")]
    listing_links = [u for u in dict.fromkeys(links)
                     if any(h in u.lower() for h in _LISTING_HINTS)][:3]
    if not listing_links:
        # Headless render sometimes yields no anchors on SPAs — fall back to
        # the near-universal listing paths job boards use (generic, not
        # site-specific; 404s are harmless).
        base = start_url.rstrip("/")
        listing_links = [f"{base}{p}" for p in
                         ("/jobs", "/find-jobs", "/search-jobs", "/careers",
                          "/vacancies")]
    for u in listing_links:
        pages.append(u)
        try:
            r = await crawler.arun(u, config=cfg)
            events += list(r.network_requests or [])
        except Exception as e:
            print(f"[SNIFF] {u} failed: {e}")
    print(f"[SNIFF] rendered {len(pages)} pages, {len(events)} network events")

    # Dedupe by URL PATTERN (path with ids collapsed + query param NAMES):
    # a scrolling list fires the same endpoint dozens of times with different
    # page params — that's one candidate, not thirty.
    seen, candidates = set(), []
    for ev in events:
        if not _is_candidate_request(ev, site_root):
            continue
        p = urlparse(ev["url"])
        pattern = (ev.get("method", "GET"),
                   re.sub(r"\d+", "*", p.netloc + p.path),
                   tuple(sorted(q.split("=")[0] for q in p.query.split("&") if q)))
        if pattern in seen:
            continue
        seen.add(pattern)
        candidates.append(ev)

    # Job-ish endpoints first — the sample budget must not be eaten by
    # locale/config/static-page data calls.
    def score(ev):
        path = urlparse(ev["url"]).path.lower()
        hits = sum(h in path for h in _LISTING_HINTS)
        return (-hits, 0 if "list" in path else 1)
    candidates.sort(key=score)

    # Re-fetch each candidate to get a JSON sample the LLM can read.
    out = []
    for ev in candidates[:40]:
        data = await asyncio.to_thread(
            _fetch_json, ev["url"], ev.get("method", "GET"), ev.get("post_data"))
        if data is None:
            continue
        out.append({"url": ev["url"], "method": ev.get("method", "GET"),
                    "post_data": ev.get("post_data"),
                    "sample": json.dumps(data)[:900]})
    print(f"[SNIFF] {len(out)} JSON endpoints of {len(candidates)} candidates")
    return out, pages


# --------------------------------------------------
# 2. MAP — DeepSeek identifies the job API among the sniffed endpoints
# --------------------------------------------------
def learn_api_map(domain: str, endpoints: list, page_urls: list) -> dict | None:
    if not endpoints:
        return None
    listing = "\n\n".join(
        f"[{i}] {e['method']} {e['url']}\nresponse sample: {e['sample']}"
        for i, e in enumerate(endpoints))
    prompt = f"""You are a web crawling expert. {domain} is a job-listing website built as a JS single-page app. Below are the JSON API endpoints its own pages called, with response samples, plus the browser page URLs.

Browser page URLs seen: {page_urls}

Identify the endpoint that returns the PAGINATED LIST OF JOB POSTINGS and, if present, the endpoint returning ONE job's full detail. Return ONLY valid JSON:

{{
  "list_endpoint": "<full URL of the job-list endpoint, WITHOUT page/limit query params>",
  "list_method": "GET",
  "page_param": "<query param name for page number, e.g. page>",
  "limit_param": "<query param name for page size, or null>",
  "items_path": "<dot-path to the array of jobs in the response, e.g. data or result.items>",
  "id_field": "<field on each item holding the job id>",
  "title_field": "<field on each item holding the job title>",
  "total_path": "<dot-path to total job count in the response, or null>",
  "detail_endpoint_template": "<full URL of one job's detail endpoint with {{id}} placeholder, or null>",
  "public_url_template": "<the browser-facing URL of one job with {{id}} placeholder. Infer from the page URLs; if unseen, GUESS the most likely convention for this site (job boards commonly use <list-page>?job={{id}}, /job/{{id}}, /job-detail/{{id}}, /jobs/{{id}})>"
}}

Rules:
- Pick the endpoint whose sample clearly contains job postings (titles, salaries, locations) — not blogs or company lists.
- NEVER pick a category/taxonomy endpoint (samples listing category names with job COUNTS, or categories wrapping a few jobs each). The items themselves must be individual job postings.
- If SEVERAL endpoints return job lists, choose the MAIN paginated one that covers ALL jobs — the one with a total count and page/limit params. NEVER choose subset feeds whose URL contains words like banner, random, related, featured, top, recommended.
- For detail_endpoint_template: look for an endpoint whose URL ends in a numeric job id and whose sample is ONE job's full record with long text fields (description, requirements). Replace that id with {{id}}.
- If nothing is a job list, return {{"list_endpoint": null}}.

Endpoints:
{listing}"""
    try:
        raw = ask_llm(prompt, timeout=120)
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        if not data.get("list_endpoint"):
            print("[API-MAP] LLM found no job-list endpoint")
            return None
        print(f"[API-MAP] list={data['list_endpoint']} "
              f"detail={data.get('detail_endpoint_template')}")
        return data
    except Exception as e:
        print(f"[API-MAP] failed ({e})")
        return None


def _dig(obj, path):
    """Resolve a dot-path like 'result.items' in nested dicts."""
    if not path:
        return obj
    for part in str(path).split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


# --------------------------------------------------
# 2b. Detail-endpoint discovery — render one job's public page and watch
#     which API call carries its id (fires only on job-open, so the list
#     sniff never sees it)
# --------------------------------------------------
async def discover_detail_endpoint(crawler, api_map: dict, sample_id,
                                   listing_pages: list) -> None:
    sid = str(sample_id)
    candidates = []
    if api_map.get("public_url_template"):
        candidates.append(api_map["public_url_template"].replace("{id}", sid))
    origin = "{0.scheme}://{0.netloc}".format(urlparse(listing_pages[0]))
    candidates += [f"{p.rstrip('/')}?job={sid}" for p in listing_pages[1:3]]
    candidates += [f"{origin}/job/{sid}", f"{origin}/job-detail/{sid}",
                   f"{origin}/jobs/{sid}"]
    cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, magic=True,
                           wait_until="networkidle", page_timeout=30000,
                           capture_network_requests=True, verbose=False)
    for page_url in list(dict.fromkeys(candidates))[:4]:
        try:
            res = await crawler.arun(page_url, config=cfg)
        except Exception:
            continue
        for ev in (res.network_requests or []):
            url = ev.get("url", "")
            if (ev.get("event_type") in (None, "request")
                    and sid in url and url != page_url
                    and not re.search(r"\.(js|css|png|jpe?g|svg|woff2?)(\?|$)", url, re.I)):
                data = _fetch_json(url)
                blob = json.dumps(data) if data else ""
                # one job's full record carries long prose fields
                if data and len(blob) > 800:
                    api_map["detail_endpoint_template"] = url.replace(sid, "{id}")
                    api_map["public_url_template"] = page_url.replace(sid, "{id}")
                    print(f"[API-MAP] detail discovered: "
                          f"{api_map['detail_endpoint_template']}")
                    return
    print("[API-MAP] no detail endpoint found — saving list-level records")


# --------------------------------------------------
# 3. HARVEST — page through the list API, fetch details, save as HTML
# --------------------------------------------------
def _json_to_html(job: dict, title: str) -> str:
    """Render a job JSON record as a simple HTML document for Phase 3.
    HTML-bearing string fields (descriptions) are embedded as-is; everything
    else is escaped text."""
    parts = [f"<h1>{html_mod.escape(title)}</h1>"]

    def render(key, val, depth=0):
        if val in (None, "", [], {}):
            return
        label = html_mod.escape(str(key))
        if isinstance(val, dict):
            parts.append(f"<h3>{label}</h3>")
            for k, v in val.items():
                render(k, v, depth + 1)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                render(f"{key}[{i}]", v, depth + 1)
        elif isinstance(val, str) and "<" in val and ">" in val:
            parts.append(f"<h3>{label}</h3><div>{val}</div>")
        else:
            parts.append(f"<p><b>{label}:</b> {html_mod.escape(str(val))}</p>")

    for k, v in job.items():
        render(k, v)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html_mod.escape(title)}</title></head>"
            f"<body>{''.join(parts)}</body></html>")


async def harvest_via_api(api_map: dict, out, max_pages: int) -> dict:
    stats = {"ok": 0, "failed": 0, "pages": 0}
    seen_ids = set()
    sep = "&" if "?" in api_map["list_endpoint"] else "?"
    page_param = api_map.get("page_param") or "page"
    method = api_map.get("list_method", "GET")
    sem = asyncio.Semaphore(6)

    def _list_url(page, limit_q):
        return f"{api_map['list_endpoint']}{sep}{page_param}={page}{limit_q}"

    # Probe the page-size param instead of trusting the LLM's guess: bigger
    # pages mean fewer requests, and some endpoints ignore paging entirely
    # unless a limit is passed.
    base = await asyncio.to_thread(_fetch_json, _list_url(1, ""), method)
    best_n = len(_dig(base, api_map.get("items_path")) or [])
    limit_q = ""
    for cand in dict.fromkeys(
            [api_map.get("limit_param"), "limit", "page_size", "per_page", "size"]):
        if not cand:
            continue
        probe = await asyncio.to_thread(_fetch_json, _list_url(1, f"&{cand}=100"), method)
        n = len(_dig(probe, api_map.get("items_path")) or [])
        if n > best_n:
            limit_q, best_n = f"&{cand}=100", n
            break
    print(f"[API-HARVEST] page size {best_n} per request"
          f"{' (' + limit_q[1:] + ')' if limit_q else ''}")

    async def fetch_detail(jid):
        async with sem:
            url = api_map["detail_endpoint_template"].replace("{id}", str(jid))
            data = await asyncio.to_thread(_fetch_json, url)
            return jid, url, data

    page, no_new_pages = 1, 0
    while out.saved < max_pages:
        data = await asyncio.to_thread(_fetch_json, _list_url(page, limit_q), method)
        items = _dig(data, api_map.get("items_path")) if data else None
        if not items:
            break
        stats["pages"] += 1
        new_ids = []
        for it in items:
            jid = it.get(api_map.get("id_field") or "id") if isinstance(it, dict) else None
            if jid is not None and jid not in seen_ids:
                seen_ids.add(jid)
                new_ids.append((jid, it))
        if not new_ids:
            # some backends reshuffle/loop pages — tolerate a few repeats
            # before concluding the pagination is exhausted
            no_new_pages += 1
            if no_new_pages >= 3:
                break
            page += 1
            continue
        no_new_pages = 0

        if api_map.get("detail_endpoint_template"):
            results = await asyncio.gather(*(fetch_detail(j) for j, _ in new_ids))
            by_id = {j: (u, d) for j, u, d in results}
        else:
            by_id = {}

        for jid, item in new_ids:
            if out.saved >= max_pages:
                break
            detail_url, detail = by_id.get(jid, (None, None))
            job = detail if isinstance(detail, dict) else item
            # unwrap common {data: {...}} envelopes
            if isinstance(job, dict) and isinstance(job.get("data"), dict):
                job = job["data"]
            if not isinstance(job, dict):
                stats["failed"] += 1
                continue
            title = str(job.get(api_map.get("title_field") or "title")
                        or item.get(api_map.get("title_field") or "title") or jid)
            public = (api_map.get("public_url_template") or "").replace("{id}", str(jid))
            save_url = public or detail_url or f"{api_map['list_endpoint']}#job-{jid}"
            # the harvest may run as a top-up after a link crawl — skip
            # postings the crawl already saved (same numeric id)
            if out.is_duplicate_posting(save_url):
                stats["dup"] = stats.get("dup", 0) + 1
                continue
            out.save(save_url, _json_to_html(job, title))
            stats["ok"] += 1
            if out.saved % 50 == 0:
                print(f"[SAVE] {out.saved}/{max_pages} jobs (API harvest)")

        total = _dig(data, api_map.get("total_path"))
        if isinstance(total, int) and len(seen_ids) >= total:
            break
        page += 1
        await asyncio.sleep(0.3)   # politeness between list pages
    return stats


# --------------------------------------------------
# Entry point used by main.py
# --------------------------------------------------
async def try_api_fallback(crawler, start_url: str, out, max_pages: int,
                           meta: dict) -> bool:
    """Full sniff→map→harvest pipeline. Returns True if jobs were saved."""
    domain = urlparse(start_url).netloc.removeprefix("www.")
    endpoints, pages = await sniff_api_endpoints(crawler, start_url)
    api_map = learn_api_map(domain, endpoints, pages)
    if not api_map:
        return False

    if not api_map.get("detail_endpoint_template"):
        # grab one real job id from the list, then watch a job-open page to
        # catch the detail API call the list sniff can't see
        first = await asyncio.to_thread(_fetch_json, api_map["list_endpoint"])
        items = _dig(first, api_map.get("items_path")) or []
        if items and isinstance(items[0], dict):
            sample_id = items[0].get(api_map.get("id_field") or "id")
            if sample_id is not None:
                await discover_detail_endpoint(crawler, api_map, sample_id, pages)
    meta["api_map"] = {k: api_map.get(k) for k in
                       ("list_endpoint", "detail_endpoint_template",
                        "public_url_template")}
    meta["api_harvest"] = await harvest_via_api(api_map, out, max_pages)
    print(f"[CRAWL] API harvest: {meta['api_harvest']}")
    return meta["api_harvest"]["ok"] > 0
