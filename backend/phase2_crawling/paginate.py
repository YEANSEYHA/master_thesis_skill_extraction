"""Generic listing-pagination harvester (plain HTTP, no browser).

Deep crawl can only follow <a href> links, but many job boards paginate
their listing pages with JS buttons — pages 2+ appear NOWHERE as hrefs in
the HTML even though the server happily renders them at ?page=N. This
module probes the common pagination conventions directly over HTTP and
collects every job-detail URL the paginated listings expose.

Site-agnostic: the only site-specific knowledge (the detail-URL regex)
is the taxonomy learned at runtime. Only useful on server-rendered
listings — a pure SPA returns no detail hrefs at all here, and falls
through to the API fallback instead.
"""
import re
import time
from urllib.parse import urljoin, urlparse

import requests

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0.0.0 Safari/537.36"}
_PAGE_PARAMS = ("page", "pg", "p")
_LISTING_HINTS = ("job", "vacan", "career", "position", "recruit",
                  "search", "find", "announce", "opportun")
_MAX_LISTINGS = 4          # distinct listing URLs to try paginating
_MAX_PAGES_PER_LISTING = 300
_STRIKES = 2               # consecutive pages yielding no new detail URL
_HREF_RE = re.compile(r"""href=["']([^"'#\s]+)""", re.I)


def _get(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=_UA, timeout=timeout)
        return r.text if r.ok else ""
    except requests.RequestException:
        return ""


def _detail_urls(html: str, base_url: str, detail_re) -> list:
    """Ordered unique job-detail URLs linked from this HTML."""
    found = {}
    for href in _HREF_RE.findall(html):
        u = urljoin(base_url, href.strip())
        if urlparse(u).scheme in ("http", "https") and detail_re.search(u):
            found[u] = None
    return list(found)


def _same_site(u: str, root: str) -> bool:
    a = urlparse(u).netloc.removeprefix("www.")
    b = urlparse(root).netloc.removeprefix("www.")
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _listing_candidates(start_url: str, detail_re) -> list:
    """(url, page-1 html) of listing pages, best (most detail links) first.

    The unfiltered master listing beats filtered variants of the same path
    (?experience=3 etc.), so per path we keep the variant with the fewest
    query params. Candidates are ranked by how many detail links page 1
    exposes — that pushes company profiles and niche listings (which match
    the hints but expose few jobs) below the real listing."""
    home = _get(start_url)

    # group hint links by path, prefer the bare (fewest-params) variant
    by_path = {}
    for href in _HREF_RE.findall(home):
        u = urljoin(start_url, href.strip()).split("#")[0]
        if not _same_site(u, start_url) or detail_re.search(u):
            continue
        p = urlparse(u)
        path = (p.path or "/").lower()
        if not any(h in path for h in _LISTING_HINTS):
            continue
        n_params = len([q for q in p.query.split("&") if q])
        best = by_path.get(p.path)
        if best is None or n_params < best[0]:
            by_path[p.path] = (n_params, u)

    scored = []
    if len(_detail_urls(home, start_url, detail_re)) >= 3:
        scored.append((len(_detail_urls(home, start_url, detail_re)),
                       start_url, home))
    for _, u in list(by_path.values())[:10]:
        html = _get(u)
        n = len(_detail_urls(html, u, detail_re))
        if n >= 5:
            scored.append((n, u, html))
        time.sleep(0.1)

    scored.sort(key=lambda t: -t[0])
    return [(u, html) for _, u, html in scored[:_MAX_LISTINGS]]


def _walk_pages(url: str, page1_html: str, detail_re, seen: set,
                budget: int) -> list:
    """Find a working page param on this listing and walk it to the end."""
    page1_ids = set(_detail_urls(page1_html, url, detail_re))
    sep = "&" if urlparse(url).query else "?"

    param, page2_html = None, ""
    for cand in _PAGE_PARAMS:
        html = _get(f"{url}{sep}{cand}=2")
        ids = set(_detail_urls(html, url, detail_re))
        # a real page 2 shows detail links AND differs from page 1
        # (sites that ignore the param echo page 1 back)
        if ids and ids != page1_ids:
            param, page2_html = cand, html
            break
        time.sleep(0.1)
    if not param:
        return []

    new_urls, strikes, page = [], 0, 2
    while page <= _MAX_PAGES_PER_LISTING and len(seen) < budget:
        html = page2_html if page == 2 else _get(f"{url}{sep}{param}={page}")
        fresh = [u for u in _detail_urls(html, url, detail_re)
                 if u not in seen]
        for u in fresh:
            seen.add(u)
            new_urls.append(u)
        strikes = 0 if fresh else strikes + 1
        if strikes >= _STRIKES:
            break
        page += 1
        time.sleep(0.15)
    return new_urls


def harvest_paginated_listings(start_url: str, detail_re,
                               budget: int = 2000) -> list:
    """All job-detail URLs reachable through paginated listing pages."""
    seen, urls = set(), []
    for url, html in _listing_candidates(start_url, detail_re):
        for u in _detail_urls(html, url, detail_re):
            if u not in seen:
                seen.add(u)
                urls.append(u)
        urls += _walk_pages(url, html, detail_re, seen, budget)
        if len(urls) >= budget:
            break
    return urls[:budget]
