"""Crawl4AI-based Phase 2 crawler, used to collect the thesis corpus
(the earlier Playwright crawler is in archive/phase2_playwright_crawler).

Dynamic across websites — no hardcoded site patterns. Per site:

  1. SEED   AsyncUrlSeeder queries the site's sitemaps + the Common Crawl
            index -> thousands of known URLs WITHOUT crawling a single page
            (includes historical/expired postings a live crawl can't see).
  2. LEARN  DeepSeek reads a sample of those URLs and returns the site's URL
            taxonomy as JSON: a job-detail regex, a list-page regex, and
            scoring keywords. This is the only site-specific knowledge and
            it is learned at runtime.
  3. FILTER seeded URLs -> job-detail URLs only (budget goes to what
            Phase 3 actually consumes).
  4. CRAWL  arun_many over the detail URLs; if seeding was thin, fall back
            to a BestFirst deep crawl scored by the learned keywords.
  5. SAVE   outputs/<run_id>/<domain>/page_N.html + index.json +
            metadata.json — same layout Phase 3 already reads.

Usage:
    python main.py https://www.bongthom.com/            # 100 pages
    python main.py https://www.bongthom.com/ 2000       # bigger budget

Note: no FlareSolverr tier here — heavily Cloudflare-protected sites remain
the job of archive/phase2_playwright_crawler. This path shines on ordinary
sites and for harvesting URL lists.
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from urllib.parse import urlparse

from crawl4ai import (
    AsyncUrlSeeder, AsyncWebCrawler, BrowserConfig, CacheMode,
    CrawlerRunConfig, SeedingConfig,
)
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from api_fallback import try_api_fallback
from config import OUTPUT_DIR
from llm import ask_llm
from paginate import harvest_paginated_listings

# Safety net if the LLM taxonomy fails — same generic pattern the main
# phase2 crawler uses for URL priority.
GENERIC_DETAIL_RE = re.compile(
    r"/(?:jobs?|job_detail|vacanc(?:y|ies)|positions?|announcements?|"
    r"recruit\w*|career\w*|opportunit\w*)[\w-]*/.+|/\d{4,}(?:$|[/?])", re.I)
GENERIC_KEYWORDS = ["job", "detail", "vacancy", "career", "position", "recruit"]

MIN_HTML_CHARS = 3000  # below this it's a shell/challenge page, not content
# Deep-crawl auto-stop: this many consecutive visits without saving a new
# posting means the site is exhausted — stop instead of burning the budget.
NO_YIELD_LIMIT = 150
CF_MARKERS = ("just a moment", "checking your browser", "cf-chl",
              "challenge-platform", "attention required")
# Expired/removed-posting pages: valid detail URL, no job content. Phrases,
# not single words, to avoid matching a live posting's own text.
DEAD_MARKERS = ("may not exist or expired", "job not found",
                "no longer available", "this job has expired",
                "position has been filled", "posting has been removed")


def is_dead_posting(html: str) -> bool:
    low = html.lower()
    return any(m in low for m in DEAD_MARKERS)


def norm_domain(url: str) -> str:
    netloc = urlparse(url).netloc or url
    return netloc.removeprefix("www.")


def dedupe_urls(urls):
    """Order-preserving dedupe, scheme/www/trailing-slash insensitive."""
    seen, out = set(), []
    for u in urls:
        p = urlparse(u)
        key = (p.netloc.removeprefix("www.") + p.path.rstrip("/")
               + ("?" + p.query if p.query else ""))
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def is_cf_challenge(html: str) -> bool:
    low = html[:4000].lower()
    return any(m in low for m in CF_MARKERS)


# --------------------------------------------------
# 1. SEED — sitemap + Common Crawl, no crawling
# --------------------------------------------------
async def seed_urls(domain: str, max_urls: int = 20000) -> list:
    urls = []
    async with AsyncUrlSeeder() as seeder:
        # Query the sources separately: a Common Crawl outage must not
        # take the sitemap down with it (crawl4ai issue #1747).
        for source in ("sitemap", "cc"):
            try:
                cfg = SeedingConfig(source=source, extract_head=False,
                                    max_urls=max_urls, verbose=False)
                res = await seeder.urls(domain, cfg)
                found = [r["url"] for r in res]
                print(f"[SEED] {source}: {len(found)} URLs")
                urls += found
            except Exception as e:
                print(f"[SEED] {source} failed: {e}")
    return dedupe_urls(urls)


# --------------------------------------------------
# 2. LEARN — DeepSeek derives the site's URL taxonomy
# --------------------------------------------------
def learn_url_taxonomy(domain: str, sample_urls: list) -> dict:
    sample = "\n".join(sample_urls[:150])
    prompt = f"""You are a web crawling expert analyzing the URL structure of {domain}, a job-listing website.

Below is a sample of URLs from this site. Derive its URL taxonomy and return ONLY valid JSON:

{{
  "detail_regex": "<Python regex matching URLs of SINGLE job postings (one job's detail page)>",
  "list_regex": "<Python regex matching job LIST / category / search / pagination pages>",
  "detail_keywords": ["<3-8 lowercase tokens that appear in job-detail URLs on this site>"]
}}

Rules:
- Regexes are matched with re.search against the FULL URL, must be valid Python re syntax.
- detail_regex must NOT match list pages, company profiles, blogs, login, or static pages.
- Prefer path structure (e.g. /job_detail/) over IDs alone.
- If you cannot identify job-detail URLs in the sample, use null for detail_regex.

URL sample:
{sample}"""

    taxonomy = {"detail_re": GENERIC_DETAIL_RE, "list_re": None,
                "keywords": GENERIC_KEYWORDS, "source": "generic-fallback"}
    try:
        raw = ask_llm(prompt, timeout=120)
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        if data.get("detail_regex"):
            taxonomy["detail_re"] = re.compile(data["detail_regex"], re.I)
            taxonomy["source"] = "llm"
        if data.get("list_regex"):
            taxonomy["list_re"] = re.compile(data["list_regex"], re.I)
        if data.get("detail_keywords"):
            taxonomy["keywords"] = [str(k).lower() for k in data["detail_keywords"]][:8]
        print(f"[LEARN] taxonomy ({taxonomy['source']}): "
              f"detail={taxonomy['detail_re'].pattern!r} keywords={taxonomy['keywords']}")
    except Exception as e:
        print(f"[LEARN] LLM taxonomy failed ({e}) — using generic patterns")
    return taxonomy


# --------------------------------------------------
# 5. SAVE — same on-disk layout as archive/phase2_playwright_crawler
# --------------------------------------------------
class SiteOutput:
    def __init__(self, run_id: str, domain: str):
        self.dir = OUTPUT_DIR / run_id / domain
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.saved = 0
        self._seen_ids = set()

    @staticmethod
    def _posting_key(url: str) -> tuple:
        """Dedupe key for a posting URL. Numeric-ID boards: the ID sequence
        in the path (print/QR/format variants share it). Slug boards (no
        long digits in the path): the path itself, so query variants of the
        same posting (?share=telegram, tracking params) collapse onto one
        key. Single-segment paths keep the query — there the query may BE
        the job selector. The fragment stays in the key so synthetic
        API-harvest URLs (…#job-<id>) on a shared endpoint path stay
        distinct per job."""
        p = urlparse(url)
        ids = tuple(re.findall(r"\d{4,}", p.path))
        if ids:
            return ids
        path = p.path.rstrip("/")
        host = p.netloc.removeprefix("www.")
        if path.count("/") >= 2:
            return (host, path, p.fragment)
        return (host, path, p.query, p.fragment)

    def is_duplicate_posting(self, url: str) -> bool:
        key = self._posting_key(url)
        if key in self._seen_ids:
            return True
        self._seen_ids.add(key)
        return False

    def has_posting(self, url: str) -> bool:
        """Non-mutating check: is this posting already saved? (Use before
        queueing URLs — is_duplicate_posting marks the key as seen, which
        would make the later save skip itself.)"""
        return self._posting_key(url) in self._seen_ids

    def save(self, url: str, html: str):
        fname = f"page_{self.saved}.html"
        (self.dir / fname).write_text(html, encoding="utf-8")
        self.index[fname] = url
        self.saved += 1
        (self.dir / "index.json").write_text(
            json.dumps(self.index, indent=2), encoding="utf-8")

    def write_metadata(self, meta: dict):
        (self.dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")


# --------------------------------------------------
# 4a. CRAWL the seeded detail URLs directly
# --------------------------------------------------
async def crawl_url_list(crawler, urls, out: SiteOutput, max_pages: int) -> dict:
    stats = {"ok": 0, "blocked": 0, "failed": 0, "thin": 0}
    # Two-speed fetch. networkidle never fires on sites with persistent
    # trackers/chat widgets — every page then burns the full timeout and
    # fails. Most job-DETAIL pages are server-rendered, so fetch fast
    # (domcontentloaded + short settle) and retry only the pages that come
    # back as thin JS shells or time out with the slow full-render config.
    cfg_fast = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, magic=True,
                                wait_until="domcontentloaded",
                                delay_before_return_html=2.0,
                                page_timeout=30000, stream=True, verbose=False)
    cfg_slow = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, magic=True,
                                wait_until="networkidle",
                                page_timeout=30000, stream=True, verbose=False)

    async def run_batch(batch, cfg, collect_retries):
        retries = []
        async for result in await crawler.arun_many(batch, config=cfg):
            if out.saved >= max_pages:
                break
            if not result.success:
                if collect_retries:
                    retries.append(result.url)
                else:
                    stats["failed"] += 1
                continue
            html = result.html or ""
            if is_cf_challenge(html):
                stats["blocked"] += 1
                continue
            if len(html) < MIN_HTML_CHARS:
                if collect_retries:
                    retries.append(result.url)
                else:
                    stats["thin"] += 1
                continue
            if is_dead_posting(html):
                stats["dead"] = stats.get("dead", 0) + 1
                continue
            if out.is_duplicate_posting(result.url):
                stats["dup"] = stats.get("dup", 0) + 1
                continue
            out.save(result.url, html)
            stats["ok"] += 1
            if out.saved % 20 == 0:
                print(f"[SAVE] {out.saved}/{max_pages} pages")
        return retries

    batch = urls[:max_pages - out.saved]
    retries = await run_batch(batch, cfg_fast, collect_retries=True)
    if retries and out.saved < max_pages:
        print(f"[CRAWL] {len(retries)} thin/failed pages — retrying with full JS render")
        await run_batch(retries[:max_pages - out.saved], cfg_slow,
                        collect_retries=False)
    return stats


# --------------------------------------------------
# 4b. Deep-crawl fallback when seeding found little
# --------------------------------------------------
async def deep_crawl(crawler, start_url, domain, taxonomy,
                     out: SiteOutput, max_pages: int) -> dict:
    stats = {"ok": 0, "blocked": 0, "failed": 0, "visited": 0}
    no_yield = 0  # consecutive visits without a new posting saved
    # Depth must cover pagination CHAINS (page 2 -> 3 -> 4...): each hop is
    # one level, so a 40-page listing needs ~40 depth. BestFirst keeps this
    # cheap — high-scoring (job-ish) URLs are visited first regardless.
    strategy = BestFirstCrawlingStrategy(
        max_depth=50,
        include_external=False,
        max_pages=max(max_pages * 3, 300),  # visit budget > save budget
        url_scorer=KeywordRelevanceScorer(
            keywords=taxonomy["keywords"], weight=1.0),
        filter_chain=FilterChain([
            DomainFilter(allowed_domains=[domain, f"www.{domain}"]),
        ]),
    )
    # domcontentloaded + settle instead of networkidle: sites with persistent
    # trackers/chat widgets never reach networkidle, so every page would burn
    # the full timeout and fail. 2.5s settle lets JS-rendered listings paint;
    # pure-SPA sites (no links at all) are handled by the API fallback anyway.
    cfg = CrawlerRunConfig(deep_crawl_strategy=strategy, magic=True,
                           cache_mode=CacheMode.BYPASS, page_timeout=30000,
                           wait_until="domcontentloaded",
                           delay_before_return_html=2.5,
                           stream=True, verbose=False)
    async for result in await crawler.arun(start_url, config=cfg):
        if out.saved >= max_pages:
            break
        if no_yield >= NO_YIELD_LIMIT:
            print(f"[AUTO-STOP] {NO_YIELD_LIMIT} visits without a new posting "
                  f"— site looks exhausted ({out.saved} saved)")
            stats["auto_stopped"] = True
            break
        stats["visited"] += 1
        no_yield += 1
        if not result.success:
            stats["failed"] += 1
            continue
        html = result.html or ""
        if is_cf_challenge(html):
            stats["blocked"] += 1
            continue
        # jobs-only save: details always; anything else only counts as
        # link discovery for the BestFirst frontier
        if taxonomy["detail_re"].search(result.url) and len(html) >= MIN_HTML_CHARS:
            if is_dead_posting(html):
                stats["dead"] = stats.get("dead", 0) + 1
                continue
            if out.is_duplicate_posting(result.url):
                stats["dup"] = stats.get("dup", 0) + 1
                continue
            out.save(result.url, html)
            stats["ok"] += 1
            no_yield = 0
            if out.saved % 20 == 0:
                print(f"[SAVE] {out.saved}/{max_pages} pages (deep crawl)")
    return stats


# --------------------------------------------------
# Per-site pipeline
# --------------------------------------------------
async def crawl_site(start_url: str, max_pages: int, run_id: str):
    domain = norm_domain(start_url)
    out = SiteOutput(run_id, domain)
    print(f"\n{'='*60}\n[SITE] {domain} (budget: {max_pages} pages)\n{'='*60}")

    seeded = await seed_urls(domain)
    print(f"[SEED] total unique: {len(seeded)}")

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    # A tiny seed set (homepage + a few category pages) has no detail URLs
    # for the LLM to learn from — treat it like no seeds at all.
    taxonomy = learn_url_taxonomy(domain, seeded) if len(seeded) >= 30 else None

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        if taxonomy is None or taxonomy["source"] != "llm":
            # Too few seeds (or the LLM found no pattern in them) — render the
            # homepage and learn the taxonomy from its own links instead.
            print("[LEARN] seeds unusable — sampling homepage links instead")
            probe_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS,
                                         magic=True, wait_until="networkidle",
                                         page_timeout=30000, verbose=False)
            res = await crawler.arun(start_url, config=probe_cfg)
            links = [l.get("href") for l in (res.links or {}).get("internal", [])
                     if l.get("href")]
            taxonomy = (learn_url_taxonomy(domain, links) if links else
                        {"detail_re": GENERIC_DETAIL_RE, "list_re": None,
                         "keywords": GENERIC_KEYWORDS, "source": "generic-no-links"})

        details = [u for u in seeded if taxonomy["detail_re"].search(u)]
        # Sanity-check the LLM's regex against the evidence: if it matches
        # nothing in a non-trivial seed set while the generic job-URL
        # pattern matches plenty, the learned regex is wrong (e.g. wrong
        # number of path segments) — rescue with the generic pattern.
        if not details and taxonomy["source"] == "llm":
            generic_hits = [u for u in seeded if GENERIC_DETAIL_RE.search(u)]
            if len(generic_hits) >= 5:
                print(f"[LEARN] llm regex matched 0 seeds, generic matched "
                      f"{len(generic_hits)} — rescuing with generic pattern")
                taxonomy["detail_re"] = GENERIC_DETAIL_RE
                taxonomy["source"] = "generic-rescue"
                details = generic_hits
        print(f"[FILTER] {len(details)} job-detail URLs of {len(seeded)} seeded")
        meta = {"start_url": start_url, "seeded": len(seeded),
                "seeded_details": len(details),
                "taxonomy_source": taxonomy["source"],
                "detail_regex": taxonomy["detail_re"].pattern,
                "keywords": taxonomy["keywords"]}

        if details:
            print(f"[CRAWL] fetching {min(len(details), max_pages)} seeded detail URLs...")
            meta["seed_crawl"] = await crawl_url_list(crawler, details, out, max_pages)
            print(f"[CRAWL] seeded pass: {meta['seed_crawl']}")

        # No learnable link taxonomy = SPA signal (job cards are JS click
        # handlers, not <a href> links) — sniff the JSON API the site's own
        # pages call before blind-crawling links that don't exist.
        api_tried = False
        if out.saved == 0 and taxonomy["source"] != "llm":
            print("[CRAWL] no link taxonomy — trying the site's own JSON API")
            api_tried = True
            await try_api_fallback(crawler, start_url, out, max_pages, meta)

        # JS-button pagination: listing pages 2+ often exist only behind a
        # ?page=N the server renders but no <a href> exposes — deep crawl
        # can't reach them. Probe the common conventions over plain HTTP
        # and fetch whatever new detail URLs the listings reveal.
        if out.saved < max_pages and not meta.get("api_harvest", {}).get("ok"):
            page_urls = await asyncio.to_thread(
                harvest_paginated_listings, start_url,
                taxonomy["detail_re"], max_pages)
            new_urls = [u for u in page_urls if not out.has_posting(u)]
            if new_urls:
                print(f"[PAGINATE] {len(new_urls)} new detail URLs from "
                      f"listing pagination")
                meta["paginate_crawl"] = await crawl_url_list(
                    crawler, new_urls, out, max_pages)
                print(f"[CRAWL] paginate pass: {meta['paginate_crawl']}")

        if out.saved < max_pages and not meta.get("api_harvest", {}).get("ok"):
            print(f"[CRAWL] deep-crawl fallback for remaining "
                  f"{max_pages - out.saved} pages...")
            meta["deep_crawl"] = await deep_crawl(
                crawler, start_url, domain, taxonomy, out, max_pages)
            print(f"[CRAWL] deep pass: {meta['deep_crawl']}")

        # Last resort / top-up: whenever link crawling ended below budget,
        # sniff the site's JSON API. Link crawls miss postings that only
        # exist behind JS pagination buttons (jobify page 2) or that the
        # links under-expose vs the API (camhr: 1136 of 1664). Cheap on
        # already-complete sites (harvest dedupes -> 0 new saves).
        if not api_tried and out.saved < max_pages:
            print("[CRAWL] link crawl below budget — trying the site's JSON API")
            await try_api_fallback(crawler, start_url, out, max_pages, meta)

    meta["saved"] = out.saved
    out.write_metadata(meta)
    print(f"[DONE] {domain}: {out.saved} pages -> {out.dir}")
    return meta


async def main(urls, max_pages, concurrency=3):
    """Crawl up to `concurrency` sites at the same time (each site gets its
    own browser; 3 keeps memory sane on a laptop). Logs interleave."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sem = asyncio.Semaphore(concurrency)

    async def one_site(url):
        async with sem:
            try:
                await crawl_site(url, max_pages, run_id)
            except Exception as e:
                print(f"[ERROR] {url}: {e}")

    await asyncio.gather(*(one_site(u) for u in urls))


if __name__ == "__main__":
    args = sys.argv[1:]
    budget = int(args[-1]) if args and args[-1].isdigit() else 100
    sites = [a for a in args if not a.isdigit()] or ["https://www.bongthom.com/"]
    asyncio.run(main(sites, budget))
