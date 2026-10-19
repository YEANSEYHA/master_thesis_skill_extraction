import asyncio
import heapq
import os
from .renderer import RenderSession
from .links import extract_links
from .signals import content_hash, is_blocked
from .storage import save_html, save_metadata
from .pagination import has_pagination
from .navigator import suggest_navigation_qwen
from .sitemap import discover_sitemap_urls
from .db import load_crawled_urls_for_domain, mark_urls_crawled
from urllib.parse import urlparse, urljoin, parse_qs, urlencode
import re


# --------------------------------------------------
# URL Normalization
# --------------------------------------------------
def normalize_url(url: str) -> str:
    """Normalize URL to avoid crawling duplicates.
    - Remove trailing slash (except root)
    - Remove default page params (?page=1)
    - Sort query params
    - Remove fragment
    """
    parsed = urlparse(url)

    # Remove fragment
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    # Parse and sort query params, remove page=1 (it's the default)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("page", None) if params.get("page") == ["1"] else None
    params.pop("p", None) if params.get("p") == ["1"] else None
    sorted_query = urlencode(sorted(params.items()), doseq=True)

    return f"{parsed.scheme}://{parsed.netloc}{path}{'?' + sorted_query if sorted_query else ''}"


# --------------------------------------------------
# URL priority — spend the max_pages budget on job pages first
# --------------------------------------------------
# Job-detail pages (a slug/ID after a job-ish segment, a long numeric ID,
# or an id= query param). These are what Phase 3 extracts skills from.
_DETAIL_RE = re.compile(
    r"/(?:jobs?|vacanc(?:y|ies)|positions?|announcements?|recruit\w*|"
    r"career\w*|opportunit\w*)[\w-]*/.+|/\d{4,}(?:$|[/?])", re.I)
_DETAIL_QUERY_RE = re.compile(r"(?:^|&)(?:job_?)?id=\d+", re.I)
# List/pagination pages — the gateways to details, crawled second.
_LIST_RE = re.compile(r"(?:^|&)(?:page|p)=\d+", re.I)
# Never worth a fetch on a job board.
_JUNK_PATHS = (
    "/about", "/contact", "/privacy", "/term", "/policy", "/faq",
    "/login", "/register", "/signup", "/sign-up", "/signin", "/sign-in",
    "/forgot", "/password", "/wp-admin", "/cart", "/checkout",
    "/blog", "/authsocial", "/redirect", "/hiring_guide",
)

# Crawled for link discovery only — company pages link to job details but
# carry no skill content for Phase 3, so they don't consume the save budget.
_NOSAVE_PATHS = ("/company_detail", "/company_list", "/companies", "/employers")


def url_priority(url: str) -> int:
    """0 = job detail, 1 = list/pagination, 2 = everything else."""
    parsed = urlparse(url)
    path, query = parsed.path.lower(), parsed.query.lower()
    if _DETAIL_RE.search(path) or _DETAIL_QUERY_RE.search(query):
        return 0
    if _LIST_RE.search(query) or "/page/" in path:
        return 1
    return 2


def is_junk_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.startswith(j) or f"{j}/" in path for j in _JUNK_PATHS)


def is_nosave_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.startswith(j) for j in _NOSAVE_PATHS)


def url_pattern(url: str) -> str:
    """Collapse sibling pages onto one key so per-pattern decisions stick:
    any path segment containing a digit becomes '*'
    (/company_detail/view_details_5405.html -> /company_detail/*)."""
    parsed = urlparse(url)
    segments = [("*" if re.search(r"\d", s) else s)
                for s in parsed.path.lower().split("/")]
    path = "/".join(segments)
    keys = ",".join(sorted(parse_qs(parsed.query).keys()))
    return f"{path}?{keys}" if keys else path


class AutonomousCrawler:
    def __init__(self, start_url, max_pages=500, run_id="", stop_event=None):
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.run_id = run_id
        self.stop_event = stop_event  # threading.Event — set to stop crawl

        self.max_pages = max_pages
        self.crawl_delay = 1  # seconds between requests (polite crawling)

        # Priority queue: (priority, seq, url) — job details (0) come out
        # before list pages (1) before everything else (2), FIFO within a tier.
        self._seq = 0
        self.queue = [(url_priority(start_url), 0, start_url)]
        self.queued_urls = {start_url}  # O(1) lookup instead of `url in self.queue`
        self.visited_urls = set()
        self.visited_hashes = set()
        self.saved = 0
        self.failed_urls = {}  # url -> retry count
        self.discovered_patterns = set()  # avoid redundant LLM calls
        self.llm_checked_urls = set()  # avoid calling LLM on same URL twice
        # url_pattern -> count of "no pagination" verdicts; after 3 the LLM
        # fallback is skipped for that pattern (BS4 still runs every page)
        self._no_pagination_count = {}

        self._history_buffer = []  # saved URLs pending a batched DB write

        self.metadata = {
            "start_url": start_url,
            "pages": [],
            "blocked": [],
            "llm_used": 0
        }

    async def _flush_history(self):
        """Write buffered crawl-history rows in one batched DB call,
        off the event loop."""
        if not self._history_buffer:
            return
        batch, self._history_buffer = self._history_buffer, []
        await asyncio.to_thread(mark_urls_crawled, batch, self.domain, self.run_id)

    def _add_to_queue(self, url: str, priority: int = None):
        """Add normalized URL to the priority queue if new and not junk"""
        url = normalize_url(url)
        if url in self.visited_urls or url in self.queued_urls:
            return
        if is_junk_url(url):
            return
        parsed = urlparse(url)
        netloc = parsed.netloc
        # Handle www/non-www
        if netloc == self.domain or netloc == f"www.{self.domain}" or self.domain == f"www.{netloc}":
            self._seq += 1
            heapq.heappush(
                self.queue,
                (url_priority(url) if priority is None else priority,
                 self._seq, url))
            self.queued_urls.add(url)

    def _process_llm_decision(self, decision: dict, url: str):
        """Process LLM navigation discovery results"""
        if decision.get("action") != "discovered":
            return

        self.metadata["llm_used"] += 1

        # Handle pagination patterns
        for pag in decision.get("pagination", []):
            pattern = pag.get("pattern")
            if pattern in self.discovered_patterns:
                continue
            self.discovered_patterns.add(pattern)

            max_try = pag.get("max", 10)
            print(f"[NAV] Pagination: {pattern} (max={max_try})")
            for i in range(2, max_try + 1):
                candidate = pattern.replace("{n}", str(i))
                absolute = urljoin(url, candidate)
                self._add_to_queue(absolute)

        # Handle section/category URLs
        for section_url in decision.get("sections", []):
            absolute = urljoin(url, section_url)
            self._add_to_queue(absolute)
            print(f"[NAV] Section: {absolute}")

    async def crawl(self):
        # Step 0: Try sitemap.xml first — fastest way to discover all pages
        print(f"[SITEMAP] Checking for sitemap.xml...")
        sitemap_urls = discover_sitemap_urls(self.start_url, self.domain)
        if sitemap_urls:
            print(f"[SITEMAP] Found {len(sitemap_urls)} URLs from sitemap")
            for url in sitemap_urls:
                self._add_to_queue(url)
        else:
            print(f"[SITEMAP] No sitemap found, relying on BFS + LLM")
        self.metadata["_sitemap_count"] = len(sitemap_urls) if sitemap_urls else 0

        # Load ALL previously-crawled URLs for this domain (one DB call).
        # FRESH_CRAWL=1 ignores history — every page is re-crawled.
        if os.getenv("FRESH_CRAWL", "0") == "1":
            print("[HISTORY] FRESH_CRAWL=1 — ignoring crawl history, no URLs skipped")
            self._already_crawled = set()
        else:
            self._already_crawled = load_crawled_urls_for_domain(self.domain)
        self._start_already_crawled = normalize_url(self.start_url) in self._already_crawled

        # Filter queue but keep start URL for link discovery
        skip = self._already_crawled - {normalize_url(self.start_url)}
        if skip:
            self.queue = [e for e in self.queue if e[2] not in skip]
            heapq.heapify(self.queue)
            self.queued_urls -= skip
            self.visited_urls |= skip  # prevent re-adding via BFS
            print(f"[HISTORY] Skipping {len(skip)} already-crawled URLs from queue")

        is_first_page = True

        # One browser for the whole crawl (HTTP fast path + persistent tabs)
        session = await RenderSession().start()
        try:
            await self._crawl_loop(session, is_first_page)
        finally:
            await self._flush_history()
            print(f"[SESSION] Fetch methods used: {session.stats}")
            self.metadata["fetch_stats"] = session.stats
            await session.stop()

        # Save crawl metadata
        self.metadata["sitemap_urls_found"] = self.metadata.pop("_sitemap_count", 0)
        self.metadata["total_queued"] = len(self.queued_urls)
        self.metadata["total_visited"] = len(self.visited_urls)
        self.metadata["failed_retries"] = len(self.failed_urls)
        save_metadata(self.metadata, self.run_id)

    async def _crawl_loop(self, session, is_first_page):
        while self.queue and self.saved < self.max_pages:
            # Check if stop was requested
            if self.stop_event and self.stop_event.is_set():
                print(f"[STOP] Crawl stopped by user after {self.saved} pages")
                break

            prio, _, url = heapq.heappop(self.queue)
            is_detail = prio == 0

            if url in self.visited_urls:
                continue

            # Polite delay between requests (skip for first page)
            if not is_first_page:
                await asyncio.sleep(self.crawl_delay)

            # Check stop again after delay
            if self.stop_event and self.stop_event.is_set():
                print(f"[STOP] Crawl stopped by user after {self.saved} pages")
                break

            print(f"[CRAWL] [{self.saved}/{self.max_pages}] {url}")

            try:
                # Detail pages skip the long SPA waits (auto-retried with the
                # full wait inside the session if they come back as a shell)
                html = await session.render(url, quick=is_detail)

            except Exception as e:
                print(f"[ERROR] render failed: {e}")
                # Check stop after failed render
                if self.stop_event and self.stop_event.is_set():
                    print(f"[STOP] Crawl stopped by user after {self.saved} pages")
                    break
                # Retry failed URLs up to 2 times (lowest priority = last)
                retries = self.failed_urls.get(url, 0)
                if retries < 2:
                    self.failed_urls[url] = retries + 1
                    self.queued_urls.discard(url)
                    self._add_to_queue(url, priority=3)
                    print(f"[RETRY] Will retry later ({retries + 1}/2)")
                continue

            # Check stop after render
            if self.stop_event and self.stop_event.is_set():
                print(f"[STOP] Crawl stopped by user after {self.saved} pages")
                break

            # Content deduplication
            h = content_hash(html)
            if h in self.visited_hashes:
                print(f"[DEDUP] Duplicate content, skipping")
                self.visited_urls.add(url)
                continue

            # Block detection
            if is_blocked(html):
                print(f"[BLOCKED] {url}")
                self.metadata["blocked"].append(url)
                self.visited_urls.add(url)
                continue

            # Mark visited
            self.visited_urls.add(url)
            self.visited_hashes.add(h)

            # Skip saving if this is the start URL revisited for link discovery only
            is_start_revisit = (normalize_url(url) == normalize_url(self.start_url)
                                and self._start_already_crawled)
            if is_start_revisit:
                print(f"[HISTORY] Start URL already crawled — extracting links only")
            elif is_nosave_url(url):
                print(f"[NOSAVE] Company/list page — link discovery only, not saved")
            else:
                # Save HTML
                save_html(self.saved, url, html, self.start_url, self.run_id)
                self.metadata["pages"].append(url)
                self.saved += 1

                # Record in crawl_history DB (batched — a fresh Neon SSL
                # connection per page cost ~0.5s and blocked the event loop)
                self._history_buffer.append(url)
                if len(self._history_buffer) >= 20:
                    await self._flush_history()

            # 1. Normal link extraction
            links = extract_links(html, url, self.domain)

            # Check against in-memory set (no DB call)
            new_count = sum(1 for l in links if normalize_url(l) not in self.visited_urls)
            skipped = len(links) - new_count
            if skipped > 0:
                print(f"[LINKS] {len(links)} found, {skipped} already crawled → {new_count} new")
            else:
                print(f"[LINKS] {len(links)} found, all new")

            for link in links:
                self._add_to_queue(link)

            # Check stop before LLM call
            if self.stop_event and self.stop_event.is_set():
                print(f"[STOP] Crawl stopped by user after {self.saved} pages")
                break

            # 2. LLM-assisted navigation
            if url not in self.llm_checked_urls:
                should_check_llm = False

                # Always run LLM on the homepage (discovers sections/categories)
                if is_first_page:
                    should_check_llm = True
                    print(f"[NAV] Running LLM on homepage for section discovery...")
                # Skip detail pages (URLs like /job/senior-dev or /jobs/12345)
                elif is_detail:
                    pass  # detail page, no need for LLM
                # Run LLM on pages with pagination. The LLM fallback is
                # skipped once a URL pattern said "no pagination" 3 times.
                else:
                    pat = url_pattern(url)
                    misses = self._no_pagination_count.get(pat, 0)
                    if has_pagination(html, url, llm_ok=misses < 3):
                        should_check_llm = True
                        self._no_pagination_count.pop(pat, None)
                    else:
                        self._no_pagination_count[pat] = misses + 1

                if should_check_llm:
                    self.llm_checked_urls.add(url)
                    decision = suggest_navigation_qwen(html, url)
                    self._process_llm_decision(decision, url)

            is_first_page = False
