from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import asyncio
import os
import random
import requests

# Headless by default (faster, works unattended); HEADLESS=0 to watch.
HEADLESS = os.getenv("HEADLESS", "1") != "0"

# Browser-like headers for the plain-HTTP fast path.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": ("text/html,application/xhtml+xml,application/xml;"
               "q=0.9,image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

STEALTH_INIT_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins', {
        get: () => [{
            0: {type: "application/x-google-chrome-pdf", suffixes: "pdf",
                description: "Portable Document Format"},
            description: "Portable Document Format",
            filename: "internal-pdf-viewer", length: 1,
            name: "Chrome PDF Plugin"
        }]
    });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    window.chrome = { runtime: {} };
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
    window.print = () => { console.log('[BLOCKED] window.print()'); };
"""


# ==========================================
# CLOUDFLARE DETECTION HELPER
# ==========================================

def is_cloudflare_blocked(html: str) -> bool:
    """Detect if page is Cloudflare challenge"""
    if not html or len(html) < 3000:
        return False

    text = html.lower()

    cloudflare_indicators = [
        "checking your browser",
        "cf-browser-verification",
        "cf-challenge",
        "enable javascript and cookies",
        "ray id:",
    ]

    for indicator in cloudflare_indicators:
        if indicator in text:
            print(f"[CLOUDFLARE] Detected: '{indicator}'")
            return True

    if all(word in text for word in ["cloudflare", "challenge", "browser"]):
        print("[CLOUDFLARE] Challenge page detected")
        return True

    return False


# ==========================================
# PLAIN-HTTP FAST PATH (no browser)
# ==========================================

def _looks_like_real_page(html: str) -> bool:
    """Server-rendered content vs an empty JS shell / challenge page."""
    if not html or len(html) < 3000:
        return False
    if is_cloudflare_blocked(html):
        return False
    # A JS shell has almost no anchors; real list/detail pages have many.
    lower = html.lower()
    return lower.count("<a ") >= 5


def fetch_http(url: str, timeout: int = 15, cookies: dict = None,
               user_agent: str = None) -> str:
    """Plain HTTP GET with browser headers. Returns "" when the site needs
    a real browser (non-200, tiny shell, or Cloudflare challenge).
    `cookies`/`user_agent` carry a FlareSolverr cf_clearance so challenged
    domains pass at HTTP speed (the cookie is bound to the solving UA)."""
    try:
        headers = HTTP_HEADERS if not user_agent else {
            **HTTP_HEADERS, "User-Agent": user_agent}
        resp = requests.get(url, headers=headers, cookies=cookies,
                            timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and ctype:
            return ""
        return resp.text
    except Exception:
        return ""


# ==========================================
# SPA DYNAMIC CONTENT WAIT HELPER
# ==========================================

async def _wait_for_dynamic_content(page, max_wait=15000, check_interval=1500):
    """
    Wait for SPA frameworks (Angular/React/Vue) to finish rendering.
    Tracks both DOM size and link count — waits until both stabilize.
    """
    prev_size = 0
    prev_links = 0
    stable_count = 0

    for elapsed in range(0, max_wait, check_interval):
        try:
            current_size = await page.evaluate("document.body.innerHTML.length")
            current_links = await page.evaluate("document.querySelectorAll('a[href]').length")
        except Exception:
            break

        if (current_size == prev_size and current_links == prev_links
                and current_size > 0 and current_links > 0):
            stable_count += 1
            if stable_count >= 2:
                break
        else:
            stable_count = 0

        prev_size = current_size
        prev_links = current_links
        await page.wait_for_timeout(check_interval)

    # Scroll down to trigger lazy-loaded content
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        # Check if scrolling loaded more content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
    except Exception:
        pass


# ==========================================
# BLOCK DISRUPTIVE BROWSER ACTIONS
# ==========================================

async def _block_disruptive_actions(page):
    """
    Block print dialogs, PDF previews, and external app triggers.
    Must be called BEFORE navigating to the page.
    """
    # Override window.print to prevent print dialogs
    await page.add_init_script("""
        window.print = () => { console.log('[BLOCKED] window.print()'); };
    """)

    # Auto-dismiss any dialogs (print, alert, confirm, prompt)
    page.on("dialog", lambda dialog: dialog.dismiss())

    # Block PDF downloads and external app navigations via route interception
    async def block_disruptive_routes(route):
        url = route.request.url.lower()
        # Block PDF files
        if url.endswith('.pdf') or 'application/pdf' in (route.request.headers.get('accept', '')):
            print(f"[BLOCKED] PDF request: {url[:80]}")
            await route.abort()
            return
        # Block external app deeplinks
        if any(domain in url for domain in ('t.me/', 'telegram.me/', 'wa.me/', 'viber.com/', 'line.me/')):
            print(f"[BLOCKED] External app: {url[:80]}")
            await route.abort()
            return
        await route.continue_()

    await page.route("**/*", block_disruptive_routes)


# ==========================================
# ATTEMPT 1: PLAYWRIGHT STEALTH
# ==========================================

async def render_page_stealth(url: str, timeout=25000) -> str:
    """Method 1: Standard Playwright with manual stealth scripts"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-web-security",
            ]
        )

        try:
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                locale='en-US',
                timezone_id='Asia/Phnom_Penh',
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                }
            )

            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        return [
                            {
                                0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                                description: "Portable Document Format",
                                filename: "internal-pdf-viewer",
                                length: 1,
                                name: "Chrome PDF Plugin"
                            }
                        ];
                    }
                });

                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });

                window.chrome = {
                    runtime: {}
                };

                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)

            page = await context.new_page()

            # Block print dialogs, PDF viewer, and external app popups
            await _block_disruptive_actions(page)

            await asyncio.sleep(random.uniform(1, 2))

            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout)
            except PlaywrightTimeoutError:
                print(f"[WARN] Navigation timeout, continuing with partial content")

            # Wait for SPA frameworks (Angular/React/Vue) to render
            await _wait_for_dynamic_content(page)

            html = await page.content()
            return html

        finally:
            await browser.close()


# ==========================================
# ATTEMPT 2: UNDETECTED-PLAYWRIGHT
# ==========================================

async def render_page_undetected(url: str, timeout=25000) -> str:
    """Method 2: Undetected-Playwright with Malenia"""
    try:
        from undetected_playwright import Malenia
        from playwright.async_api import async_playwright
    except ImportError:
        print("[ERROR] undetected-playwright not installed!")
        return ""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)

        try:
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='Asia/Phnom_Penh',
            )

            await Malenia.apply_stealth(context)

            page = await context.new_page()

            # Block print dialogs, PDF viewer, and external app popups
            await _block_disruptive_actions(page)

            await asyncio.sleep(random.uniform(2, 4))

            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout)
                print(f"[UNDETECTED] Page loaded")
            except PlaywrightTimeoutError:
                print(f"[WARN] Undetected navigation timeout, continuing with partial content")

            # Wait for SPA frameworks to render
            await _wait_for_dynamic_content(page)

            html = await page.content()
            print(f"[UNDETECTED] Got {len(html):,} chars")
            return html

        finally:
            await browser.close()


def flaresolverr_solve(url: str) -> dict:
    """Method 3: FlareSolverr for Cloudflare bypass. Returns the full
    solution dict ({response, cookies, userAgent}) so callers can reuse the
    cf_clearance cookie, or {} on failure. Blocking — call via to_thread."""
    try:
        print(f"[FLARESOLVERR] Requesting {url}...")
        response = requests.post(
            'http://localhost:8191/v1',
            json={
                'cmd': 'request.get',
                'url': url,
                'maxTimeout': 60000
            },
            timeout=70
        )

        result = response.json()

        if result.get('status') == 'ok':
            solution = result.get('solution', {})
            html = solution.get('response', '')
            print(f"[FLARESOLVERR] ✓ Success - {len(html):,} chars")
            return solution
        error_msg = result.get('message', 'Unknown error')
        print(f"[FLARESOLVERR] ✗ Failed: {error_msg}")

    except requests.exceptions.Timeout:
        print(f"[FLARESOLVERR] ✗ Timeout (>60s)")
    except Exception as e:
        print(f"[FLARESOLVERR] ✗ Error: {e}")
    return {}


async def render_page_flaresolverr(url: str) -> str:
    """HTML-only wrapper around flaresolverr_solve (off the event loop)."""
    solution = await asyncio.to_thread(flaresolverr_solve, url)
    return solution.get('response', '')


# ==========================================
# MASTER RENDER FUNCTION (3-TIER FALLBACK)
# ==========================================

async def render_page(url: str, timeout=25000, retry_count=0) -> str:
    """
    3-tier rendering with automatic fallback:
    1. Playwright Stealth + UA Rotation (fast, 70%)
    2. Undetected + Extreme Waits (medium, 85%)
    3. FlareSolverr (slow, 95%)
    """

    # ========================================
    # ATTEMPT 1: PLAYWRIGHT STEALTH
    # ========================================
    print(f"[ATTEMPT 1/3] Playwright Stealth + UA Rotation...")
    html = await render_page_stealth(url, timeout)

    if html and len(html) > 3000 and not is_cloudflare_blocked(html):
        print(f"✓ Success with Method 1 ({len(html):,} chars)")
        return html

    if is_cloudflare_blocked(html):
        print(f"[CLOUDFLARE DETECTED] Trying Method 2...")
    else:
        print(f"[FAILED] Method 1 - trying Method 2...")

    # ========================================
    # ATTEMPT 2: UNDETECTED-PLAYWRIGHT
    # ========================================
    print(f"[ATTEMPT 2/3] Undetected-Playwright + Extreme Waits...")
    await asyncio.sleep(random.uniform(3, 5))

    html = await render_page_undetected(url, timeout)

    if html and len(html) > 3000 and not is_cloudflare_blocked(html):
        print(f"✓ Success with Method 2 ({len(html):,} chars)")
        return html

    if is_cloudflare_blocked(html):
        print(f"[CLOUDFLARE STILL PRESENT] Trying Method 3...")
    else:
        print(f"[FAILED] Method 2 - trying Method 3...")

    # ========================================
    # ATTEMPT 3: FLARESOLVERR
    # ========================================
    print(f"[ATTEMPT 3/3] FlareSolverr (Cloudflare Solver)...")
    await asyncio.sleep(random.uniform(3, 5))

    html = await render_page_flaresolverr(url)

    if html and len(html) > 3000:
        if is_cloudflare_blocked(html):
            print(f"[CLOUDFLARE STILL BLOCKED] All 3 methods failed")
        else:
            print(f"✓ Success with Method 3 ({len(html):,} chars)")
        return html
    else:
        print(f"[FAILED] Method 3 returned {len(html) if html else 0} chars")

    # ========================================
    # ALL METHODS FAILED
    # ========================================
    print(f"[FAILED] All 3 methods exhausted for {url}")
    return html if html else ""


# ==========================================
# PERSISTENT RENDER SESSION (per site crawl)
# ==========================================

class RenderSession:
    """One browser reused for a whole site crawl, with an HTTP fast path.

    Per-URL strategy (cheapest first):
      0. plain HTTP GET            (~0.5s — server-rendered sites)
      1. persistent stealth page   (~2-5s — shared Chromium, new tab per URL)
      2. undetected-playwright     (slow — fresh hardened browser)
      3. FlareSolverr              (slowest — Cloudflare solver)

    The session learns per domain: if HTTP never succeeds after a few
    attempts (JS-shell/SPA site), it stops trying HTTP for that crawl.
    `quick=True` (job-detail pages) skips the long SPA stabilization +
    lazy-load scrolling; if the quick render comes back as a shell, the
    URL is retried once with the full dynamic wait, so complex layouts
    still get fully rendered.
    """

    HTTP_MISS_LIMIT = 3  # consecutive misses with zero successes -> stop trying

    # Challenged pages where stealth/undetected failed and FlareSolverr had
    # to solve: after this many, challenged URLs skip straight to FlareSolverr.
    CF_SKIP_AFTER = 2

    def __init__(self):
        self._pw = None
        self._browser = None
        self._context = None
        self.http_ok = 0
        self.http_miss = 0
        self.http_disabled = False
        # Cloudflare clearance from FlareSolverr — one solve unlocks the
        # domain for the HTTP fast path (~30min+ cookie lifetime)
        self._cf_cookies = {}
        self._cf_ua = None
        self._cf_escalations = 0
        self.stats = {"http": 0, "stealth": 0, "undetected": 0, "flaresolverr": 0}

    async def start(self):
        self._pw = await async_playwright().start()
        await self._launch()
        return self

    async def _launch(self):
        self._browser = await self._pw.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=HTTP_HEADERS["User-Agent"],
            locale="en-US",
            timezone_id="Asia/Phnom_Penh",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        await self._context.add_init_script(STEALTH_INIT_SCRIPT)

        # Block PDFs / external-app deeplinks context-wide (once, not per page)
        async def _block(route):
            u = route.request.url.lower()
            if u.endswith(".pdf") or any(d in u for d in (
                    "t.me/", "telegram.me/", "wa.me/", "viber.com/", "line.me/")):
                await route.abort()
                return
            await route.continue_()
        await self._context.route("**/*", _block)

    async def stop(self):
        for closer in (self._context, self._browser):
            try:
                if closer:
                    await closer.close()
            except Exception:
                pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._context = self._browser = self._pw = None

    async def _render_stealth_persistent(self, url: str, quick: bool,
                                         timeout: int = 25000) -> str:
        """Render in a new tab of the shared browser; relaunch once if the
        browser died mid-crawl."""
        for attempt in (1, 2):
            try:
                page = await self._context.new_page()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[SESSION] Browser lost ({e}) — relaunching...")
                await self.stop()
                self._pw = await async_playwright().start()
                await self._launch()
        try:
            page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
            try:
                await page.goto(url, wait_until="domcontentloaded",
                                timeout=timeout)
            except PlaywrightTimeoutError:
                print("[WARN] Navigation timeout, continuing with partial content")
            if quick:
                # Detail pages: brief settle, no scroll/stability loop
                await page.wait_for_timeout(1500)
                html = await page.content()
                if not _looks_like_real_page(html):
                    # Complex/SPA detail page — pay the full wait after all
                    await _wait_for_dynamic_content(page)
                    html = await page.content()
            else:
                await _wait_for_dynamic_content(page)
                html = await page.content()
            return html
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _flaresolverr(self, url: str) -> str:
        """Solve via FlareSolverr and keep the clearance cookie + UA so
        later challenged pages pass on the plain-HTTP fast path."""
        solution = await asyncio.to_thread(flaresolverr_solve, url)
        html = solution.get("response", "")
        if html and len(html) > 3000:
            self.stats["flaresolverr"] += 1
            self._cf_escalations += 1
            cookies = {c["name"]: c["value"]
                       for c in solution.get("cookies", []) if c.get("name")}
            if cookies:
                self._cf_cookies.update(cookies)
                self._cf_ua = solution.get("userAgent") or self._cf_ua
                print(f"[CF-COOKIE] Stored {len(cookies)} clearance cookies "
                      f"— challenged pages now try HTTP first")
        return html or ""

    async def render(self, url: str, quick: bool = False) -> str:
        # 0. HTTP fast path — skipped once the site proves to be JS-only.
        # Carries the FlareSolverr clearance cookie once we have one.
        challenged_http = False
        if not self.http_disabled:
            html = await asyncio.to_thread(
                fetch_http, url,
                cookies=self._cf_cookies or None, user_agent=self._cf_ua)
            challenged_http = is_cloudflare_blocked(html) if html else False
            if not challenged_http and _looks_like_real_page(html):
                self.http_ok += 1
                self.http_miss = 0
                self.stats["http"] += 1
                return html
            if not challenged_http:
                # A challenge is not a JS-shell signal — only genuine
                # shells/errors count toward disabling the fast path
                self.http_miss += 1
                if self.http_ok == 0 and self.http_miss >= self.HTTP_MISS_LIMIT:
                    self.http_disabled = True
                    print("[HTTP] Site needs a browser — disabling HTTP fast path")

        # Challenged + stealth/undetected already proved useless on this
        # domain → skip the doomed middle steps (~20s each) entirely
        if challenged_http and self._cf_escalations >= self.CF_SKIP_AFTER:
            print("[CF-SKIP] Known-challenged domain — straight to FlareSolverr")
            html = await self._flaresolverr(url)
            if html:
                return html

        # 1. Persistent stealth browser
        try:
            html = await self._render_stealth_persistent(url, quick)
            if html and len(html) > 3000 and not is_cloudflare_blocked(html):
                self.stats["stealth"] += 1
                return html
        except Exception as e:
            print(f"[SESSION] Stealth render failed: {e}")
            html = ""

        # Challenge surfaced at the stealth tier on a known-challenged
        # domain — undetected won't do better, go straight to FlareSolverr
        if (is_cloudflare_blocked(html)
                and self._cf_escalations >= self.CF_SKIP_AFTER):
            print("[CF-SKIP] Known-challenged domain — straight to FlareSolverr")
            return await self._flaresolverr(url)

        # 2./3. Rare escalations — fresh hardened browser, then FlareSolverr
        print("[ESCALATE] Undetected-playwright...")
        html = await render_page_undetected(url)
        if html and len(html) > 3000 and not is_cloudflare_blocked(html):
            self.stats["undetected"] += 1
            return html

        print("[ESCALATE] FlareSolverr...")
        return await self._flaresolverr(url)
