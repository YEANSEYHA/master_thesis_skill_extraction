"""
Manual Web Scraper using Playwright with FlareSolverr fallback
"""

import os
import re
import time
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup, Comment

# =============================================================================
# CONFIGURATION
# =============================================================================

# Hardcoded URLs to scrape
URLS = [
    # Add your URLs here
    'https://jobify.works/jobs/1203',
]


# Change this folder name base o job card is dynamic or not

# Output folder
OUTPUT_FOLDER = "data/flutter_ios_developer"

# FlareSolverr URL (default: http://localhost:8191)
FLARESOLVERR_URL = "http://localhost:8191"

# ScraperAPI settings
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "")
SCRAPERAPI_URL = "http://api.scraperapi.com"

# Playwright settings
HEADLESS = False  # Set to False to see the browser
TIMEOUT = 90000  # 90 seconds

# =============================================================================
# CAPTCHA/BLOCK DETECTION
# =============================================================================

BLOCK_PATTERNS = [
    "captcha",
    "challenge platform",
    "cloudflare",
    "verify you are human",
    "let's confirm you are human",
    "access denied",
    "you have been blocked",
    "please wait while we verify",
    "checking your browser",
    "js challenge",
    "ray id",
    "human verification",
]

BLOCK_SELECTORS = [
    "#captcha-container",
    ".cf-browser-verification",
    "[data-captcha]",
    ".captcha",
    "#challenge-form",
    ".cf-challenge",
    "[class*='captcha']",
    "[id*='captcha']",
]

def check_duplicates(urls):
    """
    Check for duplicate URLs in a list.
    Returns list of duplicate URLs if found, otherwise returns [].
    """
    seen = set()
    duplicates = []
    for url in urls:
        if url in seen:
            duplicates.append(url)
        else:
            seen.add(url)
    return duplicates



def extract_meta_content(soup) -> str:
    """
    Extract job description from meta tags and share links as fallback
    when JS-rendered content fails to load (e.g. MyWorld Careers).
    """
    texts = []

    # 1) Check og:description and twitter:description
    for attr in [{"property": "og:description"}, {"name": "twitter:description"}]:
        tag = soup.find("meta", attrs=attr)
        if tag and tag.get("content"):
            texts.append(tag["content"].strip())

    # 2) Extract from LinkedIn/email share links (often contain job description)
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # LinkedIn share summary parameter
        if "linkedin.com/shareArticle" in href and "summary=" in href:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            if "summary" in params:
                texts.append(params["summary"][0].strip())
        # Email body parameter
        elif href.startswith("mailto:") and "body=" in href:
            from urllib.parse import parse_qs, urlparse
            body_part = href.split("body=", 1)[-1]
            from urllib.parse import unquote
            decoded = unquote(body_part).split("http")[0].strip()
            if len(decoded) > 50:
                texts.append(decoded)

    # Return the longest text found (most complete description)
    return max(texts, key=len) if texts else ""


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Extract meta/share content BEFORE removing tags (fallback for JS-rendered sites)
    meta_content = extract_meta_content(soup)

    # 1) Remove script, style, and media tags
    for tag in soup.find_all(["script", "style", "img", "video", "iframe", "svg", "noscript", "link"]):
        tag.decompose()

    # 2) Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 3) Remove navigation, header, footer, and non-content elements
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()

    # 4) Remove inline styles
    for tag in soup.find_all(style=True):
        del tag["style"]

    cleaned = str(soup)

    # 5) If main content is thin, append meta/share description as fallback
    body_text = soup.get_text(separator=" ", strip=True)
    main_content = soup.find(class_=re.compile(r"main-content|job-description|job-detail"))
    main_text = main_content.get_text(strip=True) if main_content else ""

    if len(main_text) < 100 and meta_content:
        cleaned += f'\n<!-- Fallback: extracted from meta/share links -->\n<div class="meta-fallback-content">\n<p>{meta_content}</p>\n</div>'
        print(f"    [clean_html] Injected meta fallback content ({len(meta_content)} chars)")

    return cleaned


def has_content(html: str, min_text_length: int = 100) -> bool:
    """Check if cleaned HTML has meaningful text content."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    if len(text) < min_text_length:
        return False
    return True


def is_blocked(html_content: str, debug: bool = False) -> bool:
 
    """
    Check if the HTML content indicates a CAPTCHA or block page.

    Args:
        html_content: HTML content to check
        debug: Print debug info about what triggered the block

    Returns:
        True if blocked/CAPTCHA detected, False otherwise
    """
    html_lower = html_content.lower()

    # Check for block patterns in <title> tag (most reliable indicator)
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_lower, re.IGNORECASE | re.DOTALL)
    if title_match:
        title_text = title_match.group(1).strip()
        for pattern in BLOCK_PATTERNS:
            if pattern in title_text:
                if debug:
                    print(f"    [DEBUG] Blocked: Found '{pattern}' in title: '{title_text[:50]}...'")
                return True

    # Parse HTML to get visible text only (not script/style/links)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script, style, and link tags before checking
    for tag in soup.find_all(['script', 'style', 'link', 'meta']):
        tag.decompose()

    visible_text = soup.get_text(separator=' ', strip=True).lower()

    # Check for block patterns in visible text only
    # Use more specific checks to reduce false positives
    suspicious_patterns_found = 0
    patterns_found = []
    for pattern in BLOCK_PATTERNS:
        if pattern in visible_text:
            suspicious_patterns_found += 1
            patterns_found.append(pattern)

    # If multiple block patterns found in visible text, likely blocked
    if suspicious_patterns_found >= 2:
        if debug:
            print(f"    [DEBUG] Blocked: Found {suspicious_patterns_found} patterns in visible text: {patterns_found}")
        return True

    # Check for common CAPTCHA/block selectors in DOM
    captcha_elem = soup.find(id=re.compile(r'captcha', re.I))
    challenge_elem = soup.find(class_=re.compile(r'captcha|cf-challenge|cloudflare', re.I))

    if captcha_elem or challenge_elem:
        if debug:
            print(f"    [DEBUG] Blocked: Found captcha/challenge element in DOM")
        return True

    # Additional heuristic: very short content might be a block
    visible_len = len(visible_text.strip())
    if visible_len < 200:
        if debug:
            print(f"    [DEBUG] Blocked: Too little visible text ({visible_len} chars)")
        return True

    if debug:
        print(f"    [DEBUG] Not blocked: {visible_len} chars visible text, {suspicious_patterns_found} patterns found")

    return False


# =============================================================================
# FLARESOLVERR FALLBACK
# =============================================================================

def scrape_with_flaresolverr(url: str, max_retries: int = 2) -> str:
    """
    Scrape URL using FlareSolverr to bypass Cloudflare/CAPTCHA.

    Args:
        url: URL to scrape
        max_retries: Number of retries

    Returns:
        HTML content
    """
    payload = {
        "cmd": "request.get",
        "url": url,
        "maxTimeout": 60000,
    }

    for attempt in range(max_retries):
        try:
            print(f"    [FlareSolverr] Attempt {attempt + 1}/{max_retries}")
            # FlareSolverr v1 API endpoint
            response = requests.post(f"{FLARESOLVERR_URL}/v1", json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "ok":
                    solution = data.get("solution", {})
                    return solution.get("response", "")
                else:
                    print(f"    [FlareSolverr] Error: {data.get('message', 'Unknown error')}")
            else:
                print(f"    [FlareSolverr] HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"    [FlareSolverr] Connection refused - Is FlareSolverr running at {FLARESOLVERR_URL}?")
        except Exception as e:
            print(f"    [FlareSolverr] Error: {e}")

        if attempt < max_retries - 1:
            time.sleep(2)

    return ""


# =============================================================================
# SCRAPERAPI FALLBACK
# =============================================================================

def scrape_with_scraperapi(url: str, max_retries: int = 2) -> str:
    """
    Scrape URL using ScraperAPI to bypass blocks.

    Args:
        url: URL to scrape
        max_retries: Number of retries

    Returns:
        HTML content
    """
    if not SCRAPERAPI_KEY or SCRAPERAPI_KEY == "YOUR_API_KEY_HERE":
        print("    [ScraperAPI] API key not configured")
        return ""

    params = {
        'api_key': SCRAPERAPI_KEY,
        'url': url,
        'render': 'true'  # Enable JavaScript rendering
    }

    for attempt in range(max_retries):
        try:
            print(f"    [ScraperAPI] Attempt {attempt + 1}/{max_retries}")
            response = requests.get(SCRAPERAPI_URL, params=params, timeout=120)

            if response.status_code == 200:
                return response.text
            else:
                print(f"    [ScraperAPI] HTTP {response.status_code}")

        except Exception as e:
            print(f"    [ScraperAPI] Error: {e}")

        if attempt < max_retries - 1:
            time.sleep(2)

    return ""


# =============================================================================
# PLAYWRIGHT SCRAPER
# =============================================================================

def scrape_with_playwright(url: str) -> str:
    """
    Scrape URL using Playwright browser.

    Args:
        url: URL to scrape

    Returns:
        HTML content
    """
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            # Wait for network to go idle (ensures JS-rendered content like job descriptions are loaded)
            page.goto(url, timeout=TIMEOUT, wait_until="networkidle")

            # Wait for body to be present
            page.wait_for_selector("body", timeout=30000)

            # Wait for meaningful content to render (JS frameworks may hydrate after networkidle)
            try:
                page.wait_for_function(
                    "document.body.innerText.length > 500",
                    timeout=15000
                )
            except Exception:
                print(f"    [Playwright] Warning: content may not be fully rendered")

            html_content = page.content()

            browser.close()
            return html_content

        except Exception as e:
            browser.close()
            raise e


# =============================================================================
# FILE NAME GENERATION
# =============================================================================

def generate_filename(url: str, index: int) -> str:
    """
    Generate a filename from URL.

    Args:
        url: URL to scrape
        index: Index in the URL list

    Returns:
        Filename like: page1_www.example.com_path_to_page.html
    """
    # Strip protocol
    clean_url = re.sub(r'^https?://', '', url)
    # Replace invalid filename characters with underscores
    clean_url = re.sub(r'[^a-zA-Z0-9._-]', '_', clean_url)
    # Truncate to avoid OS filename length limit (255 chars)
    clean_url = clean_url[:200]

    return f"page{index + 1}_{clean_url}.html"


# =============================================================================
# MAIN SCRAPER
# =============================================================================

def scrape_all():
    """
    Main scraping function that tries Playwright first,
    then falls back to FlareSolverr if blocked.
    """
    # Create output folder
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)

    print(f"Output folder: {output_path.absolute()}")
    print(f"Total URLs to scrape: {len(URLS)}")
    print("-" * 50)

    if not URLS:
        print("No URLs in URLS list. Please add URLs to scrape.")
        return

        # Check for duplicates
    duplicates = check_duplicates(URLS)
    if duplicates:
        print(f"Duplicates found: {duplicates}")
    else:
        print("No duplicates found.")

    results = {
        "success": [],
        "failed": [],
        "fallback_used": [],
        "scraperapi_used": []
    }

    for idx, url in enumerate(URLS):
        print(f"[{idx + 1}/{len(URLS)}] Scraping: {url}")

        filename = generate_filename(url, idx)
        output_file = output_path / filename

        html_content = ""
        method_used = ""
        success = False

        try:
            # Try Playwright first
            print(f"    [Playwright] Attempting...")
            html_content = scrape_with_playwright(url)
            print(f"    [Playwright] Got content ({len(html_content)} chars)")

            # Check if blocked by Cloudflare/CAPTCHA
            if is_blocked(html_content, debug=True):
                print(f"    [Playwright] ⚠️ BLOCKED - Cloudflare/CAPTCHA detected")
                html_content = ""  # Clear blocked content
            else:
                # Validate content quality before marking as success
                cleaned = clean_html(html_content)
                if has_content(cleaned):
                    html_content = cleaned
                    method_used = "playwright"
                    success = True
                else:
                    print(f"    [Playwright] ⚠️ No meaningful content after cleaning")
                    html_content = ""

        except Exception as e:
            print(f"    [Playwright] Error: {e}")

        # Fallback 1: Try FlareSolverr if Playwright failed or was blocked
        if not success:
            print(f"    [FlareSolverr] Trying fallback...")
            html_content = scrape_with_flaresolverr(url)

            if html_content and len(html_content) > 1000:
                print(f"    [FlareSolverr] Got content ({len(html_content)} chars)")

                # Check if FlareSolverr content is also blocked
                if is_blocked(html_content):
                    print(f"    [FlareSolverr] ⚠️ BLOCKED - Still blocked")
                    html_content = ""
                else:
                    # Validate content quality before marking as success
                    cleaned = clean_html(html_content)
                    if has_content(cleaned):
                        html_content = cleaned
                        method_used = "flaresolverr"
                        success = True
                    else:
                        print(f"    [FlareSolverr] ⚠️ No meaningful content after cleaning")
                        html_content = ""
            else:
                print(f"    [FlareSolverr] Failed to get content")

        # Fallback 2: Try ScraperAPI if FlareSolverr also failed
        if not success:
            print(f"    [ScraperAPI] Trying final fallback...")
            html_content = scrape_with_scraperapi(url)

            if html_content and len(html_content) > 1000:
                print(f"    [ScraperAPI] Got content ({len(html_content)} chars)")

                # Check if ScraperAPI content is also blocked
                if is_blocked(html_content):
                    print(f"    [ScraperAPI] ⚠️ BLOCKED - Still blocked")
                    html_content = ""
                else:
                    # Validate content quality before marking as success
                    cleaned = clean_html(html_content)
                    if has_content(cleaned):
                        html_content = cleaned
                        method_used = "scraperapi"
                        success = True
                    else:
                        print(f"    [ScraperAPI] ⚠️ No meaningful content after cleaning")
                        html_content = ""
            else:
                print(f"    [ScraperAPI] Failed to get content")

        # Save content if successfully retrieved and not blocked
        if success and html_content:
            # Content is already cleaned and validated
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"    ✓ [Saved] {filename} (method: {method_used})")

            if method_used == "playwright":
                results["success"].append(url)
            elif method_used == "flaresolverr":
                results["fallback_used"].append(url)
            elif method_used == "scraperapi":
                results["scraperapi_used"].append(url)
        else:
            print(f"    ✗ [Failed] All methods failed or blocked")
            results["failed"].append(url)

        print()

    # Summary
    print("=" * 50)
    print("SCRAPE SUMMARY")
    print("=" * 50)
    print(f"Total URLs: {len(URLS)}")
    print(f"Success (Playwright): {len(results['success'])}")
    print(f"Fallback used (FlareSolverr): {len(results['fallback_used'])}")
    print(f"ScraperAPI used: {len(results['scraperapi_used'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"\nTotal Successful: {len(results['success']) + len(results['fallback_used']) + len(results['scraperapi_used'])}")

    if results["failed"]:
        print("\nFailed URLs:")
        for url in results["failed"]:
            print(f"  - {url}")


if __name__ == "__main__":
    scrape_all()
