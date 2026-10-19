import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def _same_domain(url: str, domain: str) -> bool:
    """Check if URL belongs to the same domain (handles www/non-www)"""
    netloc = urlparse(url).netloc
    # Exact match
    if netloc == domain:
        return True
    # www.example.com vs example.com
    if netloc == f"www.{domain}" or domain == f"www.{netloc}":
        return True
    return False


def _clean_url(url: str) -> str:
    """Remove fragment and trailing whitespace"""
    return url.split("#")[0].strip()


# Skip these URL patterns — they trigger downloads, popups, or external apps
_SKIP_EXTENSIONS = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.exe', '.dmg', '.apk')
_SKIP_DOMAINS = ('t.me', 'telegram.me', 'wa.me', 'viber.com', 'line.me', 'facebook.com/sharer',
                  'twitter.com/intent', 'linkedin.com/sharing')
_SKIP_PATHS = ('/print', '/rss', '/feed', '/xml')


def _should_skip_url(url: str) -> bool:
    """Skip URLs that trigger downloads, external apps, or non-HTML content"""
    lower = url.lower()
    parsed = urlparse(lower)

    # Skip file downloads
    if any(parsed.path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return True

    # Skip external app deeplinks
    if any(domain in lower for domain in _SKIP_DOMAINS):
        return True

    # Skip print/RSS/feed pages
    if any(path in parsed.path for path in _SKIP_PATHS):
        return True

    # Skip print/qrcode query parameters (e.g., ?print=true&qrcode=true)
    query = parsed.query
    if any(param in query for param in ('print=', 'qrcode=', 'download=', 'export=')):
        return True

    # Skip RSS/XML feeds
    if parsed.path.endswith('.xml'):
        return True

    return False


def extract_links(html: str, base_url: str, domain: str):
    soup = BeautifulSoup(html, "lxml")
    links = set()

    # 1. Standard <a href="..."> links
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript", "mailto", "tel", "#", "data:", "blob:", "intent:")):
            continue
        absolute = _clean_url(urljoin(base_url, href))
        if _should_skip_url(absolute):
            continue
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https") and _same_domain(absolute, domain):
            links.add(absolute)

    # 2. Elements with data-href or data-url attributes (common in job cards)
    for attr in ["data-href", "data-url", "data-link"]:
        for el in soup.find_all(attrs={attr: True}):
            href = el[attr].strip()
            if not href:
                continue
            absolute = _clean_url(urljoin(base_url, href))
            parsed = urlparse(absolute)
            if parsed.scheme in ("http", "https") and _same_domain(absolute, domain):
                links.add(absolute)

    # 3. onclick="window.location='...'" or onclick="location.href='...'"
    onclick_pattern = re.compile(r"""(?:window\.)?location(?:\.href)?\s*=\s*['"]([^'"]+)['"]""")
    for el in soup.find_all(attrs={"onclick": True}):
        match = onclick_pattern.search(el["onclick"])
        if match:
            href = match.group(1).strip()
            absolute = _clean_url(urljoin(base_url, href))
            parsed = urlparse(absolute)
            if parsed.scheme in ("http", "https") and _same_domain(absolute, domain):
                links.add(absolute)

    return list(links)
