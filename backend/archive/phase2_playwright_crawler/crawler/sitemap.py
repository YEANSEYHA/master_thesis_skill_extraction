import re
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup


def discover_sitemap_urls(start_url: str, domain: str) -> list:
    """
    Discover all URLs from sitemap.xml and robots.txt.
    Returns list of URLs belonging to the same domain.
    """
    base = f"{urlparse(start_url).scheme}://{domain}"
    sitemap_locations = set()
    all_urls = set()

    # 1. Check robots.txt for sitemap references
    try:
        resp = requests.get(f"{base}/robots.txt", timeout=10)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    sitemap_locations.add(sitemap_url)
    except Exception as e:
        print(f"[SITEMAP] robots.txt failed: {e}")

    # 2. Try common sitemap paths
    common_paths = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/sitemap.xml"]
    for path in common_paths:
        sitemap_locations.add(f"{base}{path}")

    # 3. Parse each sitemap
    parsed_sitemaps = set()
    while sitemap_locations:
        sitemap_url = sitemap_locations.pop()
        if sitemap_url in parsed_sitemaps:
            continue
        parsed_sitemaps.add(sitemap_url)

        urls, sub_sitemaps = _parse_sitemap(sitemap_url)
        all_urls.update(urls)
        sitemap_locations.update(sub_sitemaps - parsed_sitemaps)

    # Filter to same domain only
    filtered = []
    for url in all_urls:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if netloc == domain or netloc == f"www.{domain}" or domain == f"www.{netloc}":
            filtered.append(url)

    return filtered


def _parse_sitemap(sitemap_url: str) -> tuple:
    """
    Parse a sitemap XML file.
    Returns (set of page URLs, set of sub-sitemap URLs).
    """
    urls = set()
    sub_sitemaps = set()

    try:
        resp = requests.get(sitemap_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; thesis-crawler/1.0)"
        })
        if resp.status_code != 200:
            return urls, sub_sitemaps

        content = resp.text

        # Parse as XML
        soup = BeautifulSoup(content, "lxml-xml")

        # Sitemap index — contains <sitemap><loc>...</loc></sitemap>
        for sitemap_tag in soup.find_all("sitemap"):
            loc = sitemap_tag.find("loc")
            if loc and loc.text:
                sub_sitemaps.add(loc.text.strip())

        # URL set — contains <url><loc>...</loc></url>
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            if loc and loc.text:
                urls.add(loc.text.strip())

        print(f"[SITEMAP] {sitemap_url}: {len(urls)} URLs, {len(sub_sitemaps)} sub-sitemaps")

    except Exception as e:
        print(f"[SITEMAP] Failed to parse {sitemap_url}: {e}")

    return urls, sub_sitemaps
