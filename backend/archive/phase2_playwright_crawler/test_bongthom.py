"""One-site test run: how many pages can the improved crawler pull
from bongthom.com, and how fast. Usage:

    python test_bongthom.py            # 100 pages (quick test, ~5-10 min)
    python test_bongthom.py 2000       # full-depth run
"""
import sys
import asyncio
from main import crawl_multiple_sites, ensure_flaresolverr

MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

if not ensure_flaresolverr():
    print("⚠️  Continuing WITHOUT FlareSolverr — Cloudflare pages may be skipped.")

asyncio.run(crawl_multiple_sites(
    urls=["https://www.bongthom.com/"],
    max_pages=MAX_PAGES,
    concurrency=1,
))
