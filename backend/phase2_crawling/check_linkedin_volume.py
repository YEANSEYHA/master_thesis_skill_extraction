#!/usr/bin/env python3
"""Feasibility check ONLY — count how many IT jobs LinkedIn lists for a
location, using the public guest job-search endpoint (no login).

Purpose: decide whether a LinkedIn-Cambodia crawl is worth building BEFORE
writing a scraper. It fetches the filtered result pages and counts the job
cards; it does NOT store postings.

Caveats (read before running):
  - Scraping LinkedIn violates its Terms of Service. This tool makes a small
    number of polite, rate-limited requests to READ public result counts for
    an academic feasibility estimate; it is not an evasion tool and will stop
    on the first block (HTTP 429/999).
  - The guest endpoint caps guest results (~40 pages) and rate-limits quickly,
    so the count is a floor/estimate, not an exact total.
  - Filters: f_F=it  -> job function = Information Technology
             location=<text>, optional keywords=<title words>

    python check_linkedin_volume.py --location Cambodia
    python check_linkedin_volume.py --location Cambodia --keywords "software developer"
    python check_linkedin_volume.py --location "Phnom Penh" --max-pages 20
"""
import argparse
import sys
import time
from urllib.parse import urlencode

import requests

ENDPOINT = ("https://www.linkedin.com/jobs-guest/jobs/api/"
            "seeMoreJobPostings/search")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
CARD_MARKERS = ("job-search-card", "base-card")  # one <li> per posting


def count_cards(html: str) -> int:
    # each result is a card container; count the most frequent marker present
    return max(html.count(m) for m in CARD_MARKERS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count LinkedIn IT jobs for a location (feasibility only).")
    ap.add_argument("--location", default="Cambodia")
    ap.add_argument("--keywords", default="", help="optional title keywords")
    ap.add_argument("--function", default="it", help="LinkedIn job-function code (it = IT)")
    ap.add_argument("--max-pages", type=int, default=40, help="guest cap is ~40")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between requests (be polite)")
    args = ap.parse_args()

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})

    total, start, page = 0, 0, 0
    print(f"[linkedin] location={args.location!r} function={args.function!r} "
          f"keywords={args.keywords!r}\n")
    while page < args.max_pages:
        q = {"location": args.location, "f_F": args.function, "start": start}
        if args.keywords:
            q["keywords"] = args.keywords
        url = f"{ENDPOINT}?{urlencode(q)}"
        try:
            r = sess.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"[linkedin] request error: {e} — stopping"); break
        if r.status_code in (429, 999, 403):
            print(f"[linkedin] blocked (HTTP {r.status_code}) after {page} pages "
                  f"— LinkedIn rate-limited us. Count so far is a floor."); break
        if r.status_code != 200:
            print(f"[linkedin] HTTP {r.status_code} — stopping"); break
        n = count_cards(r.text)
        if n == 0:
            print(f"[linkedin] page {page}: 0 cards — reached the end."); break
        total += n
        page += 1
        start += 25
        print(f"[linkedin] page {page:>2} (start={start-25:>4}): +{n:>2} cards  → running total {total}")
        time.sleep(args.sleep)

    print("\n" + "=" * 52)
    print(f"LinkedIn IT jobs in {args.location!r}: ~{total}"
          + (" (hit guest/rate cap — true total may be higher)"
             if page >= args.max_pages else ""))
    print("=" * 52)
    if total < 1000:
        print("→ Below 1k. A LinkedIn-Cambodia crawl will NOT reach your target;\n"
              "  the local boards (bongthom/camhr/jobnet/jobify IT category) are\n"
              "  the higher-yield, more representative source.")


if __name__ == "__main__":
    main()
