import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from llm import ask_model
from db.job_sites import get_connection, get_job_sites_by_country, insert_job_sites
from config import DB_TABLE
from pipeline.validation import validate_job_sites, print_validation_report
from pipeline.verification import verify_job_sites_with_llm, print_verification_report, save_results

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))  # single source: backend/.env

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

# How many days before re-running discovery for a country
DISCOVERY_FRESHNESS_DAYS = 30


class SearchRateLimitError(Exception):
    """Raised when all search providers hit rate limits."""
    pass


EXCLUDED_DOMAINS = {
    # Global job platforms
    "linkedin.com", "indeed.com", "glassdoor.com", "monster.com",
    "ziprecruiter.com", "careerbuilder.com", "simplyhired.com",
    # Social media
    "facebook.com", "instagram.com", "twitter.com",
    "x.com", "tiktok.com", "telegram.org", "youtube.com",
    # NGO / international org job boards
    "reliefweb.int", "devjobs.org", "unjobs.org", "impactpool.org",
    "undp.org", "unicef.org", "who.int", "worldbank.org",
    "adb.org", "ifc.org", "unfpa.org", "unops.org",
    "devex.com", "idealist.org", "ethicaljobs.com.au",
    # Search engines / aggregators
    "google.com", "bing.com", "yahoo.com", "baidu.com",
    "jooble.org", "talent.com", "careerjet.com",
    # General / non-job
    "wikipedia.org", "reddit.com", "quora.com",
    "medium.com", "blogspot.com", "wordpress.com",
}

# Keywords that indicate a specific company career page (not a job portal)
COMPANY_CAREER_KEYWORDS = {
    "hotel", "resort", "bank", "hospital", "clinic", "school",
    "university", "airline", "insurance", "telecom", "brewery",
    "restaurant", "cafe", "casino", "mall", "supermarket",
}


def get_keywords_prompt(country: str) -> str:
    """Generate prompt to get search keywords in English + local language."""
    return f"""
TASK: Generate search queries to discover local job portal websites in {country}.

I will provide 10 ENGLISH queries. You MUST generate exactly 5 LOCAL LANGUAGE queries.

THE 10 ENGLISH QUERIES ARE ALREADY DECIDED (do not change them):
1. "job portal {country}"
2. "job board {country}"
3. "{country} recruitment website"
4. "best job websites {country} 2025"
5. "top 10 job sites {country}"
6. "top hiring platforms {country}"
7. "{country} career website"
8. "{country} vacancy website"
9. "part time job portal {country}"
10. "{country} classifieds jobs"

YOUR TASK — Generate exactly 5 queries in the LOCAL LANGUAGE of {country}:
- Use the correct native script (e.g. Khmer script for Cambodia, Thai script for Thailand)
- Each query must mean something DIFFERENT:
  1. "[job website] [country]" — general job portal search
  2. "[find work] [capital city]" — city-specific job search
  3. "[hiring/recruitment website]" — recruitment platform search
  4. "[job vacancy] [country]" — vacancy listing search
  5. "[online jobs] [country]" — online/remote job search

Also provide:
- The ISO 3166-1 alpha-2 country code (e.g. "kh" for Cambodia)
- The primary language code (e.g. "km" for Khmer)
- The local Google domain (e.g. "google.com.kh")

OUTPUT FORMAT (JSON only, no other text):
{{
    "country": "{country}",
    "country_code": "xx",
    "language_code": "xx",
    "google_domain": "google.com.xx",
    "local_queries": [
        "local language query 1",
        "local language query 2",
        "local language query 3",
        "local language query 4",
        "local language query 5"
    ]
}}
"""


# Fixed English queries — these don't need LLM generation
def get_english_queries(country: str) -> list[str]:
    """Return hardcoded English search queries for a country."""
    return [
        f"job portal {country}",
        f"job board {country}",
        f"{country} recruitment website",
        f"best job websites {country} 2025",
        f"top 10 job sites {country}",
        f"top hiring platforms {country}",
        f"{country} career website",
        f"{country} vacancy website",
        f"part time job portal {country}",
        f"{country} classifieds jobs",
    ]


def generate_search_keywords(country: str, model_choice: str = "gpt") -> dict:
    """
    Step 1: Use LLM to generate local language search keywords for a country.
    English queries are hardcoded for consistency. LLM only generates local language queries.
    model_choice: "gpt" | "claude" | "glm" | "ollama"
    Returns dict with country_code, language_code, google_domain, and queries.
    """
    print(f"\n📋 STEP 1: Generating search keywords for {country} (model: {model_choice})...")

    # Hardcoded English queries — always consistent
    english_queries = get_english_queries(country)
    print(f"   ✅ {len(english_queries)} English queries (hardcoded)")

    # LLM generates only local language queries
    prompt = get_keywords_prompt(country)
    response = ask_model(prompt, model_choice=model_choice)

    # Parse JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())

            # Fix language codes not supported by SerpAPI
            # SerpAPI uses ISO 639-1 codes (e.g. "zh-TW"), not script subtags (e.g. "zh-Hant")
            LANG_CODE_FIXES = {
                "zh-HK": "zh-TW",    # Hong Kong → Traditional Chinese
                "zh-MO": "zh-TW",    # Macau → Traditional Chinese
                "zh-Hant": "zh-TW",  # Traditional Chinese (generic)
                "zh-Hans": "zh-CN",  # Simplified Chinese (generic)
                "zh-TW": "zh-TW",    # Already correct
                "zh-CN": "zh-CN",    # Already correct
            }
            lang = result.get("language_code", "")
            if lang in LANG_CODE_FIXES:
                result["language_code"] = LANG_CODE_FIXES[lang]
            elif lang.startswith("zh"):
                # Catch any other zh- variant
                result["language_code"] = "zh-TW"

            local_queries = result.get("local_queries", [])
            print(f"   ✅ {len(local_queries)} local language queries (LLM-generated)")

            # Merge: English first, then local language
            all_queries = english_queries + local_queries
            result["queries"] = all_queries

            print(f"   Country code: {result.get('country_code')}")
            print(f"   Language: {result.get('language_code')}")
            print(f"   Google domain: {result.get('google_domain')}")
            print(f"\n   Queries:")
            for i, q in enumerate(all_queries, 1):
                label = "EN" if i <= len(english_queries) else "LOCAL"
                print(f"   {i:2d}. [{label}] {q}")
            return result
        except json.JSONDecodeError:
            print("   ❌ Failed to parse LLM response")

    # Fallback: use English queries only if LLM fails
    print("   ⚠️ LLM failed — using English queries only")
    return {
        "country": country,
        "country_code": "",
        "language_code": "en",
        "google_domain": "google.com",
        "queries": english_queries,
    }


# ============================================================
# Step 2: Search via Google Custom Search (primary) + SerpAPI (fallback)
# ============================================================

# Track which providers have hit rate limits during this pipeline run
_rate_limited_providers = set()


def search_google_custom(query: str, country_code: str, language_code: str) -> list[dict]:
    """Run a single search query via Google Custom Search JSON API."""
    if "google" in _rate_limited_providers:
        return []

    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX or GOOGLE_SEARCH_CX == "YOUR_GOOGLE_CX_HERE":
        print("   ⚠️  Google Custom Search not configured, skipping")
        _rate_limited_providers.add("google")
        return []

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "gl": country_code,
        "lr": f"lang_{language_code}",
        "num": 10,  # max 10 per request for free tier
    }

    try:
        response = httpx.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=30)
        if response.status_code == 429:
            print("   ⚠️  Google Custom Search rate limit hit — switching to SerpAPI")
            _rate_limited_providers.add("google")
            return []
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        # Normalize to match SerpAPI format
        return [{"link": item.get("link", ""), "title": item.get("title", ""), "snippet": item.get("snippet", "")} for item in items]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("   ⚠️  Google Custom Search rate limit hit — switching to SerpAPI")
            _rate_limited_providers.add("google")
            return []
        print(f"   ❌ Google Custom Search error for '{query}': {e}")
        return []
    except Exception as e:
        print(f"   ❌ Google Custom Search error for '{query}': {e}")
        return []


def search_serpapi(query: str, country_code: str, language_code: str, google_domain: str) -> list[dict]:
    """Run a single search query via SerpAPI (fallback provider)."""
    if "serpapi" in _rate_limited_providers:
        return []

    params = {
        "engine": "google",
        "q": query,
        "google_domain": google_domain,
        "gl": country_code,
        "hl": language_code,
        "num": 20,
        "api_key": SERPAPI_KEY,
    }

    try:
        response = httpx.get("https://serpapi.com/search", params=params, timeout=30)
        if response.status_code == 429:
            print("   ⚠️  SerpAPI rate limit hit")
            _rate_limited_providers.add("serpapi")
            return []
        response.raise_for_status()
        data = response.json()
        return data.get("organic_results", [])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("   ⚠️  SerpAPI rate limit hit")
            _rate_limited_providers.add("serpapi")
            return []
        print(f"   ❌ SerpAPI error for '{query}': {e}")
        return []
    except Exception as e:
        print(f"   ❌ SerpAPI error for '{query}': {e}")
        return []


def search_with_fallback(query: str, country_code: str, language_code: str, google_domain: str) -> list[dict]:
    """Search using Google Custom Search first, fall back to SerpAPI if rate limited."""
    # Check if all providers are exhausted
    if "google" in _rate_limited_providers and "serpapi" in _rate_limited_providers:
        raise SearchRateLimitError("All search providers (Google Custom Search & SerpAPI) have hit rate limits. Please try again later.")

    # Try Google Custom Search first
    if "google" not in _rate_limited_providers:
        results = search_google_custom(query, country_code, language_code)
        if results:
            return results
        # If empty but not rate limited, still try SerpAPI for better coverage
        if "google" not in _rate_limited_providers:
            return results  # genuinely no results

    # Fallback to SerpAPI
    if "serpapi" not in _rate_limited_providers:
        results = search_serpapi(query, country_code, language_code, google_domain)
        if results:
            return results
        if "serpapi" in _rate_limited_providers and "google" in _rate_limited_providers:
            raise SearchRateLimitError("All search providers (Google Custom Search & SerpAPI) have hit rate limits. Please try again later.")

    return []


def extract_domain(url: str) -> str:
    """Extract clean domain from URL (e.g. 'https://www.camhr.com/jobs' → 'camhr.com')."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def search_job_sites(keywords: dict) -> list[dict]:
    """
    Step 2: Run all search queries via SerpAPI and collect unique domains.

    Args:
        keywords: Output from Step 1 (generate_search_keywords)

    Returns:
        List of unique site dicts: [{"url": ..., "name": ..., "domain": ...}, ...]
    """
    queries = keywords.get("queries", [])
    country_code = keywords.get("country_code", "")
    language_code = keywords.get("language_code", "en").split(",")[0].strip()
    google_domain = keywords.get("google_domain", "google.com")
    country = keywords.get("country", "")

    # Reset rate limit tracking for each discovery run
    _rate_limited_providers.clear()

    print(f"\n🔍 STEP 2: Searching ({len(queries)} queries)...")
    print(f"   Primary: Google Custom Search | Fallback: SerpAPI")
    print(f"   Google domain: {google_domain}")
    print(f"   Country: {country_code} | Language: {language_code}")

    # Track unique domains to avoid duplicates
    seen_domains = {}  # domain → site info

    for i, query in enumerate(queries, 1):
        print(f"\n   [{i}/{len(queries)}] Searching: {query}")

        results = search_with_fallback(query, country_code, language_code, google_domain)
        print(f"   → {len(results)} results")

        for result in results:
            url = result.get("link", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            domain = extract_domain(url)

            if not domain:
                continue

            # Skip global platforms (check if domain ends with any global platform)
            if any(domain == g or domain.endswith("." + g) for g in EXCLUDED_DOMAINS):
                continue

            # Skip Google/search engine results
            if "google." in domain or "bing." in domain or "yahoo." in domain:
                continue

            # Skip specific company career pages (hotels, banks, etc.)
            domain_lower = domain.lower()
            if any(kw in domain_lower for kw in COMPANY_CAREER_KEYWORDS):
                continue

            # Skip URLs that are just /careers or /jobs pages on a non-portal site
            path = urlparse(url).path.lower()
            if path in ("/careers", "/career", "/jobs", "/join-us", "/work-with-us"):
                continue

            # Merge subdomains into root domain (employer.fastjobs.sg → fastjobs.sg)
            parts = domain.split(".")
            country_tlds = ("kh", "my", "sg", "th", "vn", "ph", "la", "mm", "hk", "tw", "id", "bn", "tl")
            second_level_tlds = ("com", "co", "org", "gov", "net", "edu", "ac")
            if len(parts) >= 3 and parts[-1] in country_tlds and parts[-2] in second_level_tlds:
                # e.g. sub.example.com.sg → example.com.sg, mycareersfuture.gov.sg → mycareersfuture.gov.sg
                root_domain = ".".join(parts[-3:])
            elif len(parts) > 2 and parts[-1] in country_tlds:
                # e.g. employer.fastjobs.sg → fastjobs.sg
                root_domain = ".".join(parts[-2:])
            elif len(parts) > 2:
                # e.g. sub.example.com → example.com
                root_domain = ".".join(parts[-2:])
            else:
                root_domain = domain

            # Build URL: prefer www. prefix since many sites require it
            original_domain = urlparse(url).netloc.lower()
            if original_domain.startswith("www."):
                site_url = f"https://www.{root_domain}"
            else:
                site_url = f"https://{root_domain}"

            # Store first occurrence, track which queries found it
            if root_domain not in seen_domains:
                seen_domains[root_domain] = {
                    "url": site_url,
                    "domain": root_domain,
                    "name": title,
                    "snippet": snippet,
                    "found_in_queries": 1,
                    "country": country,
                }
            else:
                seen_domains[root_domain]["found_in_queries"] += 1

    # Sort by how many queries found this domain (more = more relevant)
    sites = sorted(seen_domains.values(), key=lambda x: x["found_in_queries"], reverse=True)

    print(f"\n   {'='*50}")
    print(f"   📊 Found {len(sites)} unique domains (after filtering)")
    print(f"   {'='*50}")
    for i, site in enumerate(sites[:20], 1):
        print(f"   {i:2d}. {site['domain']} (found in {site['found_in_queries']} queries)")
        print(f"       {site['name'][:60]}")

    return sites


# ============================================================
# Discovery Log — track when each country was last discovered
# ============================================================

def create_discovery_log_table():
    """Create discovery_log table if not exists."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovery_log (
                country TEXT PRIMARY KEY,
                last_discovered_at TIMESTAMP DEFAULT NOW(),
                sites_found INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"   ❌ Error creating discovery_log table: {e}")
    finally:
        if conn:
            conn.close()


def get_discovery_status(country: str) -> dict | None:
    """Check when a country was last discovered. Returns None if never."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT country, last_discovered_at, sites_found FROM discovery_log WHERE country = %s;",
            (country,)
        )
        row = cursor.fetchone()
        cursor.close()
        if row:
            return {"country": row[0], "last_discovered_at": row[1], "sites_found": row[2]}
        return None
    except Exception as e:
        print(f"   ❌ Error checking discovery status: {e}")
        return None
    finally:
        if conn:
            conn.close()


def update_discovery_log(country: str, sites_found: int):
    """Update the discovery log after a successful run."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO discovery_log (country, last_discovered_at, sites_found)
            VALUES (%s, NOW(), %s)
            ON CONFLICT (country) DO UPDATE SET
                last_discovered_at = NOW(),
                sites_found = EXCLUDED.sites_found;
        """, (country, sites_found))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"   ❌ Error updating discovery log: {e}")
    finally:
        if conn:
            conn.close()


def get_all_discovery_status() -> list:
    """Get discovery status for all countries."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT country, last_discovered_at, sites_found FROM discovery_log ORDER BY last_discovered_at DESC;")
        rows = cursor.fetchall()
        cursor.close()
        return [{"country": r[0], "last_discovered_at": r[1], "sites_found": r[2]} for r in rows]
    except Exception as e:
        print(f"   ❌ Error fetching discovery log: {e}")
        return []
    finally:
        if conn:
            conn.close()


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(country: str, force_refresh: bool = False, model_choice: str = "ollama"):
    """
    Full discovery pipeline for a country.

    model_choice: "gpt" | "claude" | "glm" | "ollama"

    1. Check DB — skip discovery if recent (< DISCOVERY_FRESHNESS_DAYS)
    2. LLM generates search keywords
    3. SerpAPI searches for job sites
    4. Filter & deduplicate
    5. Compare with DB — only process new URLs
    6. (TODO) Validate URLs
    7. (TODO) LLM verify HTML
    8. (TODO) Save confirmed sites to DB
    """
    print(f"\n{'#'*60}")
    print(f"# Processing: {country}")
    print('#'*60)

    # --- Check discovery log ---
    status = get_discovery_status(country)

    if status and not force_refresh:
        last_run = status["last_discovered_at"]
        days_ago = (datetime.now() - last_run).days
        print(f"\n   📅 Last discovered: {last_run.strftime('%Y-%m-%d')} ({days_ago} days ago)")
        print(f"   📊 Sites in DB: {status['sites_found']}")

        if days_ago < DISCOVERY_FRESHNESS_DAYS:
            print(f"   ✅ Fresh (< {DISCOVERY_FRESHNESS_DAYS} days) — skipping discovery")
            existing_sites = get_job_sites_by_country(country)
            print(f"   📋 Returning {len(existing_sites)} sites from DB")
            return existing_sites
        else:
            print(f"   ⏰ Stale (> {DISCOVERY_FRESHNESS_DAYS} days) — re-running discovery")
    elif force_refresh:
        print(f"\n   🔄 Force refresh requested — re-running discovery")
    else:
        print(f"\n   🆕 First run for {country} — starting full discovery")

    # --- Step 1: Generate search keywords ---
    keywords = generate_search_keywords(country, model_choice=model_choice)

    if not keywords:
        print("   ❌ No keywords generated, skipping")
        return []

    keywords_file = f"output/{country}_search_keywords.json"
    with open(keywords_file, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)
    print(f"\n   💾 Saved: {keywords_file}")

    # --- Step 2-3: Search via SerpAPI + filter ---
    discovered_sites = search_job_sites(keywords)

    if not discovered_sites:
        print("   ❌ No sites discovered")
        return []

    discovered_file = f"output/{country}_discovered_sites.json"
    with open(discovered_file, "w", encoding="utf-8") as f:
        json.dump(discovered_sites, f, indent=2, ensure_ascii=False)
    print(f"\n   💾 Saved: {discovered_file}")

    # # --- Compare with DB: find new sites ---
    # existing_sites = get_job_sites_by_country(country)
    # existing_urls = {site["url"] for site in existing_sites}
    # new_sites = [s for s in discovered_sites if s["url"] not in existing_urls]
    #
    # print(f"\n   📊 Discovery summary:")
    # print(f"      Total discovered: {len(discovered_sites)}")
    # print(f"      Already in DB:    {len(discovered_sites) - len(new_sites)}")
    # print(f"      New sites:        {len(new_sites)}")
    #
    # # --- Update discovery log ---
    # update_discovery_log(country, len(discovered_sites))

    new_sites = discovered_sites

    print(f"\n   📊 Discovery summary:")
    print(f"      Total discovered: {len(discovered_sites)}")

    # --- Step 4: Validate URLs ---
    if new_sites:
        print(f"\n🔗 STEP 4: Validating {len(new_sites)} URLs...")
        valid_sites, invalid_sites = asyncio.run(validate_job_sites(new_sites))
        print_validation_report(valid_sites, invalid_sites)
    else:
        valid_sites = []
        invalid_sites = []

    # --- Step 5: LLM verify HTML content ---
    if valid_sites:
        print(f"\n🤖 STEP 5: LLM verification of {len(valid_sites)} valid sites...")
        confirmed_sites, rejected_sites = asyncio.run(
            verify_job_sites_with_llm(valid_sites, model_choice=model_choice, country=country)
        )
        print_verification_report(confirmed_sites, rejected_sites)
        save_results(confirmed_sites, rejected_sites, country)
    else:
        confirmed_sites = []
        rejected_sites = []

    # --- Print final summary ---
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS: {country.upper()}")
    print(f"{'='*60}")

    if confirmed_sites:
        print(f"\n✅ CONFIRMED JOB SITES ({len(confirmed_sites)}):")
        print(f"   {'Site':<30} {'Type':<20} {'Confidence':<12} Reason")
        print(f"   {'-'*90}")
        for site in confirmed_sites:
            v = site["verification"]
            name = site["domain"][:28]
            print(f"   {name:<30} {v.site_type:<20} {v.confidence:.0%}{'':>8} {v.reason[:60]}")

    if rejected_sites:
        print(f"\n❌ REJECTED SITES ({len(rejected_sites)}):")
        print(f"   {'Site':<30} {'Type':<20} {'Confidence':<12} Reason")
        print(f"   {'-'*90}")
        for site in rejected_sites:
            v = site["verification"]
            name = site["domain"][:28]
            print(f"   {name:<30} {v.site_type:<20} {v.confidence:.0%}{'':>8} {v.reason[:60]}")

    if invalid_sites:
        print(f"\n⚠️ FAILED URL VALIDATION ({len(invalid_sites)}):")
        for site in invalid_sites:
            print(f"   {site['domain']:<30} [{site['error_type']}] {site['error']}")

    print(f"\n{'='*60}")
    print(f"   Discovered: {len(discovered_sites)} | Valid URLs: {len(valid_sites)} | Confirmed: {len(confirmed_sites)} | Rejected: {len(rejected_sites)} | Invalid: {len(invalid_sites)}")
    print(f"{'='*60}")

    # # --- Save confirmed sites to DB ---
    # if confirmed_sites:
    #     print(f"\n   💾 Saving {len(confirmed_sites)} confirmed job sites to DB...")
    #     sites_to_insert = [
    #         {
    #             "url": site.get("final_url", site["url"]),
    #             "name": site.get("name"),
    #             "type": site["verification"].site_type,
    #             "description": site.get("snippet") or site.get("description"),
    #             "confidence": site["verification"].confidence,
    #             "reason": site["verification"].reason,
    #             "features_found": site["verification"].features_found,
    #         }
    #         for site in confirmed_sites
    #     ]
    #     inserted = insert_job_sites(sites_to_insert, country=country)
    #     print(f"   ✅ DB write complete: {inserted} rows inserted/updated")
    # else:
    #     print("\n   ℹ️ No confirmed job sites to save to DB")

    return discovered_sites


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Ensure tables exist
    create_discovery_log_table()

    # Parse command line args
    countries = sys.argv[1:] if len(sys.argv) > 1 else ["singapore"]
    force = "--refresh" in countries
    countries = [c for c in countries if not c.startswith("--")]

    for country in countries:
        run_pipeline(country, force_refresh=force)
