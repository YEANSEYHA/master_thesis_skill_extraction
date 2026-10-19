import json
import re
from typing import Optional

import httpx
from config import (
    CLAUDE_MODEL,
    GLM_MODEL,
    GPT_MODEL,
    HTML_MAX_LENGTH,
    HTML_PROMPT_MAX_LENGTH,
    MIN_CONFIDENCE,
    REQUEST_TIMEOUT,
    USER_AGENT,
)
from llm.claude_client import ask_claude
from llm.glm_client import ask_glm
from llm.openai_client import ask_gpt
from llm.ollama_client import ask_ollama
from schemas import VerificationResult


# ============================================================
# HTTP + LLM Verification
# ============================================================

async def fetch_page_content(url: str, timeout: float = REQUEST_TIMEOUT) -> Optional[str]:
    """Fetch HTML content from URL for analysis."""
    headers = {"User-Agent": USER_AGENT}

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
            verify=True
        ) as client:
            response = await client.get(url, headers=headers)
            if response.status_code < 400:
                return response.text[:HTML_MAX_LENGTH]
    except Exception:
        pass

    return None


def get_html_verification_prompt(url: str, html_content: str, country: str = "") -> str:
    """Generate prompt for HTML-based verification."""
    truncated = html_content[:HTML_PROMPT_MAX_LENGTH] if html_content else "Unable to fetch"

    country_rule = f"""
6. Country relevance: This site MUST primarily serve job seekers in {country}.
   - Reject sites that are based in or primarily serve a DIFFERENT country (e.g. a Malaysia job portal found via a Singapore search)
   - Global platforms with a local presence (e.g. jobstreet.com serving Singapore) are OK
   - Set is_job_site to false if the site does not primarily serve {country}""" if country else ""

    return f"""
Analyze this website HTML and determine if it is a JOB SITE.

Target country: {country or "any"}
URL: {url}

HTML Content:
```
{truncated}
```

LOOK FOR:
1. Job listings (job titles, company names, salary, apply buttons)
2. Job search (search bar, filters for location/industry)
3. Categories like "Jobs", "Careers", "Vacancies"
4. Employer sections ("Post a Job", "Employer Login")
5. Job seeker features (CV upload, job alerts){country_rule}

OUTPUT JSON ONLY (no other text):
{{
    "is_job_site": true or false,
    "confidence": 0.0 to 1.0,
    "site_type": "job_portal" or "recruitment_agency" or "classifieds" or "not_job_site",
    "reason": "Brief explanation",
    "features_found": ["list", "of", "features"]
}}
"""


def get_metadata_verification_prompt(url: str, name: str, snippet: str, country: str = "") -> str:
    """Generate prompt for metadata-based verification (when HTML can't be fetched)."""

    country_rule = f"""
4. Country relevance: This site MUST primarily serve job seekers in {country}.
   - Reject sites based in or primarily serving a DIFFERENT country
   - Set is_job_site to false if it does not primarily serve {country}""" if country else ""

    return f"""
Analyze this website based on its URL, title, and description to determine if it is a JOB SITE.
Note: The actual page could not be fetched (likely behind Cloudflare/bot protection), so use only the metadata below.

Target country: {country or "any"}
URL: {url}
Title: {name}
Description: {snippet}

LOOK FOR clues in the title/description:
1. Job-related keywords (jobs, careers, vacancies, hiring, recruitment)
2. Job portal branding (e.g. "Find jobs", "Post a job", "Job board")
3. Classified ads that include jobs{country_rule}

OUTPUT JSON ONLY (no other text):
{{
    "is_job_site": true or false,
    "confidence": 0.0 to 1.0,
    "site_type": "job_portal" or "recruitment_agency" or "classifieds" or "not_job_site",
    "reason": "Brief explanation",
    "features_found": ["list", "of", "features"]
}}
"""


async def verify_single_site_llm(url: str, model_choice: str = "gpt", site_metadata: dict = None, country: str = "") -> VerificationResult:
    """Verify if URL is a job site using HTTP fetch + LLM analysis.

    model_choice: 'claude', 'gpt', or 'glm'
    """
    try:
        html_content = await fetch_page_content(url)

        if not html_content and site_metadata:
            # Fallback: use SerpAPI metadata when HTML can't be fetched (e.g. Cloudflare)
            prompt = get_metadata_verification_prompt(
                url,
                site_metadata.get("name", ""),
                site_metadata.get("snippet", ""),
                country=country
            )
        elif not html_content:
            return VerificationResult(
                url=url,
                is_job_site=False,
                confidence=0.0,
                site_type="unreachable",
                reason="Could not fetch page content",
                features_found=[]
            )
        else:
            prompt = get_html_verification_prompt(url, html_content, country=country)

        if model_choice == "claude":
            response = ask_claude(prompt, model=CLAUDE_MODEL)
        elif model_choice == "glm":
            response = ask_glm(prompt, model=GLM_MODEL)
        elif model_choice == "ollama":
            response = ask_ollama(prompt)
        else:
            response = ask_gpt(prompt, model=GPT_MODEL)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return VerificationResult(
                url=url,
                is_job_site=data.get("is_job_site", False),
                confidence=float(data.get("confidence", 0.0)),
                site_type=data.get("site_type", "unknown"),
                reason=data.get("reason", "No reason provided"),
                features_found=data.get("features_found", [])
            )

        return VerificationResult(
            url=url,
            is_job_site=False,
            confidence=0.0,
            site_type="parse_error",
            reason="Could not parse LLM response",
            features_found=[]
        )

    except json.JSONDecodeError as e:
        return VerificationResult(
            url=url,
            is_job_site=False,
            confidence=0.0,
            site_type="json_error",
            reason=f"JSON parse error: {str(e)}",
            features_found=[]
        )
    except Exception as e:
        return VerificationResult(
            url=url,
            is_job_site=False,
            confidence=0.0,
            site_type="error",
            reason=f"Verification error: {str(e)}",
            features_found=[]
        )


async def verify_job_sites_with_llm(
    sites: list[dict],
    model_choice: str = "gpt",
    min_confidence: float = MIN_CONFIDENCE,
    country: str = ""
) -> tuple[list[dict], list[dict]]:
    """Verify all sites using HTTP + LLM method."""
    model_name = {"claude": "Claude", "gpt": "GPT", "glm": "GLM", "ollama": "Ollama"}.get(model_choice, model_choice.upper())

    print(f"\n🤖 AI Agent Verification ({len(sites)} sites)...")
    print(f"   Method: HTTP + LLM")
    print(f"   Model: {model_name}")
    print(f"   Min Confidence: {min_confidence:.0%}")

    confirmed = []
    rejected = []

    for i, site in enumerate(sites, 1):
        url = site.get("final_url") or site.get("url")
        name = site.get("name", "Unknown")

        print(f"\n   [{i}/{len(sites)}] 🔍 Verifying: {name}")
        print(f"   URL: {url}")

        verification = await verify_single_site_llm(url, model_choice=model_choice, site_metadata=site, country=country)

        site_with_verification = {**site, "verification": verification}

        status = "✅" if verification.is_job_site else "❌"
        print(f"   {status} Result: {verification.site_type} (confidence: {verification.confidence:.0%})")
        print(f"   Reason: {verification.reason}")

        if verification.features_found:
            print(f"   Features: {', '.join(verification.features_found[:5])}")

        if verification.is_job_site and verification.confidence >= min_confidence:
            confirmed.append(site_with_verification)
        else:
            rejected.append(site_with_verification)

    return confirmed, rejected


# ============================================================
# Reports & Output
# ============================================================

def print_verification_report(confirmed: list[dict], rejected: list[dict]) -> None:
    """Print AI verification results."""
    total = len(confirmed) + len(rejected)
    separator = "=" * 60

    print(f"\n{separator}")
    print("🤖 AI AGENT VERIFICATION REPORT")
    print(separator)
    print(f"Total: {total} | Confirmed: {len(confirmed)} | Rejected: {len(rejected)}")

    if total > 0:
        print(f"Confirmation Rate: {len(confirmed) / total * 100:.1f}%")

    if confirmed:
        print(f"\n✅ CONFIRMED JOB SITES ({len(confirmed)}):")
        for site in confirmed:
            v = site["verification"]
            print(f"   • {site['name']}: {site['url']}")
            print(f"     Type: {v.site_type} | Confidence: {v.confidence:.0%}")
            if v.features_found:
                print(f"     Features: {', '.join(v.features_found[:5])}")

    if rejected:
        print(f"\n❌ REJECTED ({len(rejected)}):")
        for site in rejected:
            v = site["verification"]
            print(f"   • {site['name']}: {site['url']}")
            print(f"     Reason: {v.reason}")

    print(f"{separator}\n")


def save_results(confirmed: list[dict], rejected: list[dict], country: str) -> None:
    """Save results to JSON files in output/ directory."""
    def serialize(site: dict) -> dict:
        result = {k: v for k, v in site.items() if k not in ["verification", "validation"]}
        if "verification" in site:
            v = site["verification"]
            result["verification"] = {
                "is_job_site": v.is_job_site,
                "confidence": v.confidence,
                "site_type": v.site_type,
                "reason": v.reason,
                "features_found": v.features_found
            }
        return result

    filename = f"output/{country.lower()}_confirmed_job_sites.json"
    with open(filename, "w") as f:
        json.dump([serialize(s) for s in confirmed], f, indent=2)
    print(f"💾 Saved: {filename}")

    filename = f"output/{country.lower()}_rejected_sites.json"
    with open(filename, "w") as f:
        json.dump([serialize(s) for s in rejected], f, indent=2)
    print(f"💾 Saved: {filename}")
