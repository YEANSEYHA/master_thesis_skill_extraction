import asyncio
from urllib.parse import urlparse

import httpx

from config import REQUEST_TIMEOUT, USER_AGENT
from schemas import URLStatus, ValidationResult


def get_base_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def validate_url(url: str, timeout: float = REQUEST_TIMEOUT) -> ValidationResult:
    """Validate if a URL is accessible."""
    headers = {"User-Agent": USER_AGENT}

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
            verify=True
        ) as client:
            response = await client.get(url, headers=headers)

            if response.status_code < 400:
                return ValidationResult(
                    url=url,
                    status=URLStatus.VALID,
                    status_code=response.status_code,
                    final_url=str(response.url)
                )

            # Cloudflare challenge returns 403 — site exists but requires JS
            cf_mitigated = response.headers.get("cf-mitigated", "")
            if response.status_code == 403 and cf_mitigated == "challenge":
                return ValidationResult(
                    url=url,
                    status=URLStatus.VALID,
                    status_code=response.status_code,
                    final_url=str(response.url)
                )

            return ValidationResult(
                url=url,
                status=URLStatus.HTTP_ERROR,
                status_code=response.status_code,
                error_message=f"HTTP {response.status_code}"
            )

    except httpx.ConnectError as e:
        error_msg = str(e)
        if "Name or service not known" in error_msg or "nodename nor servname" in error_msg:
            return ValidationResult(
                url=url,
                status=URLStatus.DNS_ERROR,
                error_message="Domain does not exist"
            )
        return ValidationResult(url=url, status=URLStatus.DNS_ERROR, error_message=error_msg)

    except httpx.TimeoutException:
        return ValidationResult(url=url, status=URLStatus.TIMEOUT, error_message="Request timed out")

    except httpx.TooManyRedirects:
        return ValidationResult(url=url, status=URLStatus.REDIRECT_LOOP, error_message="Too many redirects")

    except Exception as e:
        return ValidationResult(url=url, status=URLStatus.UNKNOWN_ERROR, error_message=str(e))


async def validate_job_sites(sites: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate all job sites and separate into valid/invalid."""
    tasks = [validate_url(site["url"]) for site in sites]
    results = await asyncio.gather(*tasks)

    valid_sites = []
    invalid_sites = []

    for site, result in zip(sites, results):
        site_with_validation = {**site, "validation": result}

        if result.is_valid:
            site_with_validation["final_url"] = result.final_url
            valid_sites.append(site_with_validation)
        else:
            site_with_validation["error"] = result.error_message
            site_with_validation["error_type"] = result.status.value
            invalid_sites.append(site_with_validation)

    return valid_sites, invalid_sites


def print_validation_report(valid: list[dict], invalid: list[dict]) -> None:
    """Print a summary of validation results."""
    total = len(valid) + len(invalid)
    separator = "=" * 60

    print(f"\n{separator}")
    print("URL Validation Report")
    print(separator)
    print(f"Total: {total} | Valid: {len(valid)} | Invalid: {len(invalid)}")

    if total > 0:
        print(f"Success Rate: {len(valid) / total * 100:.1f}%")

    if valid:
        print(f"\n✅ VALID SITES ({len(valid)}):")
        for site in valid:
            print(f"   • {site['name']}: {site['url']}")

    if invalid:
        print(f"\n❌ INVALID SITES ({len(invalid)}):")
        for site in invalid:
            print(f"   • {site['name']}: {site['url']}")
            print(f"     Error: [{site['error_type']}] {site['error']}")

    print(f"{separator}\n")
