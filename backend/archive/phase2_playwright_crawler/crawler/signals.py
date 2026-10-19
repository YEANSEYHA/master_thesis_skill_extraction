import hashlib
import re


def content_hash(html: str) -> str:
    """
    Hash the meaningful content of the page, ignoring dynamic elements
    like scripts, timestamps, session tokens, and whitespace.
    """
    # Remove script/style tags and their content
    cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)

    # Remove common dynamic attributes (csrf tokens, nonces, timestamps)
    cleaned = re.sub(r'(csrf|nonce|token|timestamp|session)[^"]*"[^"]*"', '', cleaned, flags=re.IGNORECASE)

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def is_blocked(html: str) -> bool:
    """
    Detect if page is blocked by anti-bot protection.

    Returns:
        True if blocked/error, False if page loaded successfully
    """

    if not html:
        print("[BLOCK] Empty HTML")
        return True

    # Very short pages are almost always error/challenge pages
    if len(html) < 1000:
        print(f"[BLOCK] Too short ({len(html)} chars)")
        return True

    text = html.lower()

    # ========================================
    # Cloudflare challenges
    # ========================================
    cloudflare_indicators = [
        "checking your browser",
        "verify you are human",
        "enable javascript and cookies",
        "cf-browser-verification",
        "cf-challenge-running",
        "cf-challenge",
        "cf_chl_opt",
        "managed challenge",
    ]

    for phrase in cloudflare_indicators:
        if phrase in text:
            print(f"[BLOCK] Cloudflare: '{phrase}'")
            return True

    # ========================================
    # Other anti-bot services
    # ========================================
    antibot_indicators = [
        "datadome",                     # Datadome
        "perimeterx",                   # PerimeterX
        "px-captcha",                   # PerimeterX captcha
        "hcaptcha.com",                 # hCaptcha
        "recaptcha/api",                # reCAPTCHA
        "are you a robot",              # Generic bot check
        "bot protection",               # Generic
        "please complete the security check",  # Generic
    ]

    for phrase in antibot_indicators:
        if phrase in text:
            print(f"[BLOCK] Anti-bot: '{phrase}'")
            return True

    # ========================================
    # Server error pages
    # ========================================
    server_errors = [
        "java.lang.nullpointerexception",
        "an error occured in the bean",
        "stack trace is",
        "internal server error",
    ]

    for phrase in server_errors:
        if phrase in text:
            print(f"[BLOCK] Server error: '{phrase}'")
            return True

    # ========================================
    # HTTP error pages (only if page is small)
    # ========================================
    if len(html) < 5000:
        http_errors = [
            "403 forbidden",
            "access denied",
            "429 too many requests",
            "503 service unavailable",
            "502 bad gateway",
            "error 403",
            "error 429",
            "error 503",
        ]

        for error in http_errors:
            if error in text:
                print(f"[BLOCK] HTTP error: '{error}'")
                return True

    print(f"[ALLOW] Valid page - {len(html):,} chars")
    return False
