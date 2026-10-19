import json
import re
from typing import Optional, Dict
from .llm import ask_llm
from .pagination import clean_html


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """You are a web crawling assistant. Your task is to analyze a webpage and discover URL patterns to crawl ALL pages on this website.

Given a URL and its HTML, find:

1. PAGINATION PATTERNS - numbered page navigation
   Look for: page numbers (1 2 3 ... 47), next/prev links, ?page=N, /page/N
   Find the ACTUAL last page number from the HTML (e.g. "1 2 3 ... 47" means max=47)

2. CATEGORY/SECTION PATTERNS - links to different sections
   Look for: navigation menus, category lists, sidebar links, footer sitemap links
   Examples: /jobs/engineering, /jobs/marketing, /industry/banking

3. DETAIL PAGE PATTERNS - links to individual items
   Look for: job cards, article links, product cards with URLs containing IDs
   Examples: /job/12345, /partner/2480, /article/slug-name

Return a JSON array of discovered patterns. Each pattern should be one of:

For pagination (numbered pages):
{"type": "pagination", "pattern": "<URL with {n} for page number>", "max": <last_page_number>}

For categories/sections (list of specific URLs):
{"type": "section", "urls": ["<url1>", "<url2>", "<url3>"]}

If NOTHING useful is found, return:
{"action": "none"}

Examples of good responses:

{"patterns": [
  {"type": "pagination", "pattern": "/jobs?page={n}", "max": 47},
  {"type": "section", "urls": ["/jobs/phnom-penh", "/jobs/siem-reap", "/jobs/remote"]}
]}

{"patterns": [
  {"type": "pagination", "pattern": "/careers/page/{n}", "max": 12}
]}

{"action": "none"}

Rules:
- Return ONLY valid JSON, no explanation, no markdown
- For pagination max, use the ACTUAL last page number from HTML, do not guess low
- Only include section URLs that are INTERNAL to the same website
- Do not include URLs that are already obvious from normal link extraction (standard nav menus)
- Focus on URLs that reveal MORE content (categories, filters, archives)
/no_think"""


# --------------------------------------------------
# QWEN VERSION (LOCAL / FREE)
# --------------------------------------------------
def suggest_navigation_qwen(html: str, url: str) -> Dict:
    prompt = f"""
    {SYSTEM_PROMPT}

    URL:
    {url}

    HTML:
    {clean_html(html)[:15000]}

    Now return ONLY the JSON object with the discovered patterns (or
    {{"action": "none"}}), exactly as specified above. No explanation.
    """

    try:
        raw = ask_llm(prompt, timeout=120)
        print(f"[DEBUG] Navigator LLM response: {raw}")
        return _parse_llm_output(raw)

    except Exception as e:
        print("[NAV LLM ERROR]", e)
        return {"action": "none"}


# --------------------------------------------------
# SHARED PARSER
# --------------------------------------------------
def _parse_llm_output(raw: str) -> Dict:
    data = _extract_json(raw)

    if not data:
        return {"action": "none"}

    # Legacy format: single pagination pattern
    if data.get("action") == "pagination_pattern":
        pattern = data.get("pattern")
        max_pages = data.get("max", 5)
        if isinstance(pattern, str) and "{n}" in pattern:
            return {
                "action": "discovered",
                "pagination": [{"pattern": pattern, "max": int(max_pages)}],
                "sections": [],
            }

    # New format: multiple patterns
    if "patterns" in data:
        pagination = []
        sections = []

        for p in data["patterns"]:
            if p.get("type") == "pagination":
                pattern = p.get("pattern", "")
                if isinstance(pattern, str) and "{n}" in pattern:
                    pagination.append({
                        "pattern": pattern,
                        "max": int(p.get("max", 10)),
                    })
            elif p.get("type") == "section":
                urls = p.get("urls", [])
                if isinstance(urls, list):
                    sections.extend(urls)

        if pagination or sections:
            return {
                "action": "discovered",
                "pagination": pagination,
                "sections": sections,
            }

    return {"action": "none"}


# --------------------------------------------------
# JSON EXTRACTION (SAFE)
# --------------------------------------------------
def _extract_json(text: str) -> Optional[Dict]:
    text = text.strip()

    # Case 1: direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Case 2: embedded JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None
