import re
import json
from bs4 import BeautifulSoup
from .llm import ask_llm


def clean_html(html: str) -> str:
    """Remove CSS styles and JavaScript from HTML"""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["style", "script"]):
        tag.decompose()

    for tag in soup.find_all(True):
        if tag.has_attr('style'):
            del tag['style']

    return str(soup)


def has_pagination_fast(html: str) -> bool:
    """
    BeautifulSoup pagination detection.
    Covers common pagination patterns across different frameworks.
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. CLASS-BASED DETECTION
    class_patterns = [
        r'pagination',
        r'pager',
        r'page-item',
        r'page-link',
        r'page-nav',
        r'page-number',
        r'paginator',
        r'page-list',
        r'nav-page',
        r'wp-pagenavi',
        r'pagenav',
        r'pagelinks',
        r'ant-pagination',
        r'el-pagination',
        r'mui-pagination',
        r'pagination-container',
        r'page-selector',
    ]

    for pattern in class_patterns:
        if soup.find(class_=re.compile(pattern, re.I)):
            return True

    # 2. ARIA-LABEL DETECTION
    aria_patterns = [
        r'pagination',
        r'page\s*navigation',
        r'pager',
        r'navigate\s*pages',
    ]

    for pattern in aria_patterns:
        if soup.find(attrs={'aria-label': re.compile(pattern, re.I)}):
            return True

    # 3. ROLE ATTRIBUTE DETECTION
    nav = soup.find(attrs={'role': 'navigation'})
    if nav and nav.find_all('a', string=re.compile(r'^\d+$')):
        return True

    # 4. Multiple sequential numeric links
    numeric_links = soup.find_all('a', string=re.compile(r'^\s*\d+\s*$'))
    if len(numeric_links) >= 3:
        numbers = []
        for link in numeric_links:
            text = link.get_text().strip()
            if text.isdigit():
                numbers.append(int(text))
        if _has_sequential(numbers):
            return True

    # 5. Multiple sequential numeric buttons
    numeric_buttons = soup.find_all('button', string=re.compile(r'^\s*\d+\s*$'))
    if len(numeric_buttons) >= 3:
        numbers = []
        for btn in numeric_buttons:
            text = btn.get_text().strip()
            if text.isdigit():
                numbers.append(int(text))
        if _has_sequential(numbers):
            return True

    # 6. NEXT/PREVIOUS LINK PATTERNS (no longer requires numeric links nearby)
    next_prev_patterns = [
        r'^\s*next\s*$',
        r'^\s*prev(ious)?\s*$',
        r'next\s*page',
        r'previous\s*page',
        r'next\s*»',
        r'«\s*prev',
        r'›',
        r'»',
    ]

    for pattern in next_prev_patterns:
        if soup.find('a', string=re.compile(pattern, re.I)):
            return True

    # 7. URL PATTERN DETECTION
    page_url_patterns = [
        r'[?&]page=\d+',
        r'[?&]p=\d+',
        r'/page/\d+',
        r'/p/\d+',
        r'[?&]pg=\d+',
        r'[?&]pag=\d+',
        r'[?&]pagenum=\d+',
        r'[?&]offset=\d+',
    ]

    all_links = soup.find_all('a', href=True)
    page_links_count = 0
    for link in all_links:
        href = link.get('href', '')
        for pattern in page_url_patterns:
            if re.search(pattern, href, re.I):
                page_links_count += 1
                break

    if page_links_count >= 2:
        return True

    # 8. DATA ATTRIBUTES (Vue.js, React, Angular)
    data_attr_patterns = [
        {'data-pagination': True},
        {'data-page': re.compile(r'\d+')},
        {'data-pager': True},
    ]

    for attrs in data_attr_patterns:
        if soup.find(attrs=attrs):
            return True

    return False


def _has_sequential(numbers: list) -> bool:
    """Check if list contains at least 3 sequential numbers"""
    if len(numbers) < 3:
        return False
    sorted_nums = sorted(set(numbers))
    for i in range(len(sorted_nums) - 2):
        if sorted_nums[i+1] == sorted_nums[i] + 1 and sorted_nums[i+2] == sorted_nums[i] + 2:
            return True
    return False


def has_pagination(html: str, url: str, llm_ok: bool = True) -> bool:
    """Hybrid pagination detection: BeautifulSoup first, LLM fallback.

    llm_ok=False disables the LLM fallback — the caller has seen this URL
    pattern come back "no pagination" repeatedly, so a BS4 miss is final.
    """
    if has_pagination_fast(html):
        print("✓ Pagination found (BeautifulSoup)")
        return True

    if not llm_ok:
        print("BeautifulSoup: No pagination — LLM skipped (pattern known to have none)")
        return False

    print("BeautifulSoup: No pagination found, trying LLM...")

    # Truncate HTML for LLM - send first and last 10K chars
    # (pagination is usually at top or bottom of page)
    pure_html = clean_html(html)
    if len(pure_html) > 20000:
        pure_html = pure_html[:10000] + "\n...[TRUNCATED]...\n" + pure_html[-10000:]

    prompt = f"""You are a web crawling expert.

Your task is to determine whether the following webpage contains
PAGINATION used to navigate multiple pages of the SAME type of content
(e.g. job listings, articles, products).

IMPORTANT:
Pagination usually appears as a GROUP of clickable elements and often:
- Appears at the TOP or BOTTOM of the page
- Contains page numbers like: 1 2 3 4 5
- May include symbols like « » or ‹ ›
- Page numbers are usually LINKS (<a>) or BUTTONS
- One page number may appear highlighted or marked as active

DO NOT confuse pagination with:
- Dates (e.g. 20-Jan-2026)
- Job IDs
- Item numbering inside a single listing
- Tables or calendars

URL:
{url}

HTML:
{pure_html}

Return ONLY valid JSON:
{{"has_pagination": true}} or {{"has_pagination": false}}"""

    try:
        response_text = ask_llm(prompt, timeout=120)
        print(f"[DEBUG] Pagination LLM response: {response_text}")

        cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text)

        json_match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if not json_match:
            print("✗ No pagination found (LLM)")
            return False

        result = json.loads(json_match.group(0))
        llm_result = bool(result.get("has_pagination", False))

        if llm_result:
            print("✓ Pagination found (LLM)")
        else:
            print("✗ No pagination found (LLM)")

        return llm_result

    except Exception as e:
        print(f"[ERROR] LLM pagination detection failed: {e}")
        return False


def looks_like_list_page_fast(html: str) -> bool:
    """Heuristic check for job listing pages"""
    text = html.lower()
    keywords = [
        "job", "apply", "position", "vacancy", "career",
        "hiring", "recruit", "opening", "employment",
    ]
    return sum(text.count(k) for k in keywords) >= 5


def looks_like_list_page(html: str, url: str) -> bool:
    """
    Check if page is a listing page that might have pagination.
    1. Fast heuristic check (keywords)
    2. Pagination detection (BS4 fast, then LLM fallback)
    """
    # Quick keyword check first - if it doesn't look like a list page, skip
    if not looks_like_list_page_fast(html):
        return False

    # Then check for pagination
    return has_pagination(html, url)
