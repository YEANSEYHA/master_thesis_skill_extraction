import os
import re
import json
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup, Comment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "llama3.1:8b"
OUTPUT_DIR = "outputs/classification"


def read_all_htmls(folder_path: str = "../data_collection/data/job_card") -> list[dict]:
    """Read all HTML files from the given folder and return a list of dicts with filename and content."""
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".html"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({"filename": filename, "content": content})
    return results


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1) Remove script, style, and media tags
    for tag in soup.find_all(["script", "style", "img", "video", "iframe", "svg", "noscript"]):
        tag.decompose()

    # 2) Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 3) Remove navigation, header, footer, and non-content elements
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()

    return str(soup)


def extract_text_for_classification(html: str, max_chars: int = 12000) -> str:
    """Extract clean text from HTML for classification, truncated to max_chars."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text



def build_prompt(page_text: str) -> str:
    """
    Prompt for classifying single job detail pages vs job listing pages.
    Uses extracted text (not raw HTML) to keep prompt small for the model.
    """
    return f"""Does this page contain ONE main job posting with a detailed description?

Rules:
- TRUE: Page has ONE main job with detailed description/requirements. Ignore any "Recommended Jobs" or "Similar Jobs" sections.
- FALSE: Page is a job listing/search results showing MULTIPLE jobs as the main content.

Return ONLY JSON: {{"is_single_job_detail": true}} or {{"is_single_job_detail": false}}

Page content:
{page_text}""".strip()

def safe_parse_json(text: str) -> dict:
    """
    Makes JSON parsing robust if the model accidentally adds extra text.
    Extract the first {...} block and parse it.
    """
    s = text.strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Raw: {s[:200]}")
    return json.loads(m.group(0))


def ask_llm(model_name: str, prompt: str, base_url: str = "http://localhost:11434") -> dict:
    """Send prompt to Ollama LLM and return parsed JSON response."""
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    response.raise_for_status()
    raw_text = response.json().get("response", "")
    return safe_parse_json(raw_text)


def save_result(filename: str, result: dict, cleaned_html: str):
    """Save classified result to the appropriate output folder."""
    is_job = bool(result.get("is_single_job_detail", False))

    out_dir = Path(OUTPUT_DIR) / ("single_job_detail" if is_job else "non_job_details")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned HTML
    html_path = out_dir / filename
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)


def main(limit: int | None = None):
    htmls = read_all_htmls()
    if limit:
        htmls = htmls[:limit]
    logger.info(f"Processing {len(htmls)} HTML files")

    results_summary = {"single_job_detail": 0, "non_job_details": 0, "failed": 0, "no_content": 0}

    for i, item in enumerate(htmls, 1):
        filename = item["filename"]
        logger.info(f"[{i}/{len(htmls)}] Processing: {filename}")

        raw = item["content"]
        cleaned = clean_html(raw)
        page_text = extract_text_for_classification(cleaned)

        if not page_text.strip():
            logger.warning(f"  -> No content after cleaning, skipping LLM (JS-rendered page?)")
            results_summary["no_content"] += 1
            continue

        prompt = build_prompt(page_text)

        try:
            result = ask_llm(MODEL_NAME, prompt)
            is_job = bool(result.get("is_single_job_detail", False))

            logger.info(f"  -> is_single_job_detail: {is_job}")
            save_result(filename, result, cleaned)

            if is_job:
                results_summary["single_job_detail"] += 1
            else:
                results_summary["non_job_details"] += 1

        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            results_summary["failed"] += 1
            results_summary["non_job_details"] += 1

    logger.info(f"Done! Summary: {json.dumps(results_summary, indent=2)}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
