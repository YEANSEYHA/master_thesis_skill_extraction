import os
import re
import json
import logging
import requests
from pathlib import Path
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "llama3.1:8b"
DEFAULT_INPUT_DIR = "../data_collection/data/job_card"
DEFAULT_OUTPUT_DIR = "outputs/classification"


def read_all_htmls(folder_path: str = DEFAULT_INPUT_DIR) -> list[dict]:
    """Read all HTML files from the given folder and return a list of dicts with filename and content."""
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".html"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({"filename": filename, "content": content})
    return results


def read_all_htmls_nested(base_dir: str) -> list[dict]:
    """Read all HTML files from a nested folder structure (e.g. phase2 output: run_id/domain/page_*.html)."""
    results = []
    base = Path(base_dir)
    for html_file in sorted(base.rglob("page_*.html")):
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Use relative path as filename to preserve domain context
        rel_path = str(html_file.relative_to(base))
        results.append({"filename": rel_path, "content": content})
    return results

def clean_html_trafilatura(html: str) -> str:
    text = trafilatura.extract(html, include_tables=True, include_links=False, favor_recall=True)
    if not text:
        return ""
    seen = set()
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped not in seen:
            seen.add(stripped)
            result.append(line)
    return "\n".join(result)

def build_prompt(page_text: str) -> str:
    """
    Prompt for classifying single job detail pages vs job listing pages.
    Uses extracted text (not raw HTML) to keep prompt small for the model.
    """
    return f"""Does this page contain ONE main job posting with a detailed description?
Rules:
- TRUE: Page has ONE main job with detailed description/requirements. Ignore any "Recommended Jobs" or "Similar Jobs" sections.
- FALSE: Page is a job listing/search results showing MULTIPLE jobs as the main content.
- FALSE: Page is a homepage, About Us, blog, FAQ, login, or error page with no job description.
- FALSE: Page has a job title but the description is too short or missing entirely.
- If unsure, return FALSE.
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


def save_result(filename: str, result: dict, cleaned_html: str, output_dir: str = DEFAULT_OUTPUT_DIR):
    """Save classified result to the appropriate output folder."""
    is_job = bool(result.get("is_single_job_detail", False))

    out_dir = Path(output_dir) / ("single_job_detail" if is_job else "non_job_details")

    # Handle nested filenames (e.g. "domain/page_0.html") — create subdirs
    html_path = out_dir / filename
    html_path.parent.mkdir(parents=True, exist_ok=True)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)


def main(limit: int | None = None, input_dir: str | None = None, output_dir: str | None = None, nested: bool = False):
    """
    Classify HTML pages as single job detail vs non-job.

    Args:
        limit: Max number of files to process.
        input_dir: Folder with HTML files. Defaults to data_collection/data/job_card.
        output_dir: Where to save classified results. Defaults to outputs/classification.
        nested: If True, read from nested folder structure (phase2 output: domain/page_*.html).
    """
    _input = input_dir or DEFAULT_INPUT_DIR
    _output = output_dir or DEFAULT_OUTPUT_DIR

    if nested:
        htmls = read_all_htmls_nested(_input)
    else:
        htmls = read_all_htmls(_input)

    if limit:
        htmls = htmls[:limit]
    logger.info(f"Processing {len(htmls)} HTML files from {_input}")

    results_summary = {"single_job_detail": 0, "non_job_details": 0, "failed": 0, "no_content": 0}

    for i, item in enumerate(htmls, 1):
        filename = item["filename"]
        logger.info(f"[{i}/{len(htmls)}] Processing: {filename}")

        raw = item["content"]
        page_text = clean_html_trafilatura(raw)

        if not page_text.strip():
            logger.warning(f"  -> No content after cleaning, skipping LLM (JS-rendered page?)")
            results_summary["no_content"] += 1
            continue

        prompt = build_prompt(page_text)

        try:
            result = ask_llm(MODEL_NAME, prompt)
            is_job = bool(result.get("is_single_job_detail", False))

            logger.info(f"  -> is_single_job_detail: {is_job}")
            save_result(filename, result, page_text, output_dir=_output)

            if is_job:
                results_summary["single_job_detail"] += 1
            else:
                results_summary["non_job_details"] += 1

        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            results_summary["failed"] += 1
            results_summary["non_job_details"] += 1

    logger.info(f"Done! Summary: {json.dumps(results_summary, indent=2)}")
    return results_summary


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
