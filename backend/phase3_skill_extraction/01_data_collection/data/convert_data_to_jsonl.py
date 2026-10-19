
import os
import json
import re
from bs4 import BeautifulSoup, Comment


def read_all_htmls(folder_path: str = "dataset/htmls") -> list[dict]:
    """Read all HTML files from the given folder and return a list of dicts with filename and content."""
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".html"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({"filename": filename, "content": content})
    return results


def format_single_job_detail(html: str) -> dict:
    """Format HTML into the structure for single job detail classification."""
    return {
        "input": html,
        "output": {
            "is_single_job_detail": True
        }
    }

def format_non_single_job_detail(html: str) -> dict:
    """Format HTML into the structure for single job detail classification."""
    return {
        "input": html,
        "output": {
            "is_single_job_detail": False
        }
    }

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





def main():
    # Read all HTML files from job_card folder
    html_files = read_all_htmls("non_job_card")

    # Format each HTML and save to JSONL
    output_file = "fine_tuned_dataset_classify_card_detail.jsonl"

    for html_data in html_files:
        # First clean HTML, then extract text
        cleaned_html = clean_html(html_data["content"])
        extracted_text = extract_text_for_classification(cleaned_html)

        formatted_data = format_non_single_job_detail(extracted_text)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")

    print(f"Converted {len(html_files)} files to {output_file}")


if __name__ == "__main__":
    main()
