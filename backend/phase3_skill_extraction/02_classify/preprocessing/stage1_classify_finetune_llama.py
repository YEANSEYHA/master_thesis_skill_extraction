"""
Classify HTML pages using fine-tuned Llama-3.1-8B model (Unsloth + LoRA adapters).
Replaces Ollama-based classification with local fine-tuned model on RTX 5090.

Usage:
    python stage1_classify_finetune_llama.py [limit]
    python stage1_classify_finetune_llama.py 10  # process first 10 files only
"""

import os
import re
import json
import logging
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from bs4 import BeautifulSoup, Comment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============== Configuration ==============
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "../fine_tuned/classify_output_linux/adapters"
INPUT_DIR = "../data_collection/data/job_card"
OUTPUT_DIR = "outputs/classification"
MAX_SEQ_LENGTH = 4096

LLAMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}"
    "{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


# ============== HTML Processing ==============
def read_all_htmls(folder_path: str = INPUT_DIR) -> list[dict]:
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
    for tag in soup.find_all(["script", "style", "img", "video", "iframe", "svg", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()
    return str(soup)


def extract_text_for_classification(html: str, max_chars: int = 12000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def build_prompt(page_text: str) -> str:
    return f"""Does this page contain ONE main job posting with a detailed description?

Rules:
- TRUE: Page has ONE main job with detailed description/requirements. Ignore any "Recommended Jobs" or "Similar Jobs" sections.
- FALSE: Page is a job listing/search results showing MULTIPLE jobs as the main content.

Return ONLY JSON: {{"is_single_job_detail": true}} or {{"is_single_job_detail": false}}

Page content:
{page_text}"""


def safe_parse_json(text: str) -> dict:
    s = text.strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Raw: {s[:200]}")
    return json.loads(m.group(0))


def save_result(filename: str, result: dict, cleaned_html: str):
    is_job = bool(result.get("is_single_job_detail", False))
    out_dir = Path(OUTPUT_DIR) / ("single_job_detail" if is_job else "non_job_details")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / filename
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)


# ============== Model Loading ==============
def load_model():
    logger.info(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LoRA adapters from: {ADAPTER_DIR}")
    model.load_adapter(ADAPTER_DIR, adapter_name="classify")
    model.set_adapter("classify")

    FastLanguageModel.for_inference(model)
    logger.info("Model loaded and ready for inference")
    return model, tokenizer


def classify_page(model, tokenizer, page_text: str) -> dict:
    prompt = build_prompt(page_text)
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )

    prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return safe_parse_json(prediction)


# ============== Main ==============
def main(limit: int | None = None):
    model, tokenizer = load_model()

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
            logger.warning(f"  -> No content after cleaning, skipping")
            results_summary["no_content"] += 1
            continue

        try:
            result = classify_page(model, tokenizer, page_text)
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
