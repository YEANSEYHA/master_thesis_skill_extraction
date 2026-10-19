"""Same syllabus → CLO → skills run, on OpenRouter-hosted models.

One model per invocation, one report file per model, so runs can be done and
compared independently:

    python run_openrouter_report.py meta-llama/llama-3.1-8b-instruct
    python run_openrouter_report.py qwen/qwen3-8b
    python run_openrouter_report.py all          # every model in MODELS, in order

Reuses run_gpt55_report's pipeline + report writer (identical prompts, schemas
and layout), so every report_*.md is directly comparable.

Small models are the point here, and they are the ones that break JSON, so this
backend is more defensive than the frontier ones: it retries, strips ```json
fences and <think> blocks, and asks for a large max_tokens.
"""

import json
import re
import sys
import time
from pathlib import Path

import run_gpt55_report as R

HERE = Path(__file__).parent
BASE_URL = "https://openrouter.ai/api/v1"

# label → OpenRouter model id.  NOTE: OpenRouter has no 8B DeepSeek (the
# r1-distill-*-8b variants are gone); deepseek-v4-flash is its small/cheap
# model and stands in for it — it is NOT an 8B peer, so treat that row as
# "DeepSeek's small model", not as a size-matched comparison.
MODELS = {
    "llama3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen3-8b":    "qwen/qwen3-8b",
    "deepseek":    "deepseek/deepseek-v4-flash",
}

MAX_RETRIES = 3


def _api_key() -> str:
    for line in R.ENV_FILE.read_text().splitlines():
        if line.strip().startswith("OPENROUTER_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found in backend/.env")


def _clean(txt: str) -> str:
    """Small models wrap JSON in fences and/or reasoning blocks."""
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    return re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()


def make_call(model_id: str):
    """Build a call(system, user, schema) bound to one OpenRouter model."""
    from openai import OpenAI
    client = OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=600)

    def call(system: str, user: str, schema: dict) -> str:
        last = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    temperature=0,
                    max_tokens=16384,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user
                         + "\n\nReturn ONLY JSON matching this schema:\n"
                         + json.dumps(schema)},
                    ],
                )
                out = _clean(resp.choices[0].message.content)
                if not out:
                    raise RuntimeError(
                        f"empty content (finish_reason="
                        f"{resp.choices[0].finish_reason})")
                json.loads(re.search(r"\{.*\}", out, re.S).group(0))  # parse check
                return out
            except Exception as e:                      # noqa: BLE001
                last = e
                print(f"      retry {attempt}/{MAX_RETRIES}: "
                      f"{type(e).__name__}: {str(e)[:90]}", flush=True)
                time.sleep(2 * attempt)
        raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last}")

    return call


def run(label: str, model_id: str, md_dir=None) -> None:
    report = HERE / f"report_{label}.md"
    print(f"\n{'=' * 70}\n{label}  ({model_id})\n{'=' * 70}", flush=True)
    R.main(model=model_id, call=make_call(model_id), report=report, md_dir=md_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("available:", ", ".join(f"{k} = {v}" for k, v in MODELS.items()))
        raise SystemExit(1)

    md_dir = None
    targets = []
    for a in args:
        if a == "all":
            targets += list(MODELS.items())
        elif a in MODELS:
            targets.append((a, MODELS[a]))
        elif "/" in a:                       # raw OpenRouter id
            targets.append((a.split("/")[-1], a))
        else:
            md_dir = Path(a)                 # trailing path = markdown dir

    for label, model_id in targets:
        run(label, model_id, md_dir=md_dir)
