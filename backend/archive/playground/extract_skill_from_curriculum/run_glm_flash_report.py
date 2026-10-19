"""Same syllabus → CLO → skills run, on GLM's FREE tier (glm-4.5-flash).

The paid runner (run_glm_report.py) talks to api.z.ai, which serves only paid
models — there is no flash tier there. The free model lives on the mainland
platform, open.bigmodel.cn, and accepts the same ZHIPU_API_KEY.

Same reasoning trap as the paid model, and worse here: a bare "say hi" came back
with content="" and the whole budget in reasoning_content, so `thinking:
disabled` is mandatory, not an optimisation.

Retries like the OpenRouter runner — a free small model is far likelier to emit
truncated or fenced JSON than glm-4.7 was.

    python run_glm_flash_report.py [md_dir]
"""

import json
import re
import sys
import time
from pathlib import Path

import run_gpt55_report as R

MODEL = "glm-4.5-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
HERE = Path(__file__).parent
REPORT = HERE / "report_glm-flash.md"
MAX_RETRIES = 3


def _api_key() -> str:
    for line in R.ENV_FILE.read_text().splitlines():
        if line.strip().startswith("ZHIPU_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("ZHIPU_API_KEY not found in backend/.env")


def call(system: str, user: str, schema: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=600)

    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=16384,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user
                     + "\n\nReturn ONLY JSON matching this schema:\n"
                     + json.dumps(schema)},
                ],
            )
            choice = resp.choices[0]
            out = re.sub(r"^```(?:json)?|```$", "",
                         (choice.message.content or "").strip(), flags=re.M).strip()
            if not out:
                raise RuntimeError(
                    f"empty content (finish_reason={choice.finish_reason}) — "
                    "reasoning likely consumed the budget")
            json.loads(re.search(r"\{.*\}", out, re.S).group(0))   # parse check
            return out
        except Exception as e:                                     # noqa: BLE001
            last = e
            print(f"      retry {attempt}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {str(e)[:90]}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last}")


if __name__ == "__main__":
    md_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    R.main(model=MODEL, call=call, report=REPORT, md_dir=md_dir)
