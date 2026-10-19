"""Same syllabus → CLO → skills run, on z.ai's GLM-4.7.

Reuses run_gpt55_report's pipeline and report writer — only the backend call
changes — so report_glm.md is directly comparable with report_gpt55.md.

GLM notes:
  - endpoint is z.ai's international API (api.z.ai), not open.bigmodel.cn
  - `thinking: disabled`: with reasoning ON, GLM spends its whole token budget
    on reasoning_content and returns an EMPTY content (a 460s no-op was seen
    with reasoning left on)
  - max_tokens must be generous — the AI/Data-Mining syllabi produce long
    lesson lists and a truncated response is unparseable JSON

    python run_glm_report.py [md_dir]
"""

import json
import sys
from pathlib import Path

import run_gpt55_report as R

MODEL = "glm-4.7"
BASE_URL = "https://api.z.ai/api/paas/v4"
HERE = Path(__file__).parent
REPORT = HERE / "report_glm.md"


def _api_key() -> str:
    for line in R.ENV_FILE.read_text().splitlines():
        if line.strip().startswith("ZHIPU_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("ZHIPU_API_KEY not found in backend/.env")


def call(system: str, user: str, schema: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=600)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=16384,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user
             + "\n\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema)},
        ],
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"empty content (finish_reason={resp.choices[0].finish_reason}) — "
            "reasoning likely consumed the budget")
    return content


if __name__ == "__main__":
    md_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    R.main(model=MODEL, call=call, report=REPORT, md_dir=md_dir)
