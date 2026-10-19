"""Same syllabus → CLO → skills run, on the cheap direct-OpenAI models.

The point is cost: a one-time extraction of the whole programme is ~$0.46 on
nano and ~$1.69 on mini, against ~$11 on gpt-5.5. This runs the identical V8
prompts and schemas so `compare_reports.py` can score the small models against
the gpt-5.5 reference and say whether the saving costs any quality.

    python run_openai_report.py nano
    python run_openai_report.py mini
    python run_openai_report.py all
    python run_openai_report.py nano "md_docling/Year 5 Semester 1"

Reuses run_gpt55_report's pipeline + report writer, so every report_*.md stays
directly comparable.

gpt-5.x quirks this handles: the sampling params are fixed (no `temperature`),
the output budget is `max_completion_tokens` (not `max_tokens`), and a reasoning
model can burn the whole budget on thinking and return empty content — hence the
retry loop and the generous budget.
"""

import json
import re
import sys
import time
from pathlib import Path

import run_gpt55_report as R

HERE = Path(__file__).parent

# label → OpenAI model id
MODELS = {
    "nano": "gpt-5.4-nano",
    "mini": "gpt-5.4-mini",
}

MAX_RETRIES = 3
MAX_COMPLETION = 16384


def _clean(txt: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()


def make_call(model_id: str):
    """Build a call(system, user, schema) bound to one OpenAI model."""
    from openai import OpenAI
    client = OpenAI(api_key=R._api_key(), timeout=600)

    def call(system: str, user: str, schema: dict) -> str:
        last = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    max_completion_tokens=MAX_COMPLETION,
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
        elif a.startswith("gpt-"):               # raw model id
            targets.append((a, a))
        else:
            md_dir = Path(a)                     # trailing path = markdown dir

    for label, model_id in targets:
        run(label, model_id, md_dir=md_dir)
