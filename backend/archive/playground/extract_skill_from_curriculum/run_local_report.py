"""Same syllabus → CLO → skills run, on LOCAL Ollama models.

    python run_local_report.py qwen3:14b
    python run_local_report.py qwen3:8b llama3.1:8b

Ollama constrains decoding to the JSON schema, so unlike the OpenRouter path
these calls cannot emit malformed JSON — but they CAN silently truncate if
num_ctx is too small (4096 default drops the lesson plan), hence num_ctx=16384.

One report file per model: report_<model>.md, comparable with report_gpt55.md.
"""

import sys
from pathlib import Path

import run_gpt55_report as R

HERE = Path(__file__).parent


def make_call(model: str):
    import ollama

    def call(system: str, user: str, schema: dict) -> str:
        resp = ollama.chat(
            model=model,
            think=False,
            options={"temperature": 0, "num_ctx": 16384},
            format=schema,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.message.content

    return call


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    md_dir = None
    models = []
    for a in args:
        if ":" in a:          # "qwen3:14b" — an Ollama model tag
            models.append(a)
        else:                 # a bare path = the markdown directory
            md_dir = Path(a)

    for model in models:
        label = model.replace(":", "-")
        print(f"\n{'=' * 70}\n{model}\n{'=' * 70}", flush=True)
        R.main(model=model, call=make_call(model),
               report=HERE / f"report_{label}.md", md_dir=md_dir)
