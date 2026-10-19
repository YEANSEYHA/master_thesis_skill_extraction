"""Run every docling Markdown syllabus through the V8 prompts on gpt-5.5.

Same two prompts as the local pipeline (_SECTIONS_SYS, _SKILLS_SYS) and the same
schemas — only the backend is OpenAI instead of Ollama, so the output is directly
comparable with the qwen3/llama runs.

Writes ONE report file: report_gpt55.md (per-course course info, CLOs, lessons,
skills, plus a summary table and the raw JSON appended for reprocessing).

    python run_gpt55_report.py [md_dir]
"""

import json
import re
import time
from pathlib import Path

import main as P

MODEL = "gpt-5.5"
HERE = Path(__file__).parent
# NOTE: no sys.argv here — this module is imported by the other runners, and
# reading argv at import time made *their* first argument look like a md dir.
MD_DIR = HERE / "md_docling" / "Year 5 Semester 1"
REPORT = HERE / "report_gpt55.md"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _api_key():
    for line in ENV_FILE.read_text().splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found in backend/.env")


def _call(system: str, user: str, schema: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_api_key(), timeout=300)
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},   # gpt-5.x: default temperature only
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user
             + "\n\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema)},
        ],
    )
    return resp.choices[0].message.content


def _loads(txt: str) -> dict:
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def run_one(md: str, call=None) -> dict:
    """Markdown → sections (CLOs) → skills.

    `call(system, user, schema) -> str` selects the backend; defaults to gpt-5.5
    so another model only has to supply its own `call` (see run_glm_report.py).
    """
    call = call or _call
    sec = P.Sections.model_validate(
        _loads(call(P._SECTIONS_SYS, md, P.Sections.model_json_schema())))

    lessons = "\n".join(
        f"- {ls.topic}" + (f"  ({'; '.join(ls.outcomes)})" if ls.outcomes else "")
        for ls in sec.lessons)
    user = (f"### DESCRIPTION\n{sec.description or ''}\n\n### CLOs\n"
            + "\n".join(f"- {c}" for c in sec.clos)
            + f"\n\n### LESSON TOPICS\n{lessons}")
    skills = P.normalize_and_dedup(
        P.SkillList.model_validate(
            _loads(call(P._SKILLS_SYS, user, P.SkillList.model_json_schema()))).skills)

    return {
        "course": sec.course.model_dump(),
        "description": sec.description,
        "clos": sec.clos,
        "lessons": [ls.model_dump() for ls in sec.lessons],
        "skills": [s.model_dump() for s in skills],
    }


def main(model=MODEL, call=None, report=REPORT, md_dir=None) -> None:
    files = sorted((md_dir or MD_DIR).glob("*.md"))
    print(f"{len(files)} markdown files → {model}", flush=True)

    results = []
    for path in files:
        t0 = time.time()
        rec = {"file": path.name}
        try:
            rec.update(run_one(path.read_text(), call=call))
            note = f"{len(rec['skills'])} skills, {len(rec['clos'])} CLOs"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            rec.setdefault("skills", []), rec.setdefault("clos", [])
            note = f"ERROR {rec['error'][:60]}"
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"  {path.stem[:44]:<46} {rec['seconds']:>6.1f}s  {note}", flush=True)

    write_report(results, model=model, report=report, md_dir=md_dir or MD_DIR)
    print(f"\nreport → {report}")


def write_report(results: list, model=MODEL, report=REPORT, md_dir=MD_DIR) -> None:
    ok = [r for r in results if not r.get("error")]
    out = [
        f"# Syllabus → CLO → Skills report ({model})",
        "",
        f"- Source markdown: `{md_dir}`",
        f"- Pipeline: docling markdown → `parse_sections` → `extract_skills` → "
        f"`normalize_and_dedup` (V8 prompts, unchanged)",
        f"- Courses: {len(ok)}/{len(results)} succeeded · "
        f"total skills: {sum(len(r['skills']) for r in ok)} · "
        f"wall time: {sum(r['seconds'] for r in results):.0f}s",
        "",
        "## Summary",
        "",
        "| Course | Code | Credits | Category | CLOs | Lessons | Skills | Time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r.get("course") or {}
        if r.get("error"):
            out.append(f"| {r['file']} | — | — | — | — | — | ERROR | {r['seconds']}s |")
            continue
        out.append(
            f"| {c.get('title') or '—'} | {c.get('code') or '—'} | {c.get('credits') or '—'} "
            f"| {c.get('category') or '—'} | {len(r['clos'])} | {len(r['lessons'])} "
            f"| {len(r['skills'])} | {r['seconds']}s |")

    for r in results:
        c = r.get("course") or {}
        out += ["", "---", "",
                f"## {c.get('title') or r['file']}  ({c.get('code') or '—'})", ""]
        if r.get("error"):
            out += [f"**FAILED** — `{r['error']}`", ""]
            continue
        out += [f"- **File:** `{r['file']}`",
                f"- **Credits:** {c.get('credits') or '—'} · "
                f"**Programme:** {c.get('programme') or '—'} · "
                f"**Category:** {c.get('category') or '—'}",
                "", "### Description", "", (r.get("description") or "_none extracted_"),
                "", f"### Course Learning Outcomes ({len(r['clos'])})", ""]
        out += [f"{i}. {clo}" for i, clo in enumerate(r["clos"], 1)] or ["_none_"]
        out += ["", f"### Lessons ({len(r['lessons'])})", ""]
        for ls in r["lessons"]:
            week = ls.get("week") or "—"
            out.append(f"- **Week {week}** — {ls['topic']}")
            out += [f"    - {o}" for o in (ls.get("outcomes") or [])]
        out += ["", f"### Skills ({len(r['skills'])})", "",
                "| # | Skill | Source | Evidence |", "|---|---|---|---|"]
        for i, s in enumerate(r["skills"], 1):
            ev = s["evidence"].replace("|", "\\|")
            out.append(f"| {i} | {s['name']} | {s['source']} | {ev[:110]} |")

    out += ["", "---", "", "## Raw JSON", "", "```json",
            json.dumps(results, indent=2, ensure_ascii=False), "```", ""]
    report.write_text("\n".join(out))


if __name__ == "__main__":
    import sys
    main(md_dir=Path(sys.argv[1]) if len(sys.argv) > 1 else None)
