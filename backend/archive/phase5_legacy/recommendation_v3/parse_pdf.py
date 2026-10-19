"""Recommendation V3 — self-contained syllabus parser (PDF → skills).

This is a STANDALONE copy for the Recommendation V3 tab. It deliberately does
NOT import from baseline_analyze/ or any other phase-5 package — V3 owns its
full backend so it can evolve independently.

Pipeline (2 LLM calls per course):
    PDF → pdf_to_markdown() → parse_sections() → extract_skills()
        → normalize_and_dedup() → course skill list (each skill traceable).

External deps only: ollama, pymupdf4llm (+ fitz for byte uploads), pydantic.
"""

import html
import os
import re
import sys
import time
from pathlib import Path
from typing import Union

import ollama
import pymupdf4llm
from pydantic import BaseModel, Field

# Model for the two extraction calls, switchable from backend/.env:
#
#   PHASE5_LLM_MODEL=qwen3:14b        local, free, slower  (default)
#   PHASE5_LLM_MODEL=gpt-5.4-nano     hosted, ~$0.006/course
#
# Provider is inferred from the name ("gpt-*" → OpenAI, else Ollama), so the
# backend switches with one .env line. PHASE5_LLM_PROVIDER forces it. Read per
# call, not at import, so a reload picks the change up.
# (V3 keeps its own copy on purpose — see the module docstring.)
_DEFAULT_LLM = "qwen3:14b"
_OPENAI_MAX_COMPLETION = 16384
_OPENAI_RETRIES = 3


_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"   # backend/.env
_env_cache: dict = {"mtime": None, "vals": {}}


def _from_env_file(key: str) -> str | None:
    """Read `key` straight from backend/.env, re-parsing only when the file
    changes. config.py loads .env into os.environ at STARTUP, so reading the
    file live is what lets a model switch take effect without a restart."""
    try:
        mtime = _ENV_FILE.stat().st_mtime
    except OSError:
        return None
    if _env_cache["mtime"] != mtime:
        vals = {}
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip().strip('"').strip("'")
        _env_cache.update(mtime=mtime, vals=vals)
    return _env_cache["vals"].get(key) or None


def active_model() -> str:
    """The model these extractions will run on (for logs and provenance).

    backend/.env wins over the process environment on purpose: editing that file
    is how the model is switched, and it must not be shadowed by a stale value
    copied into os.environ when the server booted.
    """
    return (_from_env_file("PHASE5_LLM_MODEL")
            or os.getenv("PHASE5_LLM_MODEL")
            or _DEFAULT_LLM)


def _provider(model: str) -> str:
    return (_from_env_file("PHASE5_LLM_PROVIDER")
            or os.getenv("PHASE5_LLM_PROVIDER")
            or ("openai" if model.startswith("gpt-") else "ollama"))


def _openai_key() -> str:
    """From backend/.env or the environment — this module is used both by the
    FastAPI process (which loads .env) and by standalone scripts."""
    key = _from_env_file("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (backend/.env or env)")
    return key


def _chat_ollama(model: str, schema: dict, system: str, user: str) -> str:
    resp = ollama.chat(
        model=model,
        think=False,
        # num_ctx: Ollama defaults to 4096 tokens, which TRUNCATES our ~6k-token
        # syllabus (silently dropping the lesson plan at the end). Raise it so the
        # whole document reaches the model.
        options={"temperature": 0, "num_ctx": 16384},
        format=schema,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.message.content


def _chat_openai(model: str, schema: dict, system: str, user: str) -> str:
    """gpt-5.x has no schema-locked mode like Ollama's `format`, so the schema
    goes in the user turn and JSON mode enforces valid JSON — the exact shaping
    the model benchmark scored. Sampling params are fixed (no `temperature`),
    the budget is `max_completion_tokens`, and a reasoning model can spend it
    all thinking and return empty content — hence the retry."""
    import json as _json
    from openai import OpenAI
    client = OpenAI(api_key=_openai_key(), timeout=600)
    payload = user + "\n\nReturn ONLY JSON matching this schema:\n" + _json.dumps(schema)

    last = None
    for attempt in range(1, _OPENAI_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=_OPENAI_MAX_COMPLETION,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": payload}],
            )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                raise RuntimeError(
                    f"empty content (finish_reason={resp.choices[0].finish_reason})")
            # Callers parse with model_validate_json, which won't tolerate a
            # ```json wrapper — strip defensively.
            out = re.sub(r"^```(?:json)?|```$", "", out, flags=re.M).strip()
            m = re.search(r"\{.*\}", out, re.S)
            return m.group(0) if m else out
        except Exception as e:                          # noqa: BLE001
            last = e
            print(f"[v3:{model}] retry {attempt}/{_OPENAI_RETRIES}: "
                  f"{type(e).__name__}: {str(e)[:90]}", file=sys.stderr, flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"{model} failed after {_OPENAI_RETRIES} attempts: {last}")


def _chat(label: str, schema: dict, system: str, user: str) -> str:
    """One schema-locked call on the configured model, timed to stderr."""
    model = active_model()
    provider = _provider(model)
    print(f"[v3:{label}] → {model} ({provider}): sending {len(user):,} chars …",
          file=sys.stderr, flush=True)
    t0 = time.time()
    fn = _chat_openai if provider == "openai" else _chat_ollama
    out = fn(model, schema, system, user)
    print(f"[v3:{label}] ← done in {time.time() - t0:.1f}s ({len(out):,} chars)",
          file=sys.stderr, flush=True)
    return out


# ── Stage 1: PDF → Markdown ──────────────────────────────────────────────────────

def pdf_to_markdown(pdf_source: Union[str, Path, bytes],
                    filename: str | None = None) -> str:
    """Convert a syllabus into a Markdown string.

    Accepts either a file path (str/Path) or the raw bytes (e.g. an uploaded
    file). PDFs go through pymupdf4llm so the document's structure is preserved
    as Markdown syntax — headings become '#', and (crucially for syllabi)
    tables become pipe tables '| ... |'. That keeps the CLO and lesson-plan
    tables intact for the later LLM extraction step.

    .docx uploads fall back to docling, which produces the same pipe-table
    Markdown from Word's XML (pymupdf4llm cannot read .docx at all). For bytes
    the extension is unknowable, so `filename` is what selects the branch —
    without it, bytes are assumed to be a PDF, as before.
    """
    suffix = Path(filename or (pdf_source if isinstance(pdf_source, (str, Path))
                               else "")).suffix.lower()
    if suffix == ".docx":
        return _docx_to_markdown(pdf_source, filename)

    if isinstance(pdf_source, (str, Path)):
        return pymupdf4llm.to_markdown(str(pdf_source))
    # bytes → open as an in-memory PyMuPDF document
    import fitz
    doc = fitz.open(stream=pdf_source, filetype="pdf")
    return pymupdf4llm.to_markdown(doc)


def _docx_to_markdown(source: Union[str, Path, bytes],
                      filename: str | None = None) -> str:
    """.docx → Markdown via docling (IBM's document converter).

    docling is imported lazily and the converter cached: constructing it loads
    layout models and costs seconds, and PDF uploads must not pay that.

    Note docling requires Python < 3.14; on 3.14 the import fails and the error
    surfaces to the caller as an unsupported-upload message rather than a
    confusing ImportError.
    """
    global _DOCLING
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:                                    # noqa: BLE001
        raise RuntimeError(
            ".docx upload needs docling (`pip install docling`, Python < 3.14); "
            "please upload the syllabus as PDF instead") from e

    if _DOCLING is None:
        _DOCLING = DocumentConverter()

    if isinstance(source, (str, Path)):
        doc = _DOCLING.convert(str(source)).document
    else:
        import io
        from docling.datamodel.base_models import DocumentStream
        doc = _DOCLING.convert(DocumentStream(
            name=filename or "upload.docx", stream=io.BytesIO(source))).document
    # docling HTML-escapes '&' etc.; the LLM should see the original text.
    return html.unescape(doc.export_to_markdown())


_DOCLING = None   # cached DocumentConverter (built on first .docx upload)


# ── Stage 2: Markdown → Sections ─────────────────────────────────────────────────

class Course(BaseModel):
    title:     str | None = None
    code:      str | None = None
    credits:   float | None = None   # float, not int: ITC has 1.5/2.5-credit courses
    programme: str | None = None
    category:  str | None = None    # labour-market bucket the course prepares for
                                    # (one of the 17 LinkedIn job categories), inferred
                                    # by the LLM from the course CONTENT, not the title


class Lesson(BaseModel):
    week:     str | None = None
    topic:    str                                       # clean topic name, e.g. "Decision Trees"
    outcomes: list[str] = Field(default_factory=list)   # that week's lesson outcomes (LLOs)


class Sections(BaseModel):
    course:      Course
    description: str | None = None                      # field 13 synopsis
    clos:        list[str]                              # field 14, verbatim incl. "CLO1 (K)" prefix
    lessons:     list[Lesson]                           # field 18 rows


_SECTIONS_SYS = """\
You extract structure from a course syllabus into JSON. Copy values VERBATIM —
never paraphrase, summarize, or invent. If a field is absent, use null or [].
Syllabi vary in layout (numbered sections, tables, or headings); read by
meaning, not position. This works for a syllabus from ANY discipline
(engineering, science, health, business, law, arts) — the example below shows
field MAPPING only, not a subject area.

Fields:
  - course.title: the course name.
  - course.code: the SHORT alphanumeric course code ONLY (e.g. "GICI3AT",
    "AIECS028", "CS201"). Never include any words from the course title.
  - course.credits / course.programme: as written, if present.
  - course.category: the labour-market FIELD this course prepares students for —
    INFER it (it is rarely written in the syllabus). Judge by the COURSE CONTENT
    (description + CLOs + lesson topics), NOT the title, which can mislead (a
    "Business IT" course that teaches calculus is "education"/math, not "it").
    Choose EXACTLY ONE of these 17 keys:
      it, engineering, health, finance, sales, marketing, design, education,
      legal, hr, admin, construction, logistics, retail, hospitality,
      customer_service, other.
    Use "other" only if none fit. Computing/CS/software/data/AI/cyber courses →
    "it". Civil/mechanical/electrical/chemical → "engineering".
  - description: the course description / synopsis — the paragraph that
    summarises what the course covers. It usually follows a label like
    "Course Description", "Synopsis", or "Description", under a heading OR
    inside a table cell. Almost every syllabus has one: find it and copy it
    verbatim. Use null ONLY if no such paragraph exists.
  - clos: the Course Learning Outcomes. Copy each verbatim, keeping its
    identifier and any taxonomy tag exactly as written
    (e.g. "CLO1", "CLO1 (K)", "ILO2"). Do NOT include per-lesson outcomes here.
  - lessons: one entry per row of the weekly/session schedule — the topic
    name plus that row's lesson learning outcomes.

Example (shows the MAPPING only — your input may look different):

INPUT:
  Course Title: Data Structures   Code: CS201   Credits: 3
  Programme: Computer Science
  Description: Covers lists, stacks, queues, trees and their algorithms.
  Learning Outcomes:
    CLO1 Explain common data structures and their trade-offs.
    CLO2 Implement tree and graph algorithms.
  Weekly plan:
    Week 1 | Arrays and Linked Lists | Analyze list operations.
    Week 2 | Binary Search Trees     | Implement BST insert and delete.

OUTPUT:
{
  "course": {"title": "Data Structures", "code": "CS201", "credits": 3,
             "programme": "Computer Science", "category": "it"},
  "description": "Covers lists, stacks, queues, trees and their algorithms.",
  "clos": [
    "CLO1 Explain common data structures and their trade-offs.",
    "CLO2 Implement tree and graph algorithms."
  ],
  "lessons": [
    {"week": "1", "topic": "Arrays and Linked Lists",
     "outcomes": ["Analyze list operations."]},
    {"week": "2", "topic": "Binary Search Trees",
     "outcomes": ["Implement BST insert and delete."]}
  ]
}"""


def parse_sections(markdown: str) -> Sections:
    """Markdown → structured {course, description, clos, lessons} via qwen3."""
    out = _chat("parse_sections", Sections.model_json_schema(),
                _SECTIONS_SYS, markdown)        # whole doc — it fits in context
    return Sections.model_validate_json(out)


# ── Stage 3: Sections → Skills ───────────────────────────────────────────────────

class Skill(BaseModel):
    name:     str                                       # normalized skill, e.g. "K-Means Clustering"
    source:   str = Field(description="CLO | Lesson Topic | Description")
    evidence: str = Field(description="the exact source text the skill came from")


class SkillList(BaseModel):
    skills: list[Skill]


_SKILLS_SYS = """\
You extract a labour-market skill list from a course's sections, as JSON.
A "skill" is a concrete technology, technique, method, tool, or concept that
could appear in a job posting — expressed as a NOUN phrase, not an action.

This must work for ANY discipline — engineering (civil, electrical, chemical,
mechanical, industrial), the sciences, health, business, law, and the arts —
not only computing/IT. The examples below illustrate the LOGIC (verb→noun,
split, canonicalize, exclude activities), NOT the subject area. Extract the
skills of whatever domain the syllabus is actually about; never force a course
into computing terms.

EXAMPLE 1 (computing course):
  ### DESCRIPTION
  Introduces supervised machine learning and secure coding.
  ### CLOs
  - CLO1 Apply secure coding practices to prevent vulnerabilities.
  ### LESSON TOPICS
  - Unsupervised learning (K-mean algorithm)  (Apply the K-mean algorithm to a dataset.)
  - Supervised learning (decision tree)  (Apply the decision tree algorithm. Write a report.)
  - Secure coding for common vulnerabilities (SQL Injection, XSS)  (Complete lab exercises.)

OUTPUT 1:
{
  "skills": [
    {"name": "Secure Coding",      "source": "CLO",          "evidence": "CLO1 Apply secure coding practices to prevent vulnerabilities."},
    {"name": "K-Means",            "source": "Lesson Topic", "evidence": "Unsupervised learning (K-mean algorithm)"},
    {"name": "Decision Trees",     "source": "Lesson Topic", "evidence": "Supervised learning (decision tree)"},
    {"name": "SQL Injection",      "source": "Lesson Topic", "evidence": "Secure coding for common vulnerabilities (SQL Injection, XSS)"},
    {"name": "Cross-Site Scripting (XSS)", "source": "Lesson Topic", "evidence": "Secure coding for common vulnerabilities (SQL Injection, XSS)"}
  ]
}
(Note: "K-mean algorithm" -> "K-Means" (filler dropped); "Write a report" is
an activity and is NOT a skill.)

EXAMPLE 2 (NON-computing course — civil engineering + chemistry + business, to
show the SAME logic applies outside IT):
  ### DESCRIPTION
  Covers reinforced-concrete structural design and basic financial accounting.
  ### CLOs
  - CLO1 Design reinforced concrete beams according to load requirements.
  - CLO2 Prepare financial statements for a small enterprise.
  ### LESSON TOPICS
  - Titration methods (acid-base titration)  (Perform a titration in the lab.)
  - Structural load analysis (bending moment, shear force)  (Solve beam problems. Submit a report.)

OUTPUT 2:
{
  "skills": [
    {"name": "Reinforced Concrete Design", "source": "CLO",          "evidence": "CLO1 Design reinforced concrete beams according to load requirements."},
    {"name": "Financial Statement Preparation", "source": "CLO",     "evidence": "CLO2 Prepare financial statements for a small enterprise."},
    {"name": "Acid-Base Titration",        "source": "Lesson Topic", "evidence": "Titration methods (acid-base titration)"},
    {"name": "Bending Moment Analysis",    "source": "Lesson Topic", "evidence": "Structural load analysis (bending moment, shear force)"},
    {"name": "Shear Force Analysis",       "source": "Lesson Topic", "evidence": "Structural load analysis (bending moment, shear force)"}
  ]
}
(Same logic, different domain: "Submit a report" is an activity and is dropped;
"bending moment, shear force" is SPLIT into two skills.)

Rules:
  - NOUN form, not verbs. Turn a competency into its underlying skill noun:
    "Apply secure coding practices" -> "Secure Coding";
    "Design reinforced concrete beams" -> "Reinforced Concrete Design";
    "Prepare financial statements" -> "Financial Statement Preparation".
  - SPLIT named techniques/tools/concepts. If one line names several
    (e.g. "SQL Injection, XSS"; "bending moment, shear force"), output EACH as
    its own skill.
  - CANONICAL name: use the common job-market / professional term for that
    field. Drop filler suffixes like "Algorithm", "Algorithms", "Methods",
    "Techniques", "Model" when they only decorate a named method:
    "K-mean algorithm" -> "K-Means"; "decision tree algorithm" -> "Decision Trees";
    "titration method" -> "Titration".
  - Prefer specific, named skills from LESSON TOPICS; also include
    competencies from CLOs and the DESCRIPTION.
  - EXCLUDE generic learning activities and admin (these are NOT skills, and
    must never appear even if mentioned in a lesson): writing/reports, report
    writing, lab exercises, case studies, lectures, tutorials, quizzes,
    assignments, presentations, group projects, attendance, exams, homework,
    "review lesson", "final exam", "project presentation".
  - Clean, title-case skill name. source must be exactly one of:
    CLO, Lesson Topic, Description. evidence is the source text it came from."""


def extract_skills(sec: Sections) -> list[Skill]:
    """Structured sections → [{name, source, evidence}] in one LLM call.

    Sections are fed as clearly LABELED blocks so the model can attribute each
    skill's source reliably without needing a separate call per section.
    """
    lessons = "\n".join(
        f"- {l.topic}" + (f"  ({'; '.join(l.outcomes)})" if l.outcomes else "")
        for l in sec.lessons
    )
    user = (
        f"### DESCRIPTION\n{sec.description or ''}\n\n"
        f"### CLOs\n" + "\n".join(f"- {c}" for c in sec.clos) + "\n\n"
        f"### LESSON TOPICS\n{lessons}"
    )
    out = _chat("extract_skills", SkillList.model_json_schema(), _SKILLS_SYS, user)
    return SkillList.model_validate_json(out).skills


# ── Stage 4: Clean + Dedup ────────────────────────────────────────────────────────

_ACTIVITY_RE = re.compile(
    r"^(?:"
    r"writing|reports?|report\s+writing|lab\s+exercises?|case\s+stud(?:y|ies)|"
    r"lectures?|tutorials?|quizzes?|assignments?|presentations?|"
    r"group\s+projects?|attendance|exams?|homework|review\s+lesson|"
    r"final\s+exam|project\s+presentation|midterm|final\s+project"
    r")$",
    re.I,
)

_FILLER_RE = re.compile(
    r"\s+(?:algorithms?|methods?|techniques?|approaches?|practices?)$",
    re.I,
)

_SMALL_WORDS = {"and", "or", "of", "for", "to", "the", "a", "an", "in", "with"}

def _norm_key(name: str) -> str:
    """Canonical match key — IDENTICAL to the job side (aggregate_job_skills),
    so course "Decision Trees" lines up with job "decision trees"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _word_key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower())).strip()


def _standard_alias(name: str) -> str | None:
    """Tiny rule layer for spellings that generic title-casing cannot infer."""
    key = _word_key(name)
    compact = key.replace(" ", "")
    if re.fullmatch(r"kmeans?(?:clustering)?", compact):
        return "K-Means"
    if re.fullmatch(r"knearestneighbou?rs?", compact):
        return "K-Nearest Neighbor"
    if re.fullmatch(r"decision\s+trees?", key):
        return "Decision Trees"
    if re.fullmatch(r"gaussian\s+mixtures?(?:\s+models?)?", key):
        return "Gaussian Mixture"
    if key in {"xss", "cross site scripting"}:
        return "Cross-Site Scripting (XSS)"
    return None


def _title_token(token: str, first: bool, acronyms: set[str]) -> str:
    left = re.match(r"^[([{]+", token)
    right = re.search(r"[)\]}]+$", token)
    prefix = left.group(0) if left else ""
    suffix = right.group(0) if right else ""
    core = token[len(prefix):len(token) - len(suffix) if suffix else len(token)]

    def cap(part: str) -> str:
        upper = part.upper()
        if part.isupper() or upper in acronyms:
            return upper
        if part.lower() in _SMALL_WORDS and not first:
            return part.lower()
        return part[:1].upper() + part[1:].lower()

    return prefix + "-".join(cap(part) for part in core.split("-")) + suffix


def clean_skill_name(
    name: str,
    aliases: dict[str, str] | None = None,
    acronyms: set[str] | None = None,
) -> str | None:
    """Clean one extracted skill name; return None for non-skill activities."""
    name = re.sub(r"\s+", " ", name.strip(" \t\r\n,.;:"))
    if not name:
        return None
    if _ACTIVITY_RE.fullmatch(_word_key(name)):
        return None

    while True:
        stripped = _FILLER_RE.sub("", name)
        if stripped == name:
            break
        name = stripped

    alias_map = {_word_key(k): v for k, v in (aliases or {}).items()}
    alias = alias_map.get(_word_key(name)) or _standard_alias(name)
    if alias:
        return alias
    return " ".join(
        _title_token(tok, i == 0, acronyms or set())
        for i, tok in enumerate(name.split())
    )


def clean_skills(
    names: list[str],
    aliases: dict[str, str] | None = None,
    acronyms: set[str] | None = None,
) -> list[str]:
    """Deterministic pre-group cleanup for raw curriculum skill names."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        cleaned = clean_skill_name(name, aliases=aliases, acronyms=acronyms)
        key = _norm_key(cleaned or "")
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def normalize_and_dedup(
    skills: list[Skill],
    aliases: dict[str, str] | None = None,
    acronyms: set[str] | None = None,
) -> list[Skill]:
    """Clean names and drop duplicate skills, keeping first evidence."""
    seen: set[str] = set()
    out: list[Skill] = []
    for s in skills:
        name = clean_skill_name(s.name, aliases=aliases, acronyms=acronyms)
        key = _norm_key(name or "")
        if key and key not in seen:
            seen.add(key)
            out.append(s.model_copy(update={"name": name}))
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────────

def pdf_to_skills(pdf_source: Union[str, Path, bytes]) -> dict:
    """Full pipeline: PDF → markdown → sections → skills → deduped list."""
    md       = pdf_to_markdown(pdf_source)
    sections = parse_sections(md)
    skills   = normalize_and_dedup(extract_skills(sections))
    return {
        "course": sections.course.model_dump(),
        "skills": [s.model_dump() for s in skills],
    }


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: python parse_pdf.py <syllabus.pdf> [<syllabus2.pdf> ...]")
        raise SystemExit(1)
    for p in paths:
        result = pdf_to_skills(p)
        c = result["course"]
        print("=" * 78)
        print(f"{c['title']}  ({c['code']})  —  {len(result['skills'])} skills")
        print("=" * 78)
        for s in result["skills"]:
            print(f"  • {s['name']:<36} [{s['source']:<12}] ← {s['evidence'][:44]}")
        print()
