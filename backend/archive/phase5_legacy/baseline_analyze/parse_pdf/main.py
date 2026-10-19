"""Parse a course-syllabus PDF → Markdown → sections → skills.

Pipeline (2 LLM calls per course):
    PDF → pdf_to_markdown() → parse_sections() → extract_skills()
        → normalize_and_dedup() → course skill list (each skill traceable).
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Union

import ollama
import pymupdf4llm
from pydantic import BaseModel, Field

# Which model runs the two extraction calls. Set PHASE5_LLM_MODEL in backend/.env:
#
#   PHASE5_LLM_MODEL=qwen3:14b        local, free, slower  (default)
#   PHASE5_LLM_MODEL=gpt-5.4-nano     hosted, ~$0.006/course
#
# The provider is inferred from the name — anything starting with "gpt-" goes to
# OpenAI, everything else to Ollama — so switching backends is one line in .env
# with no code change. PHASE5_LLM_PROVIDER forces it if a name ever breaks the
# convention. Read per call, not at import, so a reload picks up a change.
_DEFAULT_LLM = "qwen3:14b"
_OPENAI_MAX_COMPLETION = 16384
_OPENAI_RETRIES = 3


_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"   # backend/.env
_env_cache: dict = {"mtime": None, "vals": {}}


def _from_env_file(key: str) -> str | None:
    """Read `key` straight from backend/.env, re-parsing only when the file
    changes. config.py loads .env into os.environ at STARTUP, so without this a
    model switch would need a server restart — reading the file live is what
    makes PHASE5_LLM_MODEL take effect on the next extraction."""
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
    """API key from backend/.env or the environment — this module is imported
    both by the FastAPI process (which loads .env) and by standalone scripts."""
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
    is appended to the user turn and JSON mode enforces valid JSON — the exact
    shaping the model benchmark scored, so quality matches those numbers.

    Its sampling params are fixed (no `temperature`), the output budget is
    `max_completion_tokens`, and a reasoning model can spend the whole budget
    thinking and return empty content — hence the retry.
    """
    from openai import OpenAI
    client = OpenAI(api_key=_openai_key(), timeout=600)
    payload = user + "\n\nReturn ONLY JSON matching this schema:\n" + json.dumps(schema)

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
            # JSON mode should return bare JSON, but strip fences defensively —
            # the callers parse with pydantic's model_validate_json, which has
            # no tolerance for a ```json wrapper.
            out = re.sub(r"^```(?:json)?|```$", "", out, flags=re.M).strip()
            m = re.search(r"\{.*\}", out, re.S)
            return m.group(0) if m else out
        except Exception as e:                          # noqa: BLE001
            last = e
            print(f"[{model}] retry {attempt}/{_OPENAI_RETRIES}: "
                  f"{type(e).__name__}: {str(e)[:90]}", file=sys.stderr, flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"{model} failed after {_OPENAI_RETRIES} attempts: {last}")


def _chat(label: str, schema: dict, system: str, user: str) -> str:
    """One schema-locked call on the configured model, timed to stderr."""
    model = active_model()
    provider = _provider(model)
    print(f"[{label}] → {model} ({provider}): sending {len(user):,} chars …",
          file=sys.stderr, flush=True)
    t0 = time.time()
    fn = _chat_openai if provider == "openai" else _chat_ollama
    out = fn(model, schema, system, user)
    print(f"[{label}] ← done in {time.time() - t0:.1f}s ({len(out):,} chars)",
          file=sys.stderr, flush=True)
    return out


# ── Stage 1: PDF → Markdown ──────────────────────────────────────────────────────

def pdf_to_markdown(pdf_source: Union[str, Path, bytes]) -> str:
    """Convert a PDF into a Markdown string.

    Accepts either a file path (str/Path) or the raw PDF bytes (e.g. an
    uploaded file). Uses pymupdf4llm so the document's structure is preserved
    as Markdown syntax — headings become '#', and (crucially for syllabi)
    tables become pipe tables '| ... |'. That keeps the CLO and lesson-plan
    tables intact for the later LLM extraction step.
    """
    if isinstance(pdf_source, (str, Path)):
        return pymupdf4llm.to_markdown(str(pdf_source))
    # bytes → open as an in-memory PyMuPDF document
    import fitz
    doc = fitz.open(stream=pdf_source, filetype="pdf")
    return pymupdf4llm.to_markdown(doc)


# ── Stage 2: Markdown → Sections ─────────────────────────────────────────────────

class Course(BaseModel):
    title:     str | None = None
    code:      str | None = None
    credits:   float | None = None   # float, not int: ITC has 1.5/2.5-credit courses
    programme: str | None = None
    category:  str | None = None    # labour-market bucket the course prepares for
                                    # (one of the 17 LinkedIn job categories), inferred
                                    # by the LLM from the course CONTENT, not the title
    year:      int | None = None    # study year (1-6) within the programme; LLM-inferred
                                    # (explicit field, availability, or course code) with
                                    # a regex fallback in infer_year()


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
  - course.year: the STUDY YEAR (an integer 1-6) this course belongs to in the
    programme. Look for an explicit field ("Year", "Course Availability",
    "Semester 7" → year 4), or read it from the course code — the FIRST digit
    after the letter prefix is the year; a second digit is the SEMESTER, not
    part of the year ("GICI3AT" → 3, "GICI31ACI" → 3 (semester 1), "GICI42NET2"
    → 4, "GICI51CC" → 5, "CS201" → 2). Use null if genuinely undeterminable.
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
             "programme": "Computer Science", "category": "it", "year": 2},
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


def infer_year(code: str | None = None, filename: str | None = None) -> int | None:
    """Regex fallback when the LLM returns course.year = null.

    First digit after the letter prefix of the course code ("GICI42NET2" → 4,
    "CS201" → 2), else a "Year 4" / "Y4_S2" pattern in the filename.
    """
    import re
    if code:
        m = re.match(r"[A-Za-z]+([1-6])", code.strip())
        if m:
            return int(m.group(1))
    if filename:
        m = re.search(r"(?i)(?<![a-z0-9])Y(?:ear)?[ _-]?([1-6])(?![0-9])", filename)
        if m:
            return int(m.group(1))
        # filenames often START with the course code ("GICI31STA_Statistics…")
        base = filename.rsplit("/", 1)[-1]
        m = re.match(r"[A-Za-z]{2,}([1-6])\d?[A-Za-z]", base)
        if m:
            return int(m.group(1))
    return None


def _salvage_sections(raw: str) -> Sections:
    """Recover from a near-valid LLM response: drop lesson rows with a null/
    missing topic (nano occasionally emits one for a blank schedule row)
    instead of failing the whole document."""
    import json as _json
    data = _json.loads(raw)
    data["lessons"] = [l for l in (data.get("lessons") or [])
                       if isinstance(l, dict) and l.get("topic")]
    return Sections.model_validate(data)


def parse_sections(markdown: str, retries: int = 1) -> Sections:
    """Markdown → structured {course, description, clos, lessons} via LLM.

    Validation failures are handled twice over: first salvage the response
    (drop null-topic lesson rows), then — if the payload is unrecoverable —
    make a fresh LLM attempt, up to `retries` times.
    """
    last_err: Exception | None = None
    for _attempt in range(retries + 1):
        out = _chat("parse_sections", Sections.model_json_schema(),
                    _SECTIONS_SYS, markdown)    # whole doc — it fits in context
        try:
            return Sections.model_validate_json(out)
        except Exception as e:
            try:
                return _salvage_sections(out)
            except Exception:
                last_err = e                    # salvage failed too → retry LLM
    raise last_err


# ── Stage 3: Sections → Skills ───────────────────────────────────────────────────

class Skill(BaseModel):
    name:     str                                       # normalized skill, e.g. "K-Means Clustering"
    source:   str = Field(description="CLO | Lesson Topic | Description")
    evidence: str = Field(description="the exact source text the skill came from")


class SkillList(BaseModel):
    skills: list[Skill]


_SKILLS_SYS = """\
You extract the labour-market skills a course teaches, as JSON. This must
work for ANY discipline — engineering, the sciences, health, business, law,
the arts — not only computing. Extract the skills of whatever domain the
syllabus is about; never force a course into computing terms.

WHAT COUNTS AS A SKILL — the job-advert test.
A skill is a concrete technology, technique, method, tool, or concept that an
employer would plausibly NAME in a job advert — expressed as a noun phrase.
  Keep: "Word Embeddings", "Finite Element Analysis", "Titration".
  Drop: "Chain Rule", "Free Body Diagram", "Loop Unrolling" — real lecture
        content, but nobody hires for them; fold each into the skill it serves.

DECISION PROCEDURE — run every candidate phrase through these steps in order:
1. DROP learning activities and admin: report writing, lab exercises, case
   studies, lectures, tutorials, quizzes, assignments, homework, exams,
   presentations, group projects, attendance, "review lesson".
2. DROP job titles, roles and organisation types ("consultant", "vendor",
   "service provider", "client", "distributor"), titles of case studies or
   student projects, and research instruments used only as coursework
   ("survey questionnaire", "interview protocol").
3. DROP soft and transversal skills: communication, teamwork, leadership,
   presentation skills, ethics, professional responsibility, critical
   thinking, problem solving, time management, lifelong learning. Extract
   only TECHNICAL skills; attitude-type CLOs usually yield nothing.
4. DROP or fold anything that fails the job-advert test above.
5. TURN verbs into the underlying skill noun:
   "Apply secure coding practices" -> "Secure Coding";
   "Design reinforced concrete beams" -> "Reinforced Concrete Design";
   "Prepare financial statements" -> "Financial Statement Preparation".
6. SPLIT siblings; NEVER decompose a skill into its stages. Split only when
   each part is independently hireable — an advert could ask for one without
   the other: "SQL Injection, XSS" -> two skills; "bending moment, shear
   force" -> two skills. Stages of ONE method are NOT split: "forward pass,
   backward pass, weight update" -> ONE skill "Neural Network Training".
   When unsure, emit the parent skill, not the parts.
7. A PRODUCT IS ONE SKILL. A table of the modules/services/parts inside one
   named product, suite, instrument or framework yields the product, not the
   parts: "pump, column, detector" -> "High-Performance Liquid Chromatography
   (HPLC)", not three entries. Name a
   component only if it is independently hireable, and then keep the product
   identity in the name ("Swift (Object Storage)") — NEVER a bare generic
   component name ("Object Storage", "Dashboard", "Image Service"): a bare
   name matches unrelated jobs and is worse than omitting the entry.

NAMING STANDARD — the same skill from two syllabi must get the same name:
- canonical job-market term; drop filler suffixes ("Algorithm", "Method",
  "Techniques", "Model") that only decorate a named method:
  "K-mean algorithm" -> "K-Means"; "decision tree algorithm" ->
  "Decision Trees"; "titration method" -> "Titration".
- the plural adverts use: "Decision Trees", "Neural Networks" — but
  "Machine Learning", "Data Mining".
- acronyms: ONE entry, expanded with the short form in brackets —
  "Principal Component Analysis (PCA)" — never two entries for the two forms.
- clean, title-case name.

FINAL REVIEW — after drafting the full list, reread it once and delete:
- any entry that is another entry plus decoration: "Welding" + "Welding
  Techniques" + "Welding Process Management" -> keep only "Welding";
- any two entries wording the same concept differently — keep the more
  job-relevant one.
Every concept appears exactly ONCE, however many sections repeat it. Do NOT
aim for any particular count — a detailed lesson plan must not yield
finer-grained skills than a terse syllabus covering the same content.

OUTPUT — JSON only: {"skills": [{"name", "source", "evidence"}, ...]}
  source   — exactly one of: CLO, Lesson Topic, Description
  evidence — the verbatim source text the skill came from

EXAMPLE 1 (computing course):
  ### DESCRIPTION
  Introduces machine learning and secure coding for the web.
  ### CLOs
  - CLO1 Apply secure coding practices to prevent common vulnerabilities.
  - CLO2 Build and evaluate machine learning models on real datasets.
  ### LESSON TOPICS
  - Machine learning fundamentals  (Explain the ML workflow. Quiz.)
  - Unsupervised learning (K-mean algorithm)  (Apply the K-mean algorithm to a dataset.)
  - Supervised learning (decision tree)  (Apply the decision tree algorithm. Write a report.)
  - Neural network training (forward pass, backward pass, weight update)  (Derive the update rule with the chain rule.)
  - Secure coding for common vulnerabilities (SQL Injection, XSS)  (Complete lab exercises.)
  - Principal component analysis for dimensionality reduction  (Reduce features with PCA.)

OUTPUT 1:
{
  "skills": [
    {"name": "Secure Coding",           "source": "CLO",          "evidence": "CLO1 Apply secure coding practices to prevent common vulnerabilities."},
    {"name": "Machine Learning",        "source": "Lesson Topic", "evidence": "Machine learning fundamentals"},
    {"name": "K-Means",                 "source": "Lesson Topic", "evidence": "Unsupervised learning (K-mean algorithm)"},
    {"name": "Decision Trees",          "source": "Lesson Topic", "evidence": "Supervised learning (decision tree)"},
    {"name": "Neural Network Training", "source": "Lesson Topic", "evidence": "Neural network training (forward pass, backward pass, weight update)"},
    {"name": "SQL Injection",           "source": "Lesson Topic", "evidence": "Secure coding for common vulnerabilities (SQL Injection, XSS)"},
    {"name": "Cross-Site Scripting (XSS)", "source": "Lesson Topic", "evidence": "Secure coding for common vulnerabilities (SQL Injection, XSS)"},
    {"name": "Principal Component Analysis (PCA)", "source": "Lesson Topic", "evidence": "Principal component analysis for dimensionality reduction"}
  ]
}
Why: machine learning appears in the description, CLO2 AND a lesson -> ONE
entry. Forward/backward pass and weight update are stages of one method ->
not split; "chain rule" fails the job-advert test -> dropped. "Quiz",
"Write a report" and "lab exercises" are activities -> dropped.
"K-mean algorithm" -> "K-Means": filler dropped, canonical name.

EXAMPLE 2 (NON-computing course — the SAME logic outside IT):
  ### DESCRIPTION
  Covers reinforced-concrete structural design and workshop practice.
  ### CLOs
  - CLO1 Design reinforced concrete beams according to load requirements.
  - CLO2 Join steel components using appropriate welding techniques.
  ### LESSON TOPICS
  - Structural load analysis (bending moment, shear force)  (Solve beam problems. Submit a report.)
  - Welding techniques and welding process management  (Weld sample joints in the workshop.)
  - Office productivity suite (word processor, spreadsheet, presentation tool)  (Prepare project documentation.)

OUTPUT 2:
{
  "skills": [
    {"name": "Reinforced Concrete Design",   "source": "CLO",          "evidence": "CLO1 Design reinforced concrete beams according to load requirements."},
    {"name": "Bending Moment Analysis",      "source": "Lesson Topic", "evidence": "Structural load analysis (bending moment, shear force)"},
    {"name": "Shear Force Analysis",         "source": "Lesson Topic", "evidence": "Structural load analysis (bending moment, shear force)"},
    {"name": "Welding",                      "source": "CLO",          "evidence": "CLO2 Join steel components using appropriate welding techniques."},
    {"name": "Office Productivity Software", "source": "Lesson Topic", "evidence": "Office productivity suite (word processor, spreadsheet, presentation tool)"}
  ]
}
Why: bending moment / shear force are independently hireable siblings ->
split. "Welding techniques" + "welding process management" are decorated
variants of one skill -> ONE "Welding". The office-suite components are parts
of one product -> the suite is one skill; no bare "Spreadsheet" entry.
"Submit a report" is an activity -> dropped."""


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
    last_err: Exception | None = None
    for _attempt in range(2):                   # one retry on invalid JSON/schema
        out = _chat("extract_skills", SkillList.model_json_schema(),
                    _SKILLS_SYS, user)
        try:
            return SkillList.model_validate_json(out).skills
        except Exception as e:
            last_err = e
    raise last_err


# ── Stage 4: Normalize + Dedup ───────────────────────────────────────────────────

def _norm_key(name: str) -> str:
    """Canonical match key — IDENTICAL to the job side (aggregate_job_skills),
    so course "Decision Trees" lines up with job "decision trees"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalize_and_dedup(skills: list[Skill]) -> list[Skill]:
    """Drop duplicate skills, keeping the first (prettiest) display label."""
    seen: set[str] = set()
    out: list[Skill] = []
    for s in skills:
        key = _norm_key(s.name)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
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
    import sys

    # pass PDF paths as args, or default to the three test syllabi
    paths = sys.argv[1:] or [
        "./school_curriculums/automa_theory_course.pdf",
        "./Introductionto Artificial Intelligence.pdf",
        "./Cyber Crime and Security Enhanced Programming.pdf",
    ]
    for p in paths:
        result = pdf_to_skills(p)
        c = result["course"]
        print("=" * 78)
        print(f"{c['title']}  ({c['code']})  —  {len(result['skills'])} skills")
        print("=" * 78)
        for s in result["skills"]:
            print(f"  • {s['name']:<36} [{s['source']:<12}] ← {s['evidence'][:44]}")
        print()
