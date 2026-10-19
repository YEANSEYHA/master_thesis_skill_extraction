"""Read the ITC Computer-Sciences syllabus folder (.docx, one file per course)."""

import html
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, Field

SYLLABUS_DIR = Path(__file__).parent / "2025 Course Syllabus - Computer Sciences"

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_text(path):
    """Extract text from a .docx with stdlib only (paragraphs + table rows)."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    body = ElementTree.fromstring(xml).find(f"{_W}body")

    def para_text(p):
        return "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()

    lines = []
    for block in body:
        if block.tag == f"{_W}p":
            text = para_text(block)
            if text:
                lines.append(text)
        elif block.tag == f"{_W}tbl":
            for row in block.iter(f"{_W}tr"):
                cells = [
                    " ".join(filter(None, (para_text(p) for p in cell.iter(f"{_W}p"))))
                    for cell in row.findall(f"{_W}tc")
                ]
                if any(cells):
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def docx_to_markdown(path):
    """.docx → Markdown with stdlib only.

    Emits the same shape V8's `pdf_to_markdown` (pymupdf4llm) produces, because
    `parse_sections` reads syllabi as Markdown: headings become '#' and tables
    become real pipe tables '| a | b |' with a header separator row — that is
    what keeps the CLO (field 14) and lesson-plan (field 18) tables intact.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    body = ElementTree.fromstring(xml).find(f"{_W}body")

    def para_text(p):
        return "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()

    def heading_level(p):
        """Word style 'Heading2'/'Titre2' → 2, else None."""
        style = p.find(f"{_W}pPr/{_W}pStyle")
        val = style.get(f"{_W}val", "") if style is not None else ""
        m = re.fullmatch(r"(?:Heading|Titre|berschrift)\s*(\d)", val, re.I)
        return int(m.group(1)) if m else None

    def cell_text(tc):
        # newlines would break the pipe row, so join a cell's paragraphs inline
        parts = [t for t in (para_text(p) for p in tc.iter(f"{_W}p")) if t]
        return " ".join(parts).replace("|", "\\|")

    out = []
    for block in body:
        if block.tag == f"{_W}p":
            text = para_text(block)
            if not text:
                continue
            level = heading_level(block)
            out.append(f"{'#' * level} {text}" if level else text)
            out.append("")
        elif block.tag == f"{_W}tbl":
            rows = []
            for tr in block.iter(f"{_W}tr"):
                cells = [cell_text(tc) for tc in tr.findall(f"{_W}tc")]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * width)
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
            out.append("")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


_docling_converter = None


def _docling():
    """Lazy singleton — building a DocumentConverter is slow, reuse it."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _suffix_of(source, filename=None):
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    suffix = Path(filename or "").suffix.lower()
    if suffix:
        return suffix
    # sniff: docx is a zip ("PK"), pdf starts with "%PDF"
    return ".docx" if source[:2] == b"PK" else ".pdf"


def pdf_or_word_to_markdown(source, filename=None, fallback=True):
    """Any syllabus (.pdf / .docx) → Markdown, via docling.

    `source` is a path (str/Path) or raw bytes; with bytes, pass `filename` (or
    a suffix like ".docx") so the format can be told apart.

    Docling handles both formats with one converter and beats the alternatives
    on the ITC syllabi: bullets inside table cells keep their "- " markers (so a
    lesson row's several LLOs stay separable instead of fusing into one run-on),
    bold survives (the Bloom verbs), and merged cells are expanded. Verified
    lossless at the word level on the Year-5 folder.

    Docling escapes '&' as '&amp;', which is undone here.

    With `fallback=True`, an unavailable docling (it needs Python < 3.14) falls
    back to the stdlib `docx_to_markdown` / pymupdf4llm, so the function still
    works — with slightly poorer tables — wherever it is imported.
    """
    suffix = _suffix_of(source, filename)
    if suffix not in (".docx", ".pdf"):
        raise ValueError(f"unsupported syllabus format: {suffix or source!r}")

    try:
        if isinstance(source, (str, Path)):
            doc = _docling().convert(str(source)).document
        else:
            from docling.datamodel.base_models import DocumentStream
            stream = DocumentStream(name=filename or f"upload{suffix}",
                                    stream=io.BytesIO(source))
            doc = _docling().convert(stream).document
        return html.unescape(doc.export_to_markdown())
    except ImportError:
        if not fallback:
            raise
        return _markdown_without_docling(source, suffix)


def _markdown_without_docling(source, suffix):
    """Fallback path: stdlib reader for .docx, pymupdf4llm for .pdf."""
    if suffix == ".docx":
        return docx_to_markdown(source if isinstance(source, (str, Path))
                                else io.BytesIO(source))
    import pymupdf4llm
    if isinstance(source, (str, Path)):
        return pymupdf4llm.to_markdown(str(source))
    import fitz
    return pymupdf4llm.to_markdown(fitz.open(stream=source, filetype="pdf"))


# ══════════════════════════════════════════════════════════════════════════════
# Markdown → CLOs → skills.  Copied from V8's recommendation_v3/parse_pdf.py
# (prompts and schemas verbatim, so results stay comparable); only the entry
# point differs — here the Markdown comes from docling, PDF *or* .docx.
# ══════════════════════════════════════════════════════════════════════════════

_LLM = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")


def _chat(label: str, schema: dict, system: str, user: str) -> str:
    """One schema-locked qwen3 call, with debug timing printed to stderr."""
    import ollama
    print(f"[{label}] → {_LLM}: sending {len(user):,} chars …",
          file=sys.stderr, flush=True)
    t0 = time.time()
    resp = ollama.chat(
        model=_LLM,
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
    out = resp.message.content
    print(f"[{label}] ← done in {time.time() - t0:.1f}s ({len(out):,} chars)",
          file=sys.stderr, flush=True)
    return out


# ── Stage 2: Markdown → Sections (this is where the CLOs come out) ───────────────

class Course(BaseModel):
    title:     str | None = None
    code:      str | None = None
    credits:   float | None = None   # float, not int: ITC has 2.5-credit courses
                                     # (an int schema makes strict backends reject them)
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
        f"- {ls.topic}" + (f"  ({'; '.join(ls.outcomes)})" if ls.outcomes else "")
        for ls in sec.lessons
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


def _title_token(token: str, first: bool, acronyms: set) -> str:
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


def clean_skill_name(name: str, aliases: dict = None, acronyms: set = None):
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


def normalize_and_dedup(skills, aliases: dict = None, acronyms: set = None):
    """Clean names and drop duplicate skills, keeping first evidence."""
    seen = set()
    out = []
    for s in skills:
        name = clean_skill_name(s.name, aliases=aliases, acronyms=acronyms)
        key = _norm_key(name or "")
        if key and key not in seen:
            seen.add(key)
            out.append(s.model_copy(update={"name": name}))
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────────

def syllabus_to_skills(source, filename=None) -> dict:
    """Full pipeline: .docx/.pdf → markdown (docling) → sections → skills.

    Same as V8's `pdf_to_skills`, but it also accepts .docx and it keeps the
    CLOs and lessons in the result instead of discarding them after stage 3.
    """
    md = pdf_or_word_to_markdown(source, filename=filename)
    sections = parse_sections(md)
    skills = normalize_and_dedup(extract_skills(sections))
    return {
        "course": sections.course.model_dump(),
        "description": sections.description,
        "clos": sections.clos,
        "lessons": [ls.model_dump() for ls in sections.lessons],
        "skills": [s.model_dump() for s in skills],
    }


def read_syllabus_folder(folder=SYLLABUS_DIR, with_text=True):
    """Read every course .docx under `folder`.

    Returns a list of dicts: {code, title, year, semester, filename, path, text}
    sorted by year/semester/code.
    """
    folder = Path(folder)
    courses = []
    for path in sorted(folder.rglob("*.docx")):
        if path.name.startswith("~$"):  # Word lock files
            continue

        stem = path.stem
        code, _, rest = stem.partition("_")
        title = re.sub(r"_?20\d\d[-_]20\d\d$", "", rest).replace("_", " ").strip()

        group = path.parent.name  # e.g. "Year 3 Semester 1"
        m = re.search(r"Year\s*(\d+)\s*Semester\s*(\d+)", group)

        courses.append(
            {
                "code": code,
                "title": title or stem,
                "year": int(m.group(1)) if m else None,
                "semester": int(m.group(2)) if m else None,
                "filename": path.name,
                "path": str(path),
                "text": docx_to_text(path) if with_text else None,
            }
        )

    courses.sort(key=lambda c: (c["year"] or 0, c["semester"] or 0, c["code"]))
    return courses


def main():
    courses = read_syllabus_folder(with_text=True)
    print(f"{len(courses)} courses in {SYLLABUS_DIR.name}\n")
    for c in courses:
        print(f"Y{c['year']}S{c['semester']}  {c['code']:<12} {c['title']:<55} "
              f"{len(c['text'] or '')} chars")


if __name__ == "__main__":
    main()
