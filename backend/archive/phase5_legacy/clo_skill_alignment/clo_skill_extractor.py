"""
Stage 3 of Phase 5 pipeline — CLO -> labour-market skill phrases.

DOMAIN-ADAPTIVE prompt that adjusts to ANY course area
(Computer Science, Engineering, Healthcare, Business, etc.) by
threading course context through every LLM call.

Takes a parsed course (output of pdf_parser.parse_pdf) and, for each
Course Learning Outcome (CLO), uses qwen3:14b to extract the specific
labour-market-relevant skills that mastering that CLO teaches.

Input shape (from pdf_parser):
    {
        "course": { "code", "title", "credits", "programme" },
        "lessons": [ ... ],
        "learning_outcomes": [ "CLO text 1", "CLO text 2", ... ]
    }

Output shape:
    {
        "course_code":   str,
        "course_title":  str,
        "course_programme": str,
        "clo_skills":    { "CLO1": [skill, ...], "CLO2": [...] },
        "skills_union":  [deduplicated union],
        "n_clos":        int,
        "n_skills":      int
    }

CLI usage:
    python clo_skill_extractor.py [pdf_path]
"""

from __future__ import annotations

import os
import re
from typing import Any

import ollama
from pydantic import BaseModel, Field


_LLM_MODEL = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")


# ---------- schema ----------

class CLOSkills(BaseModel):
    skills: list[str] = Field(
        ...,
        description="3-8 labour-market skill phrases acquired by mastering this CLO"
    )


# ---------- domain detection ----------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "computer_science": [
        "computer", "informatics", "software", "data", "ict",
        "artificial intelligence", "computing",
    ],
    "electrical_electronic": [
        "electrical", "electronic", "telecommunication", "circuit",
        "power",
    ],
    "civil_construction": [
        "civil", "construction", "architecture", "structural", "urban",
    ],
    "mechanical_industrial": [
        "mechanical", "industrial", "manufacturing", "automation",
        "robotics",
    ],
    "chemical_food": [
        "chemical", "chemistry", "food", "biotechnology", "process",
    ],
    "healthcare_life": [
        "medical", "medicine", "health", "nursing", "pharmacy",
        "biology", "life sciences",
    ],
    "business_management": [
        "business", "management", "marketing", "finance", "economics",
        "administration", "accounting",
    ],
    "education_social": [
        "education", "teaching", "social", "psychology", "sociology",
        "linguistics",
    ],
    "agriculture_environment": [
        "agriculture", "agronomy", "environment", "forestry",
    ],
}


def _detect_domain(programme: str | None, course_title: str | None) -> str:
    """Best-effort match course area from programme + title text."""
    text = " ".join(filter(None, [programme, course_title])).lower()
    if not text:
        return "general"
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "general"


# ---------- domain-specific few-shot examples ----------

_DOMAIN_EXAMPLES: dict[str, list[tuple[str, list[str]]]] = {
    "computer_science": [
        (
            "Design and implement scalable web applications using React "
            "and Node.js.",
            ["React development", "Node.js backend", "REST API design",
             "scalable architecture design", "front-end engineering"],
        ),
        (
            "Apply machine learning techniques to classify text documents "
            "using Python.",
            ["text classification", "machine learning modelling",
             "Python programming", "feature engineering",
             "model evaluation"],
        ),
    ],
    "electrical_electronic": [
        (
            "Design analog and digital circuits using SPICE simulation "
            "and breadboard prototyping.",
            ["SPICE circuit simulation", "analog circuit design",
             "digital circuit design", "breadboard prototyping",
             "schematic capture"],
        ),
    ],
    "civil_construction": [
        (
            "Analyze structural loads on multi-story buildings using FEM "
            "in AutoCAD and ANSYS.",
            ["AutoCAD drafting", "ANSYS structural simulation",
             "finite element analysis", "load distribution analysis",
             "structural design"],
        ),
    ],
    "mechanical_industrial": [
        (
            "Design mechanical assemblies in SolidWorks and validate them "
            "with stress analysis.",
            ["SolidWorks modelling", "mechanical assembly design",
             "stress analysis", "CAD documentation",
             "tolerance specification"],
        ),
    ],
    "chemical_food": [
        (
            "Evaluate food safety using HACCP principles and microbial "
            "testing procedures.",
            ["HACCP analysis", "microbial testing",
             "food safety auditing", "risk assessment",
             "laboratory procedures"],
        ),
    ],
    "healthcare_life": [
        (
            "Interpret ECG recordings and identify common arrhythmias "
            "from 12-lead readings.",
            ["ECG interpretation", "arrhythmia detection",
             "12-lead analysis", "clinical reasoning",
             "patient assessment"],
        ),
    ],
    "business_management": [
        (
            "Build financial models in Excel and present results to "
            "stakeholders using Power BI dashboards.",
            ["Excel financial modelling", "Power BI dashboards",
             "stakeholder communication", "data presentation",
             "financial analysis"],
        ),
    ],
    "education_social": [
        (
            "Design lesson plans using Bloom's taxonomy and conduct "
            "formative assessments.",
            ["lesson planning", "formative assessment design",
             "Bloom taxonomy application", "classroom facilitation",
             "learning evaluation"],
        ),
    ],
    "agriculture_environment": [
        (
            "Analyze soil samples using laboratory tests and recommend "
            "fertilization plans for sustainable yields.",
            ["soil sample analysis", "laboratory testing",
             "fertilization planning", "sustainable agriculture",
             "agronomic recommendation"],
        ),
    ],
    "general": [
        (
            "Apply project management principles to plan and execute "
            "a 3-month team project.",
            ["project planning", "team coordination",
             "milestone tracking", "stakeholder communication",
             "risk management"],
        ),
    ],
}


# ---------- Bloom domain detection ----------

def _detect_bloom_domain(clo_text: str) -> str:
    """Extract (K) / (S) / (A) marker from the start of the CLO text."""
    m = re.match(r"\s*\(?\s*([KSA])\s*\)?", clo_text)
    if m:
        return m.group(1).upper()
    # fallback: look for verb cues
    head = clo_text.lower()[:50]
    if any(v in head for v in ["analyze", "explain", "describe", "compare",
                                "characterize", "interpret"]):
        return "K"
    if any(v in head for v in ["design", "construct", "build", "implement",
                                "develop", "create"]):
        return "S"
    if any(v in head for v in ["evaluate", "appraise", "judge", "reflect",
                                "collaborate", "communicate"]):
        return "A"
    return "?"


_BLOOM_GUIDANCE: dict[str, str] = {
    "K": ("This is a KNOWLEDGE-domain CLO. Favor ANALYTICAL and "
          "CONCEPTUAL skills: analysis, comparison, interpretation, "
          "characterization, theoretical reasoning."),
    "S": ("This is a SKILL-domain CLO. Favor APPLIED and CONSTRUCTIVE "
          "skills: design, build, implement, transform, model, "
          "configure. Include named tools/technologies verbatim."),
    "A": ("This is an ATTITUDE-domain CLO. Favor SOFT and PROFESSIONAL "
          "skills: evaluation, peer review, ethical reasoning, "
          "communication, collaboration, professional responsibility."),
    "?": ("Bloom domain not detected. Use a balanced mix of analytical, "
          "applied, and professional skills based on the CLO text."),
}


# ---------- prompt builders ----------

def _build_system_prompt(course_title: str | None,
                         course_programme: str | None,
                         clo_domain: str) -> str:
    """Build a context-aware system prompt."""
    return f"""You are a curriculum skill extractor.

COURSE CONTEXT
  Course title:      {course_title or "(unknown)"}
  Course programme:  {course_programme or "(unknown)"}
  CLO Bloom domain:  {clo_domain}

{_BLOOM_GUIDANCE[clo_domain]}

TASK
Given ONE Course Learning Outcome (CLO) from this course, list 3 to 8
labour-market-relevant skill phrases that a learner acquires after
mastering it.

RULES
  1. Return between 3 and 8 skills.
  2. Each phrase is 2-5 words.
  3. NAMED TECHNOLOGIES, TOOLS, LANGUAGES, METHODS must appear verbatim
     in the skill phrase if the CLO mentions one (e.g. "Python
     programming", "AutoCAD design", "ANSYS simulation",
     "HACCP analysis", "Power BI dashboards", "DFA construction").
  4. One concept per entry. Never use "and" or "&". Split compound
     ideas into separate phrases.
  5. No parentheses inside skill phrases.
  6. Include BOTH hard (technical) AND soft (interpersonal, cognitive,
     professional) skills when the CLO implies them.
  7. AVOID GENERIC FILLERS. These are TOO VAGUE and FORBIDDEN:
       ✗ "apply X theory"          ✗ "understand X concepts"
       ✗ "solve problems"          ✗ "use techniques"
       ✗ "reason about X"          ✗ "study X behavior"
       ✗ "demonstrate knowledge"   ✗ "perform analysis"
     Replace with SPECIFIC, ACTIONABLE alternatives.
  8. CV TEST: each skill phrase should be plausible on a graduate's
     resume. "DFA minimization" passes. "apply automata theory" fails.
  9. No degrees, certifications, job titles, years of experience,
     course names, or benefits.
 10. If two phrases overlap, keep the more specific one.
 11. Return ONLY a JSON object matching the schema."""


def _build_user_prompt(clo_text: str,
                       domain: str) -> str:
    """Build a domain-specific user prompt with relevant few-shot example."""
    examples = _DOMAIN_EXAMPLES.get(domain, _DOMAIN_EXAMPLES["general"])
    ex_text = "\n\n".join(
        f'EXAMPLE\nCLO: "{ex_clo}"\n\nCorrect output:\n{{\n  "skills": [\n'
        + ",\n".join(f'    "{s}"' for s in ex_skills)
        + "\n  ]\n}"
        for ex_clo, ex_skills in examples
    )
    return f"""{ex_text}

Now extract for THIS CLO:

CLO: "{clo_text.strip()}"

Return ONLY a JSON object with a "skills" array of 3-8 phrases.
Each phrase must pass the CV TEST (rule 8)."""


# ---------- LLM call ----------

def extract_skills_from_clo(clo_text: str,
                            course_title: str | None = None,
                            course_programme: str | None = None) -> list[str]:
    """Call the LLM on a single CLO and return skill phrases."""
    domain_area = _detect_domain(course_programme, course_title)
    bloom = _detect_bloom_domain(clo_text)

    system_prompt = _build_system_prompt(course_title, course_programme, bloom)
    user_prompt = _build_user_prompt(clo_text, domain_area)

    response = ollama.chat(
        model=_LLM_MODEL,
        think=False,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        format=CLOSkills.model_json_schema(),
        options={"temperature": 0},
    )
    parsed = CLOSkills.model_validate_json(response.message.content)
    return parsed.skills


# ---------- per-course orchestration ----------

def extract_course_skills(course_data: dict[str, Any]) -> dict[str, Any]:
    """Run Stage-3 skill extraction across all CLOs of a parsed course."""
    course_meta = course_data.get("course", {}) or {}
    course_title = course_meta.get("title")
    course_programme = course_meta.get("programme")

    domain_area = _detect_domain(course_programme, course_title)
    print(f"  [domain] detected area: {domain_area}")

    clos: list[str] = course_data.get("learning_outcomes", [])

    clo_skills: dict[str, list[str]] = {}
    for i, clo_text in enumerate(clos, start=1):
        clo_id = f"CLO{i}"
        bloom = _detect_bloom_domain(clo_text)
        print(f"  -> {clo_id} [Bloom={bloom}] "
              f"({len(clo_text)} chars)...")
        skills = extract_skills_from_clo(
            clo_text,
            course_title=course_title,
            course_programme=course_programme,
        )
        clo_skills[clo_id] = skills

    # dedup-preserving union
    seen: set[str] = set()
    union: list[str] = []
    for skills in clo_skills.values():
        for s in skills:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                union.append(s)

    return {
        "course_code":      course_meta.get("code"),
        "course_title":     course_title,
        "course_programme": course_programme,
        "domain_area":      domain_area,
        "clo_skills":       clo_skills,
        "skills_union":     union,
        "n_clos":           len(clo_skills),
        "n_skills":         len(union),
    }


# ---------- CLI ----------

if __name__ == "__main__":
    import sys

    from pdf_parser import parse_pdf

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "./"
        "school_program/automa_theory_course.pdf"
    )

    print(f"[Stage 2] Parsing {path} ...")
    course = parse_pdf(path)
    print(f"[Stage 2] -> {len(course['learning_outcomes'])} CLOs extracted\n")

    print(f"[Stage 3] Extracting skills per CLO via {_LLM_MODEL}...")
    result = extract_course_skills(course)

    print()
    print("=" * 70)
    print(f"COURSE: {result['course_code']} — {result['course_title']}")
    print(f"Programme: {result['course_programme']}")
    print(f"Detected domain area: {result['domain_area']}")
    print("=" * 70)

    for clo_id, skills in result["clo_skills"].items():
        print(f"\n{clo_id} skills ({len(skills)}):")
        for s in skills:
            print(f"  • {s}")

    print()
    print("=" * 70)
    print(f"SKILLS UNION  ({result['n_skills']} unique across "
          f"{result['n_clos']} CLOs)")
    print("=" * 70)
    for s in result["skills_union"]:
        print(f"  • {s}")
