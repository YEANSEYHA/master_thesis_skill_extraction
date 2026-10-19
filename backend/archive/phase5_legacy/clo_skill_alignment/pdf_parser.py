"""
PDF parser for course syllabi — structured curriculum extraction.

PDF → pymupdf4llm Markdown → RAG chunking + semantic retrieval → qwen3:14b (schema-enforced JSON)

Output shape:
    {
        "course":            { title, code, credits, programme },
        "lessons":           [ {n, topic, covers} ],
        "learning_outcomes": [str, ...],
        "full_text":         str,
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import ollama
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


_LLM_MODEL   = os.getenv("PHASE5_LLM_MODEL", "qwen3:14b")
_EMBED_MODEL = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


# ---------- text extraction ----------

def _extract_text(pdf_source: Union[str, Path, bytes]) -> str:
    if isinstance(pdf_source, (str, Path)):
        return pymupdf4llm.to_markdown(str(pdf_source))
    else:
        import fitz
        doc = fitz.open(stream=pdf_source, filetype="pdf")
        return pymupdf4llm.to_markdown(doc)


# ---------- RAG ----------

# top-k queries: precise fields, need only the best matching chunks
_QUERIES_TOPK = [
    "course title name code credits programme degree program",
    "course learning outcomes CLO ILO intended learning objectives",
]

# threshold queries: lesson rows are many and semantically similar — retrieve ALL above score
_QUERIES_THRESHOLD = [
    "weekly schedule lesson topics timetable session plan",
    "week number lesson learning outcome topic teaching activity detailed plan",
]

_LESSON_THRESHOLD = 0.25  # cosine similarity floor for lesson chunks


def _chunk(md: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_text(md)


def _retrieve(chunks: list[str], top_k: int = 8) -> str:
    model = _get_embedder()
    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    collected: set[int] = set()

    # metadata + CLO: top-k (only need a few precise chunks)
    for query in _QUERIES_TOPK:
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = chunk_embs @ q_emb
        top = np.argsort(scores)[-top_k:][::-1]
        collected.update(top.tolist())

    # lesson plan: threshold — catch ALL week/lesson rows, not just the top one
    for query in _QUERIES_THRESHOLD:
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = chunk_embs @ q_emb
        above = np.where(scores > _LESSON_THRESHOLD)[0]
        collected.update(above.tolist())

    return "\n\n---\n\n".join(chunks[i] for i in sorted(collected))


# ---------- LLM extraction ----------

class Course(BaseModel):
    title:     str | None = None
    code:      str | None = None
    credits:   int | None = None
    programme: str | None = None


class Lesson(BaseModel):
    n:      int = Field(..., description="lesson number, 1-based")
    topic:  str = Field(..., description="topic name, verbatim from document")
    covers: str = Field(..., description="what this lesson teaches, from per-lesson outcomes")


class Syllabus(BaseModel):
    course:            Course
    lessons:           list[Lesson]
    learning_outcomes: list[str] = Field(
        ..., description="course-level CLOs — verbatim from the document"
    )


_SYSTEM_PROMPT = """\
You are a data extractor. You will receive excerpts from a course syllabus.
Extract only what is explicitly written. Never paraphrase, summarize, or invent.

RULES:
  1. COPY values verbatim. Never paraphrase or shorten.
  2. Never invent. If a field is absent return null or [].
  3. course.title     — the course name. Copy verbatim.
  4. course.code      — short alphanumeric code (e.g. GICI3AT). Copy exactly. If absent → null.
  5. course.credits   — integer credit value. Copy the number only. If absent → null.
  6. course.programme — full degree/programme name. Copy full name exactly. If absent → null.
  7. lessons[]        — each topic row in the weekly/session schedule. Copy topic names verbatim.
                        For covers, copy the per-lesson outcome text verbatim from that row.
  8. learning_outcomes[] — course-level CLOs/ILOs only. Copy each verbatim including the
                           identifier prefix (e.g. "CLO1 (K)", "LO2", "ILO3"). Count must
                           match the document exactly. Do NOT use per-lesson outcomes here."""

_USER_PROMPT = """\
Syllabus excerpts (retrieved by semantic search):
---
{context}
---

Extract the structured curriculum. Copy all values verbatim. Never invent."""


def _extract_syllabus(full_text: str) -> Syllabus:
    chunks = _chunk(full_text)
    context = _retrieve(chunks)

    print()
    print("#" * 70)
    print(f"# RAG CONTEXT  ({len(context)} chars from {len(chunks)} chunks, doc={len(full_text)})")
    print("#" * 70)
    print(context)
    print("#" * 70)
    print()

    response = ollama.chat(
        model=_LLM_MODEL,
        think=False,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT.format(context=context)},
        ],
        format=Syllabus.model_json_schema(),
        options={"temperature": 0},
    )
    return Syllabus.model_validate_json(response.message.content)


# ---------- public API ----------

def parse_pdf(pdf_source: Union[str, Path, bytes]) -> dict:
    full_text = _extract_text(pdf_source)
    print(f"[pdf_parser] extracted {len(full_text)} chars · calling {_LLM_MODEL}...")
    syllabus = _extract_syllabus(full_text)

    return {
        "course":            syllabus.course.model_dump(),
        "lessons":           [l.model_dump() for l in syllabus.lessons],
        "learning_outcomes": syllabus.learning_outcomes,
        "full_text":         full_text,
    }


# ---------- CLI test ----------

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else \
        "./school_program/automa_theory_course.pdf"

    result = parse_pdf(path)

    print("=" * 70)
    print("COURSE")
    print("=" * 70)
    print(json.dumps(result["course"], indent=2))
    print()

    print("=" * 70)
    print(f"LESSONS  ({len(result['lessons'])})")
    print("=" * 70)
    for l in result["lessons"]:
        print(f"  {l['n']:>2}. {l['topic']}")
        print(f"      covers: {l['covers']}")
    print()

    print("=" * 70)
    print(f"LEARNING OUTCOMES  ({len(result['learning_outcomes'])})")
    print("=" * 70)
    for i, o in enumerate(result["learning_outcomes"], 1):
        print(f"  {i}. {o}")
