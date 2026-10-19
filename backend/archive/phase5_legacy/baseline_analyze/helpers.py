"""Shared IO helpers for the AMC baseline (file reading)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

_HERE = Path(__file__).resolve().parent
COURSE_MD = _HERE / "software_engineering_databases.md"


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream a JSONL file, yielding one parsed record per non-empty line.

    Memory-safe for large corpora (the LinkedIn file is ~969MB) — it never
    loads the whole file, just one line at a time.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_course_md(path: Path = COURSE_MD) -> str:
    """Read the curriculum course file (markdown) and return its raw text.

    The markdown is the ITC Course Specification (title, synopsis, CLOs, lesson
    plan). For the AMC baseline this whole text is the 'course document' that the
    Naive Bayes classifier labels into a job-role class.
    """
    return path.read_text(encoding="utf-8")
