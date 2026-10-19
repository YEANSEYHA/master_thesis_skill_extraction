"""Shared setup for the phase-5 practice steps.

Style: ONE FILE PER STEP. Each step_*.py is runnable on its own
(`python3 step1_build_market.py`) and imports what it needs from here.

The practice CALLS the production engine rather than copying it, so the
numbers it prints are the pipeline's own numbers, not a re-implementation
that can drift.

The two inputs (already cleaned + segmented upstream):
  inputs/job_market.json       254 Cambodian IT postings, {input, output}
  inputs/school_curriculum.json  8 Year-5 courses, each with its CLO skills
"""

import importlib.util
import json
import os
import sys
from collections import Counter                      # noqa: F401  (re-export)
from pathlib import Path

import numpy as np                                   # noqa: F401  (re-export)

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
JOB_MARKET = INPUTS / "job_market.json"
CURRICULUM = INPUTS / "school_curriculum.json"

# backend/archive/playground/phase5_pratice -> backend/phase5_curriculum_alignment
PHASE5 = HERE.parents[2] / "phase5_curriculum_alignment"
ENGINE_PY = PHASE5 / "03_engine" / "pipeline.py"

_engine = None
_model = None
_market = None

# Which corpus the engine should treat as "the job market". The engine
# resolves this from the environment AT IMPORT TIME (pipeline.py line 86),
# so it must be set before engine() executes the module. Default there is
# "linkedin"; the practice runs on the 254 Cambodian IT postings.
os.environ.setdefault("JOB_SOURCE", "cambodia")


def engine():
    """Load backend/phase5_curriculum_alignment/03_engine/pipeline.py — the production engine.

    Loaded BY FILE PATH, not `import pipeline`, for two reasons:
      1. `03_engine` starts with a digit, so it can never be a package.
      2. backend/phase1_job_site_discovery ships its own `pipeline` package, which
         wins on sys.path and would shadow this one.
    Cached after the first call (the module builds/loads the market cache).
    """
    global _engine
    if _engine is None:
        if not ENGINE_PY.exists():
            raise SystemExit(f"phase-5 engine not found at {ENGINE_PY}")
        spec = importlib.util.spec_from_file_location("p5_engine", ENGINE_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["p5_engine"] = mod          # so its own imports resolve
        spec.loader.exec_module(mod)
        _engine = mod
    return _engine


def model():
    """The BGE bi-encoder, loaded ONCE per process.

    `pipeline.load_model()` builds a fresh SentenceTransformer on every
    call, which is wasteful when the runner below executes every step in
    one process — so cache it here.
    """
    global _model
    if _model is None:
        _model = engine().load_model()
    return _model


def market():
    """The built market, from the engine's own disk cache.

    Step 1 shows how this is built from raw postings. Every later step
    just wants the result, so it calls production's build_market(), which
    loads cache/market_cache_*.npz instead of re-clustering 3,118 strings.
    Keys used downstream: skill_base, counts, demand_dist, skill_emb,
    n_jobs.
    """
    global _market
    if _market is None:
        _market = engine().build_market(model())
    return _market


def read_job_market(path: Path = JOB_MARKET) -> list[dict]:
    """The postings as the scraper stored them: return each `output` dict."""
    return [r["output"] for r in json.loads(path.read_text())]


def read_curriculum(path: Path = CURRICULUM) -> list[dict]:
    """The 8 saved courses, each with course_title + skills[{name, source}]."""
    return json.loads(path.read_text())


def curriculum_units(courses: list[dict]) -> tuple[list, list, list]:
    """Flatten courses -> (course_names, clo_texts, clo_owner).

    `clo_owner[j]` names the course that column j of the similarity matrix
    belongs to — the bookkeeping that lets step 5 collapse 226 CLO columns
    into 8 course columns.
    """
    names, texts, owner = [], [], []
    for c in courses:
        name = c.get("course_title") or "(untitled)"
        names.append(name)
        for s in c.get("skills") or []:
            text = (s.get("name") or "").strip()
            if text:
                texts.append(text)
                owner.append(name)
    return names, texts, owner


def banner(step: str, title: str, produces: str = "") -> None:
    """Uniform header so the step files all look the same when run."""
    print("=" * 70)
    print(f"{step} — {title}")
    if produces:
        print(f"produces: {produces}")
    print("=" * 70 + "\n")


def _step_files() -> list[Path]:
    """Every step*.py in this folder, ordered by their step number."""
    def num(p: Path) -> int:
        digits = "".join(ch for ch in p.stem.split("_")[0] if ch.isdigit())
        return int(digits) if digits else 0
    return sorted(HERE.glob("step*.py"), key=num)


def run_all() -> None:
    """Run every step in order, in ONE process so the model loads once."""
    steps = _step_files()
    if not steps:
        print("no step*.py files found yet — write step1_build_market.py")
        return
    print(f"running {len(steps)} step(s): "
          f"{', '.join(s.stem for s in steps)}\n")
    for path in steps:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
            print()
        else:
            print(f"({path.name} has no main() — skipped)\n")


if __name__ == "__main__":
    run_all()
