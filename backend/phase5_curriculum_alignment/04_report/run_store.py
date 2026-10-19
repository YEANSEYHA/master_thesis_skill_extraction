# Versioned run store for the gap analysis.
#
# Every `gen_report_data.py` invocation writes one immutable directory
# under outputs/ holding BOTH sides of the experiment:
#
#   outputs/20260723T134501Z-a3f9c2/
#       report.md      human contract  — params + results in one readable file
#       report.json    machine contract — the full GAP_REPORT
#       params.json    the resolved parameters, canonical + hashed
#       inputs.json    what was fed in (curriculum files, corpus, adjudication)
#       log.txt        stdout of the run
#   outputs/index.md   ledger, one row appended per run (this one IS tracked)
#
# Why both .md and .json: the dashboard needs structured floats
# (skills[], selection.plan[], rerank_bands), which markdown cannot
# round-trip without a lossy parser. The markdown exists so `git diff`
# between two runs reads as prose, so a run is greppable, and so a
# thesis appendix is a copy-paste away.
#
# Why hash params and inputs SEPARATELY: they move independently. Fixing
# input_hash and varying config_hash is the ablation ("what did this
# parameter do?"); the reverse asks "does the method hold on new data?".
#
# Stdlib only, no imports from the API — `python3 gen_report_data.py`
# must keep working with no server and no database.

import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# phase5 root — this file lives in 04_report/, so outputs/ and the `git`
# invocation below stay anchored on the project root as before the reorg
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
INDEX = OUTPUTS / "index.md"


# --------------------------------------------------------------- hashing

def _canon(obj) -> bytes:
    """Canonical JSON — key order and spacing fixed so the hash is stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str | None:
    try:
        return _sha(path.read_bytes())
    except OSError:
        return None


def short(h: str | None, n: int = 6) -> str:
    return (h or "?")[:n]


# ------------------------------------------------------------- parameters

def resolve_params(top_n: int, cum_frac: float, spill_min: float) -> dict:
    """Every knob that changes the numbers, in one dict.

    This is the fix for the original defect: before this, editing
    MIN_SKILL_FREQ or SKILL_MERGE_THRESHOLD left NO trace in the output,
    so two report files produced by different settings were
    indistinguishable. Anything that affects results belongs here — if
    you add a parameter to the pipeline, add it here in the same commit.
    """
    import pipeline as P
    return {
        "encoder": P.MODEL_NAME,
        "covered_at": P.COVERED_AT,
        "borderline_at": P.BORDERLINE_AT,
        "host_min": P.get_thresholds()["host_min"],
        "top_n": top_n,
        "job_source": P.JOB_SOURCE,
        "segment_mode": P.SEGMENT_MODE,
        "min_skill_freq": P.MIN_SKILL_FREQ,
        "skill_merge_threshold": P.SKILL_MERGE_THRESHOLD,
        "max_skills_to_normalize": P.MAX_SKILLS_TO_NORMALIZE,
        "eps": P.EPS,
        "cache_version": P.CACHE_VERSION,
        "cum_frac": cum_frac,
        "spill_min": spill_min,
        # inline cross-encoder host verification (was a sidecar file,
        # recorded under inputs; now config, computed fresh every run)
        "reranker": P.RERANKER_NAME,
        "rerank_top_k": P.RERANK_TOP_K,
        "rerank_supported_at": P.RERANK_SUPPORTED_AT,
        "rerank_weak_at": P.RERANK_WEAK_AT,
    }


def resolve_inputs(clo_dir: Path, raw_jobs: Path,
                   adjudication_file: Path) -> dict:
    """What was fed in, with content hashes for the small files.

    The curriculum files are hashed in full — they are a few KB and they
    change constantly (every syllabus upload). The job corpus is NOT
    hashed: it is 86 MB, it changes only when the whole dataset is
    replaced, and the derived market matrix is already keyed by
    cache_version + encoder + source in its cache filename. Name + size
    is the honest, cheap identity there.
    """
    courses = []
    for f in sorted(clo_dir.glob("*.json")):
        try:
            items = json.loads(f.read_text())
        except (OSError, ValueError):
            items = []
        courses.append({"file": f.name, "n_skills": len(items),
                        "sha256": _file_sha(f)})
    curriculum = {
        "dir": str(clo_dir),
        # the API stages PostgreSQL courses into clo_skills_db/
        "mode": "db" if clo_dir.name == "clo_skills_db" else "files",
        "n_courses": len(courses),
        "courses": courses,
    }
    curriculum["hash"] = _sha(_canon([c["sha256"] for c in courses]))

    market = {
        "dataset": raw_jobs.name,
        "path": str(raw_jobs),
        "bytes": raw_jobs.stat().st_size if raw_jobs.exists() else None,
    }
    return {
        "curriculum": curriculum,
        "market": market,
        # host verification is computed inline per run (see
        # pipeline.RERANKER_NAME in params) — no rerank input file
        "adjudication": ({"file": adjudication_file.name,
                          "sha256": _file_sha(adjudication_file)}
                         if adjudication_file.exists() else None),
    }


def config_hash(params: dict) -> str:
    """Identity of the settings, independent of the data."""
    return _sha(_canon(params))


def input_hash(inputs: dict) -> str:
    """Identity of the data, independent of the parameters."""
    return _sha(_canon([
        inputs["curriculum"]["hash"],
        inputs["market"]["dataset"], inputs["market"]["bytes"],
        (inputs["adjudication"] or {}).get("sha256"),
    ]))


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(BASE_DIR), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_id(config_hash: str) -> str:
    """Sortable timestamp + config fingerprint: 20260723T134501Z-a3f9c2."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{short(config_hash)}"


def find_run(config_hash: str, in_hash: str) -> str | None:
    """run_id of an earlier run with identical params AND inputs, if any.

    Informational only — a direct `python3 gen_report_data.py` always
    recomputes, because silently skipping the work the user asked for is
    the wrong default for a script. It is here so the caller (or a later
    API layer) can offer a cache hit.
    """
    if not OUTPUTS.is_dir():
        return None
    for d in sorted(OUTPUTS.iterdir(), reverse=True):
        p = d / "params.json"
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (meta.get("config_hash") == config_hash
                and meta.get("input_hash") == in_hash):
            return meta.get("run_id")
    return None


# ------------------------------------------------------------- stdout tee

class Tee(io.TextIOBase):
    """Duplicate stdout into a buffer without swallowing it.

    Must be a tee and never a redirect: the API streams this process's
    stdout line by line into the V8 tab's terminal view.
    """

    def __init__(self, stream):
        self.stream = stream
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


# --------------------------------------------------------------- markdown

def _cell(v) -> str:
    """Table-cell text with pipes escaped so a skill name can't break the row."""
    return str(v).replace("|", "\\|").replace("\n", " ")


def _num(v, nd=2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


_STATUS_MARK = {"covered": "✓ covered", "borderline": "◑ borderline",
                "gap": "✗ gap"}


def render_md(report: dict, params: dict, inputs: dict, meta: dict) -> str:
    """The human contract: parameters and results in one readable file."""
    L: list[str] = []
    a = L.append

    a(f"# Phase 6 gap analysis — {meta['run_id']}")
    a("")
    if meta.get("label"):
        a(f"**{meta['label']}**")
        a("")
    a("| | |")
    a("|---|---|")
    a(f"| ran | {meta['started_at']} · {_num(meta.get('duration_s'), 1)} s |")
    a(f"| code | git {meta.get('git_sha') or '—'} |")
    a(f"| config hash | `{meta['config_hash']}` |")
    a(f"| input hash | `{meta['input_hash']}` |")
    if meta.get("cohort") or report.get("cohort"):
        a(f"| cohort | {meta.get('cohort') or report.get('cohort')} |")
    a("")

    # ---- inputs
    cur, mkt = inputs["curriculum"], inputs["market"]
    a("## Inputs")
    a("")
    a(f"**Market** — `{params['job_source']}` · `{mkt['dataset']}` · "
      f"{report.get('n_jobs', 0):,} jobs in segment · "
      f"segment mode `{params['segment_mode']}`")
    a("")
    coh = cur.get("cohort") or report.get("cohort")
    a(f"**Curriculum** — {cur['n_courses']} courses ({cur['mode']}) · "
      f"{report.get('n_clo_skills', 0)} CLO skills"
      + (f" · cohort **{coh}**" if coh else ""))
    a("")
    a("| course | file | CLO skills |")
    a("|---|---|---|")
    for name, c in zip(report.get("courses", []), cur["courses"]):
        a(f"| {_cell(name)} | `{_cell(c['file'])}` | {c['n_skills']} |")
    a("")
    if report.get("reranker"):
        a(f"Host verification computed inline (`{report['reranker']}`)")
        a("")

    # ---- parameters
    a("## Parameters")
    a("")
    a("| param | value |")
    a("|---|---|")
    for k, v in params.items():
        a(f"| {k} | `{v}` |")
    a("")

    # ---- results
    top_n = report.get("top_n", 0)
    a("## Results")
    a("")
    a(f"KL divergence **{report.get('kl')}** · "
      f"covered **{report.get('covered')} / {top_n}**")
    a("")
    a("### Top skills by demand")
    a("")
    a("| # | skill | jobs | coverage | status | best CLO match |")
    a("|---:|---|---:|---:|---|---|")
    for i, s in enumerate(report.get("skills", []), 1):
        a(f"| {i} | {_cell(s['skill'])} | {s['jobs']:,} | "
          f"{_num(s['program'])} | {_STATUS_MARK.get(s['status'], s['status'])} "
          f"| {_cell(s['best_course'])} — {_cell(s['best_clo'])} |")
    a("")

    # ---- the plan
    sel = report.get("selection")
    if sel:
        a("### Optimized skill plan")
        a("")
        a(f"Greedy facility-location, stop at {sel['cum_frac']:.0%} of "
          f"achievable gain → **add {sel['n_selected']} of "
          f"{len(sel['plan'])}** gaps. "
          f"KL {sel['kl0']} → {sel['plan'][-1]['kl_after'] if sel['plan'] else '—'}")
        a("")
        a("| | skill | jobs | closes | KL after | covered after | action |")
        a("|---|---|---:|---:|---:|---:|---|")
        for e in sel["plan"]:
            tag = "**ADD**" if e["selected"] else "skip"
            act = e.get("recommended_action", {})
            a(f"| {tag} | {_cell(e['skill'])} | {e['jobs']:,} | "
              f"{e['gain_pct']}% | {e['kl_after']} | {e['covered_after']} | "
              f"{_cell(act.get('title', ''))} |")
        a("")
        skipped = [e for e in sel["plan"] if not e["selected"]]
        if skipped:
            a("Not selected, and why:")
            a("")
            for e in skipped:
                why = (f"absorbed by {e['absorbed_by']}"
                       if e.get("absorbed_by")
                       else f"only closes {e['gain_pct']}% of the gap")
                a(f"- **{e['skill']}** — {why} "
                  f"(coverage after the plan {_num(e.get('cov_after_plan'))})")
            a("")

    # ---- gaps with no verified host
    no_host = [r for r in report.get("recommendations", []) if r.get("no_host")]
    if no_host:
        a("### Gaps with no verified host")
        a("")
        a("No existing CLO shows verified evidence of teaching these — "
          "they need new content rather than an extension.")
        a("")
        for r in no_host[:10]:
            a(f"- **{r['skill']}** — {r['jobs']:,} jobs, "
              f"coverage {_num(r['coverage'])} "
              f"(nearest: {_cell(r['argmax_course'])})")
        a("")

    a("---")
    a("")
    a("Generated by `gen_report_data.py`. `report.json` in this directory "
      "holds the full machine-readable report.")
    a("")
    return "\n".join(L)


# ------------------------------------------------------------ ledger + io

_INDEX_HEADER = [
    "# Phase 6 run ledger",
    "",
    "One row per `gen_report_data.py` run. The run payloads under "
    "`outputs/<run_id>/` are gitignored (reproducible from `params.json`); "
    "this ledger is tracked so the history of what was run stays versioned.",
    "",
    "| run | label | encoder | source | covered_at | top_n | KL | covered | courses | cohort |",
    "|---|---|---|---|---:|---:|---:|---|---:|---|",
]
# the ledger column layout (last two lines above) — used to migrate an older
# index.md that predates the cohort column so its header matches new rows
_LEDGER_HEAD, _LEDGER_SEP = _INDEX_HEADER[-2], _INDEX_HEADER[-1]


def append_index(report: dict, params: dict, meta: dict) -> None:
    if not INDEX.exists():
        INDEX.write_text("\n".join(_INDEX_HEADER) + "\n")
    elif "| cohort |" not in INDEX.read_text():
        # one-time header migration: older ledgers have no cohort column, so
        # bring the header + separator up to the current layout. Old rows keep
        # their 9 cells → the new trailing cohort column renders blank for them.
        lines = INDEX.read_text().splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("| run |"):
                lines[i] = _LEDGER_HEAD
                if i + 1 < len(lines):
                    lines[i + 1] = _LEDGER_SEP
                break
        INDEX.write_text("\n".join(lines) + "\n")
    cohort = report.get("cohort") or meta.get("cohort") or ""
    row = (f"| `{meta['run_id']}` | {_cell(meta.get('label') or '')} "
           f"| {params['encoder'].split('/')[-1]} | {params['job_source']} "
           f"| {params['covered_at']} | {params['top_n']} "
           f"| {report.get('kl')} "
           f"| {report.get('covered')}/{report.get('top_n')} "
           f"| {len(report.get('courses', []))} "
           f"| {_cell(cohort)} |")
    with INDEX.open("a") as fh:
        fh.write(row + "\n")


def write_run(report: dict, params: dict, inputs: dict, meta: dict,
              log: str) -> Path:
    """Write one immutable run directory and append the ledger row."""
    run_dir = OUTPUTS / meta["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    (run_dir / "params.json").write_text(json.dumps({**meta, "params": params},
                                                    indent=2))
    (run_dir / "inputs.json").write_text(json.dumps(inputs, indent=2))
    (run_dir / "report.md").write_text(render_md(report, params, inputs, meta))
    (run_dir / "log.txt").write_text(log)
    append_index(report, params, meta)
    return run_dir


def ensure_gitignore() -> None:
    """Keep the ledger tracked, the payloads out of git.

    A sweep is 27 configs x ~40 KB and every payload is reproducible from
    its params.json, so the runs themselves do not belong in history —
    but the record of what was run does.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    gi = OUTPUTS / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n!index.md\n")
