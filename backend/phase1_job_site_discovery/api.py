"""
FastAPI server for Phase 1 Job Site Discovery + Phase 2 Crawl.

Wraps the existing pipeline and DB functions as HTTP endpoints.
Run: uvicorn api:app --reload --port 8000
"""

import io
import sys
import asyncio
import hashlib
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from db.job_sites import (
    create_job_sites_table,
    get_job_sites_by_country,
    get_all_job_sites,
    get_countries_summary,
    insert_single_job_site,
    delete_job_site,
)
from db.curricula_analyses import (
    save_analysis as db_save_analysis,
    list_analyses as db_list_analyses,
    get_analysis as db_get_analysis,
    delete_analysis as db_delete_analysis,
)
from db.course_skills import (
    save_course_skills as db_save_course_skills,
    list_course_skills as db_list_course_skills,
    get_course_skills as db_get_course_skills,
    delete_course_skills as db_delete_course_skills,
)
import r2_store   # Cloudflare R2 archive for uploaded syllabi (best-effort)
from db.curriculum_documents import (
    find_by_sha256 as db_find_document,
    save_document as db_save_document,
    set_markdown as db_set_document_markdown,
    set_status as db_set_document_status,
    set_r2 as db_set_document_r2,
    list_documents as db_list_documents,
    get_document as db_get_document,
    delete_document as db_delete_document,
)
from main import (
    run_pipeline,
    create_discovery_log_table,
    get_discovery_status,
    get_all_discovery_status,
    SearchRateLimitError,
)

# Phase 2 imports
# Phase2 modules use bare `from config import ...` which would resolve to phase1's config
# since it's already cached in sys.modules. We swap it temporarily during phase2 imports.
import importlib.util
import pathlib

_phase2_dir = pathlib.Path(__file__).resolve().parent.parent / "archive" / "phase2_playwright_crawler"
sys.path.insert(0, str(_phase2_dir))

# Temporarily swap config module so phase2 gets its own config
_phase1_config = sys.modules.pop("config", None)
_p2_config_spec = importlib.util.spec_from_file_location("config", _phase2_dir / "config.py")
_p2_config = importlib.util.module_from_spec(_p2_config_spec)
_p2_config_spec.loader.exec_module(_p2_config)
sys.modules["config"] = _p2_config

from crawler import AutonomousCrawler  # noqa: E402

# Load phase2's main.py under a different module name to avoid conflict with phase1's main
_spec = importlib.util.spec_from_file_location("phase2_main", _phase2_dir / "main.py")
_phase2_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase2_main)
CrawlTracker = _phase2_main.CrawlTracker
crawl_single_site = _phase2_main.crawl_single_site
save_simple_lists = _phase2_main.save_simple_lists

# Restore phase1's config
if _phase1_config is not None:
    sys.modules["config"] = _phase1_config
else:
    sys.modules.pop("config", None)


# Track running discovery jobs and their logs
running_jobs: dict[str, dict] = {}
job_logs: dict[str, list[str]] = {}


def _warmup_embedder():
    """Load the SBERT model once at startup so the first /api/curricula/*
    request doesn't pay the cold-load cost (and doesn't race the singleton,
    which previously caused a 500 on the first analyze call)."""
    _p5_dir = (pathlib.Path(__file__).resolve().parent.parent
               / "archive" / "phase5_legacy" / "clo_skill_alignment")
    if str(_p5_dir) not in sys.path:
        sys.path.insert(0, str(_p5_dir))
    try:
        from embedder import Embedder
        Embedder()  # singleton: blocks until model is fully loaded
    except Exception as e:
        print(f"[warmup] embedder preload failed: {e}")


def _warmup_esco():
    """Load the ESCO normalizer (JobBERT + cached label embeddings) at startup
    so the first ESCO request doesn't pay the cache-load / build cost."""
    _p5_dir = (pathlib.Path(__file__).resolve().parent.parent
               / "archive" / "phase5_legacy" / "clo_skill_alignment")
    if str(_p5_dir) not in sys.path:
        sys.path.insert(0, str(_p5_dir))
    try:
        from esco_normalizer import EscoNormalizer
        EscoNormalizer()  # singleton: loads cache (or builds it once)
    except Exception as e:
        print(f"[warmup] esco normalizer preload failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_job_sites_table()
    create_discovery_log_table()
    _warmup_embedder()
    _warmup_esco()
    yield


app = FastAPI(title="Job Site Discovery API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Any local dev origin, on any port — Next dev falls back to 3001/3002 when
    # 3000 is taken, and the UI is sometimes opened over the LAN IP. A missing
    # origin here surfaces in the browser as a bare "could not connect".
    allow_origin_regex=(
        r"^http://(localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?$"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class DiscoverRequest(BaseModel):
    country: str
    refresh: bool = False
    model_choice: str = "ollama"  # "gpt" | "claude" | "glm" | "ollama"


class ManualSiteRequest(BaseModel):
    url: str
    name: str
    country: str
    site_type: str = "job_portal"
    description: str = ""


# --- Endpoints ---

@app.get("/api/sites")
def get_sites(country: str = Query(None)):
    """Get job sites, optionally filtered by country."""
    if country:
        sites = get_job_sites_by_country(country.lower())
    else:
        sites = get_all_job_sites()

    # Convert datetime objects to strings for JSON serialization
    for site in sites:
        for key, val in site.items():
            if isinstance(val, datetime):
                site[key] = val.isoformat()

    return {"sites": sites, "count": len(sites)}


@app.get("/api/sites/countries")
def get_countries():
    """Get summary of discovered countries."""
    summary = get_countries_summary()
    for row in summary:
        if row.get("avg_confidence") is not None:
            row["avg_confidence"] = round(float(row["avg_confidence"]), 2)
    return {"countries": summary}


@app.get("/api/discovery-status")
def discovery_status(country: str = Query(None)):
    """Check discovery freshness for a country or all countries."""
    if country:
        status = get_discovery_status(country.lower())
        if not status:
            return {"status": None, "message": f"No discovery run found for {country}"}
        if status.get("last_discovered_at"):
            status["last_discovered_at"] = status["last_discovered_at"].isoformat()
        return {"status": status}
    else:
        all_status = get_all_discovery_status()
        for s in all_status:
            if s.get("last_discovered_at"):
                s["last_discovered_at"] = s["last_discovered_at"].isoformat()
        return {"statuses": all_status}


class LogCapture(io.TextIOBase):
    """Captures print output into a log buffer while also writing to real stdout."""

    def __init__(self, country: str, real_stdout):
        self.country = country
        self.real_stdout = real_stdout

    def write(self, text):
        if text and text.strip():
            job_logs.setdefault(self.country, []).append(text.rstrip())
        self.real_stdout.write(text)
        return len(text)

    def flush(self):
        self.real_stdout.flush()


def _run_discovery(country: str, refresh: bool, model_choice: str = "ollama"):
    """Background task: run the full discovery pipeline with log capture."""
    job_logs[country] = []
    running_jobs[country] = {"status": "running", "started_at": datetime.now().isoformat()}

    # Capture stdout from the pipeline prints
    real_stdout = sys.stdout
    sys.stdout = LogCapture(country, real_stdout)
    try:
        results = run_pipeline(country, force_refresh=refresh, model_choice=model_choice)
        running_jobs[country] = {
            "status": "completed",
            "sites_found": len(results) if results else 0,
            "completed_at": datetime.now().isoformat(),
        }
    except SearchRateLimitError as e:
        running_jobs[country] = {
            "status": "failed",
            "error": str(e),
            "error_type": "rate_limit",
            "completed_at": datetime.now().isoformat(),
        }
    except Exception as e:
        running_jobs[country] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }
    finally:
        sys.stdout = real_stdout


@app.post("/api/discover")
def discover(req: DiscoverRequest, background_tasks: BackgroundTasks):
    """Trigger discovery pipeline for a country (runs in background)."""
    country = req.country.lower()

    if country in running_jobs and running_jobs[country].get("status") == "running":
        raise HTTPException(409, f"Discovery already running for {country}")

    background_tasks.add_task(_run_discovery, country, req.refresh, req.model_choice)
    return {"message": f"Discovery started for {country}", "country": country}


@app.get("/api/discover/status/{country}")
def discover_job_status(country: str, since: int = Query(0)):
    """Check status of a running/completed discovery job. Pass since=N to get logs after line N."""
    country = country.lower()
    if country not in running_jobs:
        return {"status": "not_started", "logs": []}

    logs = job_logs.get(country, [])
    return {
        **running_jobs[country],
        "logs": logs[since:],
        "log_count": len(logs),
    }


@app.post("/api/sites/manual")
def add_manual_site(req: ManualSiteRequest):
    """Manually insert a single job site into the database."""
    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    success = insert_single_job_site(
        url=url,
        name=req.name.strip(),
        country=req.country.strip().lower(),
        site_type=req.site_type,
        description=req.description.strip(),
        confidence=1.0,
        reason="Manually added",
    )

    if not success:
        raise HTTPException(500, "Failed to insert site")

    return {"message": f"Added {req.name}", "url": url, "country": req.country.lower()}


@app.delete("/api/sites")
def remove_site(url: str = Query(...)):
    """Delete a job site by URL."""
    success = delete_job_site(url)
    if not success:
        raise HTTPException(404, "Site not found")
    return {"message": "Deleted", "url": url}


# =====================================================================
# Phase 2 — Crawl endpoints
# =====================================================================

# Track running crawl jobs and their logs
crawl_jobs: dict[str, dict] = {}
crawl_logs: dict[str, list[str]] = {}
crawl_stop_events: dict[str, threading.Event] = {}


class CrawlRequest(BaseModel):
    urls: List[str]
    max_pages: int = 2000
    max_depth: int = 5
    crawl_delay: float = 1.0


class CrawlLogCapture(io.TextIOBase):
    """Captures print output from the crawl pipeline into a log buffer."""

    def __init__(self, job_id: str, real_stdout):
        self.job_id = job_id
        self.real_stdout = real_stdout

    def write(self, text):
        if text and text.strip():
            crawl_logs.setdefault(self.job_id, []).append(text.rstrip())
        self.real_stdout.write(text)
        return len(text)

    def flush(self):
        self.real_stdout.flush()


def _run_crawl(job_id: str, urls: list, max_pages: int, max_depth: int, crawl_delay: float):
    """Background task: run the phase2 crawl pipeline with log capture."""
    stop_event = threading.Event()
    crawl_stop_events[job_id] = stop_event
    crawl_logs[job_id] = []
    crawl_jobs[job_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "total_sites": len(urls),
    }

    real_stdout = sys.stdout
    sys.stdout = CrawlLogCapture(job_id, real_stdout)

    try:
        # Phase 2 crawl is async — run it in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tracker = CrawlTracker()
        tracker.start_time = datetime.now()
        run_id = tracker.start_time.strftime("%Y-%m-%d_%H-%M-%S")

        async def _crawl_all():
            for index, url in enumerate(urls, 1):
                # Check if stop was requested between sites
                if stop_event.is_set():
                    print(f"\n[STOP] Crawl stopped by user after {index - 1} sites")
                    break

                crawl_jobs[job_id]["current_site"] = url
                crawl_jobs[job_id]["current_index"] = index

                # Create crawler with config from the request
                try:
                    crawler = AutonomousCrawler(
                        start_url=url,
                        max_pages=max_pages,
                        run_id=run_id,
                        stop_event=stop_event,
                    )
                    crawler.crawl_delay = crawl_delay
                    if hasattr(crawler, "max_depth"):
                        crawler.max_depth = max_depth

                    await crawler.crawl()

                    if crawler.saved > 0:
                        tracker.add_success(url, crawler.saved, crawler.metadata.get("llm_used", 0))
                        if crawler.metadata.get("llm_used", 0) > 0:
                            tracker.add_pagination(url)
                        print(f"\nSuccess: {crawler.saved} pages saved")
                    else:
                        if crawler.metadata.get("blocked"):
                            tracker.add_blocked(url, "Anti-bot protection detected")
                            print(f"Blocked: No pages saved")
                        else:
                            tracker.add_failed(url, "No content extracted")
                            print(f"Warning: No pages saved")
                except Exception as e:
                    tracker.add_failed(url, str(e))
                    print(f"Failed: {e}")

                crawl_jobs[job_id]["success"] = len(tracker.results["success"])
                crawl_jobs[job_id]["blocked"] = len(tracker.results["blocked"])
                crawl_jobs[job_id]["failed"] = len(tracker.results["failed"])

                # Delay between sites
                if index < len(urls) and not stop_event.is_set():
                    await asyncio.sleep(5)

            tracker.end_time = datetime.now()
            tracker.print_summary()

            # Save reports into the run folder
            import os
            run_dir = str(_phase2_dir / "outputs" / run_id)
            os.makedirs(run_dir, exist_ok=True)
            tracker.save_report(f"{run_dir}/crawl_report.json")
            save_simple_lists(tracker, run_dir)

        loop.run_until_complete(_crawl_all())
        loop.close()

        crawl_jobs[job_id]["success"] = len(tracker.results["success"])
        crawl_jobs[job_id]["blocked"] = len(tracker.results["blocked"])
        crawl_jobs[job_id]["failed"] = len(tracker.results["failed"])

        if stop_event.is_set():
            crawl_jobs[job_id]["status"] = "stopped"
            crawl_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        else:
            # ── Auto-classify after crawl ──
            _run_classify_after_crawl(job_id, run_id, stop_event)

    except Exception as e:
        crawl_jobs[job_id]["status"] = "failed"
        crawl_jobs[job_id]["error"] = str(e)
        crawl_jobs[job_id]["completed_at"] = datetime.now().isoformat()
    finally:
        sys.stdout = real_stdout
        crawl_stop_events.pop(job_id, None)


def _run_classify_after_crawl(job_id: str, run_id: str, stop_event: threading.Event):
    """Run classification on crawl results. Called automatically after crawl completes."""
    import json as _json
    import logging as _logging

    input_dir = _phase2_dir / "outputs" / run_id
    output_dir = str(_classify_output_dir / run_id)

    crawl_jobs[job_id]["status"] = "classifying"
    crawl_jobs[job_id]["classify_progress"] = 0
    crawl_jobs[job_id]["classify_total"] = 0
    crawl_jobs[job_id]["classify_summary"] = {
        "single_job_detail": 0, "non_job_details": 0, "failed": 0, "no_content": 0
    }

    print(f"\n{'=' * 50}")
    print(f"AUTO-CLASSIFY: Starting classification on {run_id}")
    print(f"{'=' * 50}")

    # Capture logging from stage1 module
    log_handler = _logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(_logging.Formatter("%(levelname)s - %(message)s"))
    _stage1_mod.logger.addHandler(log_handler)

    try:
        htmls = _classify_read_nested(str(input_dir))
        total = len(htmls)
        crawl_jobs[job_id]["classify_total"] = total
        print(f"Found {total} HTML pages to classify")

        summary = crawl_jobs[job_id]["classify_summary"]

        for i, item in enumerate(htmls, 1):
            if stop_event.is_set():
                print(f"[STOP] Classification stopped after {i - 1} pages")
                break

            filename = item["filename"]
            crawl_jobs[job_id]["classify_progress"] = i
            crawl_jobs[job_id]["current_site"] = f"Classifying: {filename}"

            raw = item["content"]
            page_text = _classify_clean(raw)

            if not page_text.strip():
                summary["no_content"] += 1
                print(f"[{i}/{total}] {filename} -> No content")
                continue

            prompt = _classify_build_prompt(page_text)

            try:
                result = _classify_ask_llm(_CLASSIFY_MODEL, prompt)
                is_job = bool(result.get("is_single_job_detail", False))
                _classify_save_result(filename, result, page_text, output_dir=output_dir)

                if is_job:
                    summary["single_job_detail"] += 1
                    print(f"[{i}/{total}] {filename} -> JOB PAGE")
                else:
                    summary["non_job_details"] += 1
                    print(f"[{i}/{total}] {filename} -> not a job page")
            except Exception as e:
                summary["failed"] += 1
                print(f"[{i}/{total}] {filename} -> FAILED: {e}")

        status = "stopped" if stop_event.is_set() else "completed"
        crawl_jobs[job_id]["status"] = status
        crawl_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        print(f"\nClassification done! {_json.dumps(summary)}")

    except Exception as e:
        crawl_jobs[job_id]["status"] = "failed"
        crawl_jobs[job_id]["error"] = f"Classification failed: {e}"
        crawl_jobs[job_id]["completed_at"] = datetime.now().isoformat()
    finally:
        _stage1_mod.logger.removeHandler(log_handler)


@app.post("/api/crawl")
def start_crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    """Trigger crawl for selected site URLs (runs in background)."""
    if not req.urls:
        raise HTTPException(400, "No URLs provided")

    # Check if a crawl or classification is already running
    for jid, job in crawl_jobs.items():
        if job.get("status") in ("running", "classifying"):
            raise HTTPException(409, f"A job is already running (job {jid})")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[API] Crawl request received: {len(req.urls)} URLs: {req.urls}")
    background_tasks.add_task(_run_crawl, job_id, req.urls, req.max_pages, req.max_depth, req.crawl_delay)
    return {"message": f"Crawl started for {len(req.urls)} sites", "job_id": job_id}


@app.get("/api/crawl/status")
def crawl_status(since: int = Query(0)):
    """Check status of the current/latest crawl job. Pass since=N to get logs after line N."""
    if not crawl_jobs:
        return {"status": "not_started", "logs": []}

    # Get the most recent job
    job_id = max(crawl_jobs.keys())
    job = crawl_jobs[job_id]
    logs = crawl_logs.get(job_id, [])

    return {
        "job_id": job_id,
        **job,
        "logs": logs[since:],
        "log_count": len(logs),
    }


@app.post("/api/crawl/stop")
def stop_crawl():
    """Stop the currently running crawl or classification job."""
    for jid, job in crawl_jobs.items():
        if job.get("status") in ("running", "classifying"):
            stop_event = crawl_stop_events.get(jid)
            if stop_event:
                stop_event.set()
                return {"message": "Stop signal sent", "job_id": jid}
            raise HTTPException(500, "No stop event found for running job")

    raise HTTPException(404, "No crawl is currently running")


# =====================================================================
# Phase 3 — Classification endpoints
# =====================================================================

_phase3_dir = pathlib.Path(__file__).resolve().parent.parent / "phase3_skill_extraction"
_classify_output_dir = _phase3_dir / "outputs" / "classification"

# Import phase3 stage1 classify. Phase 3 is a numbered pipeline (02_classify/,
# 03_extract/, lib/) — numbered folders can't be packages, so the stage modules
# and their shared lib go on sys.path and are loaded by file path.
sys.path.insert(0, str(_phase3_dir))
sys.path.insert(0, str(_phase3_dir / "02_classify"))
sys.path.insert(0, str(_phase3_dir / "03_extract"))
sys.path.insert(0, str(_phase3_dir / "lib"))

import importlib.util as _ilu

_s1_spec = _ilu.spec_from_file_location(
    "stage1_classify", _phase3_dir / "02_classify" / "stage1_classify.py"
)
_stage1_mod = _ilu.module_from_spec(_s1_spec)
_s1_spec.loader.exec_module(_stage1_mod)

_classify_read_nested = _stage1_mod.read_all_htmls_nested
_classify_clean = _stage1_mod.clean_html_trafilatura
_classify_build_prompt = _stage1_mod.build_prompt
_classify_ask_llm = _stage1_mod.ask_llm
_classify_save_result = _stage1_mod.save_result
_CLASSIFY_MODEL = _stage1_mod.MODEL_NAME

@app.get("/api/classify/results")
def get_classify_results():
    """List classification result folders (one per crawl run)."""
    if not _classify_output_dir.exists():
        return {"runs": []}

    runs = []
    for run_dir in sorted(_classify_output_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        job_dir = run_dir / "single_job_detail"
        non_job_dir = run_dir / "non_job_details"

        job_count = sum(1 for _ in job_dir.rglob("*.html")) if job_dir.exists() else 0
        non_job_count = sum(1 for _ in non_job_dir.rglob("*.html")) if non_job_dir.exists() else 0

        runs.append({
            "run_id": run_dir.name,
            "job_pages": job_count,
            "non_job_pages": non_job_count,
            "total": job_count + non_job_count,
        })

    return {"runs": runs}


# =====================================================================
# Crawl Results — Browse saved crawl outputs
# =====================================================================

_outputs_dir = _phase2_dir / "outputs"


@app.get("/api/crawl/runs")
def get_crawl_runs():
    """List all crawl runs with their summary stats."""
    if not _outputs_dir.exists():
        return {"runs": []}

    runs = []
    for run_dir in sorted(_outputs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        run = {"run_id": run_dir.name, "date": run_dir.name.replace("_", " ", 1).replace("-", "/", 2)}

        # Read crawl_report.json if exists
        report_path = run_dir / "crawl_report.json"
        if report_path.exists():
            try:
                import json as _json
                with open(report_path) as f:
                    report = _json.load(f)
                run["summary"] = report.get("summary", {})
            except Exception:
                pass

        # Count domains (subdirectories) and total HTML pages
        domains = [d for d in run_dir.iterdir() if d.is_dir()]
        total_pages = sum(
            sum(1 for _ in d.glob("page_*.html")) for d in domains
        )
        run["domains"] = [d.name for d in domains]
        run["domain_count"] = len(domains)
        run["total_pages"] = total_pages

        runs.append(run)

    return {"runs": runs}


@app.get("/api/crawl/runs/{run_id}")
def get_crawl_run(run_id: str):
    """Get details of a specific crawl run including all domains and page counts."""
    run_dir = _outputs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run {run_id} not found")

    result = {"run_id": run_id, "domains": []}

    # Read crawl report
    report_path = run_dir / "crawl_report.json"
    if report_path.exists():
        try:
            import json as _json
            with open(report_path) as f:
                result["report"] = _json.load(f)
        except Exception:
            pass

    # List domains with page counts
    for domain_dir in sorted(run_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_info = {"domain": domain_dir.name, "pages": 0, "urls": {}}

        # Count HTML files
        html_files = list(domain_dir.glob("page_*.html"))
        domain_info["pages"] = len(html_files)

        # Read index.json for URL mapping
        index_path = domain_dir / "index.json"
        if index_path.exists():
            try:
                import json as _json
                with open(index_path) as f:
                    domain_info["urls"] = _json.load(f)
            except Exception:
                pass

        # Read metadata.json
        meta_path = domain_dir / "metadata.json"
        if meta_path.exists():
            try:
                import json as _json
                with open(meta_path) as f:
                    domain_info["metadata"] = _json.load(f)
            except Exception:
                pass

        result["domains"].append(domain_info)

    # Sort by page count descending
    result["domains"].sort(key=lambda d: d["pages"], reverse=True)
    return result


@app.get("/api/crawl/runs/{run_id}/{domain}/pages")
def get_domain_pages(run_id: str, domain: str, page: int = Query(1), per_page: int = Query(50)):
    """Get paginated list of HTML pages for a domain in a crawl run."""
    domain_dir = _outputs_dir / run_id / domain
    if not domain_dir.exists():
        raise HTTPException(404, f"Domain {domain} not found in run {run_id}")

    # Read index.json
    urls = {}
    index_path = domain_dir / "index.json"
    if index_path.exists():
        try:
            import json as _json
            with open(index_path) as f:
                urls = _json.load(f)
        except Exception:
            pass

    # List HTML files sorted by page number
    html_files = sorted(domain_dir.glob("page_*.html"), key=lambda f: int(f.stem.split("_")[1]))
    total = len(html_files)

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_files = html_files[start:end]

    pages = []
    for f in page_files:
        size_kb = round(f.stat().st_size / 1024, 1)
        pages.append({
            "filename": f.name,
            "url": urls.get(f.name, ""),
            "size_kb": size_kb,
        })

    return {
        "domain": domain,
        "run_id": run_id,
        "pages": pages,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@app.get("/api/crawl/runs/{run_id}/{domain}/view/{filename}")
def view_page(run_id: str, domain: str, filename: str):
    """Serve a saved HTML page for preview."""
    from fastapi.responses import HTMLResponse

    file_path = _outputs_dir / run_id / domain / filename
    if not file_path.exists() or not filename.endswith(".html"):
        raise HTTPException(404, "Page not found")

    content = file_path.read_text(encoding="utf-8", errors="replace")
    return HTMLResponse(content=content)


# =====================================================================
# Phase 3 Stage 2 — Extraction endpoints
# =====================================================================

_stage2_output_dir = _phase3_dir / "outputs" / "stage2"

# Import stage2 functions
_s2_spec = _ilu.spec_from_file_location(
    "stage2_skill_extraction", _phase3_dir / "03_extract" / "stage2_skill_extraction.py"
)
_stage2_mod = _ilu.module_from_spec(_s2_spec)
_s2_spec.loader.exec_module(_stage2_mod)

_extract_clean_html = _stage2_mod.clean_html_trafilatura
_extract_build_prompt = _stage2_mod.build_prompt
_extract_ask_llm = _stage2_mod.ask_llm
_extract_url_from_html = _stage2_mod.extract_url_from_html

# Track extraction jobs
extract_jobs: dict[str, dict] = {}
extract_logs: dict[str, list[str]] = {}
extract_stop_events: dict[str, threading.Event] = {}


def _normalize_posted_date(raw) -> str | None:
    """Best-effort parse of a messy scraped posted_date → ISO 'YYYY-MM-DD'.

    Scraped postings carry inconsistent date formats ('20-Feb-2026',
    'March 25, 2026', '2026-09', …); return None when unparseable so the
    frontend can exclude undated postings from the time-series panels.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "unknown", "-"):
        return None
    from datetime import datetime
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d",
                "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    import re as _re
    m = _re.search(r"(20[12]\d)", s)
    return f"{m.group(1)}-01-01" if m else None


def _latest_stage2_file():
    """Newest outputs/stage2/<run>/stage2_result.json by mtime, or None."""
    if not _stage2_output_dir.exists():
        return None
    files = list(_stage2_output_dir.glob("*/stage2_result.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


@app.get("/api/trends/scrape-postings")
def trends_scrape_postings():
    """Real Phase-3 scraped postings reshaped as [{skills, posted_date}] for the
    Skill Demand & Trends tab. Only skill-bearing records are returned; dates
    are normalized to ISO (null when unparseable)."""
    import json as _json
    f = _latest_stage2_file()
    if f is None:
        raise HTTPException(
            404, "no Phase-3 extraction results in "
                 "backend/phase3_skill_extraction/outputs/stage2/<run>/stage2_result.json. "
                 "They are not included in the repository (see DATA.md); "
                 "run Phase-3 extraction or copy a run there.")
    records = _json.loads(f.read_text(encoding="utf-8"))
    postings = []
    for r in records:
        out = (r or {}).get("output") or {}
        skills = out.get("skills") or []
        clean = [str(s).strip().lower() for s in skills if str(s).strip()]
        if not clean:
            continue
        postings.append({
            "skills": clean,
            "posted_date": _normalize_posted_date(out.get("posted_date")),
        })
    with_date = sum(1 for p in postings if p["posted_date"])
    return {
        "postings": postings,
        "total": len(postings),
        "with_date": with_date,
        "run": f.parent.name,
    }


class ExtractRequest(BaseModel):
    run_id: str


class ExtractLogCapture(io.TextIOBase):
    """Captures print output from extraction into a log buffer."""

    def __init__(self, job_id: str, real_stdout):
        self.job_id = job_id
        self.real_stdout = real_stdout

    def write(self, text):
        if text and text.strip():
            extract_logs.setdefault(self.job_id, []).append(text.rstrip())
        self.real_stdout.write(text)
        return len(text)

    def flush(self):
        self.real_stdout.flush()


def _run_extraction(job_id: str, run_id: str):
    """Background task: run stage2 extraction on classified job pages."""
    import json as _json
    import time as _time

    stop_event = threading.Event()
    extract_stop_events[job_id] = stop_event
    extract_logs[job_id] = []
    extract_jobs[job_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "run_id": run_id,
        "progress": 0,
        "total": 0,
        "extracted": 0,
        "failed": 0,
    }

    real_stdout = sys.stdout
    sys.stdout = ExtractLogCapture(job_id, real_stdout)

    try:
        # Find classified job pages
        job_pages_dir = _classify_output_dir / run_id / "single_job_detail"
        if not job_pages_dir.exists():
            extract_jobs[job_id]["status"] = "failed"
            extract_jobs[job_id]["error"] = f"No classified job pages found for run {run_id}"
            extract_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            return

        # Collect all HTML files
        html_files = sorted(job_pages_dir.rglob("*.html"))
        total = len(html_files)
        extract_jobs[job_id]["total"] = total

        if total == 0:
            extract_jobs[job_id]["status"] = "completed"
            extract_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            print("No job pages to extract from")
            return

        print(f"Starting extraction on {total} job pages from run {run_id}")

        # Prepare output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _stage2_output_dir / f"{run_id}__{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "stage2_result.json"

        all_results = []
        errors = 0
        start_time = _time.time()

        for idx, html_path in enumerate(html_files, 1):
            if stop_event.is_set():
                print(f"[STOP] Extraction stopped after {idx - 1} pages")
                break

            filename = str(html_path.relative_to(job_pages_dir))
            extract_jobs[job_id]["progress"] = idx
            extract_jobs[job_id]["current_file"] = filename

            raw_html = html_path.read_text(encoding="utf-8", errors="replace")
            cleaned_text = _extract_clean_html(raw_html)

            if not cleaned_text.strip():
                print(f"[{idx}/{total}] {filename} -> No content")
                errors += 1
                extract_jobs[job_id]["failed"] = errors
                continue

            prompt = _extract_build_prompt(cleaned_text)

            try:
                job_start = _time.time()
                response = _extract_ask_llm("llama3.1:8b", prompt)
                duration = _time.time() - job_start

                parsed = _json.loads(response)
                parsed["source_file"] = filename
                parsed["full_url"] = _extract_url_from_html(raw_html)

                all_results.append({"input": cleaned_text, "output": parsed})

                # Save incrementally
                with open(output_file, "w", encoding="utf-8") as f:
                    _json.dump(all_results, f, ensure_ascii=False, indent=2)

                title = (parsed.get("job_title") or "Unknown")[:50]
                skills_count = len(parsed.get("skills", []))

                elapsed = _time.time() - start_time
                avg = elapsed / idx
                eta = avg * (total - idx) / 60

                print(f"[{idx}/{total}] {title} | {skills_count} skills | {duration:.1f}s | ETA: {eta:.1f}min")

                extract_jobs[job_id]["extracted"] = len(all_results)

            except Exception as e:
                errors += 1
                extract_jobs[job_id]["failed"] = errors
                print(f"[{idx}/{total}] {filename} -> FAILED: {e}")

        status = "stopped" if stop_event.is_set() else "completed"
        extract_jobs[job_id]["status"] = status
        extract_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extract_jobs[job_id]["output_dir"] = str(out_dir.name)

        total_time = _time.time() - start_time
        print(f"\nExtraction {status}! {len(all_results)}/{total} extracted in {total_time/60:.1f}min ({errors} errors)")

    except Exception as e:
        extract_jobs[job_id]["status"] = "failed"
        extract_jobs[job_id]["error"] = str(e)
        extract_jobs[job_id]["completed_at"] = datetime.now().isoformat()
    finally:
        sys.stdout = real_stdout
        extract_stop_events.pop(job_id, None)


@app.post("/api/extract")
def start_extraction(req: ExtractRequest, background_tasks: BackgroundTasks):
    """Trigger skill extraction on classified job pages from a crawl run."""
    # Check if extraction is already running
    for jid, job in extract_jobs.items():
        if job.get("status") == "running":
            raise HTTPException(409, f"Extraction already running (job {jid})")

    # Check classified pages exist
    job_pages_dir = _classify_output_dir / req.run_id / "single_job_detail"
    if not job_pages_dir.exists():
        raise HTTPException(404, f"No classified job pages found for run {req.run_id}")

    job_count = sum(1 for _ in job_pages_dir.rglob("*.html"))
    if job_count == 0:
        raise HTTPException(400, "No job pages to extract from")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_tasks.add_task(_run_extraction, job_id, req.run_id)
    return {"message": f"Extraction started for {job_count} job pages", "job_id": job_id}


@app.get("/api/extract/status")
def extract_status(since: int = Query(0)):
    """Check status of the current/latest extraction job."""
    if not extract_jobs:
        return {"status": "not_started", "logs": []}

    job_id = max(extract_jobs.keys())
    job = extract_jobs[job_id]
    logs = extract_logs.get(job_id, [])

    return {
        "job_id": job_id,
        **job,
        "logs": logs[since:],
        "log_count": len(logs),
    }


@app.post("/api/extract/stop")
def stop_extraction():
    """Stop the currently running extraction job."""
    for jid, job in extract_jobs.items():
        if job.get("status") == "running":
            stop_event = extract_stop_events.get(jid)
            if stop_event:
                stop_event.set()
                return {"message": "Stop signal sent", "job_id": jid}
            raise HTTPException(500, "No stop event found")

    raise HTTPException(404, "No extraction is currently running")


@app.get("/api/extract/runs")
def get_extract_runs():
    """List all extraction result runs."""
    if not _stage2_output_dir.exists():
        return {"runs": []}

    import json as _json

    runs = []
    for run_dir in sorted(_stage2_output_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        result_file = run_dir / "stage2_result.json"
        if not result_file.exists():
            continue

        try:
            with open(result_file, encoding="utf-8") as f:
                data = _json.load(f)

            # Collect summary stats
            total_jobs = len(data)
            all_skills = []
            categories = {}
            for item in data:
                out = item.get("output", {})
                skills = out.get("skills", [])
                all_skills.extend(skills)
                cat = out.get("job_category") or "Unknown"
                if isinstance(cat, list):
                    cat = cat[0] if cat else "Unknown"
                categories[cat] = categories.get(cat, 0) + 1

            runs.append({
                "run_id": run_dir.name,
                "total_jobs": total_jobs,
                "total_skills": len(all_skills),
                "unique_skills": len(set(s.lower() for s in all_skills)),
                "categories": categories,
            })
        except Exception:
            runs.append({"run_id": run_dir.name, "total_jobs": 0, "total_skills": 0})

    return {"runs": runs}


@app.get("/api/extract/runs/{run_id}")
def get_extract_run(run_id: str):
    """Get full extraction results for a run."""
    import json as _json

    result_file = _stage2_output_dir / run_id / "stage2_result.json"
    if not result_file.exists():
        raise HTTPException(404, f"Extraction run {run_id} not found")

    with open(result_file, encoding="utf-8") as f:
        data = _json.load(f)

    # Return just the output objects (no raw input text — too large)
    jobs = []
    for item in data:
        out = item.get("output", {})
        jobs.append(out)

    return {"run_id": run_id, "jobs": jobs, "total": len(jobs)}


# =====================================================================
# Analytics — Aggregated insights from extraction results
# =====================================================================

@app.get("/api/analytics/dashboard")
def analytics_dashboard(
    run_id: str = Query(None),
    date_from: str = Query(None),
    date_to: str = Query(None),
    include_no_date: bool = Query(True),
):
    """
    Aggregate analytics from extraction results.
    If run_id is provided, analyze that run only.
    If omitted, aggregate ALL extraction runs.
    date_from / date_to: ISO date strings (YYYY-MM-DD) to filter by posted_date.
    include_no_date: whether to include jobs with no posted_date (default True).
    """
    import json as _json
    from collections import Counter
    from datetime import datetime as _dt

    if not _stage2_output_dir.exists():
        return {"error": "No extraction data", "jobs": 0}

    # Parse date bounds
    d_from = _dt.strptime(date_from, "%Y-%m-%d").date() if date_from else None
    d_to = _dt.strptime(date_to, "%Y-%m-%d").date() if date_to else None

    def _parse_posted_date(val):
        """Try to parse various posted_date formats into a date object."""
        if not val or not isinstance(val, str):
            return None
        val = val.strip()
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%d %B, %Y", "%d %B %Y", "%B %d %Y"):
            try:
                return _dt.strptime(val, fmt).date()
            except ValueError:
                continue
        return None

    # Collect all jobs
    all_jobs_raw = []

    if run_id:
        result_file = _stage2_output_dir / run_id / "stage2_result.json"
        if not result_file.exists():
            raise HTTPException(404, f"Run {run_id} not found")
        with open(result_file, encoding="utf-8") as f:
            for item in _json.load(f):
                all_jobs_raw.append(item.get("output", {}))
    else:
        for rd in _stage2_output_dir.iterdir():
            rf = rd / "stage2_result.json"
            if rf.exists():
                try:
                    with open(rf, encoding="utf-8") as f:
                        for item in _json.load(f):
                            all_jobs_raw.append(item.get("output", {}))
                except Exception:
                    pass

    # Apply date filter
    all_jobs = []
    jobs_no_date = 0
    for job in all_jobs_raw:
        parsed = _parse_posted_date(job.get("posted_date"))
        if parsed is None:
            jobs_no_date += 1
            if include_no_date:
                all_jobs.append(job)
            continue
        if d_from and parsed < d_from:
            continue
        if d_to and parsed > d_to:
            continue
        all_jobs.append(job)

    if not all_jobs:
        return {"jobs": 0, "total_before_filter": len(all_jobs_raw), "jobs_no_date": jobs_no_date}

    # ── Skill frequency ──
    skill_counter = Counter()
    for job in all_jobs:
        for s in job.get("skills", []):
            skill_counter[s.strip()] += 1

    top_skills = [{"skill": s, "count": c} for s, c in skill_counter.most_common(30)]

    # ── Google Trends style normalized skills (0-100 scale) ──
    peak = skill_counter.most_common(1)[0][1] if skill_counter else 1
    trending_skills = [
        {"skill": s, "score": round(c / peak * 100), "count": c}
        for s, c in skill_counter.most_common(15)
    ]

    # ── Category distribution ──
    cat_counter = Counter()
    for job in all_jobs:
        cat = job.get("job_category") or "Unknown"
        if isinstance(cat, list):
            cat = cat[0] if cat else "Unknown"
        cat_counter[cat] += 1
    categories = [{"category": c, "count": n} for c, n in cat_counter.most_common()]

    # ── ISCO-08 domain grouping (JobBERT similarity, Level 2) ──
    from isco_mapper import map_categories_batch as _map_isco, ISCO_SHORT_LABELS

    unique_cats = list(cat_counter.keys())
    isco_mappings = _map_isco(unique_cats)

    # Group by ISCO-08 Level 2 sub-major group
    from collections import defaultdict as _dd
    isco_domain_jobs = _dd(int)
    isco_domain_skills = _dd(lambda: Counter())
    isco_domain_meta = {}  # code -> (isco_label, short_label)
    for job in all_jobs:
        cat = job.get("job_category") or "Unknown"
        if isinstance(cat, list):
            cat = cat[0] if cat else "Unknown"
        mapping = isco_mappings.get(cat.strip())
        if not mapping or not mapping.get("isco_code"):
            continue
        code = mapping["isco_code"]
        short = ISCO_SHORT_LABELS.get(code, mapping["isco_label"])
        isco_domain_jobs[code] += 1
        isco_domain_meta[code] = (mapping["isco_label"], short)
        for s in job.get("skills", []):
            isco_domain_skills[code][s.strip()] += 1

    # Only show domains with enough jobs; merge tiny groups into "Other"
    MIN_DOMAIN_JOBS = max(10, len(all_jobs) // 100)  # at least 1% of jobs or 10
    isco_domains = []
    other_skills = Counter()
    other_jobs = 0
    for code in sorted(isco_domain_jobs, key=isco_domain_jobs.get, reverse=True):
        if isco_domain_jobs[code] >= MIN_DOMAIN_JOBS:
            full_label, short_label = isco_domain_meta[code]
            top = [{"skill": s, "count": c} for s, c in isco_domain_skills[code].most_common(10)]
            isco_domains.append({
                "domain": short_label,
                "isco_code": code,
                "isco_label": full_label,
                "job_count": isco_domain_jobs[code],
                "top_skills": top,
            })
        else:
            other_jobs += isco_domain_jobs[code]
            other_skills += isco_domain_skills[code]
    if other_jobs > 0:
        top = [{"skill": s, "count": c} for s, c in other_skills.most_common(10)]
        isco_domains.append({
            "domain": "Other",
            "isco_code": None,
            "isco_label": "Other occupations",
            "job_count": other_jobs,
            "top_skills": top,
        })

    # ── Location distribution ──
    loc_counter = Counter()
    for job in all_jobs:
        loc = job.get("location")
        if loc:
            loc_counter[loc.strip()] += 1
    locations = [{"location": l, "count": n} for l, n in loc_counter.most_common(15)]

    # ── Company distribution ──
    company_counter = Counter()
    for job in all_jobs:
        comp = job.get("company_name")
        if comp:
            company_counter[comp.strip()] += 1
    companies = [{"company": c, "count": n} for c, n in company_counter.most_common(15)]

    # ── Skills per category ──
    cat_skills = {}
    for job in all_jobs:
        cat = job.get("job_category") or "Unknown"
        if isinstance(cat, list):
            cat = cat[0] if cat else "Unknown"
        for s in job.get("skills", []):
            cat_skills.setdefault(cat, Counter())[s.strip()] += 1

    skills_by_category = {}
    for cat, counter in cat_skills.items():
        skills_by_category[cat] = [{"skill": s, "count": c} for s, c in counter.most_common(10)]

    # ── Skill co-occurrence (which skills appear together most often) ──
    from itertools import combinations
    cooccurrence = Counter()
    for job in all_jobs:
        skills = sorted(set(s.strip() for s in job.get("skills", [])))
        for pair in combinations(skills, 2):
            cooccurrence[pair] += 1
    top_pairs = [{"skills": list(pair), "count": c} for pair, c in cooccurrence.most_common(15)]

    # ── Salary insights ──
    salary_data = []
    for job in all_jobs:
        sal = job.get("salary")
        if sal:
            salary_data.append({
                "salary": sal,
                "category": job.get("job_category") or "Unknown",
                "title": job.get("job_title") or "Unknown",
            })

    # ── Curriculum recommendations ──
    # Group skills into domains and rank by demand
    domain_keywords = {
        "Programming & Software": ["programming", "development", "software", "coding", "engineering"],
        "Data & AI": ["data", "ai", "machine learning", "analytics", "database", "sql"],
        "Web & Mobile": ["react", "angular", "vue", "frontend", "backend", "web", "mobile", "api"],
        "Cloud & Infrastructure": ["cloud", "aws", "azure", "docker", "kubernetes", "infrastructure", "devops"],
        "Design & Creative": ["design", "ui", "ux", "figma", "photoshop", "creative", "canva"],
        "Business & Management": ["management", "business", "sales", "marketing", "communication", "leadership"],
        "Security & Compliance": ["security", "compliance", "audit", "risk", "regulatory"],
    }

    curriculum = {}
    for skill, count in skill_counter.most_common():
        skill_lower = skill.lower()
        placed = False
        for domain, keywords in domain_keywords.items():
            if any(kw in skill_lower for kw in keywords):
                curriculum.setdefault(domain, []).append({"skill": skill, "demand": count})
                placed = True
                break
        if not placed:
            curriculum.setdefault("Other / Specialized", []).append({"skill": skill, "demand": count})

    # Sort domains by total demand
    curriculum_sorted = []
    for domain, skills in curriculum.items():
        total = sum(s["demand"] for s in skills)
        curriculum_sorted.append({"domain": domain, "total_demand": total, "skills": skills[:10]})
    curriculum_sorted.sort(key=lambda d: d["total_demand"], reverse=True)

    return {
        "total_jobs": len(all_jobs),
        "total_before_filter": len(all_jobs_raw),
        "jobs_no_date": jobs_no_date,
        "total_skills": sum(skill_counter.values()),
        "unique_skills": len(skill_counter),
        "total_companies": len(company_counter),
        "total_locations": len(loc_counter),
        "top_skills": top_skills,
        "trending_skills": trending_skills,
        "categories": categories,
        "isco_domains": isco_domains,
        "locations": locations,
        "companies": companies,
        "skills_by_category": skills_by_category,
        "skill_cooccurrence": top_pairs,
        "salary_data": salary_data,
        "curriculum": curriculum_sorted,
    }


# =====================================================================
# Phase 5 — Curricula endpoints
# =====================================================================

_phase5_dir = pathlib.Path(__file__).resolve().parent.parent / "archive" / "phase5_legacy"
# The core methodology modules live in the clo_skill_alignment/ package.
_algo_dir = _phase5_dir / "clo_skill_alignment"
sys.path.insert(0, str(_algo_dir))

_p5_spec = _ilu.spec_from_file_location("pdf_parser", _algo_dir / "pdf_parser.py")
_pdf_parser_mod = _ilu.module_from_spec(_p5_spec)
_p5_spec.loader.exec_module(_pdf_parser_mod)
_parse_pdf = _pdf_parser_mod.parse_pdf

_runner_spec = _ilu.spec_from_file_location("phase5_runner", _algo_dir / "runner.py")
_runner_mod = _ilu.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner_mod)
_analyze_course = _runner_mod.analyze_course

_curriculum_spec = _ilu.spec_from_file_location(
    "phase5_curriculum", _algo_dir / "curriculum_runner.py"
)
_curriculum_mod = _ilu.module_from_spec(_curriculum_spec)
_curriculum_spec.loader.exec_module(_curriculum_mod)
_curriculum_analyze_one = _curriculum_mod.analyze_one
_curriculum_aggregate = _curriculum_mod.aggregate_curriculum
_MOCK_CURRICULUM = _curriculum_mod.MOCK_COURSES

_esco_spec = _ilu.spec_from_file_location(
    "phase5_esco_normalizer", _algo_dir / "esco_normalizer.py"
)
_esco_mod = _ilu.module_from_spec(_esco_spec)
_esco_spec.loader.exec_module(_esco_mod)
_esco_normalize_freq = _esco_mod.normalize_freq

# Lexical baseline (Ahadi exact-match) — alternative matching method.
_lexical_dir = _phase5_dir / "lexical_baseline"
sys.path.insert(0, str(_lexical_dir))
_lexical_spec = _ilu.spec_from_file_location(
    "phase5_lexical_runner", _lexical_dir / "lexical_runner.py"
)
_lexical_mod = _ilu.module_from_spec(_lexical_spec)
_lexical_spec.loader.exec_module(_lexical_mod)
_lexical_analyze_course = _lexical_mod.analyze_course
_lexical_analyze_one = _lexical_mod.analyze_one

# Domain router (qwen3 course → job-category classifier).
_domain_spec = _ilu.spec_from_file_location(
    "phase5_domain_router", _algo_dir / "domain_router.py"
)
_domain_mod = _ilu.module_from_spec(_domain_spec)
_domain_spec.loader.exec_module(_domain_mod)
_classify_course = _domain_mod.classify_course

# CLO → labour-market skill extractor (for symmetric ESCO matching).
_cloext_spec = _ilu.spec_from_file_location(
    "phase5_clo_extractor", _algo_dir / "clo_skill_extractor.py"
)
_cloext_mod = _ilu.module_from_spec(_cloext_spec)
_cloext_spec.loader.exec_module(_cloext_mod)
_extract_course_skills = _cloext_mod.extract_course_skills

# PDF → Markdown preview step (Recommendation V1 tab).
_parsepdf_spec = _ilu.spec_from_file_location(
    "phase5_parse_pdf_md", _phase5_dir / "baseline_analyze" / "parse_pdf" / "main.py"
)
_parsepdf_mod = _ilu.module_from_spec(_parsepdf_spec)
_parsepdf_spec.loader.exec_module(_parsepdf_mod)
_pdf_to_markdown      = _parsepdf_mod.pdf_to_markdown
_parse_sections       = _parsepdf_mod.parse_sections
_extract_skills_md    = _parsepdf_mod.extract_skills
_infer_year           = _parsepdf_mod.infer_year
_normalize_and_dedup  = _parsepdf_mod.normalize_and_dedup
# Which LLM the extraction runs on — PHASE5_LLM_MODEL in backend/.env
# (qwen3:14b local, or gpt-5.4-nano hosted). Called, not captured, so a change
# takes effect on reload without touching this module.
_extraction_model     = _parsepdf_mod.active_model

# Phase 6 — SELF-CONTAINED syllabus→skills for the Curriculum–Market Gap Analysis
# (V8) tab. phase5_curriculum_alignment/02_input_curriculum/curriculum_parse.py is a standalone copy
# the docling .docx branch merged) so phase6 does NOT reach into archive/phase5_legacy/
# for its curriculum side. The older V1 endpoints keep using _parsepdf_mod above.
_phase6_dir = pathlib.Path(__file__).resolve().parent.parent / "phase5_curriculum_alignment"
_p6_parse_spec = _ilu.spec_from_file_location(
    "phase6_curriculum_parse", _phase6_dir / "02_input_curriculum" / "curriculum_parse.py"
)
_p6_parse_mod = _ilu.module_from_spec(_p6_parse_spec)
_p6_parse_spec.loader.exec_module(_p6_parse_mod)
_p6_pdf_to_markdown     = _p6_parse_mod.pdf_to_markdown   # handles PDF AND .docx
_p6_parse_sections      = _p6_parse_mod.parse_sections
_p6_extract_skills      = _p6_parse_mod.extract_skills
_p6_normalize_and_dedup = _p6_parse_mod.normalize_and_dedup
_p6_infer_year          = _p6_parse_mod.infer_year
_p6_extraction_model    = _p6_parse_mod.active_model

# Recommendation V3 — SELF-CONTAINED PDF→skills (recommendation_v3/parse_pdf.py).
# V3 owns its full backend and does NOT reach into baseline_analyze/.
_v3_parsepdf_spec = _ilu.spec_from_file_location(
    "phase5_v3_parse_pdf", _phase5_dir / "recommendation_v3" / "parse_pdf.py"
)
_v3_parsepdf_mod = _ilu.module_from_spec(_v3_parsepdf_spec)
_v3_parsepdf_spec.loader.exec_module(_v3_parsepdf_mod)
_v3_pdf_to_markdown     = _v3_parsepdf_mod.pdf_to_markdown
_v3_parse_sections      = _v3_parsepdf_mod.parse_sections
_v3_extract_skills      = _v3_parsepdf_mod.extract_skills
_v3_normalize_and_dedup = _v3_parsepdf_mod.normalize_and_dedup

# V3 career-domain routing (domain_routing/ stage 5): course skills → fields + scores.
_v3_route_spec = _ilu.spec_from_file_location(
    "phase5_v3_route",
    _phase5_dir / "recommendation_v3" / "domain_routing" / "stage5_route.py"
)
_v3_route_mod = _ilu.module_from_spec(_v3_route_spec)
_v3_route_spec.loader.exec_module(_v3_route_mod)
_v3_selected_fields = _v3_route_mod.selected_fields
_v3_route_hybrid    = _v3_route_mod.route_hybrid    # LLM selects domains, cosine scores
_v3_all_domains     = _v3_route_mod.all_domains     # 20 career domains (UI chips)

# V3 coverage (Stage 6+7): course vs its routed domain inventory → label + gaps.
_v3_cov_spec = _ilu.spec_from_file_location(
    "phase5_v3_coverage", _phase5_dir / "recommendation_v3" / "coverage.py"
)
_v3_cov_mod = _ilu.module_from_spec(_v3_cov_spec)
_v3_cov_spec.loader.exec_module(_v3_cov_mod)
_v3_analyze = _v3_cov_mod.analyze

# V3 stage 5+6: canonical-pipeline recommend (consolidate → exact-match coverage).
_v3_rec_spec = _ilu.spec_from_file_location(
    "phase5_v3_recommend", _phase5_dir / "recommendation_v3" / "recommend.py"
)
_v3_rec_mod = _ilu.module_from_spec(_v3_rec_spec)
_v3_rec_spec.loader.exec_module(_v3_rec_mod)
_v3_recommend = _v3_rec_mod.recommend
_v3_recommend_independent = _v3_rec_mod.recommend_independent
_v3_all_categories = _v3_rec_mod.all_categories   # 17 raw job_category buckets (UI chips)

# V3 stage 4 only: competency consolidation (raw skills → canonical competencies).
_v3_consol_spec = _ilu.spec_from_file_location(
    "phase5_v3_consolidate", _phase5_dir / "recommendation_v3" / "consolidate.py"
)
_v3_consol_mod = _ilu.module_from_spec(_v3_consol_spec)
_v3_consol_spec.loader.exec_module(_v3_consol_mod)
_v3_consolidate = _v3_consol_mod.consolidate

# V3 independent grouping (de-biased, no market vocab) — recommendation_v3_independent/.
# Backs the grouping-only preview endpoint; the full pipeline (recommend.py) keeps its
# own sibling consolidate import.
_v3_indep_spec = _ilu.spec_from_file_location(
    "phase5_v3_consolidate_v3",
    _phase5_dir / "recommendation_v3_independent" / "consolidate_v3.py",
)
_v3_indep_mod = _ilu.module_from_spec(_v3_indep_spec)
_v3_indep_spec.loader.exec_module(_v3_indep_mod)
_v3_consolidate_independent = _v3_indep_mod.consolidate_independent

# V3 two-stage retrieve→verify analyze (recommendation_v3_independent/recommend_verified.py).
_v3_verify_spec = _ilu.spec_from_file_location(
    "phase5_v3_recommend_verified",
    _phase5_dir / "recommendation_v3_independent" / "recommend_verified.py",
)
_v3_verify_mod = _ilu.module_from_spec(_v3_verify_spec)
_v3_verify_spec.loader.exec_module(_v3_verify_mod)
_v3_analyze_verified = _v3_verify_mod.analyze_verified

# V7 action card (recommendation_v3_independent/action_card.py): wraps the frozen
# V6 engine for coverage + adds the relevance direction → label-free card payload
# (counts + ADD/RENAME/REVIEW actions). V6 modules are reused, not modified.
_v7_card_spec = _ilu.spec_from_file_location(
    "phase5_v7_action_card",
    _phase5_dir / "recommendation_v3_independent" / "action_card.py",
)
_v7_card_mod = _ilu.module_from_spec(_v7_card_spec)
_v7_card_spec.loader.exec_module(_v7_card_mod)
_v7_analyze_action_card = _v7_card_mod.analyze_action_card

# V7 suggest-add (recommendation_v3_independent/suggest_add.py): course-conditioned
# ADD suggestions — hard-skill filter + association lift over the IT posting
# cohort + embedding adjacency + qwen3 binary committee verdict (add/skip).
_v7_suggest_spec = _ilu.spec_from_file_location(
    "phase5_v7_suggest_add",
    _phase5_dir / "recommendation_v3_independent" / "suggest_add.py",
)
_v7_suggest_mod = _ilu.module_from_spec(_v7_suggest_spec)
_v7_suggest_spec.loader.exec_module(_v7_suggest_mod)
_v7_suggest_additions = _v7_suggest_mod.suggest_additions

# V3 ESCO normalization: map each course skill to its nearest ESCO preferred label.
_v3_esco_spec = _ilu.spec_from_file_location(
    "phase5_v3_esco", _phase5_dir / "recommendation_v3" / "esco.py"
)
_v3_esco_mod = _ilu.module_from_spec(_v3_esco_spec)
_v3_esco_spec.loader.exec_module(_v3_esco_mod)
_v3_esco_normalize = _v3_esco_mod.normalize

# V3 ESCO anchoring v2 (C1): Qwen3 retrieve top-5 candidates → qwen3:14b pick/reject.
_v3_escollm_spec = _ilu.spec_from_file_location(
    "phase5_v3_esco_llm", _phase5_dir / "recommendation_v3" / "esco_llm.py"
)
_v3_escollm_mod = _ilu.module_from_spec(_v3_escollm_spec)
_v3_escollm_spec.loader.exec_module(_v3_escollm_mod)
_v3_esco_normalize_llm = _v3_escollm_mod.normalize

# V3 ESCO-tab coverage: symmetric ESCO↔ESCO vs raw↔raw, cosine-matched.
_v3_escocov_spec = _ilu.spec_from_file_location(
    "phase5_v3_esco_coverage", _phase5_dir / "recommendation_v3" / "esco_coverage.py"
)
_v3_escocov_mod = _ilu.module_from_spec(_v3_escocov_spec)
_v3_escocov_spec.loader.exec_module(_v3_escocov_mod)
_v3_esco_analyze = _v3_escocov_mod.analyze

# AMC baseline — semantic, demand-weighted coverage (SBERT cosine + label).
_baseline_dir = _phase5_dir / "baseline_analyze"
if str(_baseline_dir) not in sys.path:
    sys.path.insert(0, str(_baseline_dir))
_baseline_spec = _ilu.spec_from_file_location("phase5_baseline_amc", _baseline_dir / "main.py")
_baseline_mod = _ilu.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline_mod)
_compute_coverage = _baseline_mod.compute_coverage

# Curriculum Recommendation pipeline (proposed) — importance-weighted
# (demand × specificity) coverage, domain-scoped to one LinkedIn category.
_cr_spec = _ilu.spec_from_file_location(
    "phase5_curric_rec", _phase5_dir / "curriculum_recommendation" / "main.py"
)
_cr_mod = _ilu.module_from_spec(_cr_spec)
_cr_spec.loader.exec_module(_cr_mod)
_cr_recommend_from_skills = _cr_mod.recommend_from_skills
_cr_classify_field = _cr_mod.classify_field
_cr_curate_suggestions = _cr_mod.curate_suggestions
_cr_recommend_v2 = _cr_mod.recommend_v2          # Method v2 (propose_pipeline_v2.md)


def _esco_map_skills(skills: list[str], threshold: float = 0.50) -> list[str]:
    """Map a list of free-text skills → unique ESCO preferred labels (cosine).

    Used to normalize the COURSE side so the match is ESCO-label ↔ ESCO-label.
    Below-threshold skills are kept raw (so nothing is silently dropped).
    """
    if not skills:
        return []
    mappings = _esco_mod.EscoNormalizer().map_skills(skills, threshold=threshold)
    seen, out = set(), []
    for m in mappings:
        lbl = m["esco_skill"]
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _course_units_for(req) -> list[str] | None:
    """Symmetric-ESCO course side: extracted CLO skills → ESCO labels (or None)."""
    if not getattr(req, "clo_esco", False):
        return None
    skills = req.course_skills
    if not skills:                      # fallback: extract now (LLM) if not cached
        try:
            skills = _extract_course_skills(req.course).get("skills_union", [])
        except Exception:
            skills = []
    return _esco_map_skills(skills) or None


# ---- Domain-conditioned matching helpers ----

def _norm_categories(v) -> list[str]:
    """Normalize a record's job_category (str | list | None) → lowercased list."""
    if isinstance(v, list):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip().lower()]
    return []


def _record_in_categories(rec: dict, catset: set[str] | None) -> bool:
    """True if no filter, or the record's job_category intersects catset."""
    if not catset:
        return True
    return bool(set(_norm_categories(rec.get("output", {}).get("job_category"))) & catset)


def _iter_stage2_records():
    """Yield every stage-2 posting record across all runs."""
    import json as _json
    if not _stage2_output_dir.exists():
        return
    for rd in _stage2_output_dir.iterdir():
        if not rd.is_dir():
            continue
        rf = rd / "stage2_result.json"
        if not rf.exists():
            continue
        try:
            with open(rf, encoding="utf-8") as f:
                for item in _json.load(f):
                    yield item
        except Exception:
            pass


def _list_job_categories(top_n: int | None = None) -> tuple[list[dict], int]:
    """Distinct job_category labels + posting counts across stage-2 outputs."""
    from collections import Counter
    counter: Counter = Counter()
    total = 0
    for rec in _iter_stage2_records():
        total += 1
        for c in _norm_categories(rec.get("output", {}).get("job_category")):
            counter[c] += 1
    items = counter.most_common(top_n)
    return [{"category": c, "count": n} for c, n in items], total


# ---- LinkedIn corpus (pre-aggregated {category: {skill: freq}} cache) ----

_LINKEDIN_INV_PATH = _algo_dir / "linkedin_inventory.json"
_linkedin_inv_cache: dict | None = None


def _load_linkedin_inventory() -> dict:
    """Lazy-load the trimmed top-N LinkedIn inventory (small, ~200KB)."""
    global _linkedin_inv_cache
    if _linkedin_inv_cache is None:
        import json as _json
        if not _LINKEDIN_INV_PATH.exists():
            raise HTTPException(404, "LinkedIn inventory not found — run build_linkedin_inventory.py + trim_inventory.py")
        _linkedin_inv_cache = _json.loads(_LINKEDIN_INV_PATH.read_text(encoding="utf-8"))
    return _linkedin_inv_cache


def _linkedin_categories() -> tuple[list[dict], int]:
    inv = _load_linkedin_inventory()
    counts = inv.get("category_job_counts", {})
    cats = [{"category": c, "count": n}
            for c, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    return cats, inv.get("total_jobs", 0)


def _linkedin_skill_freq(categories: list[str] | None,
                         weighting: str = "raw") -> tuple[dict[str, int], int]:
    """Merge skill weights across the selected LinkedIn categories (all if None).

    weighting="raw":   raw frequency f(s).
    weighting="tfidf": demand-distinctiveness f(s) × cross-category IDF(s) —
        down-weights ubiquitous skills (communication), up-weights domain-
        distinctive ones (python, aws). Follows the TF-IDF skill-specificity
        literature (Skills-First; Shi et al. 2020; RCA in Ahadi et al. 2022).
    """
    from collections import Counter
    inv = _load_linkedin_inventory()
    cats = inv.get("categories", {})
    counts = inv.get("category_job_counts", {})
    idf = inv.get("idf", {})
    selected = ([c.strip().lower() for c in categories if c and c.strip()]
                if categories else list(cats.keys()))
    merged: Counter = Counter()
    jobs = 0
    for c in selected:
        for s, n in cats.get(c, {}).items():
            w = n * idf.get(s, 1.0) if weighting == "tfidf" else n
            merged[s] += w
        jobs += counts.get(c, 0)
    if not categories:
        jobs = inv.get("total_jobs", jobs)
    return {s: int(round(v)) for s, v in merged.items()}, jobs


@app.get("/api/curricula/categories")
def curricula_categories(top_n: int = Query(40), source: str = Query("cambodia")):
    """List the job categories (with posting counts) for domain filtering."""
    if source == "linkedin":
        cats, total = _linkedin_categories()
        return {"categories": cats[:top_n], "total_jobs": total, "source": source}
    cats, total = _list_job_categories(top_n=top_n)
    return {"categories": cats, "total_jobs": total, "source": source}


class ClassifyDomainReq(BaseModel):
    course: dict
    source: str = "cambodia"


@app.post("/api/curricula/classify-domain")
def curricula_classify_domain(req: ClassifyDomainReq):
    """qwen3-suggest the job category(ies) a course belongs to, plus the full list."""
    if req.source == "linkedin":
        cats, total = _linkedin_categories()
    else:
        cats, total = _list_job_categories(top_n=40)
    try:
        suggested = _classify_course(req.course, cats)
    except Exception as e:
        raise HTTPException(500, f"Classification failed: {e}")
    return {"suggested": suggested, "categories": cats, "total_jobs": total}


@app.post("/api/curricula/upload")
async def curricula_upload(file: UploadFile = File(...)):
    """Parse a course syllabus PDF and extract learning outcomes + lessons via LLM."""
    content = await file.read()
    try:
        result = _parse_pdf(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to parse syllabus: {e}")
    # Extract labour-market skills from the CLOs once, here — reused by the
    # symmetric-ESCO matching path (avoids an LLM call on every analyze).
    course_skills: list[str] = []
    try:
        course_skills = _extract_course_skills(result).get("skills_union", []) or []
    except Exception as e:
        print(f"[upload] CLO skill extraction failed ({e}) — symmetric ESCO will fall back")
    return {
        "course": result["course"],
        "lessons": result["lessons"],
        "learning_outcomes": result["learning_outcomes"],
        "course_skills": course_skills,
    }


@app.post("/api/curricula/markdown")
async def curricula_markdown(file: UploadFile = File(...)):
    """Convert an uploaded syllabus PDF/DOCX to Markdown (V1 preview, used by V8 step 1).

    .docx goes through the V3 parser's docling branch — the conversion is a
    format concern, not a V3 one, and baseline_analyze's pdf_to_markdown only
    speaks PDF. The PDF path is untouched.
    """
    content = await file.read()
    is_docx = (file.filename or "").lower().endswith(".docx")
    try:
        markdown = (_v3_pdf_to_markdown(content, filename=file.filename)
                    if is_docx else _pdf_to_markdown(content))
    except Exception as e:
        raise HTTPException(500, f"Failed to convert syllabus to markdown: {e}")
    return {"filename": file.filename, "chars": len(markdown), "markdown": markdown}


class ExtractMarkdownReq(BaseModel):
    markdown: str


@app.post("/api/curricula/extract-skills")
def curricula_extract_skills(req: ExtractMarkdownReq):
    """Markdown → sections → skills (Stage 2-4): {course, skills[{name,source,evidence}]}."""
    try:
        sec = _parse_sections(req.markdown)
        skills = _normalize_and_dedup(_extract_skills_md(sec))
    except Exception as e:
        raise HTTPException(500, f"Skill extraction failed: {e}")
    return {
        "course": sec.course.model_dump(),
        "skills": [s.model_dump() for s in skills],
    }


# ── Saved course skills (V8 Step 1 snapshots on Neon) ──────────────────────────
class SaveCourseSkillsReq(BaseModel):
    course: dict                 # {code, title, category, ...} from extract-skills
    skills: list                 # [{name, source, evidence}]
    filename: str | None = None  # original PDF name
    markdown: str | None = None  # extracted markdown (provenance)


@app.post("/api/curricula/course-skills")
def curricula_save_course_skills(req: SaveCourseSkillsReq):
    """Persist one extracted course-skill set. Upserts by (code, title) so
    re-saving the same course replaces its skills; replay never needs Ollama."""
    title = (req.course or {}).get("title") or ""
    if not title:
        raise HTTPException(400, "course.title is required")
    if not req.skills:
        raise HTTPException(400, "skills must be non-empty")
    try:
        course = req.course or {}
        return db_save_course_skills(
            course_code=course.get("code") or "",
            course_title=title,
            category=course.get("category"),
            filename=req.filename,
            markdown=req.markdown,
            skills=req.skills,
            # code/filename regex is deterministic and beats the LLM when both
            # answer (nano read "GICI31" as year 1); LLM covers codeless syllabi
            year=_infer_year(course.get("code"), req.filename)
                 or course.get("year"),
        )
    except Exception as e:
        raise HTTPException(500, f"save course skills failed: {e}")


@app.get("/api/curricula/course-skills")
def curricula_list_course_skills(limit: int = 100):
    try:
        return {"courses": db_list_course_skills(limit=limit)}
    except Exception as e:
        raise HTTPException(500, f"list course skills failed: {e}")


@app.get("/api/curricula/course-skills/{row_id}")
def curricula_get_course_skills(row_id: int):
    try:
        row = db_get_course_skills(row_id)
    except Exception as e:
        raise HTTPException(500, f"get course skills failed: {e}")
    if not row:
        raise HTTPException(404, f"course skills {row_id} not found")
    return row


@app.delete("/api/curricula/course-skills/{row_id}")
def curricula_delete_course_skills(row_id: int):
    try:
        return {"deleted": db_delete_course_skills(row_id)}
    except Exception as e:
        raise HTTPException(500, f"delete course skills failed: {e}")


# ── Syllabus document store (V8 Step 1) — R2 archive + Postgres markdown ──────
#
# A syllabus is uploaded once, ever. Identity is the sha256 of the bytes (ITC
# ships the same document as both `..._2024-2025.docx` and `..._2024_2025.docx`,
# so filename dedupe would miss). The original goes to R2 for provenance; the
# Markdown lives in Postgres and is what every later stage reads.

def _archive_to_r2(sha: str, ext: str, content: bytes) -> tuple:
    """Best-effort archive. Returns (bucket, key, r2_status) — never raises."""
    if not r2_store.configured():
        return None, None, "skipped"
    key = r2_store.key_for(sha, ext)
    ok = r2_store.put_bytes(key, content, ext)
    return r2_store.BUCKET, key, ("stored" if ok else "failed")


@app.post("/api/curricula/documents")
async def curricula_upload_document(file: UploadFile = File(...)):
    """Upload one syllabus: hash → R2 archive → row → Markdown.

    The resume gate: a file already in the store is not re-archived or
    re-converted, but each stage is checked on its own, so a row whose R2 put
    failed (creds down) or whose conversion never finished is healed by dropping
    the same file again. Fully-done files return `cached: True` having run
    nothing at all.
    """
    content = await file.read()
    name = file.filename or "upload"
    ext = pathlib.Path(name).suffix.lower().lstrip(".")
    if ext not in ("pdf", "docx"):
        raise HTTPException(
            400, f"unsupported file type '.{ext}' — upload PDF or Word (.docx)")
    sha = hashlib.sha256(content).hexdigest()

    existing = db_find_document(sha)
    archived = bool(existing) and existing["r2_status"] in ("stored", "skipped")
    converted = bool(existing) and bool(existing["chars"])
    if archived and converted:
        return {**existing, "cached": True}

    if archived:                       # keep what is already in the bucket
        bucket, key, r2_status = (existing["r2_bucket"], existing["r2_key"],
                                  existing["r2_status"])
    else:                              # first upload, or a retry of a failed put
        bucket, key, r2_status = _archive_to_r2(sha, ext, content)

    doc = db_save_document(sha256=sha, filename=name, ext=ext,
                           size_bytes=len(content), r2_bucket=bucket,
                           r2_key=key, r2_status=r2_status)
    if converted:                      # archive-only retry — markdown is fine
        return {**doc, "cached": True}

    try:
        parser = "docling" if ext == "docx" else "pymupdf4llm"
        markdown = _p6_pdf_to_markdown(content, filename=name)  # phase6 owns V8
        doc = db_set_document_markdown(doc["id"], markdown, parser)
    except Exception as e:
        db_set_document_status(doc["id"], "error", str(e))
        raise HTTPException(500, f"Failed to convert syllabus to markdown: {e}")
    return {**doc, "cached": False}


@app.get("/api/curricula/extraction-model")
def curricula_extraction_model():
    """Which LLM skill extraction will use — set PHASE5_LLM_MODEL in backend/.env
    (`qwen3:14b` local / `gpt-5.4-nano` hosted). Lets the UI say so up front,
    since the two differ in cost and in how long a course takes."""
    model = _p6_extraction_model()
    return {"model": model,
            "provider": "openai" if model.startswith("gpt-") else "ollama"}


@app.get("/api/curricula/documents")
def curricula_list_documents(limit: int = 200):
    """Every stored syllabus, newest first — this is what rebuilds the Step 1
    queue on page load (markdown excluded; fetch it per row on demand)."""
    try:
        return {"documents": db_list_documents(limit=limit)}
    except Exception as e:
        raise HTTPException(500, f"list documents failed: {e}")


@app.get("/api/curricula/documents/{doc_id}")
def curricula_get_document(doc_id: int):
    try:
        doc = db_get_document(doc_id)
    except Exception as e:
        raise HTTPException(500, f"get document failed: {e}")
    if not doc:
        raise HTTPException(404, f"document {doc_id} not found")
    return doc


@app.get("/api/curricula/documents/{doc_id}/file")
def curricula_document_file(doc_id: int):
    """302 to a presigned R2 URL — downloads the original .docx/.pdf.

    R2 buckets are private, so the URL has to be signed at request time rather
    than stored in the row.
    """
    doc = db_get_document(doc_id, with_markdown=False)
    if not doc:
        raise HTTPException(404, f"document {doc_id} not found")
    if doc["r2_status"] != "stored" or not doc["r2_key"]:
        raise HTTPException(
            404, f"original not archived in R2 (r2_status={doc['r2_status']})")
    url = r2_store.presign(doc["r2_key"])
    if not url:
        raise HTTPException(503, "R2 not configured — cannot sign a download URL")
    return RedirectResponse(url, status_code=302)


@app.post("/api/curricula/documents/{doc_id}/extract-skills")
def curricula_document_extract_skills(doc_id: int):
    """Markdown FROM POSTGRES → qwen3 → course skills linked to the document.

    The uploaded file is never touched again — this is a DB read plus one LLM
    call, which is also what makes re-running the extraction with a different
    model cheap.
    """
    doc = db_get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"document {doc_id} not found")
    if not doc.get("markdown"):
        raise HTTPException(400, f"document {doc_id} has no markdown yet")
    model = _p6_extraction_model()
    try:
        sec = _p6_parse_sections(doc["markdown"])
        skills = _p6_normalize_and_dedup(_p6_extract_skills(sec))
    except Exception as e:
        raise HTTPException(500, f"Skill extraction failed ({model}): {e}")

    course = sec.course.model_dump()
    title = course.get("title") or ""
    if not title:
        raise HTTPException(500, "extraction produced no course title")
    skills_json = [s.model_dump() for s in skills]
    saved = db_save_course_skills(
        course_code=course.get("code") or "",
        course_title=title,
        category=course.get("category"),
        filename=doc["filename"],
        markdown=doc["markdown"],
        skills=skills_json,
        document_id=doc_id,
        # year: deterministic code/filename regex first (beats LLM when both
        # answer — nano read "GICI31" as year 1); LLM covers codeless syllabi
        year=_p6_infer_year(course.get("code"), doc["filename"]) or course.get("year"),
    )
    db_set_document_status(doc_id, "extracted")
    return {"document_id": doc_id, "course": course, "skills": skills_json,
            "saved": saved, "model": model}


@app.delete("/api/curricula/documents/{doc_id}")
def curricula_delete_document(doc_id: int):
    """Remove from the queue. The R2 object is kept on purpose — it is the
    archive, and the row can be recreated by re-uploading the same bytes."""
    try:
        return {"deleted": db_delete_document(doc_id)}
    except Exception as e:
        raise HTTPException(500, f"delete document failed: {e}")


# ── Recommendation V3 — self-contained endpoints (recommendation_v3/) ──────────

@app.post("/api/curricula/v3/markdown")
async def curricula_v3_markdown(file: UploadFile = File(...)):
    """V3: uploaded syllabus PDF/DOCX → Markdown (self-contained recommendation_v3).

    filename is forwarded because the upload arrives as bytes: the extension is
    the only way to tell a .docx (→ docling) from a PDF (→ pymupdf4llm).
    """
    content = await file.read()
    try:
        markdown = _v3_pdf_to_markdown(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(500, f"Failed to convert syllabus to markdown: {e}")
    return {"filename": file.filename, "chars": len(markdown), "markdown": markdown}


@app.post("/api/curricula/v3/extract-skills")
def curricula_v3_extract_skills(req: ExtractMarkdownReq):
    """V3: Markdown → sections → skills → career-domain routing.

    Returns {course, skills}. course carries the seeds for BOTH compare modes:
    `category` (LLM-inferred job_category bucket, e.g. "it") pre-selects the
    "Compare by job category" chips, and `fields` (routed career domain(s) with
    computed cosine scores, route.py) pre-selects "Compare by career domain".
    """
    try:
        sec = _v3_parse_sections(req.markdown)
        skills = _v3_normalize_and_dedup(_v3_extract_skills(sec))
        course = sec.course.model_dump()
        # Hybrid routing: qwen3 picks the 1–3 career domains, cosine scores them —
        # scoped to the course's job_category taxonomy (build_profiles.py).
        course["fields"] = _v3_route_hybrid(course.get("title") or "",
                                            [s.name for s in skills],
                                            category=course.get("category"))
    except Exception as e:
        raise HTTPException(500, f"Skill extraction failed: {e}")
    return {
        "course": course,
        "skills": [s.model_dump() for s in skills],
    }


@app.get("/api/curricula/v3/categories")
def curricula_v3_categories():
    """V3: the unique job_category buckets of the raw LinkedIn 1.35M corpus
    (category_vocab.json, built by build_category_vocab.py), with posting
    counts — the "Compare by job category" chip list. Selecting one scores the
    course against that category's raw demand core via recommend._core()."""
    return {"categories": _v3_all_categories()}


@app.get("/api/curricula/v3/domains")
def curricula_v3_domains(category: str | None = None):
    """V3: every curated career domain (domain_profiles.json), with posting
    counts + owning job_category — the "Compare by career domain" chip list.
    `category` scopes the list to one job_category's taxonomy (e.g. only IT
    domains for an IT course; empty until that category's taxonomy is built)."""
    return {"domains": _v3_all_domains(category)}


class AnalyzeV3Req(BaseModel):
    course_skills: list[str]
    domains: list[str] | None = None     # if omitted, route from the skills
    title: str = ""
    course_code: str = ""                 # stored on the consolidation cache (readability)
    top_k: int = 30
    markdown: str = ""                    # syllabus md → infer implied tools (python…)
    matcher: str = "verify"               # "verify" (LLM) | "cosine" (embedding-only ablation)


@app.post("/api/curricula/v3/analyze")
def curricula_v3_analyze(req: AnalyzeV3Req):
    """V3 Stage 5+6 (canonical pipeline): consolidate the course's skills into
    competencies, then exact-match the MARKET competencies against its routed
    domain(s)' top-K demand core → {coverage_pct, recommendation, covered[+evidence],
    gaps[], specialized[]}. Routes from skills if `domains` is not supplied."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v3_recommend(req.course_skills, domains, top_k=req.top_k,
                               markdown=req.markdown, course_code=req.course_code)
    except Exception as e:
        raise HTTPException(500, f"V3 analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


class ConsolidateV3Req(BaseModel):
    course_skills: list[str]
    domains: list[str] | None = None     # if omitted, route from the skills
    title: str = ""
    course_code: str = ""                 # stored on the consolidation cache (readability)


@app.post("/api/curricula/v3/consolidate")
def curricula_v3_consolidate(req: ConsolidateV3Req):
    """V3 Stage 4 ONLY: group the course's raw extracted skills into canonical
    competencies (market) + curriculum-specific competencies (specialized), each
    with its evidence (the original skills). No coverage scoring — this is the
    competency-grouping preview shown right after extraction. Routes from skills
    if `domains` is not supplied. Cached per (skills, domains)."""
    domains = req.domains
    if not domains:
        domains = [f["name"] for f in _v3_route_hybrid(req.title, req.course_skills)]
    try:
        return _v3_consolidate(req.course_skills, domains,
                               course_code=req.course_code, course_title=req.title)
    except Exception as e:
        raise HTTPException(500, f"V3 consolidate failed: {e}")


@app.post("/api/curricula/v3/independent-consolidate")
def curricula_v3_independent_consolidate(req: ConsolidateV3Req):
    """V3 proposal method 1: group raw course skills into competencies using only
    syllabus-derived skills. Job-posting data is not used in this grouping step."""
    try:
        return _v3_consolidate_independent(
            req.course_skills,
            course_code=req.course_code,
            course_title=req.title,
        )
    except Exception as e:
        raise HTTPException(500, f"V3 independent consolidate failed: {e}")


@app.post("/api/curricula/v3/independent-analyze")
def curricula_v3_independent_analyze(req: AnalyzeV3Req):
    """V3 proposal method 1: independent course competency grouping first, then
    semantic comparison against the routed job-market core."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v3_recommend_independent(
            req.course_skills,
            domains,
            top_k=req.top_k,
            markdown=req.markdown,
            course_code=req.course_code,
            course_title=req.title,
        )
    except Exception as e:
        raise HTTPException(500, f"V3 independent analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


@app.post("/api/curricula/v3/verified-analyze")
def curricula_v3_verified_analyze(req: AnalyzeV3Req):
    """V3 two-stage retrieve→verify: group (no market vocab) → embedding shortlist
    → qwen3 yes/no verification of each candidate → demand-weighted coverage."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v3_analyze_verified(
            req.course_skills,
            domains,
            top_k=req.top_k,
            markdown=req.markdown,
            course_code=req.course_code,
            course_title=req.title,
        )
    except Exception as e:
        raise HTTPException(500, f"V3 verified analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


@app.post("/api/curricula/v3/verified-raw-analyze")
def curricula_v3_verified_raw_analyze(req: AnalyzeV3Req):
    """V6: retrieve→verify on RAW course skills — no competency grouping.
    Each extracted skill is its own matching unit: embedding shortlist →
    qwen3 yes/no verification → demand-weighted coverage."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v3_analyze_verified(
            req.course_skills,
            domains,
            top_k=req.top_k,
            markdown=req.markdown,
            course_code=req.course_code,
            course_title=req.title,
            group=False,
            matcher=req.matcher,
        )
    except Exception as e:
        raise HTTPException(500, f"V3 verified-raw analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


@app.post("/api/curricula/v3/action-card")
def curricula_v3_action_card(req: AnalyzeV3Req):
    """V7: label-free ACTION CARD — the frozen V6 retrieve→verify engine gives
    field-core coverage (market→course); this adds the relevance direction
    (course→market: is each taught unit demanded anywhere in the top vocab?)
    and returns counts + typed actions (ADD / RENAME / REVIEW). No label, no
    headline % — coverage is supplementary only."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v7_analyze_action_card(
            req.course_skills,
            domains,
            top_k=req.top_k,
            markdown=req.markdown,
            course_code=req.course_code,
            course_title=req.title,
        )
    except Exception as e:
        raise HTTPException(500, f"V7 action-card analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


class SuggestAddReq(BaseModel):
    taught: list[str]                 # all taught units (in-demand + review)
    anchors: list[str]                # verified taught buckets (market wording)
    course_title: str = ""
    vocab_n: int = 200
    llm: bool = True
    force: bool = False


@app.post("/api/curricula/v3/suggest-add")
def curricula_v3_suggest_add(req: SuggestAddReq):
    """V7 third panel: course-conditioned ADD suggestions. Association lift
    mined from the IT posting cohort (demanded TOGETHER with what the course
    teaches, vs baseline) + embedding adjacency + qwen3 committee, one binary
    add/skip verdict per candidate — only approved ADDs reach the payload."""
    if not req.taught or not req.anchors:
        raise HTTPException(400, "taught and anchors must be non-empty")
    try:
        return _v7_suggest_additions(
            req.taught, req.anchors, course_title=req.course_title,
            vocab_n=req.vocab_n, llm=req.llm, force=req.force)
    except Exception as e:
        raise HTTPException(500, f"V7 suggest-add failed: {e}")


# ── Saved analyses (V6 snapshots on Neon) ──────────────────────────────────────
class SaveAnalysisReq(BaseModel):
    course_code: str = ""
    course_title: str = ""
    result: dict


@app.post("/api/curricula/v3/analyses")
def curricula_v3_save_analysis(req: SaveAnalysisReq):
    """Persist a completed analyze run (full result JSON + the settings that
    produced it). Duplicate runs (same course + settings + result) return the
    existing row instead of inserting again."""
    r = req.result or {}
    esco_fn = getattr(_v3_verify_mod, "_ESCO_NORMALIZE", None)
    settings = {
        "corpus": "linkedin",
        "domains": r.get("domains"),
        "top_k": r.get("top_k"),
        "method": r.get("method"),
        "retrieve_floor": _v3_verify_mod._match.FLOOR,
        "retrieve_top_n": _v3_verify_mod._match.TOP_N,
        "embed_model": _v3_verify_mod._match.EMBED_MODEL,
        "verify_model": _v3_verify_mod._match.VERIFY_MODEL,
        "esco_mode": ("threshold" if esco_fn is not None
                      and esco_fn.__name__ == "normalize_threshold" else "llm_pick"),
        "pipeline_version": "v3",
    }
    domains = r.get("domains") or []
    try:
        return db_save_analysis(
            course_code=req.course_code,
            course_title=req.course_title,
            domain=domains[0] if domains else None,
            coverage_pct=(r.get("coverage_pct") if r.get("coverage_pct") is not None
                          else r.get("alignment_score")),
            recommendation=r.get("recommendation"),
            settings=settings,
            result=r,
        )
    except Exception as e:
        raise HTTPException(500, f"save analysis failed: {e}")


@app.get("/api/curricula/v3/analyses")
def curricula_v3_list_analyses(course_code: str | None = None, limit: int = 50):
    try:
        return {"analyses": db_list_analyses(course_code=course_code, limit=limit)}
    except Exception as e:
        raise HTTPException(500, f"list analyses failed: {e}")


@app.get("/api/curricula/v3/analyses/{analysis_id}")
def curricula_v3_get_analysis(analysis_id: int):
    try:
        row = db_get_analysis(analysis_id)
    except Exception as e:
        raise HTTPException(500, f"get analysis failed: {e}")
    if not row:
        raise HTTPException(404, f"analysis {analysis_id} not found")
    return row


@app.delete("/api/curricula/v3/analyses/{analysis_id}")
def curricula_v3_delete_analysis(analysis_id: int):
    try:
        return {"deleted": db_delete_analysis(analysis_id)}
    except Exception as e:
        raise HTTPException(500, f"delete analysis failed: {e}")


class EscoNormalizeReq(BaseModel):
    course_skills: list[str]
    threshold: float = 0.70               # below this, keep the raw skill (avoid a bad label)
    llm: bool = False                     # True → Qwen3 top-5 retrieve + qwen3:14b pick/reject


@app.post("/api/curricula/v3/esco-normalize")
def curricula_v3_esco_normalize(req: EscoNormalizeReq):
    """Recommendation-ESCO: map each extracted course skill to an ESCO preferred
    label. Default: nearest-surface cosine with a τ gate. llm=true (V6/C1 path):
    Qwen3 retrieves top-5 candidate labels, qwen3:14b picks the correct one or
    rejects all (kept raw, mapped=false). Returns
    {normalized: [{orig, esco, score, mapped, ...}], mapped_count, total}."""
    try:
        if req.llm:
            rows = _v3_esco_normalize_llm(req.course_skills)
        else:
            rows = _v3_esco_normalize(req.course_skills, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(500, f"V3 ESCO normalize failed: {e}")
    return {
        "normalized": rows,
        "mapped_count": sum(1 for r in rows if r["mapped"]),
        "total": len(rows),
    }


class EscoAnalyzeReq(BaseModel):
    course_skills: list[str]
    domains: list[str] | None = None      # if omitted, route from the skills
    title: str = ""
    top_k: int = 15                        # # of top market-demand skills to score against
    mode: str = "esco"                     # "esco" (symmetric ESCO) | "raw"
    threshold: float = 0.70                # cosine match threshold


@app.post("/api/curricula/v3/esco-analyze")
def curricula_v3_esco_analyze(req: EscoAnalyzeReq):
    """Recommendation-ESCO coverage: course skills vs the routed domain(s)' top-K
    market demand, matched by COSINE ≥ threshold, demand-weighted. mode='raw'
    compares raw strings; mode='esco' maps BOTH sides to ESCO labels first
    (symmetric ESCO). Returns {coverage_pct, covered[+evidence], gaps[], off_core[]}.
    Routes from skills if `domains` is not supplied."""
    domains, fields = req.domains, None
    if not domains:
        fields = _v3_route_hybrid(req.title, req.course_skills)
        domains = [f["name"] for f in fields]
    try:
        result = _v3_esco_analyze(req.course_skills, domains, top_k=req.top_k,
                                  mode=req.mode, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(500, f"V3 ESCO analyze failed: {e}")
    if fields is not None:
        result["fields"] = fields
    return result


class MatchCoverageReq(BaseModel):
    course_skills: list[str]
    source: str = "linkedin"               # "linkedin" | "cambodia"
    categories: list[str] | None = None    # domain filter (job categories)
    threshold: float = 0.70                # cosine match threshold (tuned for Qwen3-Embedding)
    top_n: int = 100                       # cap demand table to top-N skills


@app.post("/api/curricula/match-coverage")
def curricula_match_coverage(req: MatchCoverageReq):
    """Semantic demand-weighted coverage (AMC baseline): course skills vs job demand.

    Returns {matched, gaps, scores{coverage_pct, demand_weighted_pct, …},
    label (MAINTAIN/REVISE/UPDATE), metadata, total_jobs}.
    """
    from collections import Counter

    catlist = [c.strip().lower() for c in (req.categories or []) if c.strip()] or None

    if req.source == "linkedin":
        freq, total_jobs = _linkedin_skill_freq(catlist, weighting="raw")
    else:
        if not _stage2_output_dir.exists():
            raise HTTPException(404, "No stage2 output (cambodia corpus) found")
        catset = set(catlist) if catlist else None
        counter: Counter = Counter()
        total_jobs = 0
        for item in _iter_stage2_records():
            if not _record_in_categories(item, catset):
                continue
            total_jobs += 1
            for s in item.get("output", {}).get("skills", []):
                counter[s.strip().lower()] += 1
        freq = dict(counter)

    if not freq:
        raise HTTPException(404, "No job skills for this corpus/category")

    try:
        result = _compute_coverage(req.course_skills, freq,
                                   threshold=req.threshold, top_n=req.top_n, min_freq=2)
    except Exception as e:
        raise HTTPException(500, f"Coverage failed: {e}")
    result["total_jobs"] = total_jobs
    return result


# cache job-vocab embeddings per (source, categories, vocab_n) — embedding the
# full category vocabulary is slow; compute once, reuse across requests.
_job_vocab_cache: dict = {}


def _job_freq_for(source: str, catlist: list[str] | None):
    """Build {skill: frequency} demand table for a corpus/category (full, untrimmed)."""
    from collections import Counter
    if source == "linkedin":
        return _linkedin_skill_freq(catlist, weighting="raw")
    if not _stage2_output_dir.exists():
        raise HTTPException(404, "No stage2 output (cambodia corpus) found")
    catset = set(catlist) if catlist else None
    counter: Counter = Counter()
    total = 0
    for item in _iter_stage2_records():
        if not _record_in_categories(item, catset):
            continue
        total += 1
        for s in item.get("output", {}).get("skills", []):
            counter[s.strip().lower()] += 1
    return dict(counter), total


class MarketValueReq(BaseModel):
    course_skills: list[str]
    source: str = "linkedin"
    categories: list[str] | None = None
    threshold: float = 0.70   # tuned for Qwen3-Embedding
    vocab_n: int = 2000   # search the top-N of the FULL vocab; <= 0 means ALL unique skills


@app.post("/api/curricula/market-value")
def curricula_market_value(req: MarketValueReq):
    """Course → job: for each COURSE skill, is the market asking for it (at any
    frequency)? Searches the full category vocabulary, not the top-100 — so niche
    technical skills (e.g. gradient boosting) aren't hidden by the freq trim.

    Returns per-course-skill {best_job_skill, score, job_freq, in_demand} + a
    market-value summary (how many of the taught skills are in real demand).
    """
    import numpy as np

    catlist = [c.strip().lower() for c in (req.categories or []) if c.strip()] or None
    freq, total_jobs = _job_freq_for(req.source, catlist)
    if not freq:
        raise HTTPException(404, "No job skills for this corpus/category")

    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    if req.vocab_n > 0:                      # <= 0 → search ALL unique skills
        items = items[:req.vocab_n]
    vocab = [s for s, _ in items]
    freqs = [f for _, f in items]

    key = (req.source, tuple(sorted(catlist or [])), req.vocab_n)
    cached = _job_vocab_cache.get(key)
    if cached and cached[0] == vocab:
        job_emb = cached[1]
    else:
        job_emb = _baseline_mod.embed_skills(vocab)   # (V, D) L2-normalized
        _job_vocab_cache[key] = (vocab, job_emb)

    course_emb = _baseline_mod.embed_skills(req.course_skills)  # (C, D)
    sims = course_emb @ job_emb.T                              # (C, V)
    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    results = []
    in_demand = 0
    demand_total = 0
    for i, cs in enumerate(req.course_skills):
        j = int(best_idx[i])
        sc = float(best_score[i])
        fr = int(freqs[j])
        hit = sc >= req.threshold
        if hit:
            in_demand += 1
            demand_total += fr
        results.append({
            "skill": cs, "best_job_skill": vocab[j], "score": round(sc, 3),
            "job_freq": fr, "in_demand": hit,
        })
    results.sort(key=lambda r: (r["in_demand"], r["job_freq"]), reverse=True)

    n = len(req.course_skills)
    return {
        "course_skills": results,
        "summary": {
            "in_demand": in_demand,
            "total": n,
            "in_demand_pct": round(100 * in_demand / n, 1) if n else 0.0,
            "demand_total": demand_total,
            "demand_avg": round(demand_total / in_demand) if in_demand else 0,
        },
        "vocab_size": len(vocab),
        "total_jobs": total_jobs,
    }


class CrRecommendReq(BaseModel):
    course_skills: list[str]
    course_title: str = ""        # used by the field classifier (auto scope)
    auto_field: bool = True       # qwen3 detects the field from the course content
    category: str = "it"          # LinkedIn job category (used for "whole category" scope)
    field: str | None = None      # explicit field override (skips classification)
    threshold: float = 0.70       # cosine match threshold (Qwen3-Embedding)
    top_n: int = 2000             # market vocab depth
    top_k: int = 50               # # core field skills the completeness % is over
    verify: bool = True           # Stage-2b LLM verifier (off = raw-cosine baseline)


@app.post("/api/curricula/cr-recommend")
def curricula_cr_recommend(req: CrRecommendReq):
    """Proposed Curriculum-Recommendation pipeline (Steps 1→5), DYNAMIC scope.

    When auto_field is set (and no explicit `field`), qwen3 classifies the course
    by its CONTENT → {field, is_foundational, role_patterns}:
      • foundational (e.g. automata, calculus) → returns "not market-facing",
        no scoring (the course-type gate).
      • applied → builds/loads that field's demand (from role_patterns) and scores
        field completeness (the fair metric) → MAINTAIN / REVISE / UPDATE.
    With auto_field off, scores against the whole `category`. Reuses already-
    extracted course skills (skips re-parsing).
    """
    if not req.course_skills:
        raise HTTPException(400, "no course skills provided")

    classification = None
    field = req.field
    role_patterns = None
    if req.auto_field and not field:
        try:
            classification = _cr_classify_field(req.course_title, req.course_skills)
        except Exception as e:
            raise HTTPException(502, f"field classification failed: {e}")
        if classification.get("is_foundational"):
            return {
                "is_foundational": True,
                "recommendation": "FOUNDATIONAL",
                "field": classification.get("field"),
                "field_name": classification.get("field_name"),
                "reason": "Foundational/theoretical course — no direct labour-market "
                          "role, so market-demand coverage does not apply.",
                "classification": classification,
                "metadata": {"n_course": len(req.course_skills)},
            }
        field = classification.get("field")
        role_patterns = classification.get("role_patterns")

    try:
        result = _cr_recommend_from_skills(
            req.course_skills, category=req.category, field=field,
            role_patterns=role_patterns,
            top_n=req.top_n, top_k=req.top_k, threshold=req.threshold,
            verify=req.verify,
            field_name=(classification or {}).get("field_name", "") if classification else "",
            course_title=req.course_title,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["is_foundational"] = False
    if classification:
        result["classification"] = classification
        result["field_name"] = classification.get("field_name")
    return result


class CrRecommendV2Req(BaseModel):
    course_skills: list[str]
    course_title: str = ""
    description: str = ""
    markdown: str = ""            # full syllabus markdown — used for tool inference only
    course_type: str | None = None  # UI override: foundational | applied | professional
    field: str | None = None      # explicit field override (skips field detection)
    threshold: float = 0.70       # auto-covered cutoff (Qwen3-Embedding)
    band_low: float = 0.55        # gray-zone floor (band verifier adjudicates ≥ this)
    top_n: int = 2000             # market vocab depth
    top_k: int = 25               # core size before the tier filter
    curate: bool = True           # LLM-curate the suggested additions


@app.post("/api/curricula/cr-recommend-v2")
def curricula_cr_recommend_v2(req: CrRecommendV2Req):
    """Method v2 (propose_pipeline_v2.md): 3-way course gate → tier-filtered,
    importance-weighted completeness + relevance → two-axis label
    (MAINTAIN / REVISE / FOCUSED / UPDATE / FOUNDATIONAL) → curated additions
    + next-course topics. Reuses V1's extraction (course_skills + markdown come
    from the existing /api/curricula/markdown + /extract-skills endpoints)."""
    if not req.course_skills:
        raise HTTPException(400, "no course skills provided")
    if req.course_type and req.course_type.lower() not in (
            "foundational", "applied", "professional"):
        raise HTTPException(400, "course_type must be foundational|applied|professional")
    try:
        return _cr_recommend_v2(
            req.course_skills, course_title=req.course_title,
            description=req.description, markdown=req.markdown,
            course_type=req.course_type, field=req.field,
            top_n=req.top_n, top_k=req.top_k, threshold=req.threshold,
            band_low=req.band_low, curate=req.curate,
        )
    except FileNotFoundError as e:
        raise HTTPException(502, f"field inventory unavailable: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))


class CrCurateReq(BaseModel):
    gaps: list[str]               # market-mined "skills to add" (demand-sorted)
    field_name: str = ""          # detected field (e.g. "Artificial Intelligence")
    course_title: str = ""
    taught_skills: list[str] = []  # what the course already teaches (level/next-step signal)
    max_keep: int = 12


@app.post("/api/curricula/cr-curate-suggestions")
def curricula_cr_curate(req: CrCurateReq):
    """qwen3 selects the best NEXT additions for THIS course from the market-mined
    gaps — scored on field fit, LEVEL fit (intro vs applied), and how well they build
    on what's already taught. Returns a subset of the input (verbatim) — never invents."""
    if not req.gaps:
        return {"suggested": []}
    try:
        kept = _cr_curate_suggestions(
            req.gaps, field_name=req.field_name,
            course_title=req.course_title, taught_skills=req.taught_skills,
            max_keep=req.max_keep,
        )
    except Exception as e:
        raise HTTPException(502, f"curation failed: {e}")
    return {"suggested": kept}


@app.get("/api/curricula/skills")
def curricula_skills(top_n: int = Query(20),
                     normalize: str = Query("esco"),
                     categories: str | None = Query(None),
                     source: str = Query("cambodia"),
                     weighting: str = Query("raw")):
    """Return top N in-demand skills from the job corpus.

    source="cambodia" (default): aggregate from Phase-3 stage2 outputs.
    source="linkedin": read the pre-aggregated LinkedIn top-N inventory.
    normalize="esco": collapse variants into ESCO preferred labels (JobBERT).
    categories: comma-separated job categories — domain-conditioned filtering.
    """
    from collections import Counter

    catlist = [c.strip().lower() for c in categories.split(",") if c.strip()] if categories else None

    if source == "linkedin":
        freq, total_jobs = _linkedin_skill_freq(catlist, weighting=weighting)
        skill_counter = Counter(freq)
    else:
        if not _stage2_output_dir.exists():
            return {"skills": [], "total_jobs": 0, "unique_skills": 0,
                    "normalize": normalize}
        catset = set(catlist) if catlist else None
        skill_counter = Counter()
        total_jobs = 0
        for item in _iter_stage2_records():
            if not _record_in_categories(item, catset):
                continue
            total_jobs += 1
            for s in item.get("output", {}).get("skills", []):
                # lowercase for consistency with the analyze path
                skill_counter[s.strip().lower()] += 1

    raw_unique = len(skill_counter)
    freq_dict = dict(skill_counter)
    mapping = []
    mapped_count = None

    if normalize == "esco":
        norm = _esco_normalize_freq(freq_dict)
        ranked = list(norm["esco_freq"].items())[:top_n]
        unique_skills = norm["esco_unique"]
        mapped_count = norm["mapped_count"]
        # raw→esco rows for the skills that survive into the top-N
        top_labels = {s for s, _ in ranked}
        mapping = [m for m in norm["mapping"] if m["esco_skill"] in top_labels]
    else:
        ranked = skill_counter.most_common(top_n)
        unique_skills = raw_unique

    skills = [
        {
            "skill": s,
            "frequency": c,
            "pct": round(100 * c / total_jobs, 1) if total_jobs > 0 else 0,
            "demand": "High" if c >= 50 else "Medium" if c >= 20 else "Low",
        }
        for s, c in ranked
    ]

    return {
        "skills": skills,
        "total_jobs": total_jobs,
        "unique_skills": unique_skills,
        "raw_unique": raw_unique,
        "normalize": normalize,
        "mapped_count": mapped_count,
        "mapping": mapping,
    }


class CurriculaAnalyzeReq(BaseModel):
    """Request body for /api/curricula/analyze.

    Skill source priority (first non-empty wins):
      1. `job_freq` (explicit dict from caller)
      2. `use_dummy=True` -> archive/phase5_legacy/dummy_job_skills.json
      3. otherwise, aggregate from Phase-3 stage2 outputs (top_n)
    """
    course: dict
    job_freq: dict[str, int] | None = None
    use_dummy: bool = False
    top_n: int = 30
    match_threshold: float = 0.60
    k_gaps: int = 5
    esco_normalize: bool = True
    method: str = "semantic"   # "semantic" (SBERT+ESCO) | "lexical" (Ahadi exact-match)
    categories: list[str] | None = None   # domain filter: only these job categories
    source: str = "cambodia"   # "cambodia" (stage2) | "linkedin" (pre-aggregated inventory)
    weighting: str = "raw"     # "raw" frequency | "tfidf" demand-distinctiveness
    clo_esco: bool = False     # symmetric ESCO: ESCO-normalize the course side too
    course_skills: list[str] | None = None   # extracted CLO skills (from /upload)


def _load_dummy_skills() -> dict[str, int]:
    """Load curated test skills from dummy_job_skills.json."""
    import json as _json

    path = _algo_dir / "dummy_job_skills.json"
    if not path.exists():
        raise HTTPException(404, f"Dummy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    skills = data.get("skills", {})
    if not skills:
        raise HTTPException(500, "dummy_job_skills.json has no 'skills' field")
    return {k: int(v) for k, v in skills.items()}


def _aggregate_phase3_skills(top_n: int,
                             categories: list[str] | None = None) -> dict[str, int]:
    """Real path — count skill frequencies across all Phase-3 outputs.

    If `categories` is given, only postings whose job_category intersects it
    are counted (domain-conditioned matching).
    """
    from collections import Counter

    if not _stage2_output_dir.exists():
        raise HTTPException(404, "No Phase 3 outputs found")

    catset = {c.strip().lower() for c in categories if c and c.strip()} if categories else None
    counter: Counter = Counter()
    for item in _iter_stage2_records():
        if not _record_in_categories(item, catset):
            continue
        for s in item.get("output", {}).get("skills", []):
            counter[s.strip().lower()] += 1
    return dict(counter.most_common(top_n))


@app.post("/api/curricula/analyze")
def curricula_analyze(req: CurriculaAnalyzeReq):
    """Run Skills-Taught-vs-Skills-Sought matching → maintain/revise/update.

    Body fields:
      course:           parsed course (from /api/curricula/upload)
      job_freq:         {skill: frequency} OR omit (see priority below)
      use_dummy:        true  -> load curated dummy_job_skills.json
                        false -> aggregate from Phase-3 outputs (default)
      top_n:            cutoff if aggregating from Phase 3 (default 30)
      match_threshold:  cosine threshold per skill (default 0.60)
      k_gaps:           number of top-frequency gaps to report (default 5)

    Returns the full recommendation report.
    """
    if req.job_freq:
        job_freq = req.job_freq
        source = "explicit"
    elif req.use_dummy:
        job_freq = _load_dummy_skills()
        source = "dummy"
    elif req.source == "linkedin":
        from collections import Counter as _C
        freq, _ = _linkedin_skill_freq(req.categories, weighting=req.weighting)
        job_freq = dict(_C(freq).most_common(req.top_n))
        source = "linkedin" + ("+tfidf" if req.weighting == "tfidf" else "")
    else:
        job_freq = _aggregate_phase3_skills(req.top_n, categories=req.categories)
        source = "phase3"

    if not job_freq:
        raise HTTPException(400, "No job-skill frequencies available")

    # ESCO normalization applies to the semantic method only; the lexical
    # baseline matches raw skill strings (no ESCO) by definition.
    if req.method == "semantic" and req.esco_normalize:
        job_freq = _esco_normalize_freq(job_freq)["esco_freq"]
        source = f"{source}+esco"

    course_units = _course_units_for(req) if req.method == "semantic" else None
    if course_units:
        source = f"{source}+cloesco"

    try:
        if req.method == "lexical":
            report = _lexical_analyze_course(
                course_data=req.course,
                job_freq=job_freq,
                k_gaps=req.k_gaps,
            )
        else:
            report = _analyze_course(
                course_data=req.course,
                job_freq=job_freq,
                match_threshold=req.match_threshold,
                k_gaps=req.k_gaps,
                course_units=course_units,
            )
    except Exception as e:
        raise HTTPException(500, f"Matcher failed: {e}")

    md = report.setdefault("metadata", {})
    md["skill_source"] = f"{source}+lexical" if req.method == "lexical" else source
    md["method"] = req.method
    return report


class CurriculumAnalyzeReq(BaseModel):
    """Body for /api/curricula/analyze-curriculum.

    Treats the uploaded `course`(s) as the programme and aggregates
    per-course matching into a single programme-level label
    (union of matched skills + frequency-weighted coverage).
    """
    course: dict
    job_freq: dict[str, int] | None = None
    use_dummy: bool = True
    top_n: int = 30
    match_threshold: float = 0.50
    k_gaps: int = 5
    esco_normalize: bool = True
    method: str = "semantic"   # "semantic" (SBERT+ESCO) | "lexical" (Ahadi exact-match)
    categories: list[str] | None = None   # domain filter: only these job categories
    source: str = "cambodia"   # "cambodia" (stage2) | "linkedin" (pre-aggregated inventory)
    weighting: str = "raw"     # "raw" frequency | "tfidf" demand-distinctiveness
    clo_esco: bool = False     # symmetric ESCO: ESCO-normalize the course side too
    course_skills: list[str] | None = None   # extracted CLO skills (from /upload)


@app.post("/api/curricula/analyze-curriculum")
def curricula_analyze_curriculum(req: CurriculumAnalyzeReq):
    """Run programme-level (curriculum) recommendation.

    Aggregates the uploaded course(s) (from /api/curricula/upload) into a
    programme-level MAINTAIN / REVISE / UPDATE label plus union-of-matched
    skills and cross-course gaps. Only real uploads are included.
    """
    # Resolve skill inventory (same logic as analyze)
    if req.job_freq:
        job_freq = req.job_freq
        source = "explicit"
    elif req.use_dummy:
        job_freq = _load_dummy_skills()
        source = "dummy"
    elif req.source == "linkedin":
        from collections import Counter as _C
        freq, _ = _linkedin_skill_freq(req.categories, weighting=req.weighting)
        job_freq = dict(_C(freq).most_common(req.top_n))
        source = "linkedin" + ("+tfidf" if req.weighting == "tfidf" else "")
    else:
        job_freq = _aggregate_phase3_skills(req.top_n, categories=req.categories)
        source = "phase3"

    if not job_freq:
        raise HTTPException(400, "No job-skill frequencies available")

    # ESCO applies to the semantic method only; lexical matches raw strings.
    if req.method == "semantic" and req.esco_normalize:
        job_freq = _esco_normalize_freq(job_freq)["esco_freq"]
        source = f"{source}+esco"
    elif req.method == "lexical":
        source = f"{source}+lexical"

    # Build the course list: only the uploaded course(s).
    # (Mock CS padding removed — programme view reflects real uploads only.)
    uploaded = req.course
    if not uploaded:
        raise HTTPException(400, "No course supplied. Upload a syllabus first.")
    all_courses = [uploaded]

    # Per-course analysis — lexical or semantic analyze_one; the programme
    # aggregation step (aggregate_curriculum) is matcher-agnostic and shared.
    _analyze_one = (_lexical_analyze_one if req.method == "lexical"
                    else _curriculum_analyze_one)
    course_units = _course_units_for(req) if req.method == "semantic" else None
    if course_units:
        source = f"{source}+cloesco"
    per_course = []
    try:
        for c in all_courses:
            if req.method == "lexical":
                per_course.append(_analyze_one(c, job_freq, k_gaps=req.k_gaps))
            else:
                per_course.append(_analyze_one(
                    c, job_freq, match_threshold=req.match_threshold,
                    k_gaps=req.k_gaps, course_units=course_units,
                ))
    except Exception as e:
        raise HTTPException(500, f"Per-course matching failed: {e}")

    # Aggregate to programme level
    try:
        report = _curriculum_aggregate(per_course, job_freq, k_gaps=req.k_gaps)
    except Exception as e:
        raise HTTPException(500, f"Curriculum aggregation failed: {e}")

    # Tag the source
    report.setdefault("metadata", {})["skill_source"] = source
    report.setdefault("metadata", {})["match_threshold"] = req.match_threshold
    report["metadata"]["method"] = req.method

    return report


# ---------------------------------------------------------------------------
# V8 — Phase 6 segment-matched gap analysis (backend/phase5_curriculum_alignment)
# ---------------------------------------------------------------------------
# The phase6 pipeline writes its results as static JS files consumed by its
# own standalone index.html (`window.X = <json>;`). We parse those same files
# here so the Next.js V8 tab always shows the latest `gen_report_data.py` run
# without a separate export step.

_PHASE6_DIR = pathlib.Path(__file__).resolve().parent.parent / "phase5_curriculum_alignment"
_PHASE6_SOURCES = ("linkedin", "cambodia")   # pipeline.JOB_SOURCES keys


def _phase6_live_report(src: str) -> pathlib.Path:
    """Live-run output for one corpus (gitignored) so the committed
    report_data*.js baselines for the standalone index.html never churn.

    Per-source: a cambodia run must not overwrite the linkedin live file,
    otherwise switching the corpus toggle shows the wrong corpus's run.
    """
    suffix = "" if src == "linkedin" else f".{src}"
    return _PHASE6_DIR / "out" / f"report_data{suffix}.js"


_PHASE6_LIVE_REPORT = _phase6_live_report("linkedin")


def _parse_window_assignments(path: pathlib.Path) -> dict:
    """Parse a `window.NAME = <json>;` data file into {NAME: value}."""
    import json as _json
    import re as _re
    text = path.read_text()
    parts = _re.split(r"^window\.(\w+)\s*=\s*", text, flags=_re.M)
    out = {}
    for name, raw in zip(parts[1::2], parts[2::2]):
        out[name] = _json.loads(raw.strip().rstrip(";"))
    return out


@app.get("/api/v8/report")
def v8_report(which: str = "latest", source: str = "linkedin"):
    """Phase-6 gap-analysis report + dataset previews for the V8 tab.

    source = "linkedin" (default, the committed US validation baseline) or
    "cambodia" (the Phase-3 Cambodian corpus). Non-default sources read
    report_data.<source>.js / preview_data.<source>.js, generated by
    `JOB_SOURCE=<source> python3 04_report/gen_report_data.py`.

    which = "baseline" (committed report_data.js — the Demo mode view),
    "live" (out/report_data.js from the last /api/v8/run, falls back to
    baseline if no run happened yet), or "latest" (newest of the two).
    live runs only apply to the default source. Returns {report, generated,
    preview: {...}}.
    """
    src = (source or "linkedin").lower()
    suffix = "" if src == "linkedin" else f".{src}"
    baseline = _PHASE6_DIR / "dashboard" / f"report_data{suffix}.js"
    preview_path = _PHASE6_DIR / "dashboard" / f"preview_data{suffix}.js"
    # every corpus gets its own live path, so a run made while the tab is
    # on cambodia is visible on cambodia (it used to be linkedin-only,
    # which silently showed the stale committed baseline instead)
    pairs = [(baseline, "baseline"), (_phase6_live_report(src), "live")]
    candidates = [(p, kind) for p, kind in pairs if p.exists()]
    if which == "baseline":
        candidates = [c for c in candidates if c[1] == "baseline"]
    elif which == "live":
        live = [c for c in candidates if c[1] == "live"]
        candidates = live or candidates  # no run yet -> baseline fallback
    if not candidates:
        raise HTTPException(
            404, f"phase6 report for source '{src}' not found — run "
                 f"`JOB_SOURCE={src} python3 04_report/gen_report_data.py` in "
                 f"backend/phase5_curriculum_alignment" if src != "linkedin" else
                 "phase6 report_data.js not found — run "
                 "`python3 04_report/gen_report_data.py` in backend/phase5_curriculum_alignment")
    report_path, generated = max(candidates, key=lambda c: c[0].stat().st_mtime)
    try:
        report = _parse_window_assignments(report_path)["GAP_REPORT"]
    except Exception as e:
        raise HTTPException(500, f"failed to parse {report_path.name}: {e}")
    preview = {"jobs": [], "job_total": 0, "courses": []}
    if preview_path.exists():
        try:
            pv = _parse_window_assignments(preview_path)
            preview = {
                "jobs": pv.get("JOB_PREVIEW", []),
                "job_total": pv.get("JOB_TOTAL", 0),
                "courses": pv.get("CURRICULUM_PREVIEW", []),
            }
        except Exception as e:
            raise HTTPException(500, f"failed to parse preview_data.js: {e}")
    return {"report": report, "generated": generated, "preview": preview}


# Live pipeline run for the V8 tab's "Real experiment" mode. One run at a
# time; stdout is buffered here and polled by the frontend terminal view.
_v8_run_state = {
    "running": False,
    "log": [],
    "returncode": None,
    "started_at": None,
    "finished_at": None,
}
_v8_run_lock = threading.Lock()


class V8RunReq(BaseModel):
    source: str = "files"          # "files" = curated clo_skills/ | "db" = saved courses
    course_ids: list[int] = []     # required when source == "db"
    # which JOB corpus to run against — distinct from `source` above,
    # which selects the CURRICULUM side. Must match the corpus the tab is
    # displaying or the run lands in a file the dashboard never reads.
    job_source: str = "linkedin"


def _v8_stage_db_courses(course_ids: list[int]) -> tuple[str, str | None, list]:
    """Export selected curriculum_course_skills rows into a staging dir in the
    exact file shape gen_report_data.py reads ([{skill, source}], filename
    prefix stripped by course_name()).

    Returns (staging_dir, cohort, manifest) where cohort is the study-year
    label of the
    selected courses ("Y3", or "Y3+Y4" when the selection spans years),
    derived from each course's own code/filename via the same infer_year used
    at parse time — so the recorded cohort always matches what actually ran,
    regardless of which Year filter the UI had active. None if no year could
    be inferred for any selected course.

    `manifest` carries the course codes/titles/years that only the DB knows —
    the staged files are bare [{skill, source}] lists, and CLO_SKILLS_DIR is
    globbed for *.json by the pipeline, so the manifest CANNOT be written
    there. It is archived into the run directory once the run_id exists."""
    import json as _json
    import re as _re
    staging = _PHASE6_DIR / "02_input_curriculum" / "clo_skills_db"
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob("*.json"):
        old.unlink()
    years: set[int] = set()
    manifest: list[dict] = []
    for i, cid in enumerate(course_ids, 1):
        row = db_get_course_skills(cid)
        if not row:
            raise HTTPException(404, f"saved course {cid} not found")
        slug = _re.sub(r"[^a-z0-9]+", "_", row["course_title"].lower()).strip("_")
        items = [{"skill": s.get("name") or s.get("skill"),
                  "source": s.get("source", "")}
                 for s in (row["skills"] or []) if s.get("name") or s.get("skill")]
        if not items:
            raise HTTPException(400, f"saved course {cid} has no skills")
        yr = _p6_infer_year(row.get("course_code"), row.get("filename"))
        if yr:
            years.add(yr)
        fname = f"{i}_{slug}.json"
        (staging / fname).write_text(_json.dumps(items, indent=2))
        # keyed on `file` by the exporter — never on position, because
        # `10_x.json` sorts before `2_x.json` once a cohort has 10+ courses
        manifest.append({"file": fname, "course_id": cid,
                         "course_code": row.get("course_code") or "",
                         "course_title": row.get("course_title") or "",
                         "year": f"Y{yr}" if yr else "",
                         "n_skills": len(items)})
    cohort = "+".join(f"Y{y}" for y in sorted(years)) if years else None
    return str(staging), cohort, manifest


@app.post("/api/v8/run")
def v8_run(req: V8RunReq | None = None):
    """Execute backend/phase5_curriculum_alignment/04_report/gen_report_data.py in the background.

    Recomputes the gap analysis and writes it to the gitignored live path
    (out/report_data.js) — the committed report_data.js baseline is never
    touched by API runs. Poll /api/v8/run/status for progress, then refetch
    /api/v8/report (which serves the newest of baseline/live).
    Uses the API server's own interpreter (same env that already loads
    sentence-transformers + PyMuPDF).

    Body (optional): {source: "files"|"db", course_ids: [...]} — "db" stages
    the selected saved courses from PostgreSQL and points the pipeline at
    them via CLO_SKILLS_DIR; default stays the curated clo_skills/ files."""
    import os
    import subprocess
    req = req or V8RunReq()
    job_src = (req.job_source or "linkedin").lower()
    if job_src not in _PHASE6_SOURCES:
        raise HTTPException(400, f"unknown job_source {job_src!r}; "
                                 f"choose one of {list(_PHASE6_SOURCES)}")
    env = dict(os.environ)
    env["JOB_SOURCE"] = job_src
    env["REPORT_OUT"] = str(_phase6_live_report(job_src))
    # non-default corpora also regenerate a preview; redirect it into out/
    # so the committed preview_data.<source>.js is never rewritten either
    env["PREVIEW_OUT"] = str(_phase6_live_report(job_src)
                             .with_name(f"preview_data.{job_src}.js"))
    cohort, manifest = None, []
    if req.source == "db":
        if not req.course_ids:
            raise HTTPException(400, "course_ids required when source='db'")
        env["CLO_SKILLS_DIR"], cohort, manifest = \
            _v8_stage_db_courses(req.course_ids)
        # the run store records the cohort in report.json / inputs.json /
        # report.md / the ledger, derived from the selected courses themselves
        if cohort:
            env["RUN_COHORT"] = cohort
    # names the run in outputs/index.md so a UI-triggered run is
    # distinguishable from a `python3 04_report/gen_report_data.py` one
    env["RUN_LABEL"] = (f"V8 tab · {job_src} · "
                        + (f"{len(req.course_ids)} saved courses"
                           + (f" · {cohort}" if cohort else "")
                           if req.source == "db" else "curated files"))

    with _v8_run_lock:
        if _v8_run_state["running"]:
            raise HTTPException(409, "phase6 pipeline run already in progress")
        _v8_run_state.update(
            running=True, log=[], returncode=None, run_id=None, exported=False,
            started_at=datetime.now().isoformat(), finished_at=None)
        if req.source == "db":
            _v8_run_state["log"].append(
                f"curriculum source: PostgreSQL ({len(req.course_ids)} saved "
                f"courses) -> {env['CLO_SKILLS_DIR']}")

    def worker():
        import json as _json
        import re as _re
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "04_report/gen_report_data.py"],
                cwd=str(_PHASE6_DIR), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in proc.stdout:
                line = line.rstrip()
                _v8_run_state["log"].append(line)
                # run_store prints "run archived -> outputs/<run_id>/ (...)"
                m = _re.search(r"run archived -> outputs/([^/\s]+)/", line)
                if m:
                    _v8_run_state["run_id"] = m.group(1)
            proc.wait()
            _v8_run_state["returncode"] = proc.returncode
            rid = _v8_run_state.get("run_id")
            if proc.returncode == 0 and rid and manifest:
                # the DB-only facts (course codes, titles, years) the pipeline
                # never sees, parked beside the run for export_run.py
                (_PHASE6_OUTPUTS / rid / "db_courses.json").write_text(
                    _json.dumps(manifest, indent=2))
        except Exception as e:
            _v8_run_state["log"].append(f"ERROR: {e}")
            _v8_run_state["returncode"] = -1
        finally:
            _v8_run_state["finished_at"] = datetime.now().isoformat()
            _v8_run_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


@app.get("/api/v8/run/status")
def v8_run_status():
    return _v8_run_state


@app.post("/api/v8/runs/{run_id}/export")
def v8_run_export(run_id: str):
    """Write the thesis bundle for an archived run (the Save button).

    Pure re-shaping of outputs/<run_id>/{params,report,inputs}.json into
    outputs/<run_id>/thesis/ — nothing is recomputed, so re-saving is
    idempotent and always agrees with report.json. Also rebuilds
    outputs/thesis_registry.csv across every exported run.
    See phase5_curriculum_alignment/04_report/THESIS_EXPORT_FORMAT.md."""
    import json as _json
    import subprocess
    run_dir = _PHASE6_OUTPUTS / run_id
    if not (run_dir / "report.json").is_file():
        raise HTTPException(404, f"no archived run {run_id!r}")
    proc = subprocess.run(
        [sys.executable, "04_report/export_run.py", run_id],
        cwd=str(_PHASE6_DIR), capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise HTTPException(
            500, f"export failed: {(proc.stderr or proc.stdout)[-500:]}")
    bundle = run_dir / "thesis"
    summary = _json.loads((bundle / "run_summary.json").read_text())
    if _v8_run_state.get("run_id") == run_id:
        _v8_run_state["exported"] = True
    return {
        "run_id": run_id,
        "cohort": summary.get("cohort"),
        "dir": f"outputs/{run_id}/thesis",
        "files": sorted(p.name for p in bundle.iterdir() if p.is_file()),
        "measured": summary.get("measured_results", {}),
        "registry": "outputs/thesis_registry.csv",
    }


# ---------------------------------------------------------------------------
# Versioned run store — outputs/<run_id>/ written by run_store.py
# ---------------------------------------------------------------------------
# Read-only here. gen_report_data.py owns writing; the filesystem is the
# source of truth so a run archived by a bare `04_report/gen_report_data.py`
# (no API, no DB) shows up in the UI exactly the same.

_PHASE6_OUTPUTS = _PHASE6_DIR / "outputs"


def _v8_run_meta(run_dir: pathlib.Path) -> dict | None:
    """params.json + the headline metrics, without the full report blob."""
    import json as _json
    try:
        meta = _json.loads((run_dir / "params.json").read_text())
        report = _json.loads((run_dir / "report.json").read_text())
    except (OSError, ValueError):
        return None
    return {
        **{k: v for k, v in meta.items() if k != "params"},
        "params": meta.get("params", {}),
        "kl": report.get("kl"),
        "covered": report.get("covered"),
        "top_n": report.get("top_n"),
        "n_jobs": report.get("n_jobs"),
        "n_courses": len(report.get("courses", [])),
        "n_clo_skills": report.get("n_clo_skills"),
        "courses": report.get("courses", []),
        "has_md": (run_dir / "report.md").is_file(),
    }


@app.get("/api/v8/runs")
def v8_runs(limit: int = 50):
    """Every archived run, newest first — the history behind Step 3."""
    if not _PHASE6_OUTPUTS.is_dir():
        return {"runs": [], "outputs_dir": str(_PHASE6_OUTPUTS)}
    runs = []
    for d in sorted((p for p in _PHASE6_OUTPUTS.iterdir() if p.is_dir()),
                    reverse=True)[:limit]:
        meta = _v8_run_meta(d)
        if meta:
            runs.append(meta)
    return {"runs": runs, "outputs_dir": str(_PHASE6_OUTPUTS)}


@app.get("/api/v8/runs/{run_id}")
def v8_run_detail(run_id: str, fmt: str = "json"):
    """One archived run. fmt=json -> full report; fmt=md -> report.md text."""
    import json as _json
    import re as _re
    # run_id goes into a path: keep it to the shape run_store mints
    if not _re.fullmatch(r"[0-9A-Za-z_-]+", run_id):
        raise HTTPException(400, "invalid run_id")
    run_dir = _PHASE6_OUTPUTS / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, f"run {run_id} not found")
    if fmt == "md":
        md = run_dir / "report.md"
        if not md.is_file():
            raise HTTPException(404, "report.md not found for this run")
        return PlainTextResponse(md.read_text())
    meta = _v8_run_meta(run_dir)
    if not meta:
        raise HTTPException(500, f"run {run_id} is missing params/report json")
    try:
        report = _json.loads((run_dir / "report.json").read_text())
        inputs = _json.loads((run_dir / "inputs.json").read_text())
    except (OSError, ValueError) as e:
        raise HTTPException(500, f"failed to read run {run_id}: {e}")
    return {**meta, "report": report, "inputs": inputs}
