"use client";

import { useState, useRef, useEffect } from "react";
import { API_BASE, ConfidenceBadge, Terminal, countryFlag } from "./ui";
import CrawlResults from "./CrawlResults";

function formatRunId(runId) {
  const parts = runId.split("__");
  const ts = parts[parts.length - 1];
  const m1 = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (m1) { const [, y, mo, d, h, mi] = m1; return `${y}-${mo}-${d} ${h}:${mi}`; }
  const m2 = ts.match(/^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$/);
  if (m2) { const [, y, mo, d, h, mi] = m2; return `${y}-${mo}-${d} ${h}:${mi}`; }
  return runId;
}

const SUB_VIEWS = [
  { key: "launch", label: "Launch Pipeline" },
  { key: "results", label: "Browse Results" },
];

export default function PipelineTab({ countries, countriesLoading, fetchCountries, onViewAnalytics }) {
  const [view, setView] = useState("launch");
  const [selectedCountry, setSelectedCountry] = useState("");
  const [countrySites, setCountrySites] = useState([]);
  const [countrySitesLoading, setCountrySitesLoading] = useState(false);
  const [selectedSites, setSelectedSites] = useState(new Set());
  const [crawling, setCrawling] = useState(false);
  const [crawlLogs, setCrawlLogs] = useState([]);
  const [crawlStatus, setCrawlStatus] = useState(null);
  const [crawlError, setCrawlError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [showResultsModal, setShowResultsModal] = useState(false);
  const [config, setConfig] = useState({ maxPages: 2000, maxDepth: 5, crawlDelay: 1 });
  const crawlLogRef = useRef(null);
  const crawlLogCountRef = useRef(0);

  // Extraction state
  const [extracting, setExtracting] = useState(false);
  const [extractStatus, setExtractStatus] = useState(null);
  const [extractLogs, setExtractLogs] = useState([]);
  const extractLogRef = useRef(null);
  const extractLogCountRef = useRef(0);

  useEffect(() => {
    if (crawlLogRef.current) {
      crawlLogRef.current.scrollTop = crawlLogRef.current.scrollHeight;
    }
  }, [crawlLogs]);

  useEffect(() => {
    if (extractLogRef.current) {
      extractLogRef.current.scrollTop = extractLogRef.current.scrollHeight;
    }
  }, [extractLogs]);

  useEffect(() => {
    fetchCountries();
  }, []);

  // --- Data fetching ---

  async function fetchCountrySites(country) {
    setCountrySitesLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sites?country=${encodeURIComponent(country)}`);
      const data = await res.json();
      setCountrySites(data.sites || []);
    } catch {
      setCountrySites([]);
    } finally {
      setCountrySitesLoading(false);
    }
  }

  function handleCountrySelect(country) {
    setSelectedCountry(country);
    setSelectedSites(new Set());
    if (country) fetchCountrySites(country);
    else setCountrySites([]);
  }

  function toggleSite(url) {
    setSelectedSites((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
  }

  function toggleAllSites() {
    if (selectedSites.size === countrySites.length) {
      setSelectedSites(new Set());
    } else {
      setSelectedSites(new Set(countrySites.map((s) => s.url)));
    }
  }

  // --- Crawl actions ---

  async function handleCrawl() {
    const urls = Array.from(selectedSites);
    if (urls.length === 0) return;

    setShowModal(false);
    setCrawling(true);
    setCrawlError(null);
    setCrawlLogs([]);
    setCrawlStatus(null);
    setExtractStatus(null);
    setExtractLogs([]);
    crawlLogCountRef.current = 0;
    try {
      const res = await fetch(`${API_BASE}/api/crawl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          urls,
          max_pages: config.maxPages,
          max_depth: config.maxDepth,
          crawl_delay: config.crawlDelay,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        pollCrawlStatus();
      } else {
        setCrawlError(data.detail || "Crawl failed to start");
        setCrawling(false);
      }
    } catch {
      setCrawlError("Could not connect to the API. Is the backend running on port 8000?");
      setCrawling(false);
    }
  }

  async function handleStop() {
    try {
      const res = await fetch(`${API_BASE}/api/crawl/stop`, { method: "POST" });
      if (res.ok) {
        setCrawlLogs((prev) => [...prev, "[STOP] Stop signal sent — waiting for current page to finish..."]);
      }
    } catch {
      setCrawlError("Could not connect to the API.");
    }
  }

  function pollCrawlStatus() {
    const interval = setInterval(async () => {
      try {
        const since = crawlLogCountRef.current;
        const res = await fetch(`${API_BASE}/api/crawl/status?since=${since}`);
        const data = await res.json();

        if (data.logs && data.logs.length > 0) {
          setCrawlLogs((prev) => [...prev, ...data.logs]);
          crawlLogCountRef.current = data.log_count;
        }

        setCrawlStatus(data);

        if (data.status === "completed" || data.status === "stopped") {
          clearInterval(interval);
          setCrawling(false);
          // Auto-extract if job pages found
          const clsSummary = data.classify_summary || {};
          if ((clsSummary.single_job_detail || 0) > 0 && data.started_at) {
            autoExtract(data);
          } else {
            setShowResultsModal(true);
          }
        } else if (data.status === "failed") {
          clearInterval(interval);
          setCrawling(false);
          setCrawlError(`Crawl failed: ${data.error}`);
        }
      } catch {
        clearInterval(interval);
        setCrawling(false);
      }
    }, 800);
  }

  // --- Extraction actions ---

  function deriveRunId(startedAt) {
    const dt = new Date(startedAt);
    return `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}_${String(dt.getHours()).padStart(2, "0")}-${String(dt.getMinutes()).padStart(2, "0")}-${String(dt.getSeconds()).padStart(2, "0")}`;
  }

  async function autoExtract(crawlData) {
    const runId = deriveRunId(crawlData.started_at);
    setExtracting(true);
    setExtractLogs([]);
    setExtractStatus(null);
    extractLogCountRef.current = 0;

    try {
      const res = await fetch(`${API_BASE}/api/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
      });
      if (!res.ok) {
        const err = await res.json();
        setExtractStatus({ status: "failed", error: err.detail });
        setExtracting(false);
        setShowResultsModal(true);
        return;
      }
      pollExtractStatus();
    } catch (e) {
      setExtractStatus({ status: "failed", error: e.message });
      setExtracting(false);
      setShowResultsModal(true);
    }
  }

  async function handleStopExtract() {
    try {
      await fetch(`${API_BASE}/api/extract/stop`, { method: "POST" });
    } catch {}
  }

  function pollExtractStatus() {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/extract/status?since=${extractLogCountRef.current}`);
        const data = await res.json();

        if (data.logs && data.logs.length > 0) {
          setExtractLogs((prev) => [...prev, ...data.logs]);
          extractLogCountRef.current = data.log_count;
        }

        setExtractStatus(data);

        if (data.status !== "running") {
          clearInterval(interval);
          setExtracting(false);
          setShowResultsModal(true);
        }
      } catch {
        clearInterval(interval);
        setExtracting(false);
      }
    }, 800);
  }

  // --- Render ---

  const showDashboard = crawling || extracting || (crawlStatus && ["completed", "stopped", "failed"].includes(crawlStatus.status));

  return (
    <div>
      {/* Header + sub-view toggle */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-gray-900">Pipeline</h1>
          <p className="mt-1 text-sm text-gray-500">
            {view === "launch"
              ? "Crawl sites, classify pages, and extract skills — all in one flow."
              : "Browse saved HTML from previous crawl runs."}
          </p>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {(crawling || extracting) && (
            <button
              onClick={extracting ? handleStopExtract : handleStop}
              className="h-9 px-4 text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg
                         hover:bg-red-100 transition-colors flex items-center gap-2 mr-2"
            >
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              Stop
            </button>
          )}
          {SUB_VIEWS.map((sv) => (
            <button
              key={sv.key}
              onClick={() => setView(sv.key)}
              className={`h-9 px-4 text-sm font-medium rounded-lg transition-colors
                ${view === sv.key
                  ? "text-gray-900 bg-gray-100"
                  : "text-gray-400 hover:text-gray-600 hover:bg-gray-50"
                }`}
            >
              {sv.label}
            </button>
          ))}
        </div>
      </div>

      {/* ======== LAUNCH VIEW ======== */}
      {view === "launch" && (
        <>
          {/* ---- PROGRESS DASHBOARD ---- */}
          {showDashboard && (
            <ProgressDashboard
              crawlStatus={crawlStatus}
              crawling={crawling}
              crawlError={crawlError}
              crawlLogs={crawlLogs}
              crawlLogRef={crawlLogRef}
              extracting={extracting}
              extractStatus={extractStatus}
              extractLogs={extractLogs}
              extractLogRef={extractLogRef}
            />
          )}

          {/* ---- STEP 1: SITE SELECTION ---- */}
          <div className={`${(crawling || extracting) ? "opacity-60 pointer-events-none" : ""} transition-opacity`}>
            <div className="flex items-center gap-3 mb-4">
              <div className="flex items-center justify-center w-6 h-6 rounded-full bg-gray-900 text-white text-xs font-medium">1</div>
              <span className="text-sm font-medium text-gray-900">Select sites to crawl</span>
            </div>

            <div className="pl-9">
              <CountryDropdown
                value={selectedCountry}
                onChange={handleCountrySelect}
                countries={countries}
                loading={countriesLoading}
                disabled={crawling || extracting}
              />
            </div>

            {selectedCountry && (
              <div className="pl-9 mt-6">
                <SiteList
                  sites={countrySites}
                  loading={countrySitesLoading}
                  selectedSites={selectedSites}
                  onToggle={toggleSite}
                  onToggleAll={toggleAllSites}
                />
              </div>
            )}
          </div>

          {/* ---- STEP 2: START PIPELINE ---- */}
          {selectedSites.size > 0 && !crawling && !extracting && (
            <div className="mt-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-gray-900 text-white text-xs font-medium">2</div>
                <span className="text-sm font-medium text-gray-900">Configure & start</span>
              </div>
              <div className="pl-9">
                <button
                  onClick={() => setShowModal(true)}
                  className="w-full h-12 text-sm font-medium text-white bg-gray-900 rounded-xl
                             hover:bg-gray-800 transition-colors flex items-center justify-center gap-3"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 3l14 9-14 9V3z" />
                  </svg>
                  Start Pipeline
                  <span className="text-xs bg-white/20 px-2 py-0.5 rounded-full">
                    {selectedSites.size} site{selectedSites.size !== 1 ? "s" : ""}
                  </span>
                </button>
              </div>
            </div>
          )}

          {/* Config modal */}
          {showModal && (
            <CrawlConfigModal
              config={config}
              setConfig={setConfig}
              selectedCount={selectedSites.size}
              onStart={handleCrawl}
              onClose={() => setShowModal(false)}
            />
          )}
        </>
      )}

      {/* ======== RESULTS VIEW ======== */}
      {view === "results" && <CrawlResults />}

      {/* ======== RESULTS MODAL ======== */}
      {showResultsModal && crawlStatus && (
        <ResultsModal
          status={crawlStatus}
          onClose={() => setShowResultsModal(false)}
          extractStatus={extractStatus}
          onViewAnalytics={onViewAnalytics}
        />
      )}
    </div>
  );
}

// --- Sub-components ---

function ProgressDashboard({ crawlStatus, crawling, crawlError, crawlLogs, crawlLogRef, extracting, extractStatus, extractLogs, extractLogRef }) {
  const s = crawlStatus || {};
  const isClassifying = s.status === "classifying";
  const isRunning = s.status === "running";
  // Job lifecycle is running -> classifying -> completed (or stopped/failed),
  // so "completed" is the only status where both phases actually finished.
  const crawlDone = s.status === "completed";

  // Crawl progress
  const crawlTotal = s.total_sites || 0;
  const crawlDoneCount = (s.success || 0) + (s.blocked || 0) + (s.failed || 0);
  const crawlPct = crawlTotal > 0 ? Math.round(crawlDoneCount / crawlTotal * 100) : 0;

  // Classify progress
  const clsTotal = s.classify_total || 0;
  const clsSummary = s.classify_summary || {};
  const clsDone = (clsSummary.single_job_detail || 0) + (clsSummary.non_job_details || 0) + (clsSummary.failed || 0) + (clsSummary.no_content || 0);
  const clsPct = clsTotal > 0 ? Math.round(clsDone / clsTotal * 100) : 0;

  // Extract progress
  const ext = extractStatus || {};
  const extTotal = ext.total || 0;
  const extPct = extTotal > 0 ? Math.round((ext.progress || 0) / extTotal * 100) : 0;

  // Phase label for terminal
  const phaseLabel = isRunning ? "Crawling"
    : isClassifying ? "Classifying"
    : extracting ? "Extracting"
    : s.status === "stopped" ? "Stopped"
    : s.status === "failed" ? "Failed"
    : "Complete";

  return (
    <div className="mb-8">
      {/* Phase indicator */}
      <div className="flex items-center gap-3 mb-4">
        <PhaseStep label="Crawl" num={1} active={isRunning} done={isClassifying || crawlDone} />
        <div className="w-8 h-px bg-gray-200" />
        <PhaseStep label="Classify" num={2} active={isClassifying} done={crawlDone} />
        <div className="w-8 h-px bg-gray-200" />
        <PhaseStep label="Extract" num={3} active={extracting} done={ext.status === "completed"} />
      </div>

      {/* Crawl progress bar */}
      {crawlTotal > 0 && (
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm font-medium text-gray-900">
              {isRunning ? `Crawling site ${s.current_index || 1} of ${crawlTotal}` : `Crawl: ${crawlDoneCount} of ${crawlTotal} sites`}
            </span>
            <span className="text-xs text-gray-400">{crawlPct}%</span>
          </div>
          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-gray-900 transition-all duration-500" style={{ width: `${Math.max(2, crawlPct)}%` }} />
          </div>
        </div>
      )}

      {/* Classify progress bar */}
      {(isClassifying || crawlDone) && clsTotal > 0 && (
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm font-medium text-gray-900">
              {isClassifying ? `Classifying page ${s.classify_progress || 0} of ${clsTotal}` : `Classify: ${clsDone} of ${clsTotal} pages`}
            </span>
            <span className="text-xs text-gray-400">{clsPct}%</span>
          </div>
          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-violet-600 transition-all duration-500" style={{ width: `${Math.max(2, clsPct)}%` }} />
          </div>
        </div>
      )}

      {/* Extract progress bar */}
      {extracting && extTotal > 0 && (
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm font-medium text-gray-900">
              Extracting page {ext.progress || 0} of {extTotal}
            </span>
            <span className="text-xs text-gray-400">{extPct}%</span>
          </div>
          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-emerald-500 animate-progress-stripe transition-all duration-500" style={{ width: `${Math.max(2, extPct)}%` }} />
          </div>
        </div>
      )}

      {/* Crawl stats */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <StatCard value={s.success || 0} label="Crawl OK" color="emerald" />
        <StatCard value={s.blocked || 0} label="Blocked" color="amber" />
        <StatCard value={s.failed || 0} label="Failed" color="red" />
      </div>

      {/* Classify stats */}
      {(isClassifying || crawlDone) && clsTotal > 0 && (
        <div className="grid grid-cols-4 gap-3 mb-4">
          <StatCard value={clsSummary.single_job_detail || 0} label="Job Pages" color="emerald" />
          <StatCard value={clsSummary.non_job_details || 0} label="Non-Job" color="gray" />
          <StatCard value={clsSummary.no_content || 0} label="No Content" color="amber" />
          <StatCard value={clsSummary.failed || 0} label="Cls Failed" color="red" />
        </div>
      )}

      {/* Extract stats */}
      {ext.status === "completed" && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <StatCard value={ext.extracted || 0} label="Extracted" color="emerald" />
          <StatCard value={ext.failed || 0} label="Ext Failed" color="red" />
          <StatCard value={ext.total || 0} label="Total Pages" color="gray" />
        </div>
      )}

      {/* Currently processing */}
      {(crawling || extracting) && (s.current_site || ext.current_file) && (
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg mb-4 ${
          extracting ? "bg-emerald-50 border border-emerald-200"
          : isClassifying ? "bg-violet-50 border border-violet-200"
          : "bg-blue-50 border border-blue-200"
        }`}>
          <span className={`w-2 h-2 rounded-full animate-pulse shrink-0 ${
            extracting ? "bg-emerald-500" : isClassifying ? "bg-violet-500" : "bg-blue-500"
          }`} />
          <span className={`text-xs truncate ${
            extracting ? "text-emerald-700" : isClassifying ? "text-violet-700" : "text-blue-700"
          }`}>
            {extracting ? ext.current_file : s.current_site}
          </span>
        </div>
      )}

      {crawlError && (
        <div className="px-4 py-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg mb-4">
          {crawlError}
        </div>
      )}

      {/* Terminal — show extract logs if extracting, otherwise crawl logs */}
      {extracting && extractLogs.length > 0 ? (
        <Terminal
          logRef={extractLogRef}
          logs={extractLogs}
          label="extraction"
          isLive={extracting}
          colorize={(line) => {
            if (line.includes("STOP")) return "text-amber-400 font-medium";
            if (line.includes("FAILED")) return "text-red-400";
            if (line.includes("skills")) return "text-emerald-400";
            if (line.includes("Starting") || line.includes("Extraction") || line.includes("===")) return "text-white font-medium";
            if (line.includes("ETA")) return "text-violet-300";
            return "text-gray-300";
          }}
        />
      ) : crawlLogs.length > 0 && (
        <Terminal logRef={crawlLogRef} logs={crawlLogs} label={`${phaseLabel.toLowerCase()} pipeline`} isLive={crawling} />
      )}
    </div>
  );
}

function PhaseStep({ label, num, active, done }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-all
        ${done ? "bg-emerald-500 text-white"
          : active ? "bg-gray-900 text-white ring-2 ring-gray-900/20"
          : "bg-gray-100 text-gray-400"
        }`}
      >
        {done ? (
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        ) : num}
      </div>
      <span className={`text-sm font-medium ${active ? "text-gray-900" : done ? "text-emerald-600" : "text-gray-400"}`}>{label}</span>
    </div>
  );
}

const STAT_COLORS = {
  emerald: "bg-emerald-50 border-emerald-200 text-emerald-700 [&_small]:text-emerald-600",
  amber: "bg-amber-50 border-amber-200 text-amber-700 [&_small]:text-amber-600",
  red: "bg-red-50 border-red-200 text-red-700 [&_small]:text-red-600",
  gray: "bg-gray-50 border-gray-200 text-gray-700 [&_small]:text-gray-600",
};

function StatCard({ value, label, color }) {
  return (
    <div className={`border rounded-xl px-4 py-3 ${STAT_COLORS[color]}`}>
      <p className="text-2xl font-semibold">{value}</p>
      <small className="text-xs mt-0.5 block">{label}</small>
    </div>
  );
}

function ResultsModal({ status, onClose, extractStatus, onViewAnalytics }) {
  const s = status || {};
  const cls = s.classify_summary || {};
  const crawlTotal = s.total_sites || 0;
  const clsDone = (cls.single_job_detail || 0) + (cls.non_job_details || 0) + (cls.failed || 0) + (cls.no_content || 0);
  const jobPct = clsDone > 0 ? Math.round((cls.single_job_detail || 0) / clsDone * 100) : 0;
  const isStopped = s.status === "stopped";

  // Check if extraction already ran
  const ext = extractStatus || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-lg mx-4 p-6 animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              extractDone ? "bg-emerald-100" : isStopped ? "bg-amber-100" : "bg-emerald-100"
            }`}>
              {isStopped && !extractDone ? (
                <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {extractDone ? "Pipeline Complete" : isStopped ? "Crawl Stopped" : "Crawl & Classify Done"}
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">{formatRunId(s.job_id || "")}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Crawl section */}
        <div className="mb-5">
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Crawl</h4>
          <div className="bg-gray-50 rounded-xl px-4 py-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Total Sites</span>
              <span className="text-sm font-semibold text-gray-900">{crawlTotal}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-emerald-600">Succeeded</span>
              <span className="text-sm font-semibold text-emerald-700">{s.success || 0}</span>
            </div>
            {(s.blocked || 0) > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-amber-600">Blocked</span>
                <span className="text-sm font-semibold text-amber-700">{s.blocked}</span>
              </div>
            )}
            {(s.failed || 0) > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-red-600">Failed</span>
                <span className="text-sm font-semibold text-red-700">{s.failed}</span>
              </div>
            )}
          </div>
        </div>

        {/* Classify section */}
        <div className="mb-5">
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Classification</h4>
          {clsDone > 0 ? (
            <div className="bg-gray-50 rounded-xl px-4 py-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Total Pages</span>
                <span className="text-sm font-semibold text-gray-900">{clsDone}</span>
              </div>
              <div className="h-px bg-gray-200 my-1" />
              <div className="flex items-center justify-between">
                <span className="text-sm text-emerald-600">Job Detail</span>
                <span className="text-sm font-semibold text-emerald-700">{cls.single_job_detail || 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Non-Job</span>
                <span className="text-sm font-semibold text-gray-600">{cls.non_job_details || 0}</span>
              </div>
              {(cls.no_content || 0) > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-amber-600">No Content</span>
                  <span className="text-sm font-semibold text-amber-700">{cls.no_content}</span>
                </div>
              )}
              {(cls.failed || 0) > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-red-600">Failed</span>
                  <span className="text-sm font-semibold text-red-700">{cls.failed}</span>
                </div>
              )}
              <div className="pt-1">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden flex">
                  <div className="h-full bg-emerald-500 rounded-l-full" style={{ width: `${jobPct}%` }} />
                </div>
                <p className="text-[11px] text-gray-400 mt-1 text-center">
                  {jobPct}% job detail pages
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 rounded-xl px-4 py-3 text-sm text-gray-400">
              No pages were classified. Check if Ollama is running with model <span className="font-mono text-xs">llama3.1:8b</span>.
            </div>
          )}
        </div>

        {/* Extraction section (if done) */}
        {extractDone && (
          <div className="mb-5 animate-fade-in-up">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Extraction</h4>
            <div className="bg-gray-50 rounded-xl px-4 py-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-emerald-600">Jobs Extracted</span>
                <span className="text-sm font-semibold text-emerald-700">{ext.extracted || 0}</span>
              </div>
              {(ext.failed || 0) > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-red-600">Failed</span>
                  <span className="text-sm font-semibold text-red-700">{ext.failed}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="pt-4 border-t border-gray-100 flex items-center gap-2">
          {/* View Analytics — show after extraction done */}
          {extractDone && onViewAnalytics && (
            <button
              onClick={() => { onClose(); onViewAnalytics(); }}
              className="flex-1 h-10 text-sm font-medium text-white bg-emerald-600 rounded-lg
                         hover:bg-emerald-700 transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
              </svg>
              View Analytics
            </button>
          )}

          <button
            onClick={onClose}
            className={`${extractDone ? "w-24" : "flex-1"} h-10 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-lg
                       hover:bg-gray-50 transition-colors`}
          >
            {extractDone ? "Close" : "Done"}
          </button>
        </div>
      </div>
    </div>
  );
}

function CountryDropdown({ value, onChange, countries, loading, disabled }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={loading || disabled}
      className="w-full h-11 px-4 text-sm bg-white border border-gray-200 rounded-lg
                 text-gray-900 appearance-none
                 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent
                 disabled:opacity-50 transition-shadow
                 bg-[url('data:image/svg+xml;charset=UTF-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%239ca3af%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M6%209l6%206%206-6%22%2F%3E%3C%2Fsvg%3E')]
                 bg-[length:12px] bg-[right_12px_center] bg-no-repeat"
    >
      <option value="">{loading ? "Loading countries..." : "Select a country"}</option>
      {countries.map((c) => (
        <option key={c.country} value={c.country}>
          {countryFlag(c.country)} {c.country.charAt(0).toUpperCase() + c.country.slice(1)} -- {c.site_count} site{c.site_count !== 1 ? "s" : ""}
        </option>
      ))}
    </select>
  );
}

function SiteList({ sites, loading, selectedSites, onToggle, onToggleAll }) {
  const [search, setSearch] = useState("");

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <span className="w-5 h-5 border-2 border-gray-200 border-t-gray-600 rounded-full animate-spin" />
        <span className="ml-3 text-sm text-gray-400">Loading sites...</span>
      </div>
    );
  }

  if (sites.length === 0) {
    return <p className="text-sm text-gray-400 text-center py-12">No sites found for this country.</p>;
  }

  const filtered = search
    ? sites.filter(
        (s) =>
          (s.name || "").toLowerCase().includes(search.toLowerCase()) ||
          s.url.toLowerCase().includes(search.toLowerCase()) ||
          (s.site_type || "").toLowerCase().includes(search.toLowerCase())
      )
    : sites;

  const allFilteredSelected = filtered.length > 0 && filtered.every((s) => selectedSites.has(s.url));

  function handleToggleFiltered() {
    if (allFilteredSelected) {
      onToggleAll();
    } else {
      if (!search) {
        onToggleAll();
      } else {
        filtered.forEach((s) => {
          if (!selectedSites.has(s.url)) onToggle(s.url);
        });
      }
    }
  }

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-100 bg-gray-50/50">
        <button
          onClick={search ? handleToggleFiltered : onToggleAll}
          className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors shrink-0"
        >
          <span
            className={`w-4 h-4 rounded border flex items-center justify-center transition-all
            ${selectedSites.size > 0 ? "bg-gray-900 border-gray-900" : "border-gray-300 bg-white"}`}
          >
            {(search ? allFilteredSelected : selectedSites.size === sites.length) ? (
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            ) : selectedSites.size > 0 ? (
              <span className="w-2 h-0.5 bg-white rounded" />
            ) : null}
          </span>
        </button>

        <div className="relative flex-1">
          <svg
            className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400"
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search sites..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full h-8 pl-8 pr-3 text-xs bg-white border border-gray-200 rounded-md
                       focus:outline-none focus:ring-1 focus:ring-gray-400 focus:border-transparent transition-all"
          />
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {selectedSites.size > 0 && (
            <span className="text-xs font-medium text-white bg-gray-900 px-2 py-0.5 rounded-full">
              {selectedSites.size} selected
            </span>
          )}
          <span className="text-xs text-gray-400">
            {search ? `${filtered.length} of ${sites.length}` : `${sites.length} sites`}
          </span>
        </div>
      </div>

      {/* Table header */}
      <div className="grid grid-cols-[40px_1fr_120px_80px] px-4 py-2 border-b border-gray-100 bg-gray-50/30">
        <span />
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Site</span>
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Type</span>
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider text-right">Confidence</span>
      </div>

      {/* Rows */}
      <div className="max-h-[420px] overflow-y-auto divide-y divide-gray-50" style={{ scrollbarWidth: "thin" }}>
        {filtered.length === 0 ? (
          <div className="py-8 text-center text-sm text-gray-400">
            No sites match &quot;{search}&quot;
          </div>
        ) : (
          filtered.map((site) => {
            const checked = selectedSites.has(site.url);
            const domain = site.url.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, "");
            return (
              <button
                key={site.url}
                onClick={() => onToggle(site.url)}
                className={`grid grid-cols-[40px_1fr_120px_80px] items-center px-4 py-2.5 text-left w-full transition-colors
                  ${checked ? "bg-gray-900/[0.03]" : "hover:bg-gray-50"}`}
              >
                <span
                  className={`w-4 h-4 rounded border flex items-center justify-center transition-all
                  ${checked ? "bg-gray-900 border-gray-900" : "border-gray-300 bg-white"}`}
                >
                  {checked && (
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </span>

                <div className="min-w-0 pr-4">
                  <p className="text-sm font-medium text-gray-900 truncate">{site.name || domain}</p>
                  <p className="text-[11px] text-gray-400 truncate">{domain}</p>
                </div>

                <div>
                  {site.site_type && (
                    <span className={`inline-block text-[11px] px-2 py-0.5 rounded-full font-medium
                      ${site.site_type === "job_portal"
                        ? "bg-blue-50 text-blue-700"
                        : site.site_type === "recruitment_agency"
                          ? "bg-purple-50 text-purple-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {site.site_type === "job_portal" ? "Portal"
                        : site.site_type === "recruitment_agency" ? "Agency"
                        : site.site_type?.replace(/_/g, " ") || "—"}
                    </span>
                  )}
                </div>

                <div className="text-right">
                  {site.confidence != null && <ConfidenceBadge value={site.confidence} />}
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

function CrawlConfigModal({ config, setConfig, selectedCount, onStart, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-md mx-4 p-6 animate-scale-in">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Pipeline Configuration</h3>
            <p className="text-sm text-gray-500 mt-0.5">
              {selectedCount} site{selectedCount !== 1 ? "s" : ""} selected
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex flex-col gap-5">
          <ConfigField
            label="Max pages per site"
            hint="Stop crawling each site after this many pages"
            type="number" min={1}
            value={config.maxPages}
            onChange={(v) => setConfig({ ...config, maxPages: v === "" ? "" : parseInt(v) || 0 })}
            onBlur={() => setConfig((c) => ({ ...c, maxPages: c.maxPages || 1 }))}
          />
          <ConfigField
            label="Max depth"
            hint="How many link levels deep to follow (1 = homepage only)"
            type="number" min={1} max={10}
            value={config.maxDepth}
            onChange={(v) => setConfig({ ...config, maxDepth: v === "" ? "" : parseInt(v) || 0 })}
            onBlur={() => setConfig((c) => ({ ...c, maxDepth: c.maxDepth || 1 }))}
          />
          <ConfigField
            label="Delay between requests"
            hint="Polite crawling delay between page requests"
            type="number" min={0} step={0.5} suffix="sec"
            value={config.crawlDelay}
            onChange={(v) => setConfig({ ...config, crawlDelay: v === "" ? "" : parseFloat(v) || 0 })}
            onBlur={() => setConfig((c) => ({ ...c, crawlDelay: c.crawlDelay === "" ? 0 : c.crawlDelay }))}
          />
        </div>

        <p className="mt-4 text-xs text-gray-400">
          After crawling, pages will be auto-classified. You can then extract skills from the results modal.
        </p>

        <div className="flex items-center gap-2 mt-5 pt-5 border-t border-gray-100">
          <button
            onClick={onClose}
            className="flex-1 h-10 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-lg
                       hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onStart}
            className="flex-1 h-10 text-sm font-medium text-white bg-gray-900 rounded-lg
                       hover:bg-gray-800 transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 3l14 9-14 9V3z" />
            </svg>
            Start Pipeline
          </button>
        </div>
      </div>
    </div>
  );
}

function ConfigField({ label, hint, suffix, value, onChange, onBlur, ...inputProps }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1.5">{label}</label>
      <div className="relative">
        <input
          {...inputProps}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
          className={`w-full h-10 px-3 ${suffix ? "pr-10" : ""} text-sm bg-gray-50 border border-gray-200 rounded-lg
                     focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent
                     focus:bg-white transition-all`}
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">{suffix}</span>
        )}
      </div>
      {hint && <p className="mt-1.5 text-xs text-gray-400">{hint}</p>}
    </div>
  );
}
