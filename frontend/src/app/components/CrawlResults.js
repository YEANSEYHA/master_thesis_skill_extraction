"use client";

import { useState, useEffect } from "react";
import { API_BASE } from "./ui";

function formatRunId(runId) {
  const parts = runId.split("__");
  const ts = parts[parts.length - 1];
  const m1 = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  const m2 = !m1 && ts.match(/^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$/);
  const m = m1 || m2;
  if (m) {
    const dt = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    return dt.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", hour12: true });
  }
  return runId;
}

export default function CrawlResults() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState(null);
  const [runDetail, setRunDetail] = useState(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [domainPages, setDomainPages] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [, setPageNum] = useState(1);   // value unread; setter resets the pager

  useEffect(() => {
    fetchRuns();
  }, []);

  async function fetchRuns() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/crawl/runs`);
      const data = await res.json();
      setRuns(data.runs || []);
    } catch {
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }

  async function selectRun(run) {
    setSelectedRun(run.run_id);
    setSelectedDomain(null);
    setDomainPages(null);
    setPreviewFile(null);
    try {
      const res = await fetch(`${API_BASE}/api/crawl/runs/${run.run_id}`);
      const data = await res.json();
      setRunDetail(data);
    } catch {
      setRunDetail(null);
    }
  }

  async function selectDomain(domain) {
    setSelectedDomain(domain);
    setPreviewFile(null);
    setPageNum(1);
    await fetchPages(domain, 1);
  }

  async function fetchPages(domain, page) {
    try {
      const res = await fetch(
        `${API_BASE}/api/crawl/runs/${selectedRun}/${domain}/pages?page=${page}&per_page=30`
      );
      const data = await res.json();
      setDomainPages(data);
    } catch {
      setDomainPages(null);
    }
  }

  function handlePageChange(newPage) {
    setPageNum(newPage);
    fetchPages(selectedDomain, newPage);
  }

  // --- Breadcrumb navigation ---
  function goToRuns() {
    setSelectedRun(null);
    setRunDetail(null);
    setSelectedDomain(null);
    setDomainPages(null);
    setPreviewFile(null);
  }

  function goToRun() {
    setSelectedDomain(null);
    setDomainPages(null);
    setPreviewFile(null);
  }

  function goToDomain() {
    setPreviewFile(null);
  }

  // --- Loading state ---
  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <span className="w-5 h-5 border-2 border-gray-200 border-t-gray-600 rounded-full animate-spin" />
        <span className="ml-3 text-sm text-gray-400">Loading crawl runs...</span>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center mx-auto mb-4">
          <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
          </svg>
        </div>
        <p className="text-sm text-gray-500">No crawl runs yet.</p>
        <p className="text-xs text-gray-400 mt-1">Start a crawl above to see results here.</p>
      </div>
    );
  }

  // --- Breadcrumbs ---
  const breadcrumbs = [{ label: "All Runs", onClick: goToRuns }];
  if (selectedRun) breadcrumbs.push({ label: formatRunId(selectedRun), onClick: goToRun });
  if (selectedDomain) breadcrumbs.push({ label: selectedDomain, onClick: goToDomain });
  if (previewFile) breadcrumbs.push({ label: previewFile });

  return (
    <div>
      {/* Breadcrumb */}
      {selectedRun && (
        <nav className="flex items-center gap-1.5 text-sm mb-4 flex-wrap">
          {breadcrumbs.map((bc, i) => (
            <span key={i} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-gray-300">/</span>}
              {bc.onClick ? (
                <button
                  onClick={bc.onClick}
                  className="text-gray-500 hover:text-gray-900 transition-colors"
                >
                  {bc.label}
                </button>
              ) : (
                <span className="text-gray-900 font-medium">{bc.label}</span>
              )}
            </span>
          ))}
        </nav>
      )}

      {/* Preview mode */}
      {previewFile && selectedRun && selectedDomain && (
        <div>
          <iframe
            src={`${API_BASE}/api/crawl/runs/${selectedRun}/${selectedDomain}/view/${previewFile}`}
            className="w-full border border-gray-200 rounded-xl bg-white"
            style={{ height: "70vh" }}
            sandbox="allow-same-origin"
            title="Page preview"
          />
        </div>
      )}

      {/* Domain pages list */}
      {!previewFile && selectedDomain && domainPages && (
        <DomainPagesView
          data={domainPages}
          onPreview={(filename) => setPreviewFile(filename)}
          onPageChange={handlePageChange}
          runId={selectedRun}
        />
      )}

      {/* Run detail — domain list */}
      {!previewFile && !selectedDomain && runDetail && (
        <RunDetailView detail={runDetail} onSelectDomain={selectDomain} />
      )}

      {/* Runs list */}
      {!selectedRun && <RunsList runs={runs} onSelect={selectRun} />}
    </div>
  );
}

// --- Runs list ---
function RunsList({ runs, onSelect }) {
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
          {runs.length} Crawl Run{runs.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="divide-y divide-gray-50">
        {runs.map((run) => {
          const s = run.summary || {};
          return (
            <button
              key={run.run_id}
              onClick={() => onSelect(run)}
              className="w-full px-4 py-3.5 text-left hover:bg-gray-50 transition-colors flex items-center gap-4"
            >
              <div className="w-9 h-9 rounded-lg bg-gray-100 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">{formatRunId(run.run_id)}</p>
                <p className="text-xs text-gray-400 mt-0.5">
                  {run.domain_count} domain{run.domain_count !== 1 ? "s" : ""}
                  {run.total_pages ? ` \u00B7 ${run.total_pages} HTML pages` : ""}
                </p>
              </div>
              {/* Mini stats */}
              <div className="flex items-center gap-2 shrink-0">
                {s.success > 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium">
                    {s.success} ok
                  </span>
                )}
                {s.blocked > 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 font-medium">
                    {s.blocked} blocked
                  </span>
                )}
                {s.failed > 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-red-50 text-red-700 font-medium">
                    {s.failed} failed
                  </span>
                )}
              </div>
              <svg className="w-4 h-4 text-gray-300 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// --- Run detail view ---
function RunDetailView({ detail, onSelectDomain }) {
  const report = detail.report || {};
  const summary = report.summary || {};

  return (
    <div>
      {/* Summary cards */}
      {Object.keys(summary).length > 0 && (
        <div className="grid grid-cols-4 gap-3 mb-5">
          <MiniStat label="Sites" value={summary.total_sites || 0} />
          <MiniStat label="Pages" value={summary.total_pages || 0} />
          <MiniStat label="Success" value={summary.success || 0} color="emerald" />
          <MiniStat label="Blocked" value={summary.blocked || 0} color="amber" />
        </div>
      )}

      {/* Domain list */}
      <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
        <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
            {detail.domains?.length || 0} Domains
          </span>
        </div>
        <div className="divide-y divide-gray-50 max-h-[500px] overflow-y-auto" style={{ scrollbarWidth: "thin" }}>
          {(detail.domains || []).map((d) => {
            const meta = d.metadata || {};
            const status = meta.crawl_status || meta.status;
            return (
              <button
                key={d.domain}
                onClick={() => onSelectDomain(d.domain)}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors flex items-center gap-4"
              >
                {/* Status dot */}
                <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                  status === "success" ? "bg-emerald-500"
                  : status === "blocked" ? "bg-amber-500"
                  : status === "failed" ? "bg-red-500"
                  : "bg-gray-300"
                }`} />

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{d.domain}</p>
                  {meta.base_url && (
                    <p className="text-[11px] text-gray-400 truncate mt-0.5">{meta.base_url}</p>
                  )}
                </div>

                <span className="text-xs text-gray-500 shrink-0">
                  {d.pages} page{d.pages !== 1 ? "s" : ""}
                </span>

                <svg className="w-4 h-4 text-gray-300 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// --- Domain pages view ---
function DomainPagesView({ data, onPreview, onPageChange, runId }) {
  const pages = data.pages || [];

  return (
    <div>
      <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
        {/* Header */}
        <div className="grid grid-cols-[1fr_80px_60px] px-4 py-2 border-b border-gray-100 bg-gray-50/50">
          <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">URL</span>
          <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider text-right">Size</span>
          <span />
        </div>

        {/* Rows */}
        <div className="divide-y divide-gray-50 max-h-[500px] overflow-y-auto" style={{ scrollbarWidth: "thin" }}>
          {pages.length === 0 ? (
            <div className="py-8 text-center text-sm text-gray-400">No pages saved.</div>
          ) : (
            pages.map((p) => (
              <div
                key={p.filename}
                className="grid grid-cols-[1fr_80px_60px] items-center px-4 py-2.5 hover:bg-gray-50 transition-colors"
              >
                <div className="min-w-0 pr-4">
                  <p className="text-xs font-medium text-gray-700 truncate">
                    {p.filename}
                  </p>
                  {p.url && (
                    <p className="text-[11px] text-gray-400 truncate mt-0.5">{p.url}</p>
                  )}
                </div>
                <span className="text-xs text-gray-400 text-right">{p.size_kb} KB</span>
                <div className="flex justify-end">
                  <button
                    onClick={() => onPreview(p.filename)}
                    className="text-xs text-blue-600 hover:text-blue-800 font-medium transition-colors"
                  >
                    View
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Pagination */}
      {data.total_pages > 1 && (
        <div className="flex items-center justify-between mt-3 px-1">
          <span className="text-xs text-gray-400">
            Page {data.page} of {data.total_pages} ({data.total} pages total)
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange(data.page - 1)}
              disabled={data.page <= 1}
              className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-white border border-gray-200 rounded-md
                         hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Prev
            </button>
            <button
              onClick={() => onPageChange(data.page + 1)}
              disabled={data.page >= data.total_pages}
              className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-white border border-gray-200 rounded-md
                         hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value, color }) {
  const colors = {
    emerald: "text-emerald-700",
    amber: "text-amber-700",
    red: "text-red-700",
  };
  return (
    <div className="bg-gray-50 rounded-lg px-3 py-2.5 border border-gray-100">
      <p className={`text-lg font-semibold ${colors[color] || "text-gray-900"}`}>{value}</p>
      <p className="text-[11px] text-gray-400 mt-0.5">{label}</p>
    </div>
  );
}
