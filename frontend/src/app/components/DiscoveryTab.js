"use client";

import { useState, useRef, useEffect } from "react";
import { API_BASE, apiFetch, ConfidenceBadge, SiteTypeBadge, Terminal, countryFlag } from "./ui";

export default function DiscoveryTab({ countries, countriesLoading, fetchCountries }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [discovering, setDiscovering] = useState(false);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [showManual, setShowManual] = useState(false);
  const [manualForm, setManualForm] = useState({ url: "", name: "", country: "", description: "" });
  const [manualLoading, setManualLoading] = useState(false);
  const [manualSuccess, setManualSuccess] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleting, setDeleting] = useState(null);
  const [rateLimitModal, setRateLimitModal] = useState(null);
  const logRef = useRef(null);
  const logCountRef = useRef(0);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    fetchCountries();
  }, []);

  async function handleSearch(countryOverride) {
    const country = (countryOverride || query).trim().toLowerCase();
    if (!country) return;

    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch(`/api/sites?country=${encodeURIComponent(country)}`);
      setResults(data.sites || []);
    } catch (e) {
      setError(e.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleDiscover() {
    const country = query.trim().toLowerCase();
    if (!country) return;

    setDiscovering(true);
    setError(null);
    setLogs([]);
    logCountRef.current = 0;
    try {
      await apiFetch(`/api/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ country, refresh: false }),
      });
      pollDiscoveryStatus(country);
    } catch (e) {
      setError(e.message);
      setDiscovering(false);
    }
  }

  async function handleManualInsert() {
    const { url, name, country, description } = manualForm;
    if (!url.trim() || !name.trim() || !country.trim()) return;

    setManualLoading(true);
    setManualSuccess(null);
    setError(null);
    try {
      const data = await apiFetch(`/api/sites/manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, name, country, description }),
      });
      setManualSuccess(data.message);
      setManualForm({ url: "", name: "", country: "", description: "" });
      if (query.trim().toLowerCase() === country.trim().toLowerCase()) {
        handleSearch();
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setManualLoading(false);
    }
  }

  async function handleDelete(url) {
    setDeleting(url);
    try {
      await apiFetch(`/api/sites?url=${encodeURIComponent(url)}`, { method: "DELETE" });
      setResults((prev) => prev?.filter((s) => s.url !== url) || null);
      setDeleteConfirm(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setDeleting(null);
    }
  }

  function pollDiscoveryStatus(country) {
    const interval = setInterval(async () => {
      try {
        const since = logCountRef.current;
        const res = await fetch(`${API_BASE}/api/discover/status/${encodeURIComponent(country)}?since=${since}`);
        const data = await res.json();

        if (data.logs && data.logs.length > 0) {
          setLogs((prev) => [...prev, ...data.logs]);
          logCountRef.current = data.log_count;
        }

        if (data.status === "completed") {
          clearInterval(interval);
          setDiscovering(false);
          handleSearch();
        } else if (data.status === "failed") {
          clearInterval(interval);
          setDiscovering(false);
          if (data.error_type === "rate_limit") {
            setRateLimitModal(data.error);
          } else {
            setError(`Discovery failed: ${data.error}`);
          }
        }
      } catch {
        clearInterval(interval);
        setDiscovering(false);
      }
    }, 2000);
  }

  return (
    <>
      {/* Section header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold tracking-tight text-gray-900">Job Site Discovery</h1>
        <p className="mt-1 text-sm text-gray-500">Search discovered job portals or run the pipeline to find new ones.</p>
      </div>

      {/* Country selector + search */}
      <div className="flex gap-2 mb-3">
        <select
          value={query}
          onChange={(e) => { setQuery(e.target.value); if (e.target.value) { handleSearch(e.target.value); if (showManual) setManualForm(f => ({ ...f, country: e.target.value })); } }}
          className="h-11 px-3 text-sm bg-white border border-gray-200 rounded-lg text-gray-900
                     focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent
                     transition-shadow appearance-none
                     bg-[url('data:image/svg+xml;charset=UTF-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%239ca3af%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M6%209l6%206%206-6%22%2F%3E%3C%2Fsvg%3E')]
                     bg-[length:12px] bg-[right_10px_center] bg-no-repeat pr-8"
        >
          <option value="">Select country</option>
          {countries.map((c) => (
            <option key={c.country} value={c.country}>
              {countryFlag(c.country)} {c.country.charAt(0).toUpperCase() + c.country.slice(1)} ({c.site_count})
            </option>
          ))}
        </select>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Or type a new country..."
          className="flex-1 h-11 px-4 text-sm bg-white border border-gray-200 rounded-lg
                     placeholder:text-gray-400
                     focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent
                     transition-shadow"
        />
        <button
          onClick={() => handleSearch()}
          disabled={loading || !query.trim()}
          className="h-11 px-6 text-sm font-medium text-white bg-gray-900 rounded-lg
                     hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Searching
            </span>
          ) : "Search"}
        </button>
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 mb-8">
        <button
          onClick={handleDiscover}
          disabled={discovering || !query.trim()}
          className="h-9 px-4 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded-lg
                     hover:bg-gray-50 hover:border-gray-300
                     disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {discovering ? (
            <span className="flex items-center gap-2">
              <span className="w-3 h-3 border-2 border-gray-400/30 border-t-gray-500 rounded-full animate-spin" />
              Discovering...
            </span>
          ) : "Discover New Sites"}
        </button>
        <button
          onClick={() => { if (!showManual && query.trim()) setManualForm(f => ({ ...f, country: query.trim() })); setShowManual(!showManual); setManualSuccess(null); }}
          className={`h-9 px-4 text-xs font-medium rounded-lg border transition-all
            ${showManual
              ? "text-gray-900 bg-gray-100 border-gray-300"
              : "text-gray-700 bg-white border-gray-200 hover:bg-gray-50 hover:border-gray-300"
            }`}
        >
          + Add Manually
        </button>
        {discovering && (
          <span className="text-xs text-gray-400 ml-1">Pipeline running — this may take a few minutes</span>
        )}
      </div>

      {/* Manual insert form */}
      {showManual && (
        <div className="mb-8 p-5 bg-white border border-gray-200 rounded-xl">
          <h3 className="text-sm font-medium text-gray-900 mb-4">Add a job site manually</h3>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <input type="text" placeholder="URL *" value={manualForm.url}
              onChange={(e) => {
                const url = e.target.value;
                const update = { ...manualForm, url };
                try {
                  const host = new URL(url.startsWith("http") ? url : "https://" + url).hostname;
                  update.name = host.replace(/^www\./, "").replace(/\.(com|org|net|io|co|jobs)$/i, "");
                } catch {}
                setManualForm(update);
              }}
              className="col-span-2 h-10 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent focus:bg-white transition-all" />
            <input type="text" placeholder="Site name *" value={manualForm.name}
              onChange={(e) => setManualForm({ ...manualForm, name: e.target.value })}
              className="h-10 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent focus:bg-white transition-all" />
            <input type="text" placeholder="Country *" value={manualForm.country}
              onChange={(e) => setManualForm({ ...manualForm, country: e.target.value })}
              className="h-10 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent focus:bg-white transition-all" />
            <input type="text" placeholder="Description (optional)" value={manualForm.description}
              onChange={(e) => setManualForm({ ...manualForm, description: e.target.value })}
              className="col-span-2 h-10 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent focus:bg-white transition-all" />
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleManualInsert}
              disabled={manualLoading || !manualForm.url.trim() || !manualForm.name.trim() || !manualForm.country.trim()}
              className="h-9 px-5 text-sm font-medium text-white bg-gray-900 rounded-lg hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {manualLoading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Adding...
                </span>
              ) : "Add Site"}
            </button>
            {manualSuccess && <span className="text-sm text-emerald-600">{manualSuccess}</span>}
          </div>
        </div>
      )}

      {/* Terminal log */}
      {logs.length > 0 && (
        <div className="mb-8">
          <Terminal logRef={logRef} logs={logs} label="discovery pipeline" isLive={discovering} />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 px-4 py-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg">
          {error}
        </div>
      )}

      {/* Results */}
      {results !== null && (
        <div>
          {results.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-gray-400">No job sites found for &ldquo;{query}&rdquo;</p>
              <p className="text-xs text-gray-300 mt-1">Try clicking &ldquo;Discover New Sites&rdquo; to run the pipeline</p>
            </div>
          ) : (
            <>
              <div className="flex items-baseline justify-between mb-4">
                <h2 className="text-sm font-medium text-gray-900">
                  {results.length} site{results.length !== 1 ? "s" : ""} found
                </h2>
                <span className="text-xs text-gray-400">Sorted by confidence</span>
              </div>

              <div className="flex flex-col gap-1.5">
                {results.map((site) => (
                  <div
                    key={site.url}
                    className={`flex items-center gap-3 px-3 py-2.5 bg-white border rounded-lg transition-all group relative
                      ${deleteConfirm === site.url ? "border-red-300 bg-red-50/50" : "border-gray-200 hover:border-gray-300"}`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <a href={site.url} target="_blank" rel="noopener noreferrer"
                          className="text-sm font-medium text-gray-900 truncate hover:underline underline-offset-2 decoration-gray-300">
                          {site.name || site.url}
                        </a>
                        {site.site_type && <SiteTypeBadge type={site.site_type} />}
                      </div>
                      <p className="text-xs text-gray-400 truncate mt-0.5">{site.url}</p>
                    </div>
                    {site.confidence != null && <ConfidenceBadge value={site.confidence} />}

                    {deleteConfirm === site.url ? (
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button onClick={() => handleDelete(site.url)} disabled={deleting === site.url}
                          className="h-7 px-2.5 text-xs font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors">
                          {deleting === site.url ? (
                            <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block" />
                          ) : "Delete"}
                        </button>
                        <button onClick={() => setDeleteConfirm(null)}
                          className="h-7 px-2 text-xs text-gray-500 hover:text-gray-700 transition-colors">
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button onClick={() => setDeleteConfirm(site.url)}
                        className="opacity-0 group-hover:opacity-100 shrink-0 h-7 w-7 flex items-center justify-center rounded-md text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all"
                        title="Remove site">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Rate Limit Error Modal */}
      {rateLimitModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full mx-4 overflow-hidden">
            <div className="px-6 pt-6 pb-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Search API Rate Limit</h3>
              </div>
              <p className="text-sm text-gray-600 leading-relaxed">{rateLimitModal}</p>
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">
                  <span className="font-medium text-gray-700">Tip:</span> Google Custom Search resets daily (100 queries/day). SerpAPI resets monthly. Try again later or upgrade your plan.
                </p>
              </div>
            </div>
            <div className="px-6 pb-6 pt-2">
              <button
                onClick={() => setRateLimitModal(null)}
                className="w-full h-10 text-sm font-medium text-white bg-gray-900 rounded-lg hover:bg-gray-800 transition-colors"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
