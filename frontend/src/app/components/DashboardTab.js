"use client";

import { Fragment, useEffect, useState } from "react";
import { API_BASE, Spinner } from "./ui";

// Postgres returns UTC ("2026-07-06 08:10:11+00:00"); parse it so display is LOCAL time.
function parseTs(s) {
  if (!s) return null;
  const d = new Date(String(s).replace(" ", "T"));
  return isNaN(d) ? null : d;
}

function timeAgo(s) {
  const d = parseTs(s);
  if (!d) return s ? String(s).slice(0, 16) : "—";
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hr${hrs > 1 ? "s" : ""} ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  return d.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function fullTime(s) {
  const d = parseTs(s);
  return d ? d.toLocaleString() : s || "—";
}

export default function DashboardTab() {
  const [rows, setRows] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [openId, setOpenId] = useState(null); // expanded row id
  const [detail, setDetail] = useState(null); // full snapshot for openId
  const [detailLoading, setDetailLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false); // Input 1 / Input 2 modal

  async function load() {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/api/curricula/v3/analyses?limit=100`);
      if (!r.ok) throw new Error(`Fetching saved analyses failed (${r.status})`);
      setRows((await r.json()).analyses);
    } catch (e) {
      setError(e.message || "Failed to load saved analyses");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function toggle(id) {
    setModalOpen(false);
    if (openId === id) {
      setOpenId(null);
      setDetail(null);
      return;
    }
    setOpenId(id);
    setDetail(null);
    setDetailLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/curricula/v3/analyses/${id}`);
      if (r.ok) setDetail(await r.json());
    } catch {}
    setDetailLoading(false);
  }

  async function remove(id) {
    try {
      await fetch(`${API_BASE}/api/curricula/v3/analyses/${id}`, {
        method: "DELETE",
      });
      if (openId === id) {
        setOpenId(null);
        setDetail(null);
      }
      load();
    } catch {}
  }

  // course-unit label -> market skills it was verified to cover (modal ✓ marks)
  const coversBy = {};
  for (const c of detail?.result?.covered || []) {
    (coversBy[c.course_competency] = coversBy[c.course_competency] || []).push(
      c.skill
    );
  }

  const fmtSettings = (s) =>
    s
      ? [
          s.esco_mode && `esco ${s.esco_mode}`,
          s.retrieve_floor != null && `floor ${s.retrieve_floor}`,
          s.top_k != null && `top-${s.top_k}`,
          s.pipeline_version,
        ]
          .filter(Boolean)
          .join(" · ")
      : "—";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Dashboard — saved analyses
          </h2>
          <p className="text-sm text-gray-500">
            Course–market alignment runs stored on Neon, replayed from the DB —
            no pipeline re-run.
          </p>
        </div>
        <button
          onClick={load}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-sm text-gray-500">
          <Spinner size={16} /> Loading saved analyses…
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {!loading && !error && !(rows?.length > 0) && (
        <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-8 text-center text-sm text-gray-500">
          No saved analyses yet — run a course in the Curriculum–Market
          Alignment_V6 tab and click “Save result”.
        </div>
      )}

      {rows?.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50 text-left text-xs uppercase tracking-wide text-gray-500">
              <tr>
                <th className="px-4 py-2">#</th>
                <th className="px-4 py-2">Course</th>
                <th className="px-4 py-2">Domain</th>
                <th className="px-4 py-2">Coverage</th>
                <th className="px-4 py-2">Settings</th>
                <th className="px-4 py-2">Saved at</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 bg-white">
              {rows.map((a) => (
                <Fragment key={a.id}>
                  <tr
                    onClick={() => toggle(a.id)}
                    className="cursor-pointer hover:bg-gray-50/70"
                  >
                    <td className="px-4 py-2 text-gray-400">{a.id}</td>
                    <td className="px-4 py-2">
                      <span className="font-medium text-gray-900">
                        {a.course_code || "—"}
                      </span>{" "}
                      <span className="text-gray-500">{a.course_title}</span>
                    </td>
                    <td className="px-4 py-2 text-gray-600">
                      {a.domain ? a.domain.replace(/_/g, " ") : "—"}
                    </td>
                    <td className="px-4 py-2 font-medium text-gray-900">
                      {a.coverage_pct != null ? `${a.coverage_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500">
                      {fmtSettings(a.settings)}
                    </td>
                    <td
                      className="whitespace-nowrap px-4 py-2 text-xs text-gray-500"
                      title={fullTime(a.created_at)}
                    >
                      {timeAgo(a.created_at)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          remove(a.id);
                        }}
                        className="text-xs text-gray-400 hover:text-red-600"
                      >
                        delete
                      </button>
                    </td>
                  </tr>

                  {openId === a.id && (
                    <tr>
                      <td colSpan={7} className="bg-gray-50/60 px-6 py-4">
                        {detailLoading && (
                          <div className="flex items-center gap-2 text-sm text-gray-500">
                            <Spinner size={14} /> Loading snapshot…
                          </div>
                        )}
                        {!detailLoading && detail && (
                          <div className="space-y-3">
                            <div className="space-y-1.5">
                              <p className="text-xs font-medium uppercase tracking-wide text-emerald-600">
                                ✓ Covered ({detail.result?.covered?.length || 0})
                              </p>
                              <div className="flex flex-wrap gap-1.5">
                                {(detail.result?.covered || []).map((c) => (
                                  <span
                                    key={c.skill}
                                    className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 ring-1 ring-emerald-200"
                                    title={c.reason || ""}
                                  >
                                    {c.skill}
                                    {c.demand != null ? ` · ${c.demand}` : ""}
                                  </span>
                                ))}
                                {!(detail.result?.covered || []).length && (
                                  <span className="text-xs text-gray-400">none</span>
                                )}
                              </div>
                            </div>

                            <div className="space-y-1.5">
                              <p className="text-xs font-medium uppercase tracking-wide text-red-600">
                                ✗ Gaps ({detail.result?.gaps?.length || 0})
                              </p>
                              <div className="flex flex-wrap gap-1.5">
                                {(detail.result?.gaps || []).map((g) => (
                                  <span
                                    key={g.skill}
                                    className="rounded-full bg-red-50 px-2 py-0.5 text-xs text-red-700 ring-1 ring-red-200"
                                  >
                                    {g.skill}
                                    {g.demand != null ? ` · ${g.demand}` : ""}
                                  </span>
                                ))}
                                {!(detail.result?.gaps || []).length && (
                                  <span className="text-xs text-gray-400">none</span>
                                )}
                              </div>
                            </div>

                            <div className="flex items-center justify-between gap-3">
                              <p className="text-xs text-gray-400">
                                method: {detail.result?.method || "—"} · off-core:{" "}
                                {(detail.result?.extra_market?.length || 0) +
                                  (detail.result?.specialized?.length || 0)}{" "}
                                · saved {timeAgo(detail.created_at)}
                              </p>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setModalOpen(true);
                                }}
                                className="rounded-lg border border-gray-300 px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50"
                              >
                                View inputs — Course Side ↔ Market Side
                              </button>
                            </div>
                          </div>
                        )}
                        {!detailLoading && !detail && (
                          <p className="text-sm text-gray-400">
                            Could not load snapshot.
                          </p>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Input 1 / Input 2 modal — the two skill lists stored in the snapshot */}
      {modalOpen && detail && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => setModalOpen(false)}
        >
          <div
            className="max-h-[85vh] w-full max-w-4xl overflow-y-auto rounded-2xl bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-gray-900">
                  {detail.course_code || "—"}{" "}
                  <span className="font-normal text-gray-600">
                    {detail.course_title}
                  </span>
                </h3>
                <p className="mt-1 text-xs text-gray-500">
                  {detail.domain ? detail.domain.replace(/_/g, " ") : "—"} ·
                  coverage{" "}
                  {detail.coverage_pct != null
                    ? `${detail.coverage_pct}%`
                    : "—"}{" "}
                  · saved{" "}
                  <span title={fullTime(detail.created_at)}>
                    {timeAgo(detail.created_at)}
                  </span>
                </p>
              </div>
              <button
                onClick={() => setModalOpen(false)}
                className="rounded-lg border border-gray-300 px-2.5 py-1 text-sm text-gray-600 hover:bg-gray-50"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-gray-200 px-4 py-3">
                <p className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400">
                  Input 1 — Course Side (
                  {detail.result?.esco?.length || 0} skills · ✓ covers demand /
                  ✗ off-core)
                </p>
                <ol className="space-y-1 text-sm text-gray-700">
                  {(detail.result?.esco || []).map((r, i) => {
                    const covers = coversBy[r.esco] || [];
                    return (
                      <li key={`${r.orig}-${i}`} title={r.reason || ""}>
                        {i + 1}.{" "}
                        <span
                          className={
                            covers.length
                              ? "font-medium text-emerald-700"
                              : "text-slate-400"
                          }
                        >
                          {covers.length ? "✓" : "✗"} {r.orig}
                        </span>{" "}
                        <span className="text-gray-400">
                          (ESCO = {r.mapped ? r.esco : "kept raw"})
                        </span>
                        {covers.length > 0 && (
                          <span className="text-xs text-emerald-600">
                            {" "}
                            → covers {covers.join(" · ")}
                          </span>
                        )}
                      </li>
                    );
                  })}
                  {!(detail.result?.esco || []).length && (
                    <li className="text-xs text-gray-400">
                      course-side ESCO rows not stored in this snapshot
                    </li>
                  )}
                </ol>
              </div>

              <div className="rounded-xl border border-gray-200 px-4 py-3">
                <p className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400">
                  Input 2 — Market Side (top-
                  {detail.result?.top_k ?? "K"} demand)
                </p>
                <ol className="space-y-1 text-sm text-gray-700">
                  {[
                    ...(detail.result?.covered || []),
                    ...(detail.result?.gaps || []),
                  ]
                    .sort((a, b) => (b.demand || 0) - (a.demand || 0))
                    .map((x, i) => (
                      <li key={`${x.skill}-${i}`}>
                        {i + 1}. {x.skill}{" "}
                        <span className="text-gray-400">
                          {x.esco ? `(ESCO = ${x.esco}) ` : ""}· {x.demand}
                        </span>
                      </li>
                    ))}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
