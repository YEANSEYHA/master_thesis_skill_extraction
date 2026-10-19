"use client";

import { useState, useEffect } from "react";
import { API_BASE, Spinner } from "./ui";
import placeholder from "./recV3Placeholder.json";

// DEV: when true, preload the baked-in result (markdown + skills) for
// "Introduction to Artificial Intelligence" so the page renders instantly
// without an upload or any qwen3/LLM call. Flip to false for the real flow.
const USE_PLACEHOLDER = true;

export default function RecommendationESCOTab() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(USE_PLACEHOLDER ? placeholder.markdown : null);
  const [skills, setSkills] = useState(USE_PLACEHOLDER ? placeholder.skills : null);
  const [usingPlaceholder, setUsingPlaceholder] = useState(USE_PLACEHOLDER);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [skillsError, setSkillsError] = useState("");
  const [view, setView] = useState("extract"); // "extract" | "analyze"
  const [esco, setEsco] = useState(null);          // ESCO normalization result
  const [normalizing, setNormalizing] = useState(false);
  const [normError, setNormError] = useState("");

  // After skills are extracted: normalize every skill to its nearest ESCO
  // preferred label and show the mapping below the skills table.
  useEffect(() => {
    if (!skills?.skills?.length) {
      setEsco(null);
      setNormalizing(false);
      return;
    }
    let alive = true;
    setEsco(null);
    setNormError("");
    setNormalizing(true);
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/curricula/v3/esco-normalize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            course_skills: skills.skills.map((s) => s.name),
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || `ESCO normalize failed (${r.status})`);
        }
        if (alive) setEsco(await r.json());
      } catch (e) {
        if (alive) setNormError(e.message || "ESCO normalize failed");
      } finally {
        if (alive) setNormalizing(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [skills]);

  function selectFile(f) {
    if (!f) return;
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      return;
    }
    setFile(f);
    setResult(null);
    setError("");
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  }

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    setSkills(null);
    setSkillsError("");
    setUsingPlaceholder(false);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/api/curricula/v3/markdown`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Upload failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleExtractSkills() {
    if (!result?.markdown) return;
    setSkillsLoading(true);
    setSkillsError("");
    setSkills(null);
    try {
      const res = await fetch(`${API_BASE}/api/curricula/v3/extract-skills`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ markdown: result.markdown }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Extraction failed (${res.status})`);
      }
      setSkills(await res.json());
    } catch (e) {
      setSkillsError(e.message || "Extraction failed");
    } finally {
      setSkillsLoading(false);
    }
  }

  // Step 2: the ESCO analyze page
  if (view === "analyze" && skills) {
    return (
      <AnalyzeESCO
        course={skills.course}
        skills={skills.skills}
        onBack={() => setView("extract")}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
          Curriculum–Market Alignment_V4
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Method V4 · ESCO-normalized curriculum-market matching
        </p>
      </div>

      {usingPlaceholder && (
        <div className="flex items-center justify-between rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800">
          <span>
            Dev placeholder loaded (no upload / no LLM) —{" "}
            <span className="font-medium">{placeholder.markdown.filename}</span>
          </span>
          <button
            onClick={() => {
              setUsingPlaceholder(false);
              setResult(null);
              setSkills(null);
            }}
            className="rounded-md bg-white px-2 py-1 font-medium text-amber-800 ring-1 ring-amber-200 hover:bg-amber-100"
          >
            Clear &amp; upload real PDF
          </button>
        </div>
      )}

      {/* Drag-and-drop zone */}
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl
                    border-2 border-dashed px-6 py-10 text-center transition-colors
                    ${
                      dragging
                        ? "border-gray-900 bg-gray-100"
                        : "border-gray-300 bg-gray-50 hover:bg-gray-100"
                    }`}
      >
        <input
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
        <svg
          className="h-8 w-8 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
          />
        </svg>
        <p className="text-sm font-medium text-gray-700">
          {file ? file.name : "Drag & drop a PDF here, or click to choose"}
        </p>
        <p className="text-xs text-gray-400">Course syllabus PDF</p>
      </label>

      {/* Action row */}
      <div className="flex items-center justify-end">
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="inline-flex items-center gap-2 rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white
                     hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {loading && <Spinner size={14} />}
          {loading ? "Converting…" : "Convert to Markdown"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Markdown output */}
      {result && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span className="font-medium text-gray-700">{result.filename}</span>
            <span>{result.chars.toLocaleString()} chars</span>
          </div>
          <pre
            className="max-h-[60vh] overflow-auto rounded-xl border border-gray-200 bg-gray-50
                       px-4 py-3 font-mono text-[12px] leading-5 text-gray-800 whitespace-pre-wrap"
          >
            {result.markdown}
          </pre>

          {/* Extract skills from the markdown */}
          <div className="flex justify-end pt-1">
            <button
              onClick={handleExtractSkills}
              disabled={skillsLoading}
              className="inline-flex items-center gap-2 rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white
                         hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {skillsLoading && <Spinner size={14} />}
              {skillsLoading ? "Extracting… (~1–2 min)" : "Extract the Data from this Markdown"}
            </button>
          </div>
        </div>
      )}

      {skillsError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {skillsError}
        </div>
      )}

      {/* Extracted skills */}
      {skills && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span className="flex items-center gap-2 font-medium text-gray-700">
              <span>
                {skills.course?.title}
                {skills.course?.code ? `  (${skills.course.code})` : ""}
              </span>
              {skills.course?.category && (
                <span
                  className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-medium text-violet-700"
                  title="Labour-market category inferred from the syllabus by the LLM"
                >
                  {skills.course.category}
                </span>
              )}
            </span>
            <span>{skills.skills.length} skills</span>
          </div>

          {/* Routed career domain(s) — computed cosine scores (route.py) */}
          {skills.course?.fields?.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-gray-500">Routed career domain(s):</span>
              {skills.course.fields.map((f, i) => (
                <span
                  key={f.name}
                  title="Cosine similarity of the course skills to this career-domain profile"
                  className={`rounded-full px-2 py-0.5 font-medium ${
                    i === 0
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-emerald-50 text-emerald-600"
                  }`}
                >
                  {f.name.replace(/_/g, " ")} · {f.score.toFixed(2)}
                </span>
              ))}
            </div>
          )}

          <div className="overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-left text-xs text-gray-500">
                <tr>
                  <th className="px-4 py-2 font-medium">Skill</th>
                  <th className="px-4 py-2 font-medium">Source</th>
                  <th className="px-4 py-2 font-medium">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {skills.skills.map((s, i) => (
                  <tr key={i} className="border-t border-gray-100">
                    <td className="px-4 py-2 font-medium text-gray-900">{s.name}</td>
                    <td className="px-4 py-2">
                      <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-600">
                        {s.source}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500 truncate max-w-[22rem]">
                      {s.evidence}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Auto ESCO normalization — appears right after extraction */}
          {normalizing && (
            <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-500">
              <Spinner size={14} /> Normalizing skills to nearest ESCO label…
            </div>
          )}
          {normError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {normError}
            </div>
          )}
          {esco?.normalized?.length > 0 && <EscoNormalization data={esco} />}

          <div className="flex justify-end pt-1">
            <button
              onClick={() => setView("analyze")}
              className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-gray-700"
            >
              Next: Analyze (ESCO) →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ESCO normalization result, split into two tables: matched ESCO (above) and
// kept-raw / no-confident-match (below).
function EscoNormalization({ data }) {
  const matched = data.normalized.filter((r) => r.mapped);
  const raw = data.normalized.filter((r) => !r.mapped);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span className="font-medium text-gray-700">ESCO normalization</span>
        <span>
          {data.mapped_count}/{data.total} matched to ESCO
          {raw.length > 0 && ` · ${raw.length} kept raw`}
        </span>
      </div>

      {/* Matched to ESCO */}
      <div className="space-y-1.5">
        <p className="text-xs font-semibold uppercase tracking-wide text-emerald-600">
          ✓ Matched to ESCO ({matched.length})
        </p>
        {matched.length ? (
          <div className="overflow-hidden rounded-xl border border-emerald-100">
            <table className="w-full text-sm">
              <thead className="bg-emerald-50/50 text-left text-xs text-gray-500">
                <tr>
                  <th className="px-4 py-2 font-medium">Course skill</th>
                  <th className="px-4 py-2 font-medium">ESCO label</th>
                  <th className="px-4 py-2 font-medium text-right">Cosine</th>
                </tr>
              </thead>
              <tbody>
                {matched.map((r, i) => (
                  <tr key={i} className="border-t border-emerald-50">
                    <td className="px-4 py-2 font-medium text-gray-900">{r.orig}</td>
                    <td className="px-4 py-2 text-gray-700">{r.esco}</td>
                    <td className="px-4 py-2 text-right text-xs text-emerald-600">
                      {r.score.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="pl-1 text-sm text-gray-400">None</p>
        )}
      </div>

      {/* Kept raw — no confident ESCO match */}
      <div className="space-y-1.5">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          • Kept raw — no confident ESCO match ({raw.length})
        </p>
        {raw.length ? (
          <div className="overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-left text-xs text-gray-500">
                <tr>
                  <th className="px-4 py-2 font-medium">Course skill</th>
                  <th className="px-4 py-2 font-medium text-right">Best cosine</th>
                </tr>
              </thead>
              <tbody>
                {raw.map((r, i) => (
                  <tr key={i} className="border-t border-gray-100">
                    <td className="px-4 py-2 font-medium text-gray-900">{r.orig}</td>
                    <td className="px-4 py-2 text-right text-xs text-gray-400">
                      {r.score.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="pl-1 text-sm text-gray-400">None</p>
        )}
      </div>
    </div>
  );
}

/* ── Step 2: ESCO analyze — route → domain market → demand-weighted coverage ──── */


const _SEV_STYLE = {
  critical: { text: "text-red-600", chip: "bg-red-50 text-red-700 ring-1 ring-red-100" },
  medium: { text: "text-amber-600", chip: "bg-amber-50 text-amber-700 ring-1 ring-amber-100" },
  low: { text: "text-gray-500", chip: "bg-gray-50 text-gray-600 ring-1 ring-gray-200" },
};

// One tone per set — keeps the blocks visually consistent.
const _SET = {
  emerald: { bar: "border-l-emerald-400", title: "text-emerald-700",
             chip: "bg-emerald-50 text-emerald-700 ring-emerald-100", num: "text-emerald-500/70" },
  amber:   { bar: "border-l-amber-400", title: "text-amber-700",
             chip: "bg-amber-50 text-amber-700 ring-amber-100", num: "text-amber-500/70" },
  red:     { bar: "border-l-red-400", title: "text-red-700",
             chip: "bg-red-50 text-red-700 ring-red-100", num: "text-red-500/70" },
  slate:   { bar: "border-l-slate-300", title: "text-slate-600",
             chip: "bg-slate-50 text-slate-700 ring-slate-200", num: "text-slate-400" },
};

// A labelled, color-coded card for one set. `stack` lays children vertically
// (competency rows with evidence); default is a wrapping row of chips (gaps).
function SkillSet({ tone, icon, title, subtitle, count, stack, children }) {
  const s = _SET[tone];
  return (
    <div className={`rounded-xl border border-gray-100 border-l-4 ${s.bar} bg-white px-4 py-3`}>
      <h3 className={`text-sm font-semibold ${s.title}`}>
        {icon} {title} <span className="text-gray-400">({count})</span>
        {subtitle && (
          <span className="ml-1 font-normal text-gray-400">— {subtitle}</span>
        )}
      </h3>
      <div className={`mt-2 ${stack ? "flex flex-col gap-1.5" : "flex flex-wrap gap-1.5"}`}>
        {count ? children : <span className="text-sm text-gray-400">None</span>}
      </div>
    </div>
  );
}

// A single skill chip, optionally with its demand count, styled to its set's tone.
function Chip({ tone, label, demand, title }) {
  const s = _SET[tone];
  return (
    <span title={title} className={`rounded-full px-3 py-1 text-sm ring-1 ${s.chip}`}>
      {label}
      {demand != null && (
        <span className={`ml-1 text-xs ${s.num}`}>📊{demand.toLocaleString()}</span>
      )}
    </span>
  );
}

// A canonical competency with its evidence (the original course skills it groups).
// Click to expand: "✓ Machine Learning · 6 techniques  ▸ K-Means · Decision Trees…".
function CompetencyRow({ tone, icon, label, demand, evidence, unit = "items", note }) {
  const [open, setOpen] = useState(false);
  const s = _SET[tone];
  const n = evidence?.length || 0;
  return (
    <div className={`rounded-lg px-3 py-1.5 ring-1 ${s.chip}`}>
      <button
        type="button"
        onClick={() => n && setOpen((o) => !o)}
        className={`flex w-full items-center gap-2 text-left text-sm ${n ? "cursor-pointer" : "cursor-default"}`}
      >
        <span className="font-medium">{icon} {label}</span>
        {note && (
          <span className="rounded-full bg-white/70 px-1.5 py-0.5 text-[10px] text-gray-500 ring-1 ring-gray-200">
            {note}
          </span>
        )}
        {n > 0 && <span className={`text-xs ${s.num}`}>· {n} {unit}</span>}
        {demand != null && (
          <span className={`ml-auto text-xs ${s.num}`}>📊{demand.toLocaleString()}</span>
        )}
        {n > 0 && (
          <span className={`${demand != null ? "" : "ml-auto "}text-xs text-gray-400`}>
            {open ? "▾" : "▸"}
          </span>
        )}
      </button>
      {open && n > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1 pl-5">
          {evidence.map((e, i) => (
            <span
              key={i}
              className="rounded-full bg-white/70 px-2 py-0.5 text-xs text-gray-600 ring-1 ring-gray-200"
            >
              {e}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function AnalyzeESCO({ course, skills, onBack }) {
  const routed = course?.fields || [];
  // default to the single highest-scoring routed domain (first chip)
  const _topDomain = routed.length
    ? [...routed].sort((a, b) => b.score - a.score)[0].name
    : null;
  const [selected, setSelected] = useState(_topDomain ? [_topDomain] : []);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);
  const [topK, setTopK] = useState(15);   // # of top market skills to score against
  const [mode, setMode] = useState("esco"); // "esco" (symmetric ESCO) | "raw"
  const selKey = selected.join("|");

  function toggleDomain(name) {
    setSelected((prev) =>
      prev.includes(name)
        ? prev.length > 1
          ? prev.filter((n) => n !== name)   // keep at least one selected
          : prev
        : [...prev, name]
    );
  }

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selected.length) return;
      setLoading(true);
      setError("");
      try {
        const r = await fetch(`${API_BASE}/api/curricula/v3/esco-analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            course_skills: skills.map((s) => s.name),
            domains: selected,
            title: course?.title || "",
            top_k: topK,
            mode,
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || `Analyze failed (${r.status})`);
        }
        if (alive) setRes(await r.json());
      } catch (e) {
        if (alive) setError(e.message || "Analyze failed");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selKey, topK, mode]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
            Method V4 ESCO — {course?.title}
            {course?.code ? `  (${course.code})` : ""}
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            course skills and market skills normalized to ESCO · cosine-matched ·
            demand-weighted coverage
          </p>
        </div>
        <button
          onClick={onBack}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          ← Back
        </button>
      </div>

      {/* Domain selector — score against one routed domain, the other, or both */}
      {routed.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-500">Compare against:</span>
          {routed.map((f) => {
            const on = selected.includes(f.name);
            return (
              <button
                key={f.name}
                onClick={() => toggleDomain(f.name)}
                title="Toggle this career domain in/out of the market"
                className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
                  on
                    ? "bg-gray-900 text-white"
                    : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
                }`}
              >
                {on ? "✓ " : ""}
                {f.name.replace(/_/g, " ")} · {f.score.toFixed(2)}
              </button>
            );
          })}
        </div>
      )}

      {/* Match mode — raw↔raw vs symmetric ESCO↔ESCO (both cosine-matched) */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Normalize:</span>
        {[
          { key: "raw", label: "Raw ↔ Raw", title: "Compare the raw skill strings on both sides" },
          { key: "esco", label: "ESCO ↔ ESCO", title: "Map both sides to ESCO preferred labels first (symmetric ESCO), then compare" },
        ].map((m) => (
          <button
            key={m.key}
            onClick={() => setMode(m.key)}
            title={m.title}
            className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
              mode === m.key
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Market depth — how many of the top in-demand skills to score against */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Top market skills:</span>
        {[10, 15, 20].map((n) => (
          <button
            key={n}
            onClick={() => setTopK(n)}
            title="How many of the most in-demand skills to compare the course against"
            className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
              topK === n
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-sm text-gray-500">
          <Spinner size={16} /> Computing coverage against the routed domain market…
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {res && (
        <>
          {/* small, secondary coverage metric */}
          <div>
            <h3 className="text-sm font-semibold text-gray-600">Market Gap Analysis</h3>
            <p className="mt-0.5 text-sm text-gray-500">
              Coverage: {res.alignment_score}%
              <span className="ml-2 text-xs text-gray-400">
                ({mode === "esco" ? "ESCO ↔ ESCO" : "Raw ↔ Raw"} · cosine ≥ {res.threshold})
              </span>
            </p>
          </div>

          {/* the three sets — course skills vs market demand */}
          <div className="space-y-3">
            <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
              Skill comparison — course vs{" "}
              {res.domains.map((d) => d.replace(/_/g, " ")).join(" + ")} job market
            </p>

            <p className="pl-1 text-sm text-gray-500">
              {skills.length} course skill{skills.length === 1 ? "" : "s"} vs top-
              {res.top_k} demand ·{" "}
              <span className="font-medium text-emerald-600">
                {res.covered.length} demand skill
                {res.covered.length === 1 ? "" : "s"} covered
              </span>
              ,{" "}
              <span className="font-medium text-slate-500">
                {res.off_core?.length || 0} course skill
                {(res.off_core?.length || 0) === 1 ? "" : "s"} off-core
              </span>
            </p>

            {/* Aligned — market demand skills the course covers (≥ τ) */}
            <SkillSet
              tone="emerald"
              icon="✓"
              title="Covered demand"
              subtitle="in demand and taught"
              count={res.covered.length}
              stack
            >
              {res.covered.map((c) => (
                <CompetencyRow
                  key={c.skill}
                  tone="emerald"
                  icon="✓"
                  label={c.skill}
                  demand={c.demand}
                  evidence={c.evidence}
                  unit={c.evidence.length === 1 ? "course skill" : "course skills"}
                  note={mode === "esco" && c.esco !== c.skill ? `ESCO: ${c.esco}` : undefined}
                />
              ))}
            </SkillSet>

            {/* Off-core — course skills that match NO top-K demand skill */}
            <SkillSet
              tone="slate"
              icon="•"
              title="Off-core course skills"
              subtitle={`taught, no match in the top-${res.top_k} demand`}
              count={res.off_core?.length || 0}
            >
              {(res.off_core || []).map((x) => (
                <Chip
                  key={x.skill}
                  tone="slate"
                  label={mode === "esco" && x.esco !== x.skill ? `${x.skill} → ${x.esco}` : x.skill}
                  title={`best cosine to any top-${res.top_k} demand skill: ${x.score}`}
                />
              ))}
            </SkillSet>

            {/* Market gaps — top-demand skills the course does NOT cover */}
            <SkillSet
              tone="red"
              icon="✗"
              title="Market gaps"
              subtitle="in demand, but not taught"
              count={res.gaps.length}
            >
              {res.gaps.map((g) => (
                <Chip key={g.skill} tone="red" label={g.skill} demand={g.demand} />
              ))}
            </SkillSet>
          </div>

          {/* --- commented out for now: "Market gaps by urgency" + "Consider adding" --- */}
          {false && (
            <>
              {/* market gaps ranked by urgency — a drill-down of the red set above */}
              <div className="rounded-xl border border-gray-100 bg-gray-50/50 px-4 py-3">
                <h3 className="text-sm font-semibold text-gray-700">
                  Market gaps by urgency
                  <span className="ml-1 font-normal text-gray-400">
                    — the gaps above, ranked by demand
                  </span>
                </h3>
                <div className="mt-2 space-y-2">
                  {["critical", "medium", "low"].map((sev) => {
                    const items = res.gaps.filter((g) => g.severity === sev);
                    if (!items.length) return null;
                    const style = _SEV_STYLE[sev];
                    return (
                      <div key={sev}>
                        <div
                          className={`text-xs font-semibold uppercase tracking-wide ${style.text}`}
                        >
                          {sev} ({items.length})
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1.5">
                          {items.map((g) => (
                            <span
                              key={g.skill}
                              className={`rounded-lg px-3 py-1 text-sm ${style.chip}`}
                            >
                              {g.skill}{" "}
                              <span className="text-xs opacity-60">
                                📊{g.demand.toLocaleString()}
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* suggestions */}
              {res.suggestions?.length > 0 && (
                <div className="rounded-xl border border-indigo-100 bg-indigo-50/40 px-5 py-3">
                  <div className="text-sm font-semibold text-indigo-700">
                    Consider adding
                  </div>
                  <div className="mt-1 text-sm text-gray-700">
                    {res.suggestions.join("  ·  ")}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
