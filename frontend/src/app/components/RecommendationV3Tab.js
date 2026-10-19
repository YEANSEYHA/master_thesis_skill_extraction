"use client";

import { useState, useEffect } from "react";
import { API_BASE, Spinner } from "./ui";
import placeholder from "./recV3Placeholder.json";

// DEV: when true, preload the baked-in result (markdown + skills) for
// "Introduction to Artificial Intelligence" so the page renders instantly
// without an upload or any qwen3/LLM call. Flip to false for the real flow.
const USE_PLACEHOLDER = true;

const METHOD_CONFIGS = {
  canonical: {
    id: "canonical",
    title: "Curriculum–Market Alignment",
    subtitle: "Method v3 · CLO skills vs labour-market demand",
    analyzeEndpoint: "/api/curricula/v3/analyze",
    nextLabel: "Next: Analyze (Method v3) →",
    analyzeHeading: "Method v3",
    analyzeSubtitle:
      "canonical competencies → demand-weighted coverage against the selected market",
    analyzeLoading: "Computing coverage against the selected market…",
    comparisonLabel: "Skill comparison",
    offCoreNote: (topK) => `taught, outside the top-${topK} scored demand`,
    specializedNote: "off-market / foundational",
  },
  independent: {
    id: "independent",
    title: "Curriculum–Market Alignment_V3",
    subtitle: "Proposal method 1 · independent course competency grouping",
    analyzeEndpoint: "/api/curricula/v3/independent-analyze",
    nextLabel: "Next: Analyze (Independent Method) →",
    analyzeHeading: "Independent Method",
    analyzeSubtitle:
      "course-only competency grouping → semantic comparison against labour-market demand",
    analyzeLoading: "Comparing independent course competencies against the selected market…",
    comparisonLabel: "Independent competency comparison",
    offCoreNote: (topK) => `course-side competencies outside the top-${topK} scored demand`,
    specializedNote: "course-side · outside scored core",
  },
  verified: {
    id: "verified",
    title: "Curriculum–Market Alignment_V5",
    subtitle: "Proposal method 2 · retrieve (embedding) → verify (LLM yes/no)",
    analyzeEndpoint: "/api/curricula/v3/verified-analyze",
    nextLabel: "Next: Analyze (Retrieve → Verify) →",
    analyzeHeading: "Retrieve → Verify",
    analyzeSubtitle:
      "course-only grouping → embedding shortlist (recall) → LLM verification (precision)",
    analyzeLoading: "Shortlisting candidates, then verifying each with qwen3…",
    comparisonLabel: "Verified competency comparison",
    offCoreNote: (topK) => `course-side competencies not verified against the top-${topK} demand`,
    specializedNote: "course-side · not a verified match",
  },
  verified_raw: {
    id: "verified_raw",
    title: "Curriculum–Market Alignment_V6",
    subtitle:
      "Proposal method 2b · raw skills → retrieve (embedding) → verify (LLM yes/no) — no competency grouping",
    analyzeEndpoint: "/api/curricula/v3/verified-raw-analyze",
    nextLabel: "Next: Analyze (Retrieve → Verify) →",
    analyzeHeading: "Retrieve → Verify (raw skills)",
    analyzeSubtitle:
      "raw course skills → embedding shortlist (recall) → LLM verification (precision) — no grouping stage",
    analyzeLoading: "Shortlisting candidates, then verifying each with qwen3…",
    comparisonLabel: "Verified skill comparison",
    offCoreNote: (topK) =>
      `taught in the course, but not in the top-${topK} market demand — includes foundational content, not necessarily useless`,
    specializedNote: "course-side · not a verified match",
  },
};

export default function RecommendationV3Tab({ variant = "canonical" }) {
  const method = METHOD_CONFIGS[variant] || METHOD_CONFIGS.canonical;
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
  // grouping-only preview (independent V3): course skills → competencies, no market
  const [groups, setGroups] = useState(null);
  const [grouping, setGrouping] = useState(false);
  const [groupingError, setGroupingError] = useState("");
  // C1 (V6 only): ESCO-anchor the extracted skills — raw → ESCO label mapping
  const [esco, setEsco] = useState(null);
  const [escoLoading, setEscoLoading] = useState(false);
  const [escoError, setEscoError] = useState("");

  // Button-triggered (no auto-fetch): Qwen3 top-5 retrieve → qwen3:14b pick/reject
  async function handleEscoNormalize() {
    if (!skills?.skills?.length) return;
    setEscoLoading(true);
    setEscoError("");
    setEsco(null);
    try {
      const res = await fetch(`${API_BASE}/api/curricula/v3/esco-normalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          course_skills: skills.skills.map((s) => s.name),
          llm: true, // C1: Qwen3 top-5 retrieve → qwen3:14b pick/reject (V6 path)
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `ESCO normalize failed (${res.status})`);
      }
      setEsco(await res.json());
    } catch (e) {
      setEscoError(e.message || "ESCO normalize failed");
    } finally {
      setEscoLoading(false);
    }
  }

  function selectFile(f) {
    if (!f) return;
    // .docx is accepted too — the backend routes it through docling.
    if (!/\.(pdf|docx)$/i.test(f.name)) {
      setError("Please select a PDF or DOCX file.");
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
    setGroups(null);
    setGroupingError("");
    setEsco(null);
    setEscoError("");
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
    setGroups(null);
    setGroupingError("");
    setEsco(null);
    setEscoError("");
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

  // Grouping-only preview: course skills → competencies (no market vocabulary).
  // Backed by consolidate_v3.py via /api/curricula/v3/independent-consolidate.
  async function handleGroup() {
    if (!skills?.skills?.length) return;
    setGrouping(true);
    setGroupingError("");
    setGroups(null);
    try {
      const res = await fetch(`${API_BASE}/api/curricula/v3/independent-consolidate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          course_skills: skills.skills.map((s) => s.name),
          title: skills.course?.title || "",
          course_code: skills.course?.code || "",
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Grouping failed (${res.status})`);
      }
      setGroups(await res.json());
    } catch (e) {
      setGroupingError(e.message || "Grouping failed");
    } finally {
      setGrouping(false);
    }
  }

  // Step 2: the V3 analyze page
  if (view === "analyze" && skills) {
    return (
      <AnalyzeV3
        course={skills.course}
        skills={skills.skills}
        markdown={result?.markdown || ""}
        onBack={() => setView("extract")}
        method={method}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
          {method.title}
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          {method.subtitle}
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
          accept="application/pdf,.pdf,.docx"
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
          {file ? file.name : "Drag & drop a PDF or DOCX here, or click to choose"}
        </p>
        <p className="text-xs text-gray-400">Course syllabus (PDF or DOCX)</p>
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

          {/* Routed career domain(s) — computed cosine scores (route.py); seeds
              the "Compare by career domain" mode on the analyze page */}
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

          {/* ESCO anchoring — button-triggered; nothing shown until clicked */}
          {method.id === "verified_raw" && (
            <div className="space-y-2 pt-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium uppercase tracking-wide text-gray-400">
                  ESCO anchoring — Qwen3 top-5 retrieve → qwen3:14b pick/reject
                </span>
                <button
                  onClick={handleEscoNormalize}
                  disabled={escoLoading}
                  className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700
                             hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
                >
                  {escoLoading && <Spinner size={14} />}
                  {escoLoading
                    ? "Normalizing… (~1 min first time)"
                    : esco
                      ? "Re-normalize"
                      : "Normalize to ESCO"}
                </button>
              </div>

              {escoError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {escoError}
                </div>
              )}

              {esco && (
                <p className="pl-1 text-xs text-gray-400">
                  {esco.mapped_count}/{esco.total} skills mapped — rejected skills
                  are kept raw for matching. Blue = exact, violet = broader category.
                </p>
              )}

              {/* Original skill → ESCO skill mapping list */}
              {esco?.normalized?.length > 0 && (
                <div className="space-y-1.5 rounded-xl border border-gray-200 bg-gray-50/50 px-4 py-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
                    Original skill → ESCO skill
                  </p>
                  {esco.normalized.map((r) => (
                    <div
                      key={r.orig}
                      className="flex flex-wrap items-center gap-2 text-sm"
                      title={r.reason || ""}
                    >
                      <span className="text-gray-700">{r.orig}</span>
                      <span className="text-gray-400">→</span>
                      {r.mapped ? (
                        <span
                          className={`font-medium ${
                            r.match === "broader" ? "text-violet-700" : "text-sky-700"
                          }`}
                        >
                          {r.esco}
                        </span>
                      ) : (
                        <span className="text-gray-500">kept raw</span>
                      )}
                      <span className="text-xs text-gray-400">
                        {typeof r.score === "number" ? r.score.toFixed(2) : ""}
                        {r.mapped ? (r.match === "broader" ? " · broad" : " · exact") : ""}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Grouping-only preview — course-only competencies (no market vocab) */}
          {(method.id === "independent" || method.id === "verified") && (
            <div className="space-y-2 pt-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium uppercase tracking-wide text-gray-400">
                  Competency groups — course-only (no market vocabulary)
                </span>
                <button
                  onClick={handleGroup}
                  disabled={grouping}
                  className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700
                             hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
                >
                  {grouping && <Spinner size={14} />}
                  {grouping ? "Grouping…" : groups ? "Re-group" : "Group into competencies"}
                </button>
              </div>

              {groupingError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {groupingError}
                </div>
              )}

              {groups && (
                <>
                  <p className="pl-1 text-sm text-gray-500">
                    {skills.skills.length} skill{skills.skills.length === 1 ? "" : "s"} →{" "}
                    <span className="font-medium text-gray-700">
                      {groups.competencies?.length || 0} competenc
                      {(groups.competencies?.length || 0) === 1 ? "y" : "ies"}
                    </span>
                  </p>
                  <div className="flex flex-col gap-1.5">
                    {(groups.competencies || []).map((c) => (
                      <CompetencyRow
                        key={c.competency}
                        tone="emerald"
                        icon="◆"
                        label={c.competency}
                        evidence={c.evidence}
                        unit="skills"
                      />
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          <div className="flex justify-end pt-1">
            <button
              onClick={() => setView("analyze")}
              className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-gray-700"
            >
              {method.nextLabel}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Step 2: V3 analyze — route → domain market → demand-weighted coverage ──── */


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

function AnalyzeV3({ course, skills, markdown, onBack, method }) {
  const routed = course?.fields || [];
  const _topDomain = routed.length
    ? [...routed].sort((a, b) => b.score - a.score)[0].name
    : null;
  // Two compare modes, each with its own chip set + selection:
  //   "category" — the 17 raw job_category buckets of the corpus
  //   "domain"   — the 20 curated career domains (previous process, routed+scored)
  const [compareBy, setCompareBy] = useState("category");
  const [selCats, setSelCats] = useState(
    course?.category ? [course.category] : []
  );
  const [selDoms, setSelDoms] = useState(_topDomain ? [_topDomain] : []);
  const selected = compareBy === "category" ? selCats : selDoms;
  const setSelected = compareBy === "category" ? setSelCats : setSelDoms;
  const [allCategories, setAllCategories] = useState(null); // [{name, n_postings}]
  const [allDomains, setAllDomains] = useState(null);       // [{name, n_postings}]
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);
  const [topK, setTopK] = useState(15);   // # of top market skills to score against
  const [corpus, setCorpus] = useState("linkedin");  // job-posting source to compare against
  // V6: match decision engine — "verify" (LLM yes/no) | "cosine" (embedding-only ablation)
  const [matcher, setMatcher] = useState("verify");
  // V6: raw → ESCO view of the top-K market skills (LLM-normalized, cached server-side)
  const [marketEsco, setMarketEsco] = useState(null);
  const [marketEscoLoading, setMarketEscoLoading] = useState(false);
  // Saved analyses (V6): each completed run is auto-saved to Neon; the list lets
  // you replay a stored snapshot without re-running the pipeline.
  const [saved, setSaved] = useState(null);          // {id, created_at, duplicate}
  const [savedBusy, setSavedBusy] = useState(false);
  const [savedList, setSavedList] = useState(null);  // summary rows
  const [snapshot, setSnapshot] = useState(null);    // set while viewing a stored run
  const selKey = selected.join("|");

  async function loadSavedList() {
    try {
      const q = course?.code ? `?course_code=${encodeURIComponent(course.code)}` : "";
      const r = await fetch(`${API_BASE}/api/curricula/v3/analyses${q}`);
      if (r.ok) setSavedList((await r.json()).analyses);
    } catch {}
  }

  useEffect(() => {
    if (method.id === "verified_raw") loadSavedList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load both chip lists once: the corpus's unique job_category buckets and
  // the curated career domains (scoped to the course's category when known —
  // other faculties' domains appear once their taxonomy is built).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const dq = course?.category
          ? `?category=${encodeURIComponent(course.category)}`
          : "";
        const [rc, rd] = await Promise.all([
          fetch(`${API_BASE}/api/curricula/v3/categories`),
          fetch(`${API_BASE}/api/curricula/v3/domains${dq}`),
        ]);
        if (rc.ok && alive) setAllCategories((await rc.json()).categories || []);
        if (rd.ok && alive) setAllDomains((await rd.json()).domains || []);
      } catch {}
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadSnapshot(id) {
    try {
      const r = await fetch(`${API_BASE}/api/curricula/v3/analyses/${id}`);
      if (!r.ok) return;
      const row = await r.json();
      setRes(row.result);
      setSnapshot({ id: row.id, created_at: row.created_at, settings: row.settings });
      setError("");
    } catch {}
  }

  async function deleteSnapshot(id) {
    try {
      await fetch(`${API_BASE}/api/curricula/v3/analyses/${id}`, { method: "DELETE" });
      if (snapshot?.id === id) setSnapshot(null);
      loadSavedList();
    } catch {}
  }

  async function saveResult() {
    if (!res || savedBusy) return;
    setSavedBusy(true);
    try {
      const r = await fetch(`${API_BASE}/api/curricula/v3/analyses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          course_code: course?.code || "",
          course_title: course?.title || "",
          result: res,
        }),
      });
      if (r.ok) {
        setSaved(await r.json());
        loadSavedList();
      }
    } catch {}
    setSavedBusy(false);
  }

  function toggleDomain(name) {
    setSelected((prev) =>
      prev.includes(name)
        ? prev.length > 1
          ? prev.filter((n) => n !== name)   // keep at least one selected
          : prev
        : [...prev, name]
    );
  }

  // Course-Side list number(s) for each matching unit (ESCO label, or raw name
  // when kept raw) — lets the Aligned table point back to "13." in the list.
  const courseNo = {};
  (res?.esco || []).forEach((r, i) => {
    const label = r.mapped ? r.esco : r.orig;
    (courseNo[label] = courseNo[label] || []).push(i + 1);
  });

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selected.length) {
        // no category yet (older extraction without course.category) — wait
        // for the user to pick a chip instead of spinning forever
        setLoading(false);
        setRes(null);
        return;
      }
      setLoading(true);
      setError("");
      try {
        const r = await fetch(`${API_BASE}${method.analyzeEndpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            course_skills: skills.map((s) => s.name),
            domains: selected,
            title: course?.title || "",
            course_code: course?.code || "",
            markdown: markdown || "",
            top_k: topK,
            matcher,
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || `Analyze failed (${r.status})`);
        }
        const data = await r.json();
        if (alive) {
          setRes(data);
          setSnapshot(null);            // live run, not a replay
          setSaved(null);               // fresh run — not saved until the button
        }
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
  }, [selKey, topK, matcher]);

  // V6: normalize the top-K market skills to ESCO for the raw → ESCO display
  useEffect(() => {
    if (method.id !== "verified_raw" || !res) {
      setMarketEsco(null);
      return;
    }
    const names = [...(res.covered || []), ...(res.gaps || [])].map((x) => x.skill);
    if (!names.length) {
      setMarketEsco(null);
      return;
    }
    let alive = true;
    (async () => {
      setMarketEscoLoading(true);
      try {
        const r = await fetch(`${API_BASE}/api/curricula/v3/esco-normalize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ course_skills: names, llm: true }),
        });
        if (!r.ok) throw new Error(`market ESCO normalize failed (${r.status})`);
        const d = await r.json();
        if (alive)
          setMarketEsco(Object.fromEntries(d.normalized.map((x) => [x.orig, x])));
      } catch {
        if (alive) setMarketEsco(null);
      } finally {
        if (alive) setMarketEscoLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [res, method.id]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
            {method.analyzeHeading} — {course?.title}
            {course?.code ? `  (${course.code})` : ""}
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            {method.analyzeSubtitle}
          </p>
        </div>
        <button
          onClick={onBack}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          ← Back
        </button>
      </div>

      {/* Job-posting source — which labour-market corpus to compare against */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Job-posting data:</span>
        <button
          onClick={() => setCorpus("linkedin")}
          title="LinkedIn — 1.35M global postings (the corpus currently wired)"
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            corpus === "linkedin"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {corpus === "linkedin" ? "✓ " : ""}LinkedIn data
        </button>
        <button
          disabled
          title="Cambodian job-posting dataset — not yet wired (see Next Action Step)"
          className="cursor-not-allowed rounded-full px-3 py-1 text-sm font-medium bg-white text-gray-300 ring-1 ring-gray-200"
        >
          Cambodia job posting
          <span className="ml-1.5 text-[10px] uppercase tracking-wide text-gray-300">
            soon
          </span>
        </button>
      </div>

      {/* Compare mode — raw job_category buckets vs curated career domains */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Compare by:</span>
        <button
          onClick={() => setCompareBy("category")}
          title="The unique job_category buckets of the raw corpus — market core = raw skill demand"
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            compareBy === "category"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {compareBy === "category" ? "✓ " : ""}Job category
        </button>
        <button
          onClick={() => setCompareBy("domain")}
          title="Curated career domains (previous process) — the course is routed to 1–3 domains with cosine scores"
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            compareBy === "domain"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {compareBy === "domain" ? "✓ " : ""}Career domain
        </button>
      </div>

      {/* Chip set for the active mode */}
      {compareBy === "category" ? (
        (allCategories || []).length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm text-gray-500">Compare against:</span>
            {allCategories.map((c) => {
              const on = selected.includes(c.name);
              return (
                <button
                  key={c.name}
                  onClick={() => toggleDomain(c.name)}
                  title={
                    c.n_postings != null
                      ? `${c.n_postings.toLocaleString()} postings`
                      : "Job category"
                  }
                  className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
                    on
                      ? "bg-gray-900 text-white"
                      : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
                  }`}
                >
                  {on ? "✓ " : ""}
                  {c.name.replace(/_/g, " ")}
                </button>
              );
            })}
          </div>
        )
      ) : (
        (() => {
          // previous process: routed domain(s) first with their cosine score,
          // then the remaining curated domains alphabetically
          const routedScore = Object.fromEntries(
            routed.map((f) => [f.name, f.score])
          );
          const postings = Object.fromEntries(
            (allDomains || []).map((d) => [d.name, d.n_postings])
          );
          const chips = [
            ...[...routed].sort((a, b) => b.score - a.score).map((f) => f.name),
            ...(allDomains || [])
              .map((d) => d.name)
              .filter((n) => !(n in routedScore))
              .sort(),
          ];
          if (!chips.length)
            return (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800">
                No career-domain taxonomy built for
                {course?.category ? ` "${course.category}"` : " this category"}{" "}
                yet (see CATEGORY_DOMAINS in build_profiles.py) — use the Job
                category mode instead.
              </div>
            );
          return (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-gray-500">Compare against:</span>
              {chips.map((name) => {
                const on = selected.includes(name);
                const score = routedScore[name];
                const n = postings[name];
                return (
                  <button
                    key={name}
                    onClick={() => toggleDomain(name)}
                    title={
                      (score != null
                        ? `Routed domain (cosine ${score.toFixed(2)})`
                        : "Career domain") +
                      (n != null ? ` — ${n.toLocaleString()} postings` : "")
                    }
                    className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
                      on
                        ? "bg-gray-900 text-white"
                        : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
                    }`}
                  >
                    {on ? "✓ " : ""}
                    {name.replace(/_/g, " ")}
                    {score != null ? ` · ${score.toFixed(2)}` : ""}
                  </button>
                );
              })}
            </div>
          );
        })()
      )}

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

      {/* V6: match-decision engine — LLM verify vs embedding-only (ablation) */}
      {method.id === "verified_raw" && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-500">Match decision:</span>
          <button
            onClick={() => setMatcher("verify")}
            title="Retrieve top-2 by cosine, then qwen3:14b judges each pair (yes/no)"
            className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
              matcher === "verify"
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
            }`}
          >
            {matcher === "verify" ? "✓ " : ""}LLM verify
          </button>
          <button
            onClick={() => setMatcher("cosine")}
            title="Embedding-only ablation: covered iff best Qwen3 cosine ≥ 0.70 — no LLM"
            className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
              matcher === "cosine"
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
            }`}
          >
            {matcher === "cosine" ? "✓ " : ""}Cosine only (τ=0.70)
          </button>
        </div>
      )}

      {/* V6: Course side + Market side (raw skill → its ESCO anchor), side by side */}
      {method.id === "verified_raw" && res && (
        <div className="grid items-start gap-3 lg:grid-cols-2">
        {/* Course side */}
        {res.esco?.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white px-5 py-4">
            <h3 className="text-xl text-gray-500">Course Side</h3>
            <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-gray-800">
              {res.esco.map((r) => (
                <li key={r.orig} title={r.reason || ""}>
                  {r.orig}{" "}
                  <span className="text-gray-500">
                    (ESCO = {r.mapped ? r.esco : "kept raw"})
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {/* Market side */}
        <div className="rounded-xl border border-gray-200 bg-white px-5 py-4">
          <h3 className="text-xl text-gray-500">Market Side</h3>
          {marketEscoLoading && (
            <p className="mt-1 flex items-center gap-2 text-xs text-gray-400">
              <Spinner size={12} /> normalizing market skills to ESCO… (first time
              only, then cached)
            </p>
          )}
          {![...(res.covered || []), ...(res.gaps || [])].length ? (
            <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
              No labour-market inventory for{" "}
              <span className="font-medium">
                {res.domains?.map((d) => d.replace(/_/g, " ")).join(", ") || "this domain"}
              </span>
              . This course did not route to a market category — likely a
              foundational / theory course, or the domain isn’t one of the
              built inventories. Nothing to compare against, so coverage is
              undefined (not 0%).
            </div>
          ) : (
            <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-gray-800">
              {[...(res.covered || []), ...(res.gaps || [])]
                .map((x) => ({ skill: x.skill, demand: x.demand }))
                .sort((a, b) => b.demand - a.demand)
                .map((row, i) => {
                  const r = marketEsco?.[row.skill];
                  return (
                    <li key={`${row.skill}-${i}`} title={r?.reason || ""}>
                      {row.skill}{" "}
                      <span className="text-gray-500">
                        {r
                          ? `(ESCO = ${r.mapped ? r.esco : "kept raw"})`
                          : "(…)"}
                      </span>
                    </li>
                  );
                })}
            </ol>
          )}
        </div>
        </div>
      )}

      {method.id === "verified_raw" && savedList?.length > 0 && (
        <div className="space-y-1.5 rounded-xl border border-gray-200 bg-gray-50/50 px-4 py-3">
          <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
            Saved analyses — stored on Neon, click to replay without re-running
          </p>
          <ul className="divide-y divide-gray-100">
            {savedList.map((a) => (
              <li key={a.id} className="flex items-center justify-between gap-3 py-1.5 text-sm">
                <button
                  onClick={() => loadSnapshot(a.id)}
                  className="text-left text-gray-700 hover:text-gray-900 hover:underline"
                >
                  #{a.id} · {a.course_code || a.course_title || "untitled"} ·{" "}
                  {a.domain || "—"} ·{" "}
                  {a.coverage_pct != null ? `${a.coverage_pct}%` : "—"}
                  <span className="text-gray-400">
                    {" "}· {(a.created_at || "").slice(0, 16)}
                  </span>
                </button>
                <button
                  onClick={() => deleteSnapshot(a.id)}
                  className="text-xs text-gray-400 hover:text-red-600"
                >
                  delete
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-sm text-gray-500">
          <Spinner size={16} /> {method.analyzeLoading}
        </div>
      )}

      {!loading && !res && !error && !selected.length && (
        <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-sm text-gray-500">
          Select a {compareBy === "category" ? "job category" : "career domain"}{" "}
          above to compare the course against.
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {res && (
        <>
          {snapshot && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800">
              Snapshot #{snapshot.id} · saved {(snapshot.created_at || "").slice(0, 16)} —
              replayed from DB, not recomputed
              {snapshot.settings
                ? ` · floor ${snapshot.settings.retrieve_floor} · esco ${snapshot.settings.esco_mode} · ${snapshot.settings.pipeline_version}`
                : ""}
            </div>
          )}
          {/* small, secondary coverage metric + save action */}
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-gray-600">Market Gap Analysis</h3>
              <p className="mt-0.5 text-sm text-gray-500">
                Coverage: {res.alignment_score}%
              </p>
            </div>
            {method.id === "verified_raw" && !snapshot && (
              <button
                onClick={saveResult}
                disabled={savedBusy || !!saved}
                className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700
                           hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
              >
                {savedBusy && <Spinner size={14} />}
                {saved
                  ? `Saved ✓ #${saved.id}${saved.duplicate ? " (already saved)" : ""}`
                  : savedBusy
                    ? "Saving…"
                    : "Save result"}
              </button>
            )}
          </div>

          {/* the three sets — canonical competencies vs market demand */}
          <div className="space-y-3">
            <p className="text-xs font-medium uppercase tracking-wide text-gray-400">
              {method.comparisonLabel} — course vs{" "}
              {res.domains.map((d) => d.replace(/_/g, " ")).join(" + ")} job market
            </p>

            {/* where all the raw course skills landed after consolidation */}
            <p className="pl-1 text-sm text-gray-500">
              {skills.length} course skill{skills.length === 1 ? "" : "s"}
              {method.id === "verified_raw" ? (
                <>
                  {" "}→{" "}
                  <span className="font-medium text-gray-700">
                    {res.competencies?.length || 0} ESCO-anchored skill
                    {(res.competencies?.length || 0) === 1 ? "" : "s"}
                  </span>
                </>
              ) : (
                <>
                  {" "}→{" "}
                  <span className="font-medium text-gray-700">
                    {res.competencies?.length || 0} competenc
                    {(res.competencies?.length || 0) === 1 ? "y" : "ies"}
                  </span>
                </>
              )}{" "}
              ·{" "}
              <span className="font-medium text-emerald-600">
                {res.covered.length} aligned
              </span>{" "}
              to top-{res.top_k} demand,{" "}
              <span className="font-medium text-slate-500">
                {(res.extra_market?.length || 0) + (res.specialized?.length || 0)} not in demand
              </span>
            </p>

            {/* Aligned — canonical competencies that ARE top-demand (the tick group) */}
            {method.id === "verified_raw" ? (
              /* V6: two-column view — what the COURSE teaches vs the MARKET skill it matched */
              <div className="rounded-xl border border-gray-100 border-l-4 border-l-emerald-400 bg-white px-4 py-3">
                <h3 className="text-sm font-semibold text-emerald-700">
                  ✓ Aligned competencies{" "}
                  <span className="text-gray-400">({res.covered.length})</span>
                  <span className="ml-1 font-normal text-gray-400">
                    — taught and in demand
                  </span>
                </h3>
                {res.covered.length ? (
                  <table className="mt-2 w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-gray-400">
                        <th className="w-2/5 py-1 pr-3 font-medium">
                          Course teaches (ESCO unit + topics)
                        </th>
                        <th className="py-1 font-medium">
                          Matched market skill (match % · demand)
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {res.covered.map((c, i) => (
                        <tr
                          key={`${c.skill}-${i}`}
                          className="border-t border-gray-100 align-top"
                          title={[c.reason, c.esco ? `ESCO: ${c.esco}` : ""]
                            .filter(Boolean)
                            .join("\n")}
                        >
                          <td className="py-2 pr-3">
                            <div className="font-medium text-gray-800">
                              {courseNo[c.course_competency]?.length > 0 && (
                                <span className="mr-1.5 rounded bg-gray-100 px-1.5 py-0.5 text-xs font-semibold tabular-nums text-gray-500">
                                  {courseNo[c.course_competency].join(", ")}
                                </span>
                              )}
                              {c.course_competency}
                            </div>
                            {c.evidence?.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {c.evidence.map((e, j) => (
                                  <span
                                    key={j}
                                    className="rounded-full bg-gray-50 px-2 py-0.5 text-xs text-gray-500 ring-1 ring-gray-200"
                                  >
                                    {e}
                                  </span>
                                ))}
                              </div>
                            )}
                          </td>
                          <td className="py-2">
                            <span className="font-medium text-emerald-700">
                              ✓ {c.skill}
                            </span>
                            {typeof c.score === "number" && (
                              <span className="ml-2 text-xs tabular-nums text-gray-500">
                                match {Math.round(c.score * 100)}%
                              </span>
                            )}
                            <span className="ml-2 text-xs tabular-nums text-emerald-500/80">
                              📊{c.demand?.toLocaleString()}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="mt-2 text-sm text-gray-400">None</p>
                )}
              </div>
            ) : (
              <SkillSet
                tone="emerald"
                icon="✓"
                title="Aligned competencies"
                subtitle="taught and in demand"
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
                    unit="techniques"
                  />
                ))}
              </SkillSet>
            )}
            {res.inferred_tools?.length > 0 && (
              <p className="pl-1 text-xs text-gray-400">
                Tools inferred from the syllabus (counted as taught):{" "}
                {res.inferred_tools.join(", ")}
              </p>
            )}

            {/* Market gaps — top-demand concepts the course does NOT cover */}
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

            {/* Off-core — taught competencies that DON'T align to the scored core
                (market-relevant but below top-K, or off-market/foundational) */}
            <SkillSet
              tone="slate"
              icon="•"
              title="Not in demand"
              subtitle={method.offCoreNote(res.top_k)}
              count={(res.extra_market?.length || 0) + (res.specialized?.length || 0)}
              stack
            >
              {(res.extra_market || []).map((x) => (
                <CompetencyRow
                  key={`m-${x.competency}`}
                  tone="slate"
                  icon="•"
                  label={x.competency}
                  evidence={x.evidence}
                  unit="techniques"
                  note={`in market · below top-${res.top_k}`}
                />
              ))}
              {(res.specialized || []).map((sp, i) => (
                <CompetencyRow
                  key={`s-${sp.competency || sp.label || i}`}
                  tone="slate"
                  icon="•"
                  label={sp.competency || sp.label}
                  evidence={sp.evidence}
                  unit="topics"
                  note={method.specializedNote}
                />
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
