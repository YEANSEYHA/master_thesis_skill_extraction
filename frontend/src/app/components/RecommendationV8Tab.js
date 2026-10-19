"use client";

import { Fragment, useState, useEffect, useRef, useCallback } from "react";
import { API_BASE, Spinner, Terminal } from "./ui";

/* ════════════════════════════════════════════════════════════════════════════
   V8 — SEGMENT-MATCHED GAP ANALYSIS (backend/phase5_curriculum_alignment)

   Renders the phase6 pipeline output served via /api/v8/report — the newest
   of the committed dashboard/report_data.js baseline and out/report_data.js (the
   gitignored live output that /api/v8/run regenerates).
   Three sections, same content as phase6's standalone index.html:
     1. Job-market dataset preview (input 1)
     2. Course curriculum preview (input 2)
     3. Gap dashboard: stat tiles → optimized skill plan → coverage heatmap
   Interpretation rules (backend/phase5_curriculum_alignment/readme.md): coverage is max COSINE, never a
   percentage; thresholds are encoder-specific and come from the report.
   ════════════════════════════════════════════════════════════════════════════ */

const STATUS = {
  gap: { icon: "▲", cls: "text-red-600" },
  borderline: { icon: "◆", cls: "text-amber-600" },
  covered: { icon: "●", cls: "text-emerald-600" },
};

/* ── cold-backend handling ────────────────────────────────────────────────────
   A freshly started uvicorn does not bind port 8000 until it has imported the
   BGE encoder and the market cache — about two minutes. Every fetch in that
   window rejects at the network layer, and the tab used to render a bare red
   "Failed to fetch" plus a misleading "No saved courses yet", which reads as a
   broken backend rather than a booting one. Treat a network-level rejection as
   "still waking": amber, self-retrying, no red. Real HTTP errors (4xx/5xx) are
   untouched and still surface immediately.

   A reached-but-failing server is NOT offline, so an error we raised ourselves
   from a non-ok response is tagged `http` and always shown in full — otherwise
   "Load failed (500)" would be swallowed into a silent retry loop. Browsers
   reject genuine network failures with a TypeError (Chrome "Failed to fetch",
   Safari "Load failed"), which is what the instanceof check catches. */
const httpError = (msg) => Object.assign(new Error(msg), { http: true });
const isOffline = (e) =>
  !e?.http &&
  (e instanceof TypeError ||
    /failed to fetch|networkerror|connection refused/i.test(e?.message || ""));
const WAKE_RETRY_MS = 4000;
const WAKE_MAX_TRIES = 45; // ≈3 min, comfortably past a cold boot
const OFFLINE_MSG = `Backend not reachable at ${API_BASE}. Start it with: cd backend/phase1_job_site_discovery && uvicorn api:app --reload --port 8000`;

// PROGRAM-cell verification marks (cross-encoder joint read)
const VERIFY_MARK = {
  SUPPORTED: { glyph: "✓", cls: "text-emerald-600", label: "supported" },
  WEAK: { glyph: "◑", cls: "text-amber-600", label: "weak" },
  "NOT SUPPORTED": { glyph: "×", cls: "text-red-600", label: "demoted" },
};

// Curriculum-relationship badges for the skill plan
const REL_BADGE = {
  content_exists: {
    label: "✓ RELEVANT CONTENT EXISTS",
    cls: "border-emerald-600 text-emerald-700",
  },
  foundation: {
    label: "↗ FOUNDATION EXISTS",
    cls: "border-sky-600 text-sky-700",
  },
  none: {
    label: "＋ NO VERIFIED HOST",
    cls: "border-gray-500 text-gray-600",
  },
  review: {
    label: "? REVIEW REQUIRED",
    cls: "border-violet-500 text-violet-600",
  },
};

// Threshold-aware evidence bands (BGE cosine). Band edges follow the ACTIVE
// thresholds from the report, so recalibration moves them. Fg/bg pairs all
// pass 4.5:1 text contrast (checked with the WCAG formula).
// THREE bands only, one per real decision. The former 0.60 and 0.85 edges
// were display-only sugar with no calibration behind them; the 250-pair human
// study validated borderline (0.6903) and covered (0.7368) but never
// evidenced those two, so they are gone.
function evidenceBands(th) {
  const b = th?.borderline ?? 0.69;
  const c = th?.covered ?? 0.73;
  // show thresholds exactly (calibrated values have 4 decimals); cells stay
  // 2dp — statuses are decided from full precision, not the displayed value
  return [
    { min: c, bg: "#3572B2", fg: "#FFFFFF",
      range: `≥ ${c}`, name: "covered" },
    { min: b, bg: "#F6C85F", fg: "#3D3000",
      range: `${b} – <${c}`,
      name: "borderline" },
    { min: -Infinity, bg: "#F2F4F7", fg: "#344054",
      range: `< ${b}`, name: "not covered" },
  ];
}

function bandStyle(bands, v) {
  const b = bands.find((x) => v >= x.min);
  return { background: b.bg, color: b.fg };
}

// Backward compatibility: old report entries carry only `route` — derive the
// relationship / action objects from it, null-safe throughout.
function relationshipOf(e) {
  if (e.relationship) return e.relationship;
  const rt = e.route || {};
  const base = {
    course: rt.where || null,
    clo: rt.clo || null,
    score: rt.score != null ? rt.score : null,
    addition_required: true,
    evidence_source: "fallback",
  };
  if (rt.kind === "extend")
    return {
      ...base, kind: "content_exists", target_status: "partial",
      reason: "Verified related content exists, but the target skill is " +
        "still below coverage.",
    };
  if (rt.kind === "partial")
    return {
      ...base, kind: "foundation", target_status: "missing",
      reason: "A related foundation exists, but the target skill is not " +
        "explicitly evidenced.",
    };
  return {
    ...base, kind: "none", target_status: "missing",
    reason: "No existing CLO shows verified evidence of teaching this skill.",
  };
}

function actionOf(e, rel) {
  if (e.recommended_action) return e.recommended_action;
  const kind = { content_exists: "extend", foundation: "add_module",
    review: "review_route" }[rel.kind] || "new_content";
  return {
    kind,
    title: `Add ${e.skill} content`,
    detail: rel.course
      ? `Build on ${rel.course}; do not duplicate existing content.`
      : `Introduce a CLO or module that explicitly teaches ${e.skill}.`,
  };
}

/* ── small shared pieces ─────────────────────────────────────────────────── */

function Badge({ label, cls }) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded border px-1.5 py-0.5
                  text-[10px] font-bold tracking-wide ${cls}`}
    >
      {label}
    </span>
  );
}

function SkillChips({ skills, cls }) {
  if (!skills?.length) return <span className="text-gray-400">—</span>;
  return (
    <span className="flex flex-wrap gap-1.5">
      {skills.map((s, i) => (
        <span
          key={i}
          className={`rounded-full px-2 py-0.5 text-[11px] ${
            cls || "bg-sky-50 text-sky-700"
          }`}
        >
          {s}
        </span>
      ))}
    </span>
  );
}

function StatTile({ value, label }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white px-4 py-3">
      <div className="text-2xl font-semibold tabular-nums text-gray-900">
        {value}
      </div>
      <div className="mt-0.5 text-xs text-gray-500">{label}</div>
    </div>
  );
}

function SectionTitle({ children, sub }) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-900">{children}</h2>
      {sub && <p className="mt-0.5 text-sm text-gray-500">{sub}</p>}
    </div>
  );
}

/* ── 1. job-market preview ───────────────────────────────────────────────── */

// The two job-market corpora the V8 gap analysis can run against.
// Cambodia (the local corpus) is the default and leads the selector.
const JOB_SOURCES = [
  {
    key: "cambodia",
    label: "Cambodia Job Postings",
    hint: "Phase-3 crawl, 4,876 postings",
  },
  {
    key: "linkedin",
    label: "LinkedIn (US)",
    hint: "140K US postings — validation baseline",
  },
];

// Standalone corpus picker — lives above the preview so it stays usable even
// when the selected source has no generated report yet (the fetch 404s).
function SourceSelector({ value, onChange }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-gray-500">Job-market source:</span>
      {JOB_SOURCES.map((s) => (
        <button
          key={s.key}
          onClick={() => onChange(s.key)}
          title={s.hint}
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            value === s.key
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {value === s.key ? "✓ " : ""}
          {s.label}
        </button>
      ))}
    </div>
  );
}

function JobPreview({ jobs, total, source, dataset }) {
  const file = dataset || (source === "cambodia"
    ? "stage2_result.json" : "it_job_linkedin.json");
  const dir = source === "cambodia"
    ? "phase3 stage2/api_full" : "01_input_jobmarket";
  return (
    <section className="space-y-3">
      <SectionTitle
        sub={
          <>
            {dir}/{file} — showing {jobs.length} of{" "}
            <span className="font-medium text-gray-700">
              {total.toLocaleString()}
            </span>{" "}
            usable postings (first 4 extracted skills per job)
          </>
        }
      >
        Job-Market Dataset Preview
      </SectionTitle>
      <div className="overflow-hidden rounded-xl border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs text-gray-500">
            <tr>
              <th className="px-4 py-2 font-medium">Job title</th>
              <th className="px-4 py-2 font-medium">Company</th>
              <th className="px-4 py-2 font-medium">Location</th>
              <th className="px-4 py-2 font-medium">Skills (first 4)</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job, i) => (
              <tr key={i} className="border-t border-gray-100 align-top">
                <td className="px-4 py-2 font-medium text-gray-900">
                  {job.job_title || "(untitled)"}
                </td>
                <td className="px-4 py-2 text-gray-600">
                  {job.company_name || <span className="text-gray-400">—</span>}
                </td>
                <td className="px-4 py-2 text-gray-600">
                  {job.location || <span className="text-gray-400">—</span>}
                </td>
                <td className="px-4 py-2">
                  <SkillChips skills={job.skills} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ── 2. curriculum preview ───────────────────────────────────────────────── */

function CourseRow({ course }) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? course.all_skills : course.sample_skills;
  const hidden = course.n_skills - course.sample_skills.length;
  return (
    <tr className="border-t border-gray-100 align-top">
      <td className="px-4 py-2">
        <div className="font-medium text-gray-900">{course.course}</div>
        <div className="text-xs text-gray-400">{course.file}</div>
      </td>
      <td className="px-4 py-2 text-right tabular-nums text-gray-600">
        {course.n_skills}
      </td>
      <td className="px-4 py-2">
        <SkillChips skills={shown} cls="bg-emerald-50 text-emerald-700" />
        {hidden > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            className="mt-1.5 block rounded-md border border-gray-200 px-2 py-0.5
                       text-xs text-gray-500 hover:bg-gray-50"
          >
            {expanded ? "show less ▲" : `show all ${course.n_skills} skills ▼`}
          </button>
        )}
      </td>
    </tr>
  );
}

function CurriculumPreview({ courses }) {
  const totalCLO = courses.reduce((n, c) => n + c.n_skills, 0);
  return (
    <section className="space-y-3">
      <SectionTitle
        sub={
          <>
            02_input_curriculum/clo_skills/ — {courses.length} courses,{" "}
            <span className="font-medium text-gray-700">{totalCLO}</span>{" "}
            curated CLO skills total (first 5 shown per course)
          </>
        }
      >
        Course Curriculum Preview
      </SectionTitle>
      <div className="overflow-hidden rounded-xl border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs text-gray-500">
            <tr>
              <th className="px-4 py-2 font-medium">Course</th>
              <th className="px-4 py-2 text-right font-medium"># skills</th>
              <th className="px-4 py-2 font-medium">Sample skills</th>
            </tr>
          </thead>
          <tbody>
            {courses.map((c, i) => (
              <CourseRow key={i} course={c} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ── 3a. optimized skill plan ────────────────────────────────────────────── */

function EvidenceRow({ k, val }) {
  return (
    <div>
      <span className="font-medium text-gray-700">{k}: </span>
      {val == null || val === "" ? "—" : val}
    </div>
  );
}

function EvidenceDetails({ info, rel }) {
  const v = info?.verify;
  return (
    <div className="rounded-lg bg-gray-50 px-4 py-3 text-xs leading-6 text-gray-600">
      <EvidenceRow k="current cosine coverage"
        val={info?.program != null ? info.program.toFixed(2) : null} />
      <EvidenceRow k="current status" val={info?.status} />
      <EvidenceRow k="closest retrieved CLO"
        val={info?.best_clo ? `"${info.best_clo}" (${info.best_course})` : null} />
      <EvidenceRow
        k="reranker"
        val={
          v
            ? `${v.argmax_rerank != null ? v.argmax_rerank.toFixed(2) : "—"} ` +
              `on closest CLO; best candidate ` +
              `${v.best_rerank != null ? v.best_rerank.toFixed(2) : "—"} — ` +
              v.verdict
            : "no reranker data"
        }
      />
      <EvidenceRow k="relationship decision"
        val={`${rel.kind} · target skill ${rel.target_status}`} />
      <EvidenceRow k="evidence source" val={rel.evidence_source} />
      {rel.reason && <EvidenceRow k="reason" val={rel.reason} />}
    </div>
  );
}

const PROJ_TIP =
  "Projection assumes that an explicit CLO teaching the target skill is added.";

function PlanRow({ e, n, topN, info }) {
  const [open, setOpen] = useState(false);
  const rel = relationshipOf(e);
  const act = actionOf(e, rel);
  const rb = REL_BADGE[rel.kind] || REL_BADGE.none;
  return (
    <>
      <tr className="border-t border-gray-100 align-top">
        <td className="px-4 py-2.5">
          <span
            className="inline-block h-6 w-6 rounded-full bg-gray-900 text-center
                       text-xs font-bold leading-6 text-white"
          >
            {n}
          </span>
        </td>
        <td className="px-4 py-2.5">
          <div className="font-medium text-gray-900">{e.skill}</div>
          <div className="text-xs tabular-nums text-gray-400">
            {e.jobs.toLocaleString()} postings
          </div>
        </td>
        <td className="px-4 py-2.5 tabular-nums text-gray-700">
          {e.gain_pct.toFixed(1)}%
        </td>
        <td className="min-w-[190px] px-4 py-2.5">
          <Badge label={rb.label} cls={rb.cls} />
          {rel.course && (
            <div className="mt-1 text-xs text-gray-600">
              {rel.clo ? `"${rel.clo}" in ${rel.course}` : rel.course}
              {rel.score != null ? ` (${rel.score.toFixed(2)})` : ""}
            </div>
          )}
          {rel.reason && (
            <div className="mt-0.5 text-xs text-gray-400">{rel.reason}</div>
          )}
        </td>
        <td className="min-w-[200px] px-4 py-2.5">
          {rel.addition_required !== false && (
            <Badge label="ADD REQUIRED" cls="border-orange-700 text-orange-700" />
          )}
          <div className="mt-1 text-xs font-semibold text-gray-800">
            {act.title}
          </div>
          {act.detail && (
            <div className="mt-0.5 text-xs text-gray-400">{act.detail}</div>
          )}
        </td>
        <td className="px-4 py-2.5" title={PROJ_TIP}>
          <div className="tabular-nums text-gray-700">
            {e.covered_after}/{topN} covered
          </div>
          <div className="text-xs tabular-nums text-gray-400">
            KL {e.kl_after.toFixed(3)}
          </div>
          <button
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            className="mt-1 rounded-md border border-gray-200 px-2 py-0.5
                       text-xs text-gray-500 hover:bg-gray-50"
          >
            {open ? "hide evidence ▴" : "view evidence ▾"}
          </button>
        </td>
      </tr>
      {open && (
        <tr className="border-t border-gray-100">
          <td className="px-4 py-2" />
          <td className="px-4 py-2" colSpan={5}>
            <EvidenceDetails info={info} rel={rel} />
          </td>
        </tr>
      )}
    </>
  );
}

const PLAN_PREVIEW = 5; // rows shown before "Load more"

function SelectionPlan({ sel, topN, skillsInfo }) {
  const chosen = sel.plan.filter((e) => e.selected);
  const skipped = sel.plan.filter((e) => !e.selected);
  const byName = Object.fromEntries((skillsInfo || []).map((s) => [s.skill, s]));
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? chosen : chosen.slice(0, PLAN_PREVIEW);
  const hiddenCount = chosen.length - visible.length;
  return (
    <div className="space-y-3">
      <p className="text-sm text-gray-500">
        Not every missing skill deserves a course slot. The plan below adds
        skills in the order that closes the most market gap (demand-weighted,
        over the whole job segment) and stops at{" "}
        {Math.round(sel.cum_frac * 100)}% of the achievable improvement — the
        rest cost a slot for under ~1% gain each.
      </p>
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs text-gray-500">
            <tr>
              <th className="px-4 py-2 font-medium">Step</th>
              <th className="px-4 py-2 font-medium">Target skill</th>
              <th className="px-4 py-2 font-medium">Modelled gap closed</th>
              <th className="px-4 py-2 font-medium">Curriculum relationship</th>
              <th className="px-4 py-2 font-medium">Recommended action</th>
              <th className="px-4 py-2 font-medium" title={PROJ_TIP}>
                Projected effect
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((e, i) => (
              <PlanRow key={e.skill} e={e} n={i + 1} topN={topN}
                info={byName[e.skill]} />
            ))}
          </tbody>
        </table>
      </div>

      {chosen.length > PLAN_PREVIEW && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={() => setShowAll((v) => !v)}
            className="rounded-lg border border-gray-300 bg-white px-4 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50"
          >
            {showAll
              ? "Show less"
              : `Load more (${hiddenCount} more skill${hiddenCount === 1 ? "" : "s"})`}
          </button>
        </div>
      )}

      {skipped.length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-3">
          <div className="text-sm font-semibold text-gray-800">
            Not selected — below the benefit cutoff (committee may still
            consider):
          </div>
          <ul className="mt-1.5 space-y-1 text-sm text-gray-500">
            {skipped.map((e) => (
              <li key={e.skill}>
                ▪ {e.skill} ({e.jobs.toLocaleString()} postings) — would close
                only {e.gain_pct.toFixed(1)}% more
                {e.absorbed_by
                  ? `; already largely absorbed after adding "${e.absorbed_by}"`
                  : ""}
                {e.cov_after_plan !== undefined
                  ? ` · coverage after the plan: ${e.cov_after_plan.toFixed(2)}`
                  : ""}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-xs leading-5 text-gray-400">
        Method: greedy demand-weighted coverage maximization
        (facility-location objective, near-optimal guarantee). &quot;Modelled
        gap closed&quot; = share of today&apos;s total demand-weighted gap; KL
        = program alignment after the step. {PROJ_TIP} Target coverage and
        curriculum relationship are reported separately — a related foundation
        does not mean that the target skill is already taught. Relationship
        decisions guide where new content may build on the existing
        curriculum; they do not change the optimized ranking. Gains already
        discount partial coverage (measured above the current cosine, not from
        zero). Grouping into courses = committee step.
      </p>
    </div>
  );
}

/* ── 3b. coverage heatmap ────────────────────────────────────────────────── */

function EvidenceLegend({ bands }) {
  return (
    <div className="space-y-1.5">
      <div className="text-xs font-semibold text-gray-700">
        Semantic evidence (BGE cosine), not percentage.
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {bands
          .slice()
          .reverse()
          .map((b) => (
            <span
              key={b.name}
              className="rounded-md border border-gray-200 px-2 py-0.5 text-[11px]"
              style={{ background: b.bg, color: b.fg }}
            >
              {b.range} {b.name}
            </span>
          ))}
        <span className="text-[11px] text-gray-500">
          · PROGRAM badge:{" "}
          <span className="font-bold text-emerald-600">✓ supported</span> /{" "}
          <span className="font-bold text-amber-600">◑ weak</span> /{" "}
          <span className="font-bold text-red-600">× demoted</span>{" "}
          (cross-encoder)
        </span>
      </div>
    </div>
  );
}

/* Cell inspector — answers "what were these two texts?" for a clicked cell.
   The score is a max over that course's CLOs, so the only honest thing to
   show is the CLO that won it. `clo` is absent on reports generated before
   per_course_clo existed; say so rather than rendering a blank. */
function CellInspector({ pick, bands, thresholds, onClose }) {
  if (!pick) return null;
  const band = bands.find((x) => pick.score >= x.min);
  const isBest = pick.course === pick.bestCourse;
  return (
    <div className="rounded-xl border border-gray-300 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-gray-400">
            {pick.isProgram ? "Programme best match" : "Cell inspector"}
          </div>
          <div className="mt-0.5 text-base font-semibold text-gray-900">
            <span className="text-blue-800">{pick.skill}</span>
            <span className="mx-2 text-gray-400">×</span>
            <span>{pick.course}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="text-2xl font-semibold tabular-nums text-gray-900">
              {pick.score.toFixed(2)}
            </div>
            <div
              className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase"
              style={{ background: band.bg, color: band.fg }}
            >
              {band.name}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close inspector"
            className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-500
                       hover:bg-gray-50"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
            Market skill
          </div>
          <div className="mt-1 text-sm text-gray-900">{pick.skill}</div>
          <div className="mt-1 text-xs text-gray-500">
            {pick.jobs.toLocaleString()} job postings ask for it
          </div>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
            Closest CLO in {pick.course}
          </div>
          {pick.clo ? (
            <div className="mt-1 text-sm text-gray-900">“{pick.clo}”</div>
          ) : (
            <div className="mt-1 text-sm italic text-gray-400">
              not stored in this report — regenerate it to record which CLO
              matched
            </div>
          )}
        </div>
      </div>

      {!pick.isProgram && !isBest && (
        <div className="mt-3 text-xs leading-5 text-gray-500">
          Strongest match anywhere in the programme is{" "}
          <span className="font-semibold text-gray-700">
            {pick.program.toFixed(2)}
          </span>{" "}
          in <span className="font-semibold text-gray-700">{pick.bestCourse}</span>{" "}
          (“{pick.bestClo}”), which is what the PROGRAM column and the status
          label use.
        </div>
      )}
      {pick.isProgram && pick.verify && (
        <div className="mt-3 text-xs leading-5 text-gray-500">
          Cross-encoder joint read:{" "}
          <span className="font-semibold text-gray-700">
            {pick.verify.argmax_rerank.toFixed(2)} ({pick.verify.verdict})
          </span>
          . Relevance is a separate scale from cosine — never compare the two
          numbers directly.
        </div>
      )}

      <div className="mt-3 border-t border-gray-100 pt-2 text-xs leading-5 text-gray-400">
        This score is the highest cosine similarity between the market skill and
        any CLO in {pick.course} — semantic evidence, not a percentage, and it
        means the CLO <em>addresses</em> the skill rather than teaches it to
        competence. At or above {thresholds.covered} counts as covered,{" "}
        {thresholds.borderline}–{(thresholds.covered - 0.01).toFixed(2)} as
        borderline.
      </div>
    </div>
  );
}

const HEATMAP_TOPS = [20, 25, 30]; // selectable "show top-N skills" options

function Heatmap({ report }) {
  const bands = evidenceBands(report.thresholds);
  const cellCls =
    "w-16 cursor-pointer rounded border-2 border-white px-1 py-1.5 text-center " +
    "text-xs tabular-nums hover:!border-gray-900";
  const available = report.skills.length;
  const [topShow, setTopShow] = useState(25);
  // clicked cell -> inspector panel. null = closed.
  const [pick, setPick] = useState(null);
  const shown = report.skills.slice(0, topShow);
  const openCell = (s, course) =>
    setPick({
      skill: s.skill,
      jobs: s.jobs,
      status: s.status,
      course: course ?? s.best_course,
      score: course ? s.per_course[course] : s.program,
      // per_course_clo is a newer field — older reports simply lack it
      clo: course ? s.per_course_clo?.[course] : s.best_clo,
      isProgram: !course,
      bestCourse: s.best_course,
      bestClo: s.best_clo,
      program: s.program,
      verify: s.verify,
    });
  return (
    <section className="space-y-3">
      <SectionTitle>Coverage heatmap — skills × courses</SectionTitle>
      <EvidenceLegend bands={bands} />
      <div className="flex items-center gap-2 text-xs">
        <span className="font-medium text-gray-500">Show top:</span>
        {HEATMAP_TOPS.map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => setTopShow(n)}
            className={`rounded-lg border px-3 py-1 font-medium transition-colors ${
              topShow === n
                ? "border-gray-900 bg-gray-900 text-white"
                : "border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            {n}
          </button>
        ))}
        <span className="text-gray-400">
          showing {shown.length} of {available} skills
          {topShow > available
            ? ` — only ${available} in this report (raise TOP_N to see ${topShow})`
            : ""}
        </span>
      </div>
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs text-gray-500">
            <tr>
              <th className="sticky left-0 z-10 bg-gray-50 px-4 py-2 font-medium">
                Market skill
              </th>
              <th className="px-4 py-2 text-right font-medium">Jobs</th>
              {report.courses.map((c) => (
                <th key={c} className="px-2 py-2 text-center font-medium">
                  {c}
                </th>
              ))}
              <th className="border-l-2 border-gray-300 px-2 py-2 text-center font-medium">
                PROGRAM
              </th>
              <th className="px-4 py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((s) => {
              const st = STATUS[s.status];
              const vm = s.verify && VERIFY_MARK[s.verify.verdict];
              return (
                <tr key={s.skill} className="border-t border-gray-100">
                  <td className="sticky left-0 z-10 whitespace-nowrap bg-white px-4 py-1.5 font-medium text-gray-900">
                    {s.skill}
                  </td>
                  <td className="px-4 py-1.5 text-right tabular-nums text-gray-600">
                    {s.jobs.toLocaleString()}
                  </td>
                  {report.courses.map((c) => (
                    <td
                      key={c}
                      className={cellCls}
                      style={bandStyle(bands, s.per_course[c])}
                      title={`${s.skill} × ${c}: ${s.per_course[c]} — click to inspect`}
                      onClick={() => openCell(s, c)}
                    >
                      {s.per_course[c].toFixed(2)}
                    </td>
                  ))}
                  <td
                    className={`${cellCls} border-l-2 !border-l-gray-300`}
                    style={bandStyle(bands, s.program)}
                    onClick={() => openCell(s, null)}
                    title={
                      `best CLO [${s.best_course}]: ${s.best_clo}` +
                      (s.verify
                        ? ` · joint-read ${s.verify.argmax_rerank.toFixed(2)} (${s.verify.verdict})`
                        : "") +
                      " — click to inspect"
                    }
                  >
                    {s.program.toFixed(2)}
                    {vm && (
                      <span
                        className={`ml-1 inline-block h-4 w-4 rounded-full border
                                    border-gray-200 bg-white/95 text-center text-[10px]
                                    font-bold leading-4 ${vm.cls}`}
                      >
                        {vm.glyph}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-1.5">
                    <span className={`font-semibold ${st.cls}`}>
                      {st.icon} {s.status}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <CellInspector
        pick={pick}
        bands={bands}
        thresholds={report.thresholds}
        onClose={() => setPick(null)}
      />
      <p className="text-xs leading-5 text-gray-400">
        Click any cell to see the market skill and the CLO it was compared
        against. Cells show the strongest semantic evidence found in each course. The
        PROGRAM column is the primary coverage result. Scores are cosine
        similarities, not percentages. · status: ▲ gap &lt;{" "}
        {report.thresholds.borderline} ≤ ◆ borderline &lt;{" "}
        {report.thresholds.covered} ≤ ● covered · hosts are only named when
        coverage ≥ {report.thresholds.host_min}
        {" (below that: “new module needed”) "}
        {report.reranker
          ? `· host suggestions verified by cross-encoder ${report.reranker} ` +
            `(✓ confirmed / × demoted; relevance is a separate scale from ` +
            `cosine, bands ≥ ${report.rerank_bands.supported} supported / ≥ ` +
            `${report.rerank_bands.weak} weak, provisional) `
          : ""}
        · dataset: {report.dataset}
      </p>
    </section>
  );
}

/* ── 3. gap dashboard ────────────────────────────────────────────────────── */

function GapDashboard({ report }) {
  // older report_data bundles predate the demand_weighted_coverage field —
  // re-derive it from the per-skill rows so cached runs display correctly
  const topNJobs = report.skills.reduce((s, x) => s + x.jobs, 0);
  const coveredJobs = report.skills.reduce(
    (s, x) => s + (x.status === "covered" ? x.jobs : 0),
    0,
  );
  const dwc =
    report.demand_weighted_coverage ??
    (topNJobs ? coveredJobs / topNJobs : 0);
  return (
    <div className="space-y-6">
      <SectionTitle
        sub={
          <>
            {report.courses.length} courses ({report.n_clo_skills} CLO skills)
            vs {report.n_jobs.toLocaleString()} relevant postings — top{" "}
            {report.top_n} technical market skills
          </>
        }
      >
        Gap Analysis Dashboard
      </SectionTitle>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          value={`${(dwc * 100).toFixed(1)}%`}
          label="market demand covered (demand-weighted)"
        />
        <StatTile
          value={`${report.covered} / ${report.top_n}`}
          label="top demanded skills covered by the program"
        />
        <StatTile
          value={
            report.selection
              ? `${report.selection.n_selected} / ${report.selection.plan.length}`
              : report.recommendations.length
          }
          label="gaps worth adding (of all gaps found)"
        />
        <StatTile
          value={report.kl.toFixed(3)}
          label={
            report.source === "cambodia"
              ? "KL divergence (distributional diagnostic; null = 0.399)"
              : "KL divergence (distributional diagnostic)"
          }
        />
      </div>

      <div className="rounded-2xl border border-gray-200 border-l-4 border-l-gray-900 bg-white px-5 py-4 shadow-sm">
        <h3 className="text-base font-semibold text-gray-900">
          Recommended actions — skills to add, optimized order
        </h3>
        <div className="mt-2">
          {report.selection ? (
            <SelectionPlan
              sel={report.selection}
              topN={report.top_n}
              skillsInfo={report.skills}
            />
          ) : (
            <p className="text-sm text-gray-500">
              no selection data — rerun gen_report_data.py
            </p>
          )}
        </div>
      </div>

      <Heatmap report={report} />
    </div>
  );
}

/* ── tab root ────────────────────────────────────────────────────────────── */

export default function RecommendationV8Tab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  // "run" by default: the live pipeline is the working path now; "demo"
  // reads back the last saved report without executing anything
  const [mode, setMode] = useState("run"); // "demo" | "run"
  // Which job-market corpus drives the gap analysis: "linkedin" (US baseline)
  // or "cambodia" (Phase-3 Cambodian crawl). Refetches the whole report.
  const [jobSource, setJobSource] = useState("cambodia");
  // Step 2 — which curriculum feeds the pipeline run
  // "db" by default: uploaded syllabi are the working path now, the curated
  // clo_skills/ files are the frozen thesis baseline you opt back into
  const [courseSel, setCourseSel] = useState({ source: "db", courseIds: [] });
  const [coursesTick, setCoursesTick] = useState(0); // bumped when Step 1 saves

  // Demo mode pins the committed baseline report; run mode shows the last
  // live pipeline run (falls back to baseline if none exists yet).
  const [waking, setWaking] = useState(false); // backend booting, auto-retrying
  const wakeTries = useRef(0); // guard (sync); the state below only displays it
  const [wakeAttempt, setWakeAttempt] = useState(0);
  const wakeTimer = useRef(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const which = mode === "run" ? "live" : "baseline";
      const res = await fetch(
        `${API_BASE}/api/v8/report?which=${which}&source=${jobSource}`
      );
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw httpError(detail.detail || `Load failed (${res.status})`);
      }
      setData(await res.json());
      setWaking(false);
      wakeTries.current = 0;
    } catch (e) {
      // Network-level failure = the backend has not bound its port yet.
      // Keep waiting quietly instead of shouting an error at a cold start.
      if (isOffline(e) && wakeTries.current < WAKE_MAX_TRIES) {
        wakeTries.current += 1;
        setWakeAttempt(wakeTries.current);
        setWaking(true);
        setError("");
        wakeTimer.current = setTimeout(load, WAKE_RETRY_MS);
      } else {
        setWaking(false);
        setError(isOffline(e) ? OFFLINE_MSG : e.message || "Load failed");
      }
    } finally {
      setLoading(false);
    }
  }, [mode, jobSource]);

  // Drop any pending wake-retry when the query changes or the tab unmounts,
  // so a stale timer cannot refetch the previous mode/source.
  useEffect(() => {
    wakeTries.current = 0;
    setWakeAttempt(0);
    load();
    return () => clearTimeout(wakeTimer.current);
  }, [load]);

  const retry = () => {
    wakeTries.current = 0;
    setWakeAttempt(0);
    load();
  };

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
          Curriculum–Market Gap Analysis
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Segment-matched: ITC curriculum vs labour-market demand → coverage, KL
          alignment, and a ranked plan of skills to add · data from backend/phase5_curriculum_alignment
        </p>
      </div>

      {/* Demo (saved report) vs Real experiment (run the pipeline live) */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-gray-500">Mode:</span>
        <button
          onClick={() => setMode("demo")}
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            mode === "demo"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {mode === "demo" ? "✓ " : ""}Demo (saved report)
        </button>
        <button
          onClick={() => setMode("run")}
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            mode === "run"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {mode === "run" ? "✓ " : ""}Real experiment (run pipeline)
        </button>
      </div>

      {/* Which job-market corpus feeds the gap analysis */}
      {mode === "demo" && (
        <SourceSelector value={jobSource} onChange={setJobSource} />
      )}

      {mode === "run" && (
        <>
          {/* Step 1 saves courses; Step 2 lists them — the tick tells Step 2 to
              refetch instead of showing a stale "no saved courses yet" */}
          <PdfMarkdownRow
            onCoursesChanged={() => setCoursesTick((t) => t + 1)}
          />
          <CourseSelectRow
            value={courseSel}
            onChange={setCourseSel}
            reloadKey={coursesTick}
          />
          <LiveRun selection={courseSel} jobSource={jobSource} onDone={load} />
        </>
      )}

      {loading && !waking && (
        <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-6 text-sm text-gray-500">
          <Spinner size={16} /> Loading phase6 report…
        </div>
      )}

      {waking && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-800">
          <Spinner size={16} />
          <div className="space-y-1">
            <p className="font-medium">
              Waiting for the backend to finish starting…
            </p>
            <p className="text-amber-700">
              A cold start loads the BGE encoder and the market cache before it
              accepts requests — usually under two minutes. Retrying
              automatically ({wakeAttempt}/{WAKE_MAX_TRIES}).
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="space-y-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <p className="whitespace-pre-line">{error}</p>
          <button
            onClick={retry}
            className="rounded-md bg-white px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-red-200 hover:bg-red-100"
          >
            Retry
          </button>
        </div>
      )}

      {data && !loading && (
        <>
          {mode === "demo" && (
            <>
              <JobPreview
                jobs={data.preview.jobs}
                total={data.preview.job_total}
                source={jobSource}
                dataset={data.report?.dataset}
              />
              <CurriculumPreview courses={data.preview.courses} />
            </>
          )}
          {data.report ? (
            <>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ring-1 ${
                    data.generated === "live"
                      ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
                      : "bg-gray-100 text-gray-600 ring-gray-200"
                  }`}
                >
                  {data.generated === "live"
                    ? "● live run output"
                    : "● committed baseline"}
                </span>
                <span>
                  {data.generated === "live"
                    ? "latest /api/v8/run result (gitignored out/report_data.js)"
                    : "backend/phase5_curriculum_alignment/dashboard/report_data.js"}
                </span>
              </div>
              <GapDashboard report={data.report} />
            </>
          ) : (
            <p className="text-sm text-gray-500">
              No gap report found — run the pipeline in Real experiment mode.
            </p>
          )}
        </>
      )}
    </div>
  );
}

/* ── real-experiment step 1: drop a syllabus (PDF/.docx) → extract Markdown ── */

/** curriculum_documents row → queue item. markdown and the skill list are left
 *  null and fetched only when a row is expanded: the list endpoint omits them
 *  because markdown runs ~100 KB per syllabus. */
function docToItem(doc) {
  return {
    id: `doc:${doc.id}`,
    docId: doc.id,
    name: doc.filename,
    chars: doc.chars || 0,
    status: doc.status,
    r2Status: doc.r2_status,
    course: doc.course || null,
    markdown: null,
    skills: null,
    savedInfo: doc.course?.id ? { id: doc.course.id, created: false } : null,
    file: null,
    work: "",
    error: doc.error || "",
    showMd: false,
    cached: false,
  };
}

/** converted, but no course extracted from it yet */
const needsSkills = (x) => !!x.docId && !!x.chars && !x.course && !x.skills;

function PdfMarkdownRow({ onCoursesChanged }) {
  // One entry per syllabus — ITC issues ~39 course specs, so the step is a
  // queue, and each item carries its own error so one bad file never blocks the
  // batch. Items are backed by curriculum_documents rows: the queue is seeded
  // from GET /documents on mount, so a refresh (or another machine) restores
  // everything without re-uploading, re-converting or re-running the LLM.
  // item = {id, docId, name, chars, status, r2Status, course, markdown, skills,
  //         savedInfo, file, work, error, showMd}
  const [items, setItems] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(""); // "" | "markdown" | "skills" | "save"
  const [error, setError] = useState(""); // queue-level (rejected files)
  const [savedCourses, setSavedCourses] = useState([]); // rows from Postgres
  const [loadingSavedId, setLoadingSavedId] = useState(null);
  const [llm, setLlm] = useState(null); // {model, provider} — PHASE5_LLM_MODEL
  const [showDone, setShowDone] = useState(false); // finished files stay folded

  // The batch runners walk a ref snapshot, not `items`: they await between
  // files and each per-file patch would otherwise hand them a stale array.
  const itemsRef = useRef(items);
  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  const refreshSaved = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/curricula/course-skills`);
      if (res.ok) setSavedCourses((await res.json()).courses || []);
    } catch {
      /* backend down — picker just stays empty */
    }
  }, []);

  // Rebuild the queue from the document store. Runs once on mount; this is what
  // makes "upload everything again" unnecessary.
  const loadQueue = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/curricula/documents`);
      if (!res.ok) return;
      const docs = (await res.json()).documents || [];
      setItems(docs.map(docToItem));
    } catch {
      /* backend down — queue starts empty, uploads still work once it's up */
    }
  }, []);

  useEffect(() => {
    refreshSaved();
    loadQueue();
    fetch(`${API_BASE}/api/curricula/extraction-model`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setLlm)
      .catch(() => {});
  }, [refreshSaved, loadQueue]);

  const patch = useCallback((id, p) => {
    setItems((cur) => cur.map((it) => (it.id === id ? { ...it, ...p } : it)));
  }, []);

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    const ok = (f) => /\.(pdf|docx)$/i.test(f.name);
    const rejected = incoming.filter((f) => !ok(f));
    setError(
      rejected.length
        ? `Skipped ${rejected.length} unsupported file${
            rejected.length > 1 ? "s" : ""
          } (PDF or Word .docx only): ${rejected.map((f) => f.name).join(", ")}`
        : ""
    );
    setItems((cur) => {
      const seen = new Set(cur.map((it) => it.id));
      const fresh = [];
      for (const f of incoming.filter(ok)) {
        const id = `${f.name}:${f.size}:${f.lastModified}`;
        if (seen.has(id)) continue; // same file dropped twice
        seen.add(id);
        fresh.push({
          id,
          docId: null,
          name: f.name,
          size: f.size,
          chars: 0,
          status: "queued",
          r2Status: null,
          course: null,
          markdown: null,
          skills: null,
          savedInfo: null,
          file: f,
          work: "",
          error: "",
          showMd: false,
        });
      }
      return [...cur, ...fresh];
    });
  }

  // Upload → the backend hashes the bytes and decides what actually has to run.
  // A file already in the store comes back `cached: true` having done nothing.
  async function convertItem(id) {
    const it = itemsRef.current.find((x) => x.id === id);
    if (!it?.file) return;
    patch(id, { work: "converting", error: "" });
    try {
      const form = new FormData();
      form.append("file", it.file);
      const res = await fetch(`${API_BASE}/api/curricula/documents`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Upload failed (${res.status})`);
      }
      const doc = await res.json();
      // The same document may already sit in the queue under a different local
      // id (seeded from the DB, or dropped twice under two filenames) — the
      // sha256 only reveals that server-side, so collapse the duplicate here.
      setItems((cur) => {
        const dupe = cur.find((x) => x.id !== id && x.docId === doc.id);
        const merged = { ...docToItem(doc), work: "", cached: !!doc.cached };
        return dupe
          ? cur
              .filter((x) => x.id !== id)
              .map((x) => (x.docId === doc.id ? { ...x, ...merged } : x))
          : cur.map((x) => (x.id === id ? { ...merged, id } : x));
      });
    } catch (e) {
      patch(id, { work: "", error: e.message || "Upload failed" });
    }
  }

  // Reads the markdown from Postgres server-side — the uploaded file is never
  // touched again — and persists the skills in the same call.
  async function skillsForItem(id) {
    const it = itemsRef.current.find((x) => x.id === id);
    if (!it?.docId) return;
    patch(id, { work: "extracting", error: "" });
    try {
      const res = await fetch(
        `${API_BASE}/api/curricula/documents/${it.docId}/extract-skills`,
        { method: "POST" }
      );
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Skill extraction failed (${res.status})`);
      }
      const out = await res.json();
      patch(id, {
        work: "",
        skills: { course: out.course, skills: out.skills },
        savedInfo: out.saved,
        status: "extracted",
        course: {
          id: out.saved?.id,
          title: out.course?.title,
          code: out.course?.code,
          category: out.course?.category,
          skills_count: (out.skills || []).length,
        },
        showMd: false,
      });
      refreshSaved();
      onCoursesChanged?.(); // the course is now selectable in Step 2
    } catch (e) {
      patch(id, { work: "", error: e.message || "Skill extraction failed" });
    }
  }

  // Markdown and skill lists are fetched only when a row is expanded — the list
  // endpoint deliberately omits them (markdown is ~100 KB per syllabus).
  async function toggleMarkdown(id) {
    const it = itemsRef.current.find((x) => x.id === id);
    if (!it) return;
    if (it.markdown != null || !it.docId) {
      patch(id, { showMd: !it.showMd });
      return;
    }
    patch(id, { work: "loading", error: "" });
    try {
      const res = await fetch(`${API_BASE}/api/curricula/documents/${it.docId}`);
      if (!res.ok) throw new Error(`Load failed (${res.status})`);
      const doc = await res.json();
      patch(id, { work: "", markdown: doc.markdown || "", showMd: true });
    } catch (e) {
      patch(id, { work: "", error: e.message || "Load failed" });
    }
  }

  async function showSkills(id) {
    const it = itemsRef.current.find((x) => x.id === id);
    if (!it?.course?.id) return;
    patch(id, { work: "loading", error: "" });
    try {
      const res = await fetch(
        `${API_BASE}/api/curricula/course-skills/${it.course.id}`
      );
      if (!res.ok) throw new Error(`Load failed (${res.status})`);
      const row = await res.json();
      patch(id, {
        work: "",
        skills: {
          course: {
            code: row.course_code,
            title: row.course_title,
            category: row.category,
          },
          skills: row.skills || [],
        },
        savedInfo: { id: row.id, created: false },
      });
    } catch (e) {
      patch(id, { work: "", error: e.message || "Load failed" });
    }
  }

  // Batches run one file at a time on purpose: skill extraction is a ~1–2 min
  // qwen3 call, and firing 39 of them at Ollama at once would just thrash it.
  async function runBatch(stage, pick, run) {
    setBusy(stage);
    for (const it of itemsRef.current.filter(pick)) await run(it.id);
    setBusy("");
  }

  const convertAll = () =>
    runBatch("markdown", (x) => x.file && !x.docId, convertItem);
  const skillsAll = () =>
    runBatch("skills", needsSkills, skillsForItem);

  // Removing drops the curriculum_documents row so it does not come back on the
  // next refresh. The R2 object survives — re-uploading the same file restores
  // the row without another archive round trip.
  async function removeItem(id) {
    const it = itemsRef.current.find((x) => x.id === id);
    setItems((cur) => cur.filter((x) => x.id !== id));
    if (!it?.docId) return;
    try {
      await fetch(`${API_BASE}/api/curricula/documents/${it.docId}`, {
        method: "DELETE",
      });
    } catch {
      /* list refresh will show the truth */
    }
  }

  async function loadSaved(id) {
    setLoadingSavedId(id);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/curricula/course-skills/${id}`);
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Load failed (${res.status})`);
      }
      const row = await res.json();
      // Courses saved before the document store exist without a file behind
      // them (document_id NULL) — they open read-only: markdown and skills, no
      // convert/extract buttons.
      const loaded = {
        id: `db:${row.id}`,
        docId: null,
        name: row.filename || row.course_title,
        size: 0,
        chars: (row.markdown || "").length,
        status: "extracted",
        r2Status: null,
        course: {
          id: row.id,
          code: row.course_code,
          title: row.course_title,
          category: row.category,
          skills_count: (row.skills || []).length,
        },
        markdown: row.markdown || "",
        skills: {
          course: {
            code: row.course_code,
            title: row.course_title,
            category: row.category,
          },
          skills: row.skills || [],
        },
        savedInfo: { id: row.id, created: false },
        file: null,
        work: "",
        error: "",
        showMd: false,
      };
      setItems((cur) => [...cur.filter((x) => x.id !== loaded.id), loaded]);
    } catch (e) {
      setError(e.message || "Load failed");
    } finally {
      setLoadingSavedId(null);
    }
  }

  async function deleteSaved(id) {
    try {
      await fetch(`${API_BASE}/api/curricula/course-skills/${id}`, {
        method: "DELETE",
      });
      setItems((cur) =>
        cur
          .filter((x) => x.id !== `db:${id}`)
          .map((x) => (x.savedInfo?.id === id ? { ...x, savedInfo: null } : x))
      );
      refreshSaved();
      onCoursesChanged?.();
    } catch {
      /* list refresh will show the truth */
    }
  }

  const nToConvert = items.filter((x) => x.file && !x.docId).length;
  const nToSkills = items.filter(needsSkills).length;
  const nStored = items.filter((x) => x.docId).length;
  // "done" = stored, converted AND a course extracted from it. Those need no
  // decision, so they collapse; everything else stays a full card.
  const isDone = (x) => !!x.docId && !!x.chars && !!x.course && !x.error;
  const pending = items.filter((x) => !isDone(x));
  const done = items.filter(isDone);
  const doneSkills = done.reduce((n, x) => n + (x.course?.skills_count || 0), 0);
  // Courses with no document behind them (saved before the document store, or
  // whose syllabus row was deleted) — the only ones the chip list still needs.
  const docCourseIds = new Set(items.map((x) => x.course?.id).filter(Boolean));
  const orphanCourses = savedCourses.filter((c) => !docCourseIds.has(c.id));

  return (
    <section className="space-y-3 rounded-2xl border border-gray-200 bg-white px-5 py-4 shadow-sm">
      <div>
        <h2 className="text-base font-semibold text-gray-900">
          Step 1 — curriculum input: syllabus PDF / Word → Markdown
        </h2>
        <p className="mt-0.5 text-xs text-gray-500">
          drop one or many course syllabi (PDF or Word .docx — the format ITC
          issues) → original archived to R2, Markdown stored in PostgreSQL →
          extract skills — saved courses become selectable in Step 2 below.{" "}
          <span className="text-gray-400">
            uploaded once, ever: the queue below is restored from the database on
            every visit, and re-dropping a stored file costs nothing
          </span>
          {nStored > 0 && (
            <span className="font-medium text-gray-600">
              {" "}
              · {nStored} stored
            </span>
          )}
        </p>
      </div>

      <div className="flex flex-wrap items-stretch gap-3">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`flex min-w-[240px] flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl
                      border-2 border-dashed px-4 py-3 text-center transition-colors
                      ${
                        dragging
                          ? "border-gray-900 bg-gray-100"
                          : "border-gray-300 bg-gray-50 hover:bg-gray-100"
                      }`}
        >
          <input
            type="file"
            multiple
            accept="application/pdf,.pdf,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = ""; // so the same file can be re-picked after removal
            }}
          />
          <svg
            className="h-5 w-5 shrink-0 text-gray-400"
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
          <span className="text-sm font-medium text-gray-700">
            {/* "queued" means waiting to upload — rows restored from the DB are
                already done and must not be counted as pending work */}
            {nToConvert > 0
              ? `${nToConvert} waiting to upload${
                  nStored ? ` · ${nStored} already stored` : ""
                } — drop more, or click to add`
              : nStored > 0
              ? `${nStored} syllab${nStored > 1 ? "i" : "us"} stored — drop more, or click to add`
              : "Drag & drop syllabi (PDF or Word .docx — several at once), or click to choose"}
          </span>
        </label>
        <button
          onClick={convertAll}
          disabled={!nToConvert || !!busy}
          className="inline-flex items-center gap-2 self-center rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white
                     hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {busy === "markdown" && <Spinner size={14} />}
          {busy === "markdown"
            ? "Uploading…"
            : `Upload & Extract${nToConvert > 1 ? ` (${nToConvert})` : ""}`}
        </button>
      </div>

      {/* batch actions — the whole queue in one click, one file at a time */}
      {nToSkills > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={skillsAll}
            disabled={!!busy}
            className="inline-flex items-center gap-1.5 rounded-lg bg-gray-900 px-3 py-1.5 font-medium text-white
                       hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            {busy === "skills" && <Spinner size={12} />}
            {busy === "skills"
              ? "Extracting skills…"
              : `Extract skills for all (${nToSkills})`}
          </button>
          <span className="text-gray-400">
            {llm ? (
              <>
                model{" "}
                <span className="font-mono text-gray-600">{llm.model}</span>{" "}
                ({llm.provider === "openai" ? "~20s/course" : "~1–2 min/course"},
                set PHASE5_LLM_MODEL in backend/.env) ·{" "}
              </>
            ) : null}
            saves to PostgreSQL automatically
          </span>
        </div>
      )}

      {/* Only courses with no syllabus behind them — everything else is already
          shown on its own file row, and duplicating it was just noise. */}
      {orphanCourses.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-[11px] font-medium uppercase tracking-wide text-gray-400">
            Saved without a source file — older uploads
          </div>
          <div className="flex flex-wrap gap-1.5">
            {orphanCourses.map((c) => (
              <span
                key={c.id}
                className="inline-flex items-center gap-1 rounded-full bg-white py-1 pr-1 pl-2.5 text-xs text-gray-700 ring-1 ring-gray-200"
              >
                <button
                  onClick={() => loadSaved(c.id)}
                  disabled={loadingSavedId !== null}
                  title={`${c.filename || ""} · ${c.skills_count} skills · saved ${c.updated_at}`}
                  className="font-medium hover:underline disabled:cursor-wait"
                >
                  {loadingSavedId === c.id ? "loading…" : c.course_title}
                  {c.course_code && (
                    <span className="ml-1 font-mono text-[10px] text-gray-400">
                      {c.course_code}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => deleteSaved(c.id)}
                  title="delete from DB"
                  className="rounded-full px-1 text-gray-400 hover:bg-red-50 hover:text-red-600"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Anything still needing a decision gets a full card … */}
      {pending.map((it) => (
        <SyllabusItem
          key={it.id}
          item={it}
          busy={!!busy}
          onConvert={() => convertItem(it.id)}
          onSkills={() => skillsForItem(it.id)}
          onShowSkills={() => showSkills(it.id)}
          onToggleMd={() => toggleMarkdown(it.id)}
          onRemove={() => removeItem(it.id)}
        />
      ))}

      {/* … everything finished collapses to one line, because on a return visit
          the whole corpus is done and eight tall cards is all noise. */}
      {done.length > 0 && (
        <div className="rounded-xl border border-gray-200">
          <button
            onClick={() => setShowDone((v) => !v)}
            className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left text-xs hover:bg-gray-50"
          >
            <span className="text-gray-600">
              <span className="font-medium text-emerald-700">✓ {done.length} done</span>
              {" — "}
              {doneSkills.toLocaleString()} skills extracted, archived to R2,
              selectable in Step 2
            </span>
            <span className="shrink-0 font-medium text-gray-500">
              {showDone ? "hide files ▴" : "show files ▾"}
            </span>
          </button>
          {showDone && (
            <div className="border-t border-gray-200">
              {done.map((it) => (
                <SyllabusRow
                  key={it.id}
                  item={it}
                  busy={!!busy}
                  onSkills={() => skillsForItem(it.id)}
                  onShowSkills={() => showSkills(it.id)}
                  onToggleMd={() => toggleMarkdown(it.id)}
                  onRemove={() => removeItem(it.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/* ── a finished syllabus, one compact line (expandable) ─────────────────────── */

function SyllabusRow({ item, busy, onSkills, onShowSkills, onToggleMd, onRemove }) {
  const { docId, chars, course, skills, savedInfo, work } = item;
  const locked = busy || !!work;
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(t);
  }, [armed]);

  const act = "rounded px-1.5 py-0.5 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className="border-b border-gray-100 last:border-b-0">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-xs">
        <a
          href={`${API_BASE}/api/curricula/documents/${docId}/file`}
          target="_blank"
          rel="noreferrer"
          title="download the original from R2"
          className="min-w-0 flex-1 truncate font-medium text-gray-700 hover:underline"
        >
          {item.name}
        </a>
        <span className="w-40 shrink-0 truncate text-gray-600">
          {course?.title || "—"}
        </span>
        <span className="w-16 shrink-0 text-right tabular-nums text-gray-500">
          {course?.skills_count ? `${course.skills_count} sk` : "—"}
        </span>
        <span className="w-20 shrink-0 text-right tabular-nums text-gray-400">
          {chars.toLocaleString()}
        </span>
        <span className="flex shrink-0 items-center gap-0.5 text-gray-500">
          <button onClick={onToggleMd} disabled={locked} className={act}>
            {item.showMd ? "hide md" : "md"}
          </button>
          {course && !skills && (
            <button onClick={onShowSkills} disabled={locked} className={act}>
              skills
            </button>
          )}
          <button onClick={onSkills} disabled={locked} className={act}>
            {work === "extracting" ? "extracting…" : "re-extract"}
          </button>
          <button
            onClick={() => (armed ? onRemove() : setArmed(true))}
            disabled={locked}
            title="delete this syllabus from PostgreSQL (the R2 original is kept)"
            className={
              armed
                ? "rounded bg-red-50 px-1.5 py-0.5 font-medium text-red-600 ring-1 ring-red-300"
                : `${act} text-gray-400 hover:text-red-600`
            }
          >
            {armed ? "delete?" : "×"}
          </button>
        </span>
      </div>

      {item.error && (
        <div className="mx-4 mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {item.error}
        </div>
      )}
      {item.showMd && item.markdown != null && (
        <pre className="mx-4 mb-2 max-h-72 overflow-auto rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 font-mono text-[11px] leading-5 whitespace-pre-wrap text-gray-800">
          {item.markdown}
        </pre>
      )}
      {skills && (
        <div className="mx-4 mb-2">
          <SkillsPreview data={skills} savedInfo={savedInfo} />
        </div>
      )}
    </div>
  );
}

/* ── one stored syllabus: upload → skills, with its own status + error line ── */

const R2_NOTE = {
  stored: { text: "archived", cls: "text-gray-400" },
  failed: { text: "R2 archive failed — re-drop the file to retry", cls: "text-amber-600" },
  skipped: { text: "not archived (R2 not configured)", cls: "text-gray-400" },
  pending: { text: "archive pending", cls: "text-gray-400" },
};

function SyllabusItem({
  item,
  busy,
  onConvert,
  onSkills,
  onShowSkills,
  onToggleMd,
  onRemove,
}) {
  const { docId, chars, skills, course, savedInfo, work } = item;
  const locked = busy || !!work; // a batch is running, or this row is mid-call
  const stored = !!docId && !!chars;
  const r2 = item.r2Status ? R2_NOTE[item.r2Status] : null;
  // Removing a stored row DELETES it from PostgreSQL — and since the queue is
  // rebuilt from the DB, there is no "hide locally" to fall back on. So the
  // button arms first and deletes on the second click.
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(t);
  }, [armed]);

  return (
    <div className="space-y-2 rounded-xl border border-gray-200 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
        <span className="min-w-0">
          {stored ? (
            <span className="font-medium text-emerald-700">✓ stored</span>
          ) : work === "converting" ? (
            <span className="font-medium text-gray-600">uploading…</span>
          ) : (
            <span className="font-medium text-gray-400">queued</span>
          )}{" "}
          {docId ? (
            <a
              href={`${API_BASE}/api/curricula/documents/${docId}/file`}
              target="_blank"
              rel="noreferrer"
              title="download the original from R2"
              className="font-medium text-gray-700 hover:underline"
            >
              {item.name}
            </a>
          ) : (
            <span className="font-medium text-gray-700">{item.name}</span>
          )}
          {chars > 0 && ` · ${chars.toLocaleString()} chars`}
          {item.cached && (
            <span className="ml-1 text-emerald-600">· reused, nothing re-run</span>
          )}
          {r2 && <span className={`ml-1 ${r2.cls}`}>· {r2.text}</span>}
          {course?.title && (
            <span className="ml-1 text-gray-600">
              → {course.title}
              {course.skills_count ? ` · ${course.skills_count} skills` : ""}
            </span>
          )}
          {work === "extracting" && " · extracting skills… (~1–2 min)"}
          {work === "loading" && " · loading…"}
        </span>
        <span className="flex items-center gap-2">
          {stored && (
            <button
              onClick={onToggleMd}
              disabled={locked}
              className="rounded-md bg-white px-2 py-1 font-medium text-gray-600 ring-1 ring-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed"
            >
              {item.showMd ? "hide markdown" : "show markdown"}
            </button>
          )}
          {course && !skills && (
            <button
              onClick={onShowSkills}
              disabled={locked}
              className="rounded-md bg-white px-2 py-1 font-medium text-gray-600 ring-1 ring-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed"
            >
              show skills
            </button>
          )}
          {!docId && item.file && (
            <button
              onClick={onConvert}
              disabled={locked}
              className="inline-flex items-center gap-1.5 rounded-md bg-gray-900 px-2.5 py-1 font-medium text-white
                         hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {work === "converting" && <Spinner size={12} />}
              Upload & Extract
            </button>
          )}
          {stored && (
            <button
              onClick={onSkills}
              disabled={locked}
              className="inline-flex items-center gap-1.5 rounded-md bg-gray-900 px-2.5 py-1 font-medium text-white
                         hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {work === "extracting" && <Spinner size={12} />}
              {work === "extracting"
                ? "Extracting skills…"
                : course || skills
                ? "Re-extract skills"
                : "Extract Skills"}
            </button>
          )}
          <button
            onClick={() => (armed || !docId ? onRemove() : setArmed(true))}
            disabled={locked}
            title={
              docId
                ? "delete this syllabus from PostgreSQL — the markdown and its extracted skills go with it (the R2 original is kept, so re-uploading the file restores it)"
                : "remove from queue"
            }
            className={`rounded-full px-1.5 disabled:cursor-not-allowed ${
              armed
                ? "bg-red-50 px-2 font-medium text-red-600 ring-1 ring-red-300"
                : "text-gray-400 hover:bg-red-50 hover:text-red-600"
            }`}
          >
            {armed ? "delete?" : "×"}
          </button>
        </span>
      </div>

      {item.error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          {item.error}
        </div>
      )}

      {item.showMd && item.markdown != null && (
        <pre className="max-h-72 overflow-auto rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 font-mono text-[12px] leading-5 whitespace-pre-wrap text-gray-800">
          {item.markdown}
        </pre>
      )}

      {/* extraction persists the course itself, so there is no Save button */}
      {skills && <SkillsPreview data={skills} savedInfo={savedInfo} />}
    </div>
  );
}

/* ── skills preview: extracted course skills + save-to-DB (wiring next) ───── */

const SOURCE_BADGE = {
  weekly_content: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  lesson_topic: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  clo: "bg-blue-50 text-blue-700 ring-blue-200",
  description: "bg-amber-50 text-amber-700 ring-amber-200",
  tools: "bg-purple-50 text-purple-700 ring-purple-200",
};

const sourceKey = (s) => String(s || "").toLowerCase().replace(/\s+/g, "_");

function SkillsPreview({ data, onSave, saving, savedInfo }) {
  const course = data.course || {};
  const skills = data.skills || [];
  const bySource = {};
  for (const s of skills) (bySource[s.source] ||= []).push(s);

  return (
    <div className="space-y-3 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-gray-900">
            {course.title || "Untitled course"}
            {course.code && (
              <span className="ml-2 font-mono text-xs font-normal text-gray-500">
                {course.code}
              </span>
            )}
          </div>
          <div className="text-xs text-gray-500">
            {skills.length} skills extracted
            {course.category && ` · category: ${course.category}`}
          </div>
        </div>
        <span className="flex items-center gap-2">
          {savedInfo && (
            <span className="text-xs font-medium text-emerald-700">
              ✓ {savedInfo.created ? "saved" : "in DB"} (id {savedInfo.id})
            </span>
          )}
          {/* onSave is omitted on the document path — extraction already
              persisted the course, so there is nothing left to save */}
          {onSave && (
            <button
              onClick={onSave}
              disabled={saving || !!savedInfo}
              className="inline-flex items-center gap-1.5 rounded-lg bg-gray-900 px-3.5 py-1.5 text-sm font-medium text-white
                         hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {saving && <Spinner size={13} />}
              {saving
                ? "Saving…"
                : savedInfo
                ? "Saved to PostgreSQL"
                : "Save to PostgreSQL"}
            </button>
          )}
        </span>
      </div>

      {Object.entries(bySource).map(([source, list]) => (
        <div key={source} className="space-y-1.5">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-gray-400">
            <span
              className={`rounded-md px-1.5 py-0.5 font-medium normal-case ring-1 ${
                SOURCE_BADGE[sourceKey(source)] ||
                "bg-gray-100 text-gray-600 ring-gray-200"
              }`}
            >
              {source}
            </span>
            <span>{list.length}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {list.map((s, i) => (
              <span
                key={`${s.name}-${i}`}
                title={s.evidence || ""}
                className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-gray-700 ring-1 ring-gray-200"
              >
                {s.name}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── expanded course row: every extracted skill next to its source line ───── */

function SkillEvidenceList({ skills }) {
  if (!skills) {
    return <p className="text-xs text-gray-400">loading skills…</p>;
  }
  if (skills.length === 0) {
    return <p className="text-xs text-gray-400">no skills stored for this course</p>;
  }
  return (
    <div className="space-y-1.5">
      <p className="text-[11px] uppercase tracking-wide text-gray-400">
        {skills.length} skill{skills.length === 1 ? "" : "s"} — each with the
        syllabus text it was extracted from
      </p>
      <ul className="grid gap-1.5 sm:grid-cols-2">
        {skills.map((s, i) => (
          <li
            key={`${s.name}-${i}`}
            className="rounded-lg bg-white px-3 py-2 ring-1 ring-gray-200"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-gray-900">
                {s.name}
              </span>
              {s.source && (
                <span
                  className={`rounded-md px-1.5 py-0.5 text-[10px] font-medium ring-1 ${
                    SOURCE_BADGE[sourceKey(s.source)] ||
                    "bg-gray-100 text-gray-600 ring-gray-200"
                  }`}
                >
                  {s.source}
                </span>
              )}
            </div>
            {s.evidence ? (
              <p className="mt-1 text-xs leading-snug text-gray-600">
                “{s.evidence}”
              </p>
            ) : (
              <p className="mt-1 text-xs italic text-gray-300">
                no evidence line recorded
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ── real-experiment step 2: select the curriculum courses from Postgres ──── */

function CourseSelectRow({ value, onChange, reloadKey = 0 }) {
  const [courses, setCourses] = useState([]);
  const [listError, setListError] = useState("");
  const [expandedId, setExpandedId] = useState(null);
  const [skillCache, setSkillCache] = useState({}); // id -> [{name, source, evidence}]

  // A cold backend must not read as "you have no courses" — that sent people
  // back to Step 1 to re-upload syllabi that were already saved.
  const [listWaking, setListWaking] = useState(false);
  const listTries = useRef(0);
  const listTimer = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/curricula/course-skills`);
      if (!res.ok) throw httpError(`list failed (${res.status})`);
      setCourses((await res.json()).courses || []);
      setListError("");
      setListWaking(false);
      listTries.current = 0;
    } catch (e) {
      if (isOffline(e) && listTries.current < WAKE_MAX_TRIES) {
        listTries.current += 1;
        setListWaking(true);
        setListError("");
        listTimer.current = setTimeout(refresh, WAKE_RETRY_MS);
      } else {
        setListWaking(false);
        setListError(
          isOffline(e) ? OFFLINE_MSG : e.message || "Could not load saved courses"
        );
      }
    }
  }, []);

  // reloadKey changes every time Step 1 extracts or deletes a course
  useEffect(() => {
    listTries.current = 0;
    refresh();
    return () => clearTimeout(listTimer.current);
  }, [refresh, reloadKey]);

  const { source, courseIds } = value;
  const [yearFilter, setYearFilter] = useState(null); // null = all years
  const setSource = (s) => onChange({ ...value, source: s });
  const toggleId = (id) =>
    onChange({
      ...value,
      courseIds: courseIds.includes(id)
        ? courseIds.filter((x) => x !== id)
        : [...courseIds, id],
    });

  // distinct study years present in the DB (LLM-inferred at parse time,
  // course-code regex fallback) — drives the filter chips + quick select
  const years = [...new Set(courses.map((c) => c.year).filter(Boolean))].sort();
  const visible =
    yearFilter == null ? courses : courses.filter((c) => c.year === yearFilter);

  // header checkbox = select/clear the VISIBLE (year-filtered) rows only
  const allSelected =
    visible.length > 0 && visible.every((c) => courseIds.includes(c.id));
  const toggleAll = () => {
    const visIds = visible.map((c) => c.id);
    onChange({
      ...value,
      courseIds: allSelected
        ? courseIds.filter((id) => !visIds.includes(id))
        : [...new Set([...courseIds, ...visIds])],
    });
  };
  // cumulative quick select (Y3, Y3+4, Y3+4+5 …) — mirrors the thesis
  // programme-level runs R1/R2/R3
  const selectYearsUpTo = (y) =>
    onChange({
      ...value,
      courseIds: courses.filter((c) => c.year && c.year <= y).map((c) => c.id),
    });

  const selected = courses.filter((c) => courseIds.includes(c.id));
  const selectedSkills = selected.reduce((n, c) => n + (c.skills_count || 0), 0);

  // A year chip only FILTERS the table — it deliberately does not touch the
  // selection (so you can hand-pick across years). The trap: ticks made under
  // a previous filter stay active while being invisible, which silently turned
  // an intended Y5-only run into a 16-course Y4+Y5 one. Surface anything
  // selected but out of view, with a one-click clear.
  const visibleIds = new Set(visible.map((c) => c.id));
  const hidden = selected.filter((c) => !visibleIds.has(c.id));
  const clearHidden = () =>
    onChange({
      ...value,
      courseIds: courseIds.filter((id) => visibleIds.has(id)),
    });
  // the cohort the run store will actually record, shown before you run it
  const cohort =
    [...new Set(selected.map((c) => c.year).filter(Boolean))]
      .sort()
      .map((y) => `Y${y}`)
      .join("+") || null;

  async function togglePreview(id) {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (!skillCache[id]) {
      try {
        const res = await fetch(`${API_BASE}/api/curricula/course-skills/${id}`);
        if (res.ok) {
          const row = await res.json();
          // keep the whole skill object — the preview shows the verbatim
          // syllabus line each skill was extracted from, so a wrong skill can
          // be traced back to the sentence that produced it
          setSkillCache((c) => ({ ...c, [id]: row.skills || [] }));
        }
      } catch {
        /* preview is best-effort */
      }
    }
  }

  return (
    <section className="space-y-3 rounded-2xl border border-gray-200 bg-white px-5 py-4 shadow-sm">
      <div>
        <h2 className="text-base font-semibold text-gray-900">
          Step 2 — course selection: choose the curriculum for the run
        </h2>
        <p className="mt-0.5 text-xs text-gray-500">
          pick which saved courses (PostgreSQL) the phase6 pipeline compares
          against the job market
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-gray-500">Source:</span>
        <button
          onClick={() => setSource("db")}
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            source === "db"
              ? "bg-gray-900 text-white"
              : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
          }`}
        >
          {source === "db" ? "✓ " : ""}Database — selected courses
        </button>
        {years.length > 1 && (
          <>
            <span className="ml-3 text-gray-500">Year:</span>
            <button
              onClick={() => setYearFilter(null)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                yearFilter == null
                  ? "bg-gray-900 text-white"
                  : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
              }`}
            >
              All
            </button>
            {years.map((y) => (
              <button
                key={y}
                onClick={() => setYearFilter(yearFilter === y ? null : y)}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                  yearFilter === y
                    ? "bg-gray-900 text-white"
                    : "bg-white text-gray-500 ring-1 ring-gray-300 hover:bg-gray-50"
                }`}
              >
                Y{y}
              </button>
            ))}
            <span className="ml-3 text-gray-400">⚡</span>
            {years.map((y, i) => (
              <button
                key={`cum-${y}`}
                onClick={() => selectYearsUpTo(y)}
                title={`select every course up to year ${y} (cumulative programme run)`}
                className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-gray-600 ring-1 ring-gray-300 hover:bg-gray-50"
              >
                {years
                  .slice(0, i + 1)
                  .map((yy) => `Y${yy}`)
                  .join("+")}
              </button>
            ))}
          </>
        )}
      </div>

      {source === "db" && (
        <>
          {hidden.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2.5 text-sm text-amber-900">
              <span>
                <span className="font-semibold">
                  {hidden.length} course{hidden.length === 1 ? "" : "s"} still
                  selected outside this filter
                </span>{" "}
                ({[...new Set(hidden.map((c) => (c.year ? `Y${c.year}` : "—")))]
                  .sort()
                  .join(", ")}
                ) — the run would use all {selected.length}, not just what you
                see here.
              </span>
              <button
                onClick={clearHidden}
                className="rounded-full bg-amber-900 px-3 py-1 text-xs font-medium text-white hover:bg-amber-800"
              >
                Keep only visible
              </button>
            </div>
          )}
          {listWaking && (
            <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <Spinner size={14} /> Waiting for the backend to finish starting —
              your saved courses will appear here.
            </div>
          )}
          {listError && (
            <div className="whitespace-pre-line rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {listError}
            </div>
          )}
          {!listError && !listWaking && courses.length === 0 && (
            <p className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-500">
              No saved courses yet — extract and save one in Step 1 first.
            </p>
          )}
          {courses.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-gray-200">
              <table className="w-full text-left text-sm">
                <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
                  <tr>
                    <th className="w-10 px-3 py-2">
                      <input
                        type="checkbox"
                        checked={allSelected}
                        onChange={toggleAll}
                        className="h-4 w-4 accent-gray-900"
                      />
                    </th>
                    <th className="px-3 py-2 font-medium">Course</th>
                    <th className="px-3 py-2 font-medium">Code</th>
                    <th className="px-3 py-2 font-medium">Year</th>
                    <th className="px-3 py-2 font-medium">Skills</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {visible.map((c) => (
                    <Fragment key={c.id}>
                      <tr
                        // double-click anywhere on the row does what clicking
                        // the title does — expand the skills + their evidence
                        onDoubleClick={() => togglePreview(c.id)}
                        className={
                          courseIds.includes(c.id) ? "bg-gray-50" : "bg-white"
                        }
                      >
                        <td className="px-3 py-2">
                          <input
                            type="checkbox"
                            checked={courseIds.includes(c.id)}
                            onChange={() => toggleId(c.id)}
                            className="h-4 w-4 accent-gray-900"
                          />
                        </td>
                        <td className="px-3 py-2">
                          <button
                            onClick={() => togglePreview(c.id)}
                            title="click (or double-click the row) to show skills + evidence"
                            className="font-medium text-gray-900 hover:underline"
                          >
                            {c.course_title}
                          </button>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-gray-500">
                          {c.course_code}
                        </td>
                        <td className="px-3 py-2">
                          {c.year ? (
                            <span className="rounded-md bg-gray-100 px-1.5 py-0.5 text-xs font-medium text-gray-700 ring-1 ring-gray-200">
                              Y{c.year}
                            </span>
                          ) : (
                            <span className="text-xs text-gray-300">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-gray-600">
                          {c.skills_count}
                        </td>
                      </tr>
                      {expandedId === c.id && (
                        <tr className="bg-gray-50/60">
                          <td />
                          <td colSpan={4} className="px-3 pt-0 pb-3">
                            <SkillEvidenceList skills={skillCache[c.id]} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-xs text-gray-500">
            Selected:{" "}
            <span className="font-medium text-gray-800">
              {selected.length} course{selected.length === 1 ? "" : "s"} ·{" "}
              {selectedSkills} CLO skills
              {cohort ? ` · cohort ${cohort}` : ""}
            </span>{" "}
            → feeds “Run pipeline” below
          </p>
        </>
      )}
    </section>
  );
}

/* ── real-experiment mode: run gen_report_data.py live ───────────────────── */

function runLogColor(line) {
  if (line.includes("ERROR") || line.includes("Traceback")) return "text-red-400";
  if (line.startsWith("written") || line.includes("SELECTION")) return "text-emerald-400";
  if (line.startsWith("Loading")) return "text-blue-400";
  return "text-gray-400";
}

function LiveRun({ selection, jobSource, onDone }) {
  const [run, setRun] = useState(null); // /api/v8/run/status payload
  const [runError, setRunError] = useState("");
  const [reloaded, setReloaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(null); // /export payload
  const [saveError, setSaveError] = useState("");
  const logRef = useRef(null);

  // Resume display if a run is already in progress (or just finished)
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v8/run/status`);
        if (r.ok) {
          const s = await r.json();
          if (s.running || s.log?.length) setRun(s);
        }
      } catch {}
    })();
  }, []);

  // Poll while running; reload the report once on success
  useEffect(() => {
    if (!run?.running) return;
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v8/run/status`);
        if (!r.ok) return;
        const s = await r.json();
        setRun(s);
        if (!s.running) {
          clearInterval(t);
          if (s.returncode === 0) {
            onDone();
            setReloaded(true);
          }
        }
      } catch {}
    }, 1500);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.running]);

  // Keep the terminal scrolled to the newest line
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [run?.log?.length]);

  const dbEmpty =
    selection?.source === "db" && (selection?.courseIds || []).length === 0;

  // Save = write the thesis bundle for the finished run (run_summary.json +
  // the 4 CSVs) into outputs/<run_id>/thesis/ and rebuild the registry.
  async function saveRun() {
    const runId = run?.run_id;
    if (!runId) return;
    setSaving(true);
    setSaveError("");
    try {
      const r = await fetch(`${API_BASE}/api/v8/runs/${runId}/export`, {
        method: "POST",
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `Save failed (${r.status})`);
      }
      setSaved(await r.json());
    } catch (e) {
      setSaveError(e.message || "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function startRun() {
    setRunError("");
    setReloaded(false);
    setSaved(null);
    setSaveError("");
    try {
      const r = await fetch(`${API_BASE}/api/v8/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: selection?.source || "db",
          course_ids: selection?.courseIds || [],
          // must match the corpus the dashboard is showing, or the run
          // writes a file this view never reads
          job_source: jobSource,
        }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `Run failed (${r.status})`);
      }
      setRun({ running: true, log: [], returncode: null });
    } catch (e) {
      setRunError(e.message || "Run failed");
    }
  }

  const finished = run && !run.running && run.returncode != null;
  // only an archived, successful run has a bundle to write
  const canSave = !!(finished && run.returncode === 0 && run.run_id);

  return (
    <section className="space-y-3 rounded-2xl border border-gray-200 bg-white px-5 py-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-gray-900">
            Step 3 — real experiment: run the phase6 pipeline
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">
            executes 04_report/gen_report_data.py in backend/phase5_curriculum_alignment on the Step 2
            curriculum ({(selection?.courseIds || []).length} selected DB
            course{(selection?.courseIds || []).length === 1 ? "" : "s"}):
            loads the BGE encoder + cached
            market matrix, recomputes coverage / KL / greedy selection into a
            gitignored live report (the committed baseline is untouched), then
            reloads the dashboard below
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={startRun}
            disabled={run?.running || dbEmpty}
            title={dbEmpty ? "select at least one course in Step 2" : undefined}
            className="inline-flex items-center gap-2 rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white
                       hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            {run?.running && <Spinner size={14} />}
            {run?.running ? "Running…" : "▶ Run pipeline"}
          </button>
          <button
            onClick={saveRun}
            disabled={!canSave || saving}
            title={
              canSave
                ? "write run_summary.json + the 4 thesis CSVs into outputs/<run_id>/thesis/"
                : "finish a successful run first"
            }
            className="inline-flex items-center gap-2 rounded-lg px-4 py-1.5 text-sm font-medium ring-1
                       disabled:cursor-not-allowed disabled:bg-white disabled:text-gray-300 disabled:ring-gray-200
                       bg-white text-gray-900 ring-gray-300 hover:bg-gray-50"
          >
            {saving && <Spinner size={14} />}
            {saved ? "✓ Saved" : saving ? "Saving…" : "⤓ Save record"}
          </button>
        </div>
      </div>

      {runError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {runError}
        </div>
      )}

      {run && (
        <Terminal
          logRef={logRef}
          logs={run.log || []}
          label="gen_report_data.py"
          isLive={!!run.running}
          colorize={runLogColor}
        />
      )}

      {finished && run.returncode === 0 && (
        <p className="text-sm font-medium text-emerald-700">
          ✓ pipeline finished
          {run.finished_at
            ? ` at ${new Date(run.finished_at).toLocaleTimeString()}`
            : ""}
          {reloaded ? " — fresh report loaded below" : ""}
          {run.run_id ? (
            <span className="ml-1 font-normal text-gray-500">
              · archived as <code>{run.run_id}</code>
            </span>
          ) : null}
        </p>
      )}

      {saveError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {saveError}
        </div>
      )}

      {saved && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
          <p className="font-medium">
            ✓ saved {saved.cohort ? `${saved.cohort} ` : ""}record —{" "}
            <code>{saved.dir}/</code>
          </p>
          <p className="mt-1 text-xs text-emerald-700">
            {(saved.files || []).join(" · ")}
          </p>
          {saved.measured && (
            <p className="mt-1 text-xs text-emerald-700">
              measured: KL {saved.measured.kl_divergence} · covered{" "}
              {saved.measured.covered} · borderline {saved.measured.borderline}{" "}
              · gap {saved.measured.gap} · registry updated (
              <code>{saved.registry}</code>)
            </p>
          )}
        </div>
      )}
      {finished && run.returncode !== 0 && (
        <p className="text-sm font-medium text-red-700">
          ✗ pipeline failed (exit code {run.returncode}) — see log above
        </p>
      )}
    </section>
  );
}
