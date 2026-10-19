// React preview: first 5 postings of 01_input_jobmarket/it_job_linkedin.json.
// Data comes from preview_data.js (window.JOB_PREVIEW), regenerated with:
//   python3 -c "see conversation / scratchpad make_preview script"
// No JSX (no build step) — components use React.createElement, aliased to h.

const { createRoot } = ReactDOM;
const h = React.createElement;

const styles = {
  page: { fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
          maxWidth: 880, margin: "0 auto", padding: "28px 20px",
          color: "#0b0b0b", background: "#f9f9f7" },
  sub: { color: "#52514e", marginTop: 4 },
  table: { borderCollapse: "collapse", width: "100%", fontSize: 14,
           background: "#fcfcfb", border: "1px solid rgba(11,11,11,.10)",
           borderRadius: 10 },
  th: { textAlign: "left", padding: "8px 10px", color: "#52514e",
        borderBottom: "1px solid #e1e0d9", fontWeight: 600 },
  td: { padding: "7px 10px", borderBottom: "1px solid #eeede8" },
  skill: { display: "inline-block", background: "#cde2fb", color: "#0b0b0b",
           borderRadius: 6, padding: "1px 8px", marginRight: 6,
           fontSize: 12.5 },
  skillCourse: { display: "inline-block", background: "#d2efe3",
                 color: "#0b0b0b", borderRadius: 6, padding: "1px 8px",
                 marginRight: 6, fontSize: 12.5 },
  dim: { color: "#898781" },
  expandBtn: { background: "none", border: "1px solid rgba(11,11,11,.15)",
               borderRadius: 6, padding: "2px 10px", marginTop: 6,
               fontSize: 12.5, color: "#2a78d6", cursor: "pointer",
               display: "block" },
};

function SkillTags({ skills, tagStyle }) {
  if (!skills || skills.length === 0)
    return h("span", { style: styles.dim }, "—");
  return skills.map((s, i) =>
    h("span", { key: i, style: tagStyle || styles.skill }, s));
}

function PreviewRow({ job }) {
  return h("tr", null,
    h("td", { style: styles.td }, job.job_title || "(untitled)"),
    h("td", { style: styles.td }, job.company_name ||
      h("span", { style: styles.dim }, "—")),
    h("td", { style: styles.td }, job.location ||
      h("span", { style: styles.dim }, "—")),
    h("td", { style: styles.td }, h(SkillTags, { skills: job.skills })),
  );
}

function CourseRow({ course }) {
  const [expanded, setExpanded] = React.useState(false);
  const shown = expanded ? course.all_skills : course.sample_skills;
  const hidden = course.n_skills - course.sample_skills.length;
  return h("tr", null,
    h("td", { style: { ...styles.td, verticalAlign: "top" } }, course.course,
      h("div", { style: { ...styles.dim, fontSize: 12 } }, course.file)),
    h("td", { style: { ...styles.td, textAlign: "right",
                       verticalAlign: "top" } }, course.n_skills),
    h("td", { style: styles.td },
      h("div", { style: { lineHeight: 1.9 } },
        h(SkillTags, { skills: shown, tagStyle: styles.skillCourse })),
      hidden > 0 && h("button",
        { style: styles.expandBtn,
          onClick: () => setExpanded(!expanded),
          "aria-expanded": expanded },
        expanded ? "show less ▲" : `show all ${course.n_skills} skills ▼`),
    ),
  );
}

// ---- gap-analysis dashboard (data from report_data.js) ----

const STATUS = {
  gap: { color: "#d03b3b", icon: "▲" },
  borderline: { color: "#fab219", icon: "◆" },
  covered: { color: "#0ca30c", icon: "●" },
};

// Threshold-aware evidence bands (BGE cosine). Band edges follow the
// ACTIVE thresholds from report_data.js, so recalibration moves them.
// THREE bands only, one per real decision. The former 0.60 and 0.85
// edges were display-only sugar with no calibration behind them; the
// 250-pair human study validated borderline (0.6903) and covered
// (0.7368) but never evidenced those two, so they are gone.
function evidenceBands(th) {
  const b = th ? th.borderline : 0.69;   // borderline threshold
  const c = th ? th.covered : 0.73;      // covered threshold
  // show thresholds exactly (calibrated values have 4 decimals); cells stay
  // 2dp — statuses are decided from full precision, not the displayed value
  return [
    { min: c, bg: "#4C8BCB", fg: "#FFFFFF",
      range: `≥ ${c}`, name: "covered" },
    { min: b, bg: "#F6C85F", fg: "#3D3000",
      range: `${b} – <${c}`,
      name: "borderline" },
    { min: -Infinity, bg: "#F2F4F7", fg: "#344054",
      range: `< ${b}`, name: "not covered" },
  ];
}

function bandStyle(bands, v) {
  const b = bands.find(x => v >= x.min);
  return { background: b.bg, color: b.fg };
}

// PROGRAM-cell verification marks (only when reranker data exists)
const VERIFY_MARK = {
  "SUPPORTED": { glyph: "✓", color: "#0ca30c", label: "supported" },
  "WEAK": { glyph: "◑", color: "#b97e0a", label: "weak" },
  "NOT SUPPORTED": { glyph: "×", color: "#d03b3b", label: "demoted" },
};

function EvidenceLegend({ bands }) {
  return h("div", { style: { margin: "8px 0 12px" } },
    h("div", { style: { fontWeight: 600, fontSize: 12.5 } },
      "Semantic evidence (BGE cosine), not percentage."),
    h("div", { style: { display: "flex", flexWrap: "wrap", gap: 6,
                        alignItems: "center", marginTop: 6 } },
      bands.slice().reverse().map(b =>
        h("span", { key: b.name, style: { background: b.bg, color: b.fg,
            borderRadius: 6, padding: "2px 9px", fontSize: 11.5,
            border: "1px solid rgba(11,11,11,.12)" } },
          `${b.range} ${b.name}`)),
      h("span", { style: { fontSize: 11.5, color: "#52514e" } },
        "· PROGRAM badge: ",
        h("span", { style: { color: VERIFY_MARK.SUPPORTED.color,
                             fontWeight: 700 } }, "✓ supported"),
        " / ",
        h("span", { style: { color: VERIFY_MARK.WEAK.color,
                             fontWeight: 700 } }, "◑ weak"),
        " / ",
        h("span", { style: { color: VERIFY_MARK["NOT SUPPORTED"].color,
                             fontWeight: 700 } }, "× demoted"),
        " (cross-encoder)")));
}

function StatTile({ value, label }) {
  return h("div", { style: { flex: "1 1 130px", background: "#fcfcfb",
      border: "1px solid rgba(11,11,11,.10)", borderRadius: 10,
      padding: "12px 16px" } },
    h("div", { style: { fontSize: 24, fontWeight: 650 } }, value),
    h("div", { style: { ...styles.dim, fontSize: 12.5 } }, label));
}

// Optimized skill plan (report.selection). Two questions are shown
// SEPARATELY: what related curriculum content exists (relationship)
// vs what must be added (recommended action) — a related foundation
// is never presented as proof the target skill is already taught.
const REL_BADGE = {
  content_exists: { label: "✓ RELEVANT CONTENT EXISTS", color: "#1B7F3A" },
  foundation:     { label: "↗ FOUNDATION EXISTS",       color: "#2F6FB0" },
  none:           { label: "＋ NO VERIFIED HOST",        color: "#344054" },
  review:         { label: "? REVIEW REQUIRED",          color: "#7A5AF8" },
};
const ADD_REQUIRED = "#C4320A";

function Badge({ label, color }) {
  return h("span", { style: { display: "inline-block", color,
      border: `1px solid ${color}`, borderRadius: 5, padding: "1px 7px",
      fontSize: 10, fontWeight: 700, letterSpacing: ".04em",
      whiteSpace: "nowrap" } }, label);
}

// Backward compatibility: old report_data.js entries carry only
// `route` — derive the new objects from it, null-safe throughout.
function relationshipOf(e) {
  if (e.relationship) return e.relationship;
  const rt = e.route || {};
  const base = { course: rt.where || null, clo: rt.clo || null,
    score: rt.score != null ? rt.score : null,
    addition_required: true, evidence_source: "fallback" };
  if (rt.kind === "extend")
    return { ...base, kind: "content_exists", target_status: "partial",
      reason: "Verified related content exists, but the target skill " +
              "is still below coverage." };
  if (rt.kind === "partial")
    return { ...base, kind: "foundation", target_status: "missing",
      reason: "A related foundation exists, but the target skill is " +
              "not explicitly evidenced." };
  return { ...base, kind: "none", target_status: "missing",
    reason: "No existing CLO shows verified evidence of teaching " +
            "this skill." };
}

function actionOf(e, rel) {
  if (e.recommended_action) return e.recommended_action;
  const kind = { content_exists: "extend", foundation: "add_module",
                 review: "review_route" }[rel.kind] || "new_content";
  return { kind, title: `Add ${e.skill} content`,
    detail: rel.course
      ? `Build on ${rel.course}; do not duplicate existing content.`
      : `Introduce a CLO or module that explicitly teaches ${e.skill}.` };
}

function EvidenceDetails({ info, rel }) {
  const v = info && info.verify;
  const row = (k, val) => h("div", null, h("b", null, `${k}: `),
    val == null || val === "" ? "—" : val);
  return h("div", { style: { fontSize: 12.5, color: "#52514e",
      lineHeight: 1.7, padding: "8px 12px", background: "#f4f4f0",
      borderRadius: 8 } },
    row("current cosine coverage",
      info && info.program != null ? info.program.toFixed(2) : null),
    row("current status", info ? info.status : null),
    row("closest retrieved CLO", info && info.best_clo
      ? `"${info.best_clo}" (${info.best_course})` : null),
    row("reranker", v
      ? `${v.argmax_rerank != null ? v.argmax_rerank.toFixed(2) : "—"} `
        + `on closest CLO; best candidate `
        + `${v.best_rerank != null ? v.best_rerank.toFixed(2) : "—"} `
        + `— ${v.verdict}`
      : "no reranker data"),
    row("relationship decision",
      `${rel.kind} · target skill ${rel.target_status}`),
    row("evidence source", rel.evidence_source),
    rel.reason ? row("reason", rel.reason) : null);
}

function PlanRow({ e, n, topN, info }) {
  const [open, setOpen] = React.useState(false);
  const rel = relationshipOf(e);
  const act = actionOf(e, rel);
  const rb = REL_BADGE[rel.kind] || REL_BADGE.none;
  const projTip = "Projection assumes that an explicit CLO teaching " +
    "the target skill is added.";
  const small = { fontSize: 12, marginTop: 3 };
  return h(React.Fragment, null,
    h("tr", null,
      h("td", { style: styles.td },
        h("span", { style: { display: "inline-block", width: 22,
            height: 22, borderRadius: "50%", background: "#2a78d6",
            color: "#fff", fontWeight: 700, fontSize: 12,
            lineHeight: "22px", textAlign: "center" } }, n)),
      h("td", { style: styles.td }, h("b", null, e.skill),
        h("div", { style: { ...styles.dim, fontSize: 12 } },
          `${e.jobs.toLocaleString()} postings`)),
      h("td", { style: styles.td }, `${e.gain_pct.toFixed(1)}%`),
      h("td", { style: { ...styles.td, minWidth: 190 } },
        h(Badge, rb),
        rel.course && h("div", { style: small },
          rel.clo ? `"${rel.clo}" in ${rel.course}` : rel.course,
          rel.score != null ? ` (${rel.score.toFixed(2)})` : ""),
        rel.reason && h("div", { style: { ...styles.dim, fontSize: 12,
                                          marginTop: 2 } }, rel.reason)),
      h("td", { style: { ...styles.td, minWidth: 200 } },
        rel.addition_required !== false
          && h(Badge, { label: "ADD REQUIRED", color: ADD_REQUIRED }),
        h("div", { style: { ...small, fontWeight: 600 } }, act.title),
        act.detail && h("div", { style: { ...styles.dim, fontSize: 12,
                                          marginTop: 2 } }, act.detail)),
      h("td", { style: styles.td, title: projTip },
        `${e.covered_after}/${topN} covered`,
        h("div", { style: { ...styles.dim, fontSize: 12 } },
          `KL ${e.kl_after.toFixed(3)}`),
        h("button", { style: { ...styles.expandBtn, marginTop: 5 },
            onClick: () => setOpen(!open), "aria-expanded": open },
          open ? "hide evidence ▴" : "view evidence ▾"))),
    open && h("tr", null,
      h("td", { style: styles.td }),
      h("td", { style: styles.td, colSpan: 5 },
        h(EvidenceDetails, { info, rel }))));
}

function SelectionPlan({ sel, topN, skillsInfo }) {
  const chosen = sel.plan.filter(e => e.selected);
  const skipped = sel.plan.filter(e => !e.selected);
  const byName = {};
  (skillsInfo || []).forEach(s => { byName[s.skill] = s; });
  const projTip = "Projection assumes that an explicit CLO teaching " +
    "the target skill is added.";
  return h("div", null,
    h("p", { style: { ...styles.dim, fontSize: 13, margin: "4px 0 10px" } },
      `Not every missing skill deserves a course slot. The plan below `,
      `adds skills in the order that closes the most market gap `,
      `(demand-weighted, over the whole job segment) and stops at `,
      `${Math.round(sel.cum_frac * 100)}% of the achievable improvement `,
      `— the rest cost a slot for under ~1% gain each.`),
    h("div", { style: { overflowX: "auto" } },
      h("table", { style: { ...styles.table, fontSize: 13 } },
        h("thead", null, h("tr", null,
          h("th", { style: styles.th }, "step"),
          h("th", { style: styles.th }, "target skill"),
          h("th", { style: styles.th }, "modelled gap closed"),
          h("th", { style: styles.th }, "curriculum relationship"),
          h("th", { style: styles.th }, "recommended action"),
          h("th", { style: styles.th, title: projTip },
            "projected effect"))),
        h("tbody", null, chosen.map((e, i) =>
          h(PlanRow, { key: e.skill, e, n: i + 1, topN,
                       info: byName[e.skill] }))))),
    skipped.length > 0 && h("div", { style: { marginTop: 12 } },
      h("div", { style: { fontWeight: 600, fontSize: 13.5 } },
        "Not selected — below the benefit cutoff ",
        "(committee may still consider):"),
      skipped.map(e => h("div",
        { key: e.skill, style: { ...styles.dim, fontSize: 13,
                                 margin: "4px 0" } },
        `▪ ${e.skill} (${e.jobs.toLocaleString()} postings) — would `,
        `close only ${e.gain_pct.toFixed(1)}% more`,
        e.absorbed_by
          ? `; already largely absorbed after adding "${e.absorbed_by}"`
          : "",
        e.cov_after_plan !== undefined
          ? ` · coverage after the plan: ${e.cov_after_plan.toFixed(2)}`
          : ""))),
    h("p", { style: { ...styles.dim, fontSize: 12, marginTop: 10 } },
      `Method: greedy demand-weighted coverage maximization `,
      `(facility-location objective, near-optimal guarantee). "Modelled `,
      `gap closed" = share of today's total demand-weighted gap; KL = `,
      `program alignment after the step. ${projTip} Target coverage `,
      `and curriculum relationship are reported separately. A related `,
      `foundation does not mean that the target skill is already `,
      `taught. Relationship decisions guide where new content may `,
      `build on the existing curriculum; they do not change the `,
      `optimized ranking. Gains already discount partial coverage `,
      `(measured above the current cosine, not from zero). Grouping `,
      `into courses = committee step.`));
}

function GapDashboard() {
  const r = window.GAP_REPORT;
  if (!r) return null;
  const bands = evidenceBands(r.thresholds);
  // older report_data bundles predate the demand_weighted_coverage field —
  // re-derive it from the per-skill rows so cached runs display correctly
  const topNJobs = r.skills.reduce((s, x) => s + x.jobs, 0);
  const coveredJobs = r.skills.reduce(
    (s, x) => s + (x.status === "covered" ? x.jobs : 0), 0);
  const dwc = r.demand_weighted_coverage ??
    (topNJobs ? coveredJobs / topNJobs : 0);
  const cell = { width: 62, textAlign: "center",
                 border: "2px solid #fcfcfb", borderRadius: 4,
                 fontSize: 12, padding: "5px 4px" };
  return h("div", null,
    h("h1", { style: { marginTop: 36 } }, "Gap Analysis Dashboard"),
    h("p", { style: styles.sub },
      `${r.courses.length} courses (${r.n_clo_skills} CLO skills) vs `,
      `${r.n_jobs.toLocaleString()} relevant postings — `,
      `top ${r.top_n} technical market skills.`),
    h("div", { style: { display: "flex", gap: 12, flexWrap: "wrap",
                        margin: "14px 0" } },
      h(StatTile, { value: `${(dwc * 100).toFixed(1)}%`,
                    label: "market demand covered (demand-weighted)" }),
      h(StatTile, { value: `${r.covered} / ${r.top_n}`,
                    label: "top demanded skills covered by the program" }),
      h(StatTile, { value: r.selection
          ? `${r.selection.n_selected} / ${r.selection.plan.length}`
          : r.recommendations.length,
                    label: "gaps worth adding (of all gaps found)" }),
      h(StatTile, { value: r.kl.toFixed(3),
                    label: r.source === "cambodia"
                      ? "KL divergence (distributional diagnostic; null = 0.399)"
                      : "KL divergence (distributional diagnostic)" })),
    h("div", { style: { background: "#fcfcfb", borderRadius: 10,
        border: "1px solid rgba(11,11,11,.10)",
        borderLeft: "4px solid #2a78d6", padding: "12px 16px" } },
      h("h3", { style: { margin: "2px 0 6px" } },
        "Recommended actions — skills to add, optimized order"),
      r.selection
        ? h(SelectionPlan, { sel: r.selection, topN: r.top_n,
                             skillsInfo: r.skills })
        : h("p", { style: styles.dim }, "no selection data — rerun ",
            "gen_report_data.py")),
    h("h3", { style: { marginTop: 22 } },
      "Coverage heatmap — skills × courses"),
    h(EvidenceLegend, { bands }),
    h("div", { style: { overflowX: "auto" } },
      h("table", { style: styles.table },
        h("thead", null, h("tr", null,
          h("th", { style: styles.th }, "market skill"),
          h("th", { style: { ...styles.th, textAlign: "right" } }, "jobs"),
          r.courses.map(c => h("th", { key: c, style: styles.th }, c)),
          h("th", { style: { ...styles.th,
              borderLeft: "3px solid #b9b8b0" } }, "PROGRAM"),
          h("th", { style: styles.th }, "status"))),
        h("tbody", null, r.skills.map(s => {
          const st = STATUS[s.status];
          const vm = s.verify && VERIFY_MARK[s.verify.verdict];
          return h("tr", { key: s.skill },
            h("td", { style: styles.td }, s.skill),
            h("td", { style: { ...styles.td, textAlign: "right" } },
              s.jobs.toLocaleString()),
            r.courses.map(c => h("td",
              { key: c, style: { ...cell, ...bandStyle(bands,
                                                       s.per_course[c]) },
                title: `${s.skill} × ${c}: ${s.per_course[c]}` },
              s.per_course[c].toFixed(2))),
            h("td", { style: { ...cell, ...bandStyle(bands, s.program),
                borderLeft: "3px solid #b9b8b0" },
                title: `best CLO [${s.best_course}]: ${s.best_clo}`
                  + (s.verify ? ` · joint-read ${
                      s.verify.argmax_rerank.toFixed(2)} (${
                      s.verify.verdict})` : "") },
              s.program.toFixed(2),
              vm && h("span", { style: { display: "inline-block",
                  marginLeft: 5, width: 15, height: 15,
                  lineHeight: "15px", borderRadius: "50%",
                  background: "rgba(255,255,255,.94)",
                  border: "1px solid rgba(11,11,11,.15)",
                  color: vm.color, fontSize: 10, fontWeight: 700,
                  textAlign: "center" } }, vm.glyph)),
            h("td", { style: styles.td },
              h("span", { style: { color: st.color, fontWeight: 600 } },
                `${st.icon} ${s.status}`)));
        })))),
    h("p", { style: { ...styles.dim, fontSize: 12, marginTop: 10 } },
      `Cells show the strongest semantic evidence found in each course. `,
      `The PROGRAM column is the primary coverage result. Scores are `,
      `cosine similarities, not percentages. · status: ▲ gap < `,
      `${r.thresholds.borderline} ≤ ◆ borderline < ${r.thresholds.covered} `,
      `≤ ● covered · hosts are only named when coverage ≥ `,
      `${r.thresholds.host_min} (below that: "new module needed") · `,
      r.reranker ? `host suggestions verified by cross-encoder `
        + `${r.reranker} (✓ confirmed / ✗ demoted; relevance is a separate `
        + `scale from cosine, bands ≥ ${r.rerank_bands.supported} supported `
        + `/ ≥ ${r.rerank_bands.weak} weak, provisional) · ` : "",
      `dataset: ${r.dataset}`));
}

function App() {
  const jobs = window.JOB_PREVIEW || [];
  const total = window.JOB_TOTAL || 0;
  const courses = window.CURRICULUM_PREVIEW || [];
  const totalCLO = courses.reduce((n, c) => n + c.n_skills, 0);
  return h("div", { style: styles.page },
    h("h1", null, "Job-Market Dataset Preview"),
    h("p", { style: styles.sub },
      `01_input_jobmarket/it_job_linkedin.json — showing ${jobs.length} of `,
      h("b", null, total.toLocaleString()),
      " usable postings (first 4 extracted skills per job)."),
    h("table", { style: styles.table },
      h("thead", null, h("tr", null,
        h("th", { style: styles.th }, "job title"),
        h("th", { style: styles.th }, "company"),
        h("th", { style: styles.th }, "location"),
        h("th", { style: styles.th }, "skills (first 4)"),
      )),
      h("tbody", null,
        jobs.map((job, i) => h(PreviewRow, { key: i, job }))),
    ),

    h("h1", { style: { marginTop: 36 } }, "Course Curriculum Preview"),
    h("p", { style: styles.sub },
      `02_input_curriculum/clo_skills/ — ${courses.length} courses, `,
      h("b", null, String(totalCLO)),
      " curated CLO skills total (first 5 shown per course)."),
    h("table", { style: styles.table },
      h("thead", null, h("tr", null,
        h("th", { style: styles.th }, "course"),
        h("th", { style: { ...styles.th, textAlign: "right" } }, "# skills"),
        h("th", { style: styles.th }, "sample skills (first 5)"),
      )),
      h("tbody", null,
        courses.map((c, i) => h(CourseRow, { key: i, course: c }))),
    ),

    h(GapDashboard),
  );
}

createRoot(document.getElementById("root")).render(h(App));
