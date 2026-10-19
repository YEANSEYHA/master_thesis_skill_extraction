"use client";

import { useState, useEffect } from "react";
import { API_BASE } from "./ui";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

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

// ISCO-08 domains come from backend via JobBERT similarity mapping
// No hardcoded domain list needed — the API returns isco_domains dynamically

const COLORS = [
  "#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626",
  "#0891b2", "#4f46e5", "#c026d3", "#ea580c", "#16a34a",
  "#9333ea", "#0d9488", "#e11d48", "#ca8a04", "#6366f1",
];

// ── Demand tier: Hot / Rising / Stable based on rank position ──
function demandTier(index, total) {
  const pct = index / total;
  if (pct < 0.2) return { label: "Hot",    cls: "bg-red-50 text-red-600 border-red-200" };
  if (pct < 0.5) return { label: "Rising", cls: "bg-orange-50 text-orange-500 border-orange-200" };
  return               { label: "Stable",  cls: "bg-gray-50 text-gray-400 border-gray-200" };
}

// ── Auto-generate 3 key insights from data ──
function buildInsights(data) {
  if (!data) return [];
  const { top_skills, categories, total_jobs, unique_skills, curriculum } = data;
  const insights = [];

  if (top_skills?.length > 0) {
    insights.push({
      icon: "🔥",
      text: `"${top_skills[0].skill}" is the #1 most demanded skill — appearing in ${top_skills[0].count} of ${total_jobs} job postings (${Math.round(top_skills[0].count / total_jobs * 100)}%).`,
    });
  }
  if (categories?.length > 0) {
    insights.push({
      icon: "📊",
      text: `"${categories[0].category}" dominates with ${categories[0].count} jobs (${Math.round(categories[0].count / total_jobs * 100)}% of all postings).`,
    });
  }
  if (unique_skills && total_jobs) {
    const depth = (unique_skills / total_jobs).toFixed(1);
    insights.push({
      icon: "🧠",
      text: `Skill diversity is ${depth} unique skills per job — ${depth > 8 ? "high variety, diverse hiring landscape." : "moderate concentration, employers agree on core skills."}`,
    });
  }
  if (curriculum?.length > 0) {
    insights.push({
      icon: "🎓",
      text: `Top curriculum priority: "${curriculum[0].domain}" with ${curriculum[0].total_demand} market mentions.`,
    });
  }
  return insights;
}

// ── Components ────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent = "#7c3aed", delay = 0 }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 animate-fade-in-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="text-2xl font-bold text-gray-900 tabular-nums">{value}</div>
      <div className="text-xs font-medium text-gray-500 mt-0.5">{label}</div>
      {sub && <div className="text-[11px] text-gray-400 mt-1">{sub}</div>}
      <div className="mt-2 h-0.5 rounded-full w-8" style={{ backgroundColor: accent }} />
    </div>
  );
}

function SectionHeader({ title, subtitle }) {
  return (
    <div className="mb-3">
      <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
    </div>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`bg-white rounded-xl border border-gray-200 p-4 ${className}`}>
      {children}
    </div>
  );
}

function SkillBar({ skill, count, maxCount, index, total, color }) {
  const pct = (count / maxCount) * 100;
  const tier = demandTier(index, total);
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-[11px] text-gray-600 w-36 truncate shrink-0">{skill}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-1.5">
        <div className="h-1.5 rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] text-gray-400 tabular-nums w-6 text-right shrink-0">{count}</span>
      <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-full border shrink-0 ${tier.cls}`}>{tier.label}</span>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-8">
      <div className="flex items-center justify-center gap-3 py-4">
        <svg className="animate-spin h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span className="text-sm text-gray-500">Loading analytics — this may take a moment on first load while the model initializes...</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[1,2,3,4,5].map(i => (
          <div key={i} className="rounded-xl border border-gray-200 p-4">
            <div className="h-7 w-16 rounded bg-gray-100 animate-pulse mb-2" />
            <div className="h-3 w-20 rounded bg-gray-100 animate-pulse" />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[1,2].map(i => (
          <div key={i} className="bg-white rounded-xl border border-gray-200 p-5 h-72">
            <div className="h-4 w-32 rounded bg-gray-100 animate-pulse mb-4" />
            <div className="space-y-3">
              {[1,2,3,4,5].map(j => (
                <div key={j} className="flex items-center gap-2">
                  <div className="h-3 w-28 rounded bg-gray-100 animate-pulse" />
                  <div className="flex-1 h-2 rounded-full bg-gray-100 animate-pulse" />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-20 animate-fade-in-up">
      <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mx-auto mb-4">
        <svg className="w-7 h-7 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
        </svg>
      </div>
      <h2 className="text-lg font-medium text-gray-900">No Extraction Data Yet</h2>
      <p className="text-sm text-gray-400 mt-1 max-w-sm mx-auto">
        Run the pipeline first — crawl sites, classify pages, then extract skills.
      </p>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function AnalyticsTab() {
  const [data,          setData]          = useState(null);
  const [loading,       setLoading]       = useState(true);
  const [runs,          setRuns]          = useState([]);
  const [selectedRun,   setSelectedRun]   = useState("");
  const [timePreset,    setTimePreset]    = useState("all");
  const [dateFrom,      setDateFrom]      = useState("");
  const [dateTo,        setDateTo]        = useState("");
  const [includeNoDate, setIncludeNoDate] = useState(true);
  const [activeSection, setActiveSection] = useState("overview"); // overview | skills | market | curriculum
  const [selectedDomain, setSelectedDomain] = useState("All");

  useEffect(() => { fetchRuns(); fetchDashboard(); }, []);

  async function fetchRuns() {
    try {
      const res = await fetch(`${API_BASE}/api/extract/runs`);
      const d = await res.json();
      setRuns(d.runs || []);
    } catch { setRuns([]); }
  }

  async function fetchDashboard(runId, fromDate, toDate, noDate) {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (runId)   params.set("run_id", runId);
      if (fromDate) params.set("date_from", fromDate);
      if (toDate)   params.set("date_to", toDate);
      if (noDate !== undefined) params.set("include_no_date", noDate);
      const url = `${API_BASE}/api/analytics/dashboard${params.toString() ? "?" + params : ""}`;
      const res = await fetch(url);
      setData(await res.json());
    } catch { setData(null); }
    finally { setLoading(false); }
  }

  function applyPreset(preset) {
    setTimePreset(preset);
    let from = "", to = "";
    if (preset !== "all") {
      const now = new Date();
      to = now.toISOString().split("T")[0];
      const d = new Date(now);
      if (preset === "1m") d.setMonth(d.getMonth() - 1);
      else if (preset === "3m") d.setMonth(d.getMonth() - 3);
      else if (preset === "6m") d.setMonth(d.getMonth() - 6);
      else if (preset === "1y") d.setFullYear(d.getFullYear() - 1);
      from = d.toISOString().split("T")[0];
    }
    setDateFrom(from); setDateTo(to);
    fetchDashboard(selectedRun || null, from, to, includeNoDate);
  }

  if (loading) return <DashboardSkeleton />;
  if (!data || !data.total_jobs) return <EmptyState />;

  const { top_skills, trending_skills, categories, isco_domains, locations, companies, skills_by_category, skill_cooccurrence, salary_data, curriculum } = data;
  const insights = buildInsights(data);

  // Build domain skills lookup from ISCO-08 backend mapping
  const groupedSkills = {};
  for (const d of (isco_domains || [])) {
    groupedSkills[d.domain] = (d.top_skills || []);
  }

  const domainList = ["All", ...(isco_domains || []).map(d => d.domain)];

  const filteredSkills = selectedDomain === "All"
    ? top_skills
    : (groupedSkills[selectedDomain] || []);
  const filteredTrending = selectedDomain === "All"
    ? (trending_skills || [])
    : (groupedSkills[selectedDomain] || []).slice(0, 12).map((s, i, arr) => ({
        skill: s.skill,
        score: Math.round((s.count / (arr[0]?.count || 1)) * 100),
        count: s.count,
      }));

  const radarData = categories.slice(0, 8).map(c => ({
    category: c.category.length > 12 ? c.category.slice(0, 12) + ".." : c.category,
    jobs: c.count,
  }));

  const SECTION_TABS = [
    { key: "overview",    label: "Overview"    },
    { key: "skills",      label: "Skills"      },
    { key: "market",      label: "Market"      },
    { key: "curriculum",  label: "Curriculum"  },
  ];

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-start justify-between animate-fade-in-up">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Skill Analytics</h2>
          <p className="text-xs text-gray-400 mt-0.5">Cambodian job market · extracted by qwen3:14b</p>
        </div>
        <select
          value={selectedRun}
          onChange={e => { setSelectedRun(e.target.value); fetchDashboard(e.target.value || null, dateFrom, dateTo, includeNoDate); }}
          className="text-xs border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-700 focus:outline-none"
        >
          <option value="">All Runs</option>
          {runs.map(r => (
            <option key={r.run_id} value={r.run_id}>{formatRunId(r.run_id)} ({r.total_jobs} jobs)</option>
          ))}
        </select>
      </div>

      {/* ── Timeframe + filter bar ── */}
      <div className="bg-white rounded-xl border border-gray-200 px-4 py-2.5 animate-fade-in-up flex flex-wrap items-center gap-2" style={{ animationDelay: "40ms" }}>
        <span className="text-xs font-medium text-gray-400 mr-1">Timeframe</span>
        {[
          { key: "all", label: "All Time" },
          { key: "1m",  label: "1M" },
          { key: "3m",  label: "3M" },
          { key: "6m",  label: "6M" },
          { key: "1y",  label: "1Y" },
          { key: "custom", label: "Custom" },
        ].map(p => (
          <button
            key={p.key}
            onClick={() => p.key !== "custom" ? applyPreset(p.key) : setTimePreset("custom")}
            className={`px-3 py-1 text-xs font-medium rounded-md border transition-colors ${
              timePreset === p.key
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-gray-50 text-gray-500 border-gray-200 hover:bg-gray-100"
            }`}
          >
            {p.label}
          </button>
        ))}
        {timePreset === "custom" && (
          <div className="flex items-center gap-2">
            <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}
              className="text-xs border border-gray-200 rounded-md px-2 py-1 bg-white focus:outline-none" />
            <span className="text-xs text-gray-400">to</span>
            <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}
              className="text-xs border border-gray-200 rounded-md px-2 py-1 bg-white focus:outline-none" />
            <button onClick={() => { setTimePreset("custom"); fetchDashboard(selectedRun || null, dateFrom, dateTo, includeNoDate); }}
              className="px-3 py-1 text-xs font-medium bg-gray-900 text-white rounded-md">Apply</button>
          </div>
        )}
        <label className="ml-auto flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={includeNoDate}
            onChange={() => { const n = !includeNoDate; setIncludeNoDate(n); fetchDashboard(selectedRun || null, dateFrom, dateTo, n); }}
            className="w-3.5 h-3.5 rounded border-gray-300 text-gray-900"
          />
          <span className="text-xs text-gray-500">Include undated jobs</span>
          {data?.jobs_no_date > 0 && <span className="text-[10px] text-gray-400">({data.jobs_no_date})</span>}
        </label>
      </div>

      {/* ── Key Insights strip ── */}
      {insights.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 animate-fade-in-up" style={{ animationDelay: "80ms" }}>
          {insights.map((ins, i) => (
            <div key={i} className="flex items-start gap-3 bg-white rounded-xl border border-gray-200 px-4 py-3">
              <span className="text-base shrink-0">{ins.icon}</span>
              <p className="text-xs text-gray-600 leading-relaxed">{ins.text}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Stat cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="Jobs Extracted"  value={data.total_jobs}      sub="total job postings"                                           accent="#7c3aed" delay={0}   />
        <StatCard label="Unique Skills"   value={data.unique_skills}    sub="distinct skills found"                                        accent="#2563eb" delay={40}  />
        <StatCard label="Skill Mentions"  value={data.total_skills}     sub={`${(data.total_skills/data.total_jobs).toFixed(1)} per job`}  accent="#059669" delay={80}  />
        <StatCard label="Companies"       value={data.total_companies}  sub="hiring companies"                                             accent="#d97706" delay={120} />
        <StatCard label="Skill Depth"     value={(data.unique_skills/data.total_jobs).toFixed(1)} sub="unique skills / job"               accent="#dc2626" delay={160} />
      </div>

      {/* ── Section tabs ── */}
      <div className="flex items-center gap-1 border-b border-gray-200 animate-fade-in-up" style={{ animationDelay: "180ms" }}>
        {SECTION_TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setActiveSection(t.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px ${
              activeSection === t.key
                ? "border-gray-900 text-gray-900"
                : "border-transparent text-gray-400 hover:text-gray-600"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════
          OVERVIEW TAB
      ══════════════════════════════════════════════ */}
      {activeSection === "overview" && (
        <div className="space-y-6 animate-fade-in-up">

          {/* Domain filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-medium text-gray-400 shrink-0">Domain</span>
            <div className="flex flex-wrap gap-1.5">
              {domainList.map(d => (
                <button
                  key={d}
                  onClick={() => setSelectedDomain(d)}
                  className={`px-3 py-1 text-xs font-medium rounded-full border transition-colors ${
                    selectedDomain === d
                      ? "bg-gray-900 text-white border-gray-900"
                      : "bg-white text-gray-500 border-gray-200 hover:border-gray-400"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
            {selectedDomain !== "All" && (
              <span className="text-[10px] text-gray-400 ml-1">
                {filteredSkills.length} skills in {selectedDomain}
              </span>
            )}
          </div>

          {/* Top skills + Trending side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Most In-Demand */}
            <div>
              <SectionHeader
                title={selectedDomain === "All" ? "Most In-Demand Skills" : `Top Skills — ${selectedDomain}`}
                subtitle={selectedDomain === "All" ? "Raw frequency across all job postings" : `Ranked by frequency within ${selectedDomain} jobs`}
              />
              <Card>
                {filteredSkills.length > 0 ? (
                  <div className="space-y-0.5">
                    {filteredSkills.slice(0, 12).map((s, i) => (
                      <SkillBar
                        key={i} skill={s.skill} count={s.count}
                        maxCount={filteredSkills[0].count} index={i}
                        total={filteredSkills.length} color="#7c3aed"
                      />
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-gray-400 py-8 text-center">No skill data for this domain.</p>
                )}
              </Card>
            </div>

            {/* Trending / Domain normalized */}
            <div>
              <SectionHeader
                title={selectedDomain === "All" ? "Trending Skills" : `Skill Intensity — ${selectedDomain}`}
                subtitle={selectedDomain === "All" ? "Normalized 0–100 score (Google Trends style · peak = 100)" : "Normalized 0–100 within this domain (peak = 100)"}
              />
              {filteredTrending.length > 0 ? (
                <Card>
                  <div className="space-y-0.5">
                    {filteredTrending.slice(0, 12).map((s, i) => (
                      <div key={i} className="flex items-center gap-2 py-1">
                        <span className="text-[11px] text-gray-600 w-36 truncate shrink-0">{s.skill}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-blue-500 transition-all duration-500" style={{ width: `${s.score}%` }} />
                        </div>
                        <span className="text-[10px] font-semibold text-blue-600 tabular-nums w-8 text-right shrink-0">{s.score}</span>
                        <span className="text-[9px] text-gray-400 shrink-0">/100</span>
                      </div>
                    ))}
                  </div>
                </Card>
              ) : (
                <Card className="flex items-center justify-center h-40">
                  <p className="text-xs text-gray-400">Not enough data — skills need 2+ mentions</p>
                </Card>
              )}
            </div>
          </div>

          {/* Category radar + co-occurrence */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {radarData.length >= 3 && (
              <div>
                <SectionHeader title="Category Radar" subtitle="Job volume by category" />
                <Card className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                      <PolarGrid stroke="#e5e7eb" />
                      <PolarAngleAxis dataKey="category" tick={{ fontSize: 10, fill: "#6b7280" }} />
                      <PolarRadiusAxis tick={{ fontSize: 9, fill: "#9ca3af" }} />
                      <Radar dataKey="jobs" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            )}

            <div>
              <SectionHeader title="Skill Co-occurrence" subtitle="Skills that appear together in the same job posting" />
              <Card>
                {skill_cooccurrence.length === 0 ? (
                  <p className="text-xs text-gray-400 py-8 text-center">Not enough data for co-occurrence analysis</p>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {skill_cooccurrence.map((pair, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200 whitespace-nowrap truncate max-w-[120px]">{pair.skills[0]}</span>
                        <span className="text-gray-300 text-xs">+</span>
                        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 whitespace-nowrap truncate max-w-[120px]">{pair.skills[1]}</span>
                        <div className="ml-auto flex items-center gap-1 shrink-0">
                          <div className="h-1 rounded-full bg-gray-200" style={{ width: `${Math.min(40, pair.count * 4)}px` }} />
                          <span className="text-[10px] text-gray-400 tabular-nums">{pair.count}×</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════
          SKILLS TAB
      ══════════════════════════════════════════════ */}
      {activeSection === "skills" && (
        <div className="space-y-6 animate-fade-in-up">

          {/* Skills by category heatmap */}
          {Object.keys(skills_by_category).length > 0 && (
            <div>
              <SectionHeader title="Skills by Job Category" subtitle="Top demanded skills within each category" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(skills_by_category).map(([cat, skills], ci) => (
                  <Card key={cat}>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: COLORS[ci % COLORS.length] }} />
                      <span className="text-xs font-semibold text-gray-800 truncate">{cat}</span>
                      <span className="ml-auto text-[10px] text-gray-400">{skills.length} skills</span>
                    </div>
                    <div className="space-y-1.5">
                      {skills.slice(0, 7).map((s, si) => {
                        const pct = (s.count / (skills[0]?.count || 1)) * 100;
                        return (
                          <div key={si} className="flex items-center gap-2">
                            <span className="text-[11px] text-gray-600 w-32 truncate shrink-0">{s.skill}</span>
                            <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: COLORS[ci % COLORS.length] }} />
                            </div>
                            <span className="text-[10px] text-gray-400 tabular-nums w-4 text-right">{s.count}</span>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Full skill list with demand tiers */}
          <div>
            <SectionHeader title="All Skills — Demand Tiers" subtitle="Hot = top 20% · Rising = top 50% · Stable = rest" />
            <Card>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1">
                {top_skills.map((s, i) => (
                  <div key={i} className="flex items-center gap-2 py-0.5">
                    <span className="text-[11px] text-gray-600 truncate flex-1">{s.skill}</span>
                    <span className="text-[10px] text-gray-400 tabular-nums shrink-0">{s.count}</span>
                    <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-full border shrink-0 ${demandTier(i, top_skills.length).cls}`}>
                      {demandTier(i, top_skills.length).label}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════
          MARKET TAB
      ══════════════════════════════════════════════ */}
      {activeSection === "market" && (
        <div className="space-y-6 animate-fade-in-up">

          {/* Categories */}
          <div>
            <SectionHeader title="Job Categories" subtitle="Distribution of all extracted job postings" />
            <Card className="h-72">
              {categories.length <= 6 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={categories} dataKey="count" nameKey="category" cx="50%" cy="45%"
                      innerRadius="45%" outerRadius="75%" paddingAngle={2}
                      label={({ category, percent }) => `${category} ${(percent * 100).toFixed(0)}%`}
                      labelLine={{ strokeWidth: 1, stroke: "#d1d5db" }} style={{ fontSize: 11 }}
                    >
                      {categories.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip formatter={(v, n) => [v, n]} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categories.map(c => ({ ...c, short: c.category.length > 22 ? c.category.slice(0, 22) + "…" : c.category }))}
                    layout="vertical" margin={{ left: 0, right: 32 }}>
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="short" width={150} tick={{ fontSize: 11, fill: "#6b7280" }} />
                    <Tooltip formatter={(v) => [v, "jobs"]} cursor={{ fill: "rgba(0,0,0,0.03)" }} />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={16}
                      label={{ position: "right", fontSize: 11, fill: "#6b7280" }}>
                      {categories.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Card>
          </div>

          {/* Locations + Companies */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {locations.length > 0 && (
              <div>
                <SectionHeader title="Top Locations" subtitle="Where jobs are concentrated" />
                <Card>
                  <div className="space-y-1.5">
                    {locations.slice(0, 10).map((loc, i) => {
                      const pct = (loc.count / locations[0].count) * 100;
                      return (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[11px] text-gray-600 w-40 truncate shrink-0">{loc.location}</span>
                          <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                            <div className="h-1.5 rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-[10px] text-gray-400 tabular-nums w-5 text-right">{loc.count}</span>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              </div>
            )}

            {companies.length > 0 && (
              <div>
                <SectionHeader title="Top Hiring Companies" subtitle="Most active employers" />
                <Card>
                  <div className="space-y-1.5">
                    {companies.slice(0, 10).map((comp, i) => {
                      const pct = (comp.count / companies[0].count) * 100;
                      return (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[11px] text-gray-600 w-40 truncate shrink-0">{comp.company}</span>
                          <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                            <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-[10px] text-gray-400 tabular-nums w-5 text-right">{comp.count}</span>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              </div>
            )}
          </div>

          {/* Salary snapshot */}
          {salary_data.length > 0 && (
            <div>
              <SectionHeader title="Salary Snapshot" subtitle="Reported salary ranges from job postings" />
              <Card className="overflow-hidden p-0">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                      <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Job Title</th>
                      <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Category</th>
                      <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Salary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {salary_data.slice(0, 15).map((s, i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-2 text-gray-700 font-medium">{s.title}</td>
                        <td className="px-4 py-2">
                          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">{s.category}</span>
                        </td>
                        <td className="px-4 py-2 text-emerald-600 font-medium">{s.salary}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════════
          CURRICULUM TAB
      ══════════════════════════════════════════════ */}
      {activeSection === "curriculum" && (
        <div className="space-y-6 animate-fade-in-up">

          {curriculum.length > 0 ? (
            <>
              {/* Key takeaway */}
              <div className="bg-gradient-to-r from-violet-50 to-blue-50 rounded-xl border border-violet-200 p-4 flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-violet-100 flex items-center justify-center shrink-0">
                  <svg className="w-4 h-4 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-1">Curriculum Priority</h4>
                  <p className="text-xs text-gray-600 leading-relaxed">
                    Based on <strong>{data.total_jobs}</strong> job postings with <strong>{data.unique_skills}</strong> unique skills,
                    the #1 curriculum priority is <strong>{curriculum[0]?.domain}</strong> ({curriculum[0]?.total_demand} mentions).
                    {curriculum.length > 1 && <> Also prioritize <strong>{curriculum[1]?.domain}</strong>{curriculum.length > 2 && <> and <strong>{curriculum[2]?.domain}</strong></>}.</>}
                  </p>
                </div>
              </div>

              {/* Domain cards */}
              <SectionHeader title="Skill Domains by Market Demand" subtitle="Skill pills sized by demand — darker = higher priority" />
              <div className="space-y-3">
                {curriculum.map((domain, di) => (
                  <Card key={domain.domain}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-gray-900">{domain.domain}</span>
                        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-violet-50 text-violet-600 border border-violet-200">
                          {domain.total_demand} mentions
                        </span>
                      </div>
                      <span className="text-[11px] font-bold text-gray-300 tabular-nums">#{di + 1}</span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-1 mb-3">
                      <div className="h-1 rounded-full bg-gradient-to-r from-violet-500 to-blue-500 transition-all duration-700"
                        style={{ width: `${(domain.total_demand / curriculum[0].total_demand) * 100}%` }} />
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {domain.skills.map((s, si) => {
                        const intensity = Math.max(0.4, s.demand / (domain.skills[0]?.demand || 1));
                        return (
                          <span key={si} className="text-[11px] font-medium px-2.5 py-1 rounded-lg border"
                            style={{
                              backgroundColor: `rgba(124,58,237,${intensity * 0.12})`,
                              borderColor: `rgba(124,58,237,${intensity * 0.3})`,
                              color: `rgba(91,33,182,${Math.max(0.6, intensity)})`,
                            }}>
                            {s.skill}
                            <span className="ml-1 text-[9px] opacity-60">{s.demand}</span>
                          </span>
                        );
                      })}
                    </div>
                  </Card>
                ))}
              </div>
            </>
          ) : (
            <Card className="py-12 text-center">
              <p className="text-sm text-gray-400">No curriculum data available yet.</p>
            </Card>
          )}
        </div>
      )}

    </div>
  );
}
