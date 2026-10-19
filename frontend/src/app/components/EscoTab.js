"use client";

import { useState, useEffect } from "react";
import { API_BASE } from "./ui";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell,
} from "recharts";

// ─── Client-side ESCO lookup table ────────────────────────────────────────────
// Maps common raw LLM skill phrases → ESCO preferred labels
// Skills not in this table fall back to the raw phrase
const ESCO_MAP = {
  // Communication / soft skills
  "communication":                    "communication",
  "communication skills":             "apply communication strategies",
  "interpersonal communication":      "interpersonal skills",
  "interpersonal skills":             "interpersonal skills",
  "teamwork":                         "work in teams",
  "team work":                        "work in teams",
  "problem solving":                  "solve problems",
  "problem-solving":                  "solve problems",
  "critical thinking":                "think critically",
  "leadership":                       "lead a team",
  "time management":                  "manage time",
  "negotiation":                      "negotiation techniques",
  "customer service":                 "customer service",
  "report writing":                   "write work-related reports",
  "presentation skills":              "give a presentation",
  "multitasking":                     "manage several tasks at the same time",

  // IT / Software
  "project management":               "project management",
  "data analysis":                    "analyse data",
  "data analytics":                   "analyse data",
  "microsoft office":                 "use office software",
  "ms office":                        "use office software",
  "use office software":              "use office software",
  "office software":                  "use office software",
  "java":                             "Java (computer programming)",
  "java programming":                 "Java (computer programming)",
  "python":                           "Python (computer programming)",
  "python programming":               "Python (computer programming)",
  "sql":                              "query databases",
  "sql querying":                     "query databases",
  "database":                         "administer database",
  "database management":              "administer database",
  "database administration":          "administer database",
  "ui design":                        "design user interface",
  "user interface design":            "design user interface",
  "ux design":                        "design user interface",
  "software development":             "develop software",
  "software prototyping":             "develop software prototype",
  "machine learning":                 "machine learning",
  "statistical analysis":             "perform statistical analysis",
  "issue tracking":                   "use issue tracking software",
  "jira":                             "use issue tracking software",
  "network administration":           "administer ICT system",
  "cybersecurity":                    "identify ICT security risks",
  "web development":                  "web programming",
  "javascript":                       "JavaScript (computer programming language)",
  "html":                             "hypertext markup language",
  "css":                              "cascading style sheets",
  "php":                              "PHP (computer programming language)",
  "c++":                              "C++ (computer programming language)",
  "c#":                               "C# (computer programming language)",

  // Finance / Accounting
  "accounting":                       "perform accounting tasks",
  "financial reporting":              "produce financial reports",
  "financial analysis":               "analyse financial performance of a company",
  "budgeting":                        "manage budgets",
  "bookkeeping":                      "bookkeeping",
  "auditing":                         "perform auditing activities",
  "tax":                              "apply tax policies",
  "payroll":                          "process payroll",

  // HR
  "recruitment":                      "recruit employees",
  "human resources":                  "manage human resources",
  "hr management":                    "manage human resources",
  "training":                         "train employees",
  "performance management":           "manage staff",
  "employee relations":               "manage staff",

  // Marketing / Sales
  "marketing":                        "develop marketing strategy",
  "marketing strategy":               "develop marketing strategy",
  "digital marketing":                "digital marketing",
  "social media":                     "manage social media",
  "content writing":                  "write content",
  "sales":                            "perform sales activities",
  "business development":             "develop business",
  "market research":                  "perform market research",
  "seo":                              "search engine optimisation techniques",
};

// ISCO-08 domains come from backend via JobBERT similarity mapping
// No hardcoded domain list needed — the API returns isco_domains dynamically

// Apply ESCO lookup: returns { skill (raw), esco (mapped label), mapped (bool) }
function applyEsco(rawSkill) {
  const key = rawSkill.toLowerCase().trim();
  const escoLabel = ESCO_MAP[key] || rawSkill;
  return { skill: rawSkill, esco: escoLabel, mapped: !!ESCO_MAP[key] };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const COLORS = [
  "#7c3aed","#2563eb","#059669","#d97706","#dc2626",
  "#0891b2","#4f46e5","#c026d3","#ea580c","#16a34a",
];

function demandTier(index, total) {
  const pct = index / total;
  if (pct < 0.2) return { label: "Hot",    cls: "bg-red-50 text-red-600 border-red-200" };
  if (pct < 0.5) return { label: "Rising", cls: "bg-orange-50 text-orange-500 border-orange-200" };
  return               { label: "Stable",  cls: "bg-gray-50 text-gray-400 border-gray-200" };
}

function Card({ children, className = "" }) {
  return <div className={`bg-white rounded-xl border border-gray-200 p-4 ${className}`}>{children}</div>;
}

function SectionHeader({ title, subtitle }) {
  return (
    <div className="mb-3">
      <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
    </div>
  );
}

function StatCard({ label, value, sub, accent = "#7c3aed", delay = 0 }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4" style={{ animationDelay: `${delay}ms` }}>
      <div className="text-2xl font-bold text-gray-900 tabular-nums">{value}</div>
      <div className="text-xs font-medium text-gray-500 mt-0.5">{label}</div>
      {sub && <div className="text-[11px] text-gray-400 mt-1">{sub}</div>}
      <div className="mt-2 h-0.5 rounded-full w-8" style={{ backgroundColor: accent }} />
    </div>
  );
}

function Skeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[1,2,3,4].map(i => (
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

const SECTION_TABS = [
  { key: "overview", label: "Overview" },
  { key: "skills",   label: "Skills"   },
  { key: "market",   label: "Market"   },
];

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function EscoTab() {
  const [data,           setData]           = useState(null);
  const [loading,        setLoading]        = useState(true);
  const [activeSection,  setActiveSection]  = useState("overview");
  const [selectedDomain, setSelectedDomain] = useState("All");

  useEffect(() => { fetchDashboard(); }, []);

  async function fetchDashboard() {
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/api/analytics/dashboard`);
      const json = await res.json();
      setData(json);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <Skeleton />;
  if (!data)   return (
    <div className="text-center py-20">
      <p className="text-sm text-gray-400">Failed to load data. Make sure the backend is running.</p>
    </div>
  );

  // Build ESCO-mapped skills from real top_skills
  const allMapped = (data.top_skills || []).map(({ skill, count }) => {
    const { esco, mapped } = applyEsco(skill);
    return { skill, esco, count, mapped };
  });

  // ISCO-08 domains from backend (JobBERT mapped)
  const iscoDomains = data.isco_domains || [];
  const domainList = ["All", ...iscoDomains.map(d => d.domain)];

  // Build skill set per domain for filtering
  const domainSkillSets = {};
  for (const d of iscoDomains) {
    domainSkillSets[d.domain] = new Set((d.top_skills || []).map(s => s.skill));
  }

  const filteredSkills = selectedDomain === "All"
    ? allMapped
    : allMapped.filter(s => domainSkillSets[selectedDomain]?.has(s.skill));

  const totalJobs     = data.total_jobs     || 0;
  const totalMentions = allMapped.reduce((s, k) => s + k.count, 0);
  const mappedCount   = allMapped.filter(s => s.mapped).length;
  const uniqueEsco    = new Set(allMapped.map(s => s.esco)).size;

  // ISCO domain data for Market tab / radar
  const categoryData = iscoDomains.map(d => ({
    category: d.domain,
    count: d.job_count,
  }));

  const radarData = categoryData.slice(0, 8).map(c => ({
    category: c.category.length > 14 ? c.category.slice(0, 14) + ".." : c.category,
    jobs: c.count,
  }));

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Analytics (ESCO Exact Match)</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Skills standardized to ESCO taxonomy · client-side lookup · JobBERT mapping coming soon
          </p>
        </div>
        <span className="text-[10px] font-medium px-2.5 py-1 rounded-full bg-green-50 text-green-600 border border-green-200">
          Live data
        </span>
      </div>

      {/* ── ESCO info banner ── */}
      <div className="flex items-start gap-3 bg-blue-50 rounded-xl border border-blue-200 px-4 py-3">
        <span className="text-base shrink-0">🗂️</span>
        <p className="text-xs text-blue-700 leading-relaxed">
          Raw LLM skill phrases are mapped to <strong>ESCO preferred labels</strong> via a keyword lookup table.
          <span className="ml-1 font-medium">{mappedCount}/{allMapped.length}</span> skills matched.
          Unmatched skills are shown as-is. Full JobBERT cosine similarity mapping planned for backend.
        </p>
      </div>

      {/* ── Stat cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Jobs Analysed"  value={totalJobs.toLocaleString()}    sub="total job postings"                          accent="#7c3aed" delay={0}   />
        <StatCard label="ESCO Skills"    value={uniqueEsco}                     sub="unique ESCO labels"                           accent="#2563eb" delay={40}  />
        <StatCard label="Skill Mentions" value={totalMentions.toLocaleString()} sub={`${(totalMentions / (totalJobs||1)).toFixed(1)} per job`} accent="#059669" delay={80}  />
      </div>

      {/* ── Section tabs ── */}
      <div className="flex items-center gap-1 border-b border-gray-200">
        {SECTION_TABS.map(t => (
          <button key={t.key} onClick={() => setActiveSection(t.key)}
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
          OVERVIEW
      ══════════════════════════════════════════════ */}
      {activeSection === "overview" && (
        <div className="space-y-6">

          {/* Domain filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-medium text-gray-400 shrink-0">Domain</span>
            <div className="flex flex-wrap gap-1.5">
              {domainList.map(d => (
                <button key={d} onClick={() => setSelectedDomain(d)}
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
              <span className="text-[10px] text-gray-400">{filteredSkills.length} ESCO skills in {selectedDomain}</span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Top ESCO skills bar */}
            <div>
              <SectionHeader
                title={selectedDomain === "All" ? "Most In-Demand ESCO Skills" : `Top ESCO Skills — ${selectedDomain}`}
                subtitle="ESCO preferred labels ranked by frequency"
              />
              <Card>
                <div className="space-y-0.5">
                  {filteredSkills.slice(0, 12).map((s, i) => {
                    const tier = demandTier(i, filteredSkills.length);
                    const pct  = filteredSkills[0] ? (s.count / filteredSkills[0].count) * 100 : 0;
                    return (
                      <div key={i} className="flex items-center gap-2 py-1">
                        <span className="text-[11px] text-gray-600 w-40 truncate shrink-0" title={s.esco}>{s.esco}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-violet-500 transition-all duration-500" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-[10px] text-gray-400 tabular-nums w-5 text-right shrink-0">{s.count}</span>
                        <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-full border shrink-0 ${tier.cls}`}>{tier.label}</span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>

            {/* Skill intensity */}
            <div>
              <SectionHeader
                title={selectedDomain === "All" ? "Skill Intensity" : `Skill Intensity — ${selectedDomain}`}
                subtitle="Normalized 0–100 within selection (peak = 100)"
              />
              <Card>
                <div className="space-y-0.5">
                  {filteredSkills.slice(0, 12).map((s, i) => {
                    const score = filteredSkills[0] ? Math.round((s.count / filteredSkills[0].count) * 100) : 0;
                    return (
                      <div key={i} className="flex items-center gap-2 py-1">
                        <span className="text-[11px] text-gray-600 w-40 truncate shrink-0" title={s.esco}>{s.esco}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-blue-500 transition-all duration-500" style={{ width: `${score}%` }} />
                        </div>
                        <span className="text-[10px] font-semibold text-blue-600 tabular-nums w-7 text-right shrink-0">{score}</span>
                        <span className="text-[9px] text-gray-400 shrink-0">/100</span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>
          </div>

          {/* Radar */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <SectionHeader title="Category Radar" subtitle="Job volume by ESCO-grouped domain" />
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

            {/* Mapping coverage */}
            <div>
              <SectionHeader title="Mapping Coverage" subtitle="Share of skills matched to ESCO vs unmatched" />
              <Card>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {[
                    { label: "Matched to ESCO",  count: mappedCount,                   color: "#7c3aed" },
                    { label: "Unmatched (raw)",   count: allMapped.length - mappedCount, color: "#d1d5db" },
                  ].map((row, i) => {
                    const pct = allMapped.length ? Math.round((row.count / allMapped.length) * 100) : 0;
                    return (
                      <div key={i} className="flex items-center gap-3 py-1">
                        <span className="text-[11px] text-gray-600 w-36 shrink-0">{row.label}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-2">
                          <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: row.color }} />
                        </div>
                        <span className="text-[10px] text-gray-500 tabular-nums w-12 text-right shrink-0">{row.count} ({pct}%)</span>
                      </div>
                    );
                  })}
                </div>

                <div className="mt-4 pt-3 border-t border-gray-100">
                  <p className="text-[10px] text-gray-400">
                    Unmatched skills will be resolved via JobBERT cosine similarity when the backend endpoint is ready.
                  </p>
                </div>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════
          SKILLS — raw → ESCO table
      ══════════════════════════════════════════════ */}
      {activeSection === "skills" && (
        <div className="space-y-6">

          {/* Domain filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-medium text-gray-400 shrink-0">Domain</span>
            <div className="flex flex-wrap gap-1.5">
              {domainList.map(d => (
                <button key={d} onClick={() => setSelectedDomain(d)}
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
          </div>

          <SectionHeader
            title="Raw → ESCO Mapping Table"
            subtitle="How each extracted skill phrase maps to a standardized ESCO label"
          />
          <Card className="p-0 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="text-left px-4 py-2.5 text-gray-500 font-medium w-[28%]">Raw LLM Skill</th>
                  <th className="text-left px-4 py-2.5 text-gray-500 font-medium w-[35%]">ESCO Label</th>
                  <th className="text-left px-4 py-2.5 text-gray-500 font-medium w-[15%]">Domain</th>
                  <th className="text-left px-4 py-2.5 text-gray-500 font-medium w-[10%]">Count</th>
                  <th className="text-left px-4 py-2.5 text-gray-500 font-medium w-[12%]">Tier</th>
                </tr>
              </thead>
              <tbody>
                {filteredSkills.map((s, i) => {
                  const tier = demandTier(i, filteredSkills.length);
                  return (
                    <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-2 text-gray-500 italic">{s.skill}</td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-1.5">
                          {s.mapped ? (
                            <>
                              <span className="text-gray-300 text-[10px]">→</span>
                              <span className="font-medium text-gray-800">{s.esco}</span>
                            </>
                          ) : (
                            <>
                              <span className="font-medium text-gray-500">{s.esco}</span>
                              <span className="text-[9px] text-amber-500 ml-1 border border-amber-200 rounded px-1">no match</span>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-2">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">{s.domain}</span>
                      </td>
                      <td className="px-4 py-2 text-gray-500 tabular-nums">{s.count}</td>
                      <td className="px-4 py-2">
                        <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-full border ${tier.cls}`}>{tier.label}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ══════════════════════════════════════════════
          MARKET
      ══════════════════════════════════════════════ */}
      {activeSection === "market" && (
        <div className="space-y-6">
          <SectionHeader title="Job Categories (ESCO-grouped)" subtitle="Distribution after grouping raw categories into broad ESCO domains" />
          <Card className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="vertical" margin={{ left: 0, right: 40 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="category" width={120} tick={{ fontSize: 11, fill: "#6b7280" }} />
                <Tooltip formatter={v => [v, "jobs"]} cursor={{ fill: "rgba(0,0,0,0.03)" }} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={18}
                  label={{ position: "right", fontSize: 11, fill: "#6b7280" }}>
                  {categoryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

    </div>
  );
}
