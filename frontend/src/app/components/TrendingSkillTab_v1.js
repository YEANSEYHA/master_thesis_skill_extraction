"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { real_data as dummy_data } from "./real_data";
// v1 — backed by real extracted data (phase3 stage2_result.json)

const LINE_COLORS = [
  "#7c3aed",
  "#2563eb",
  "#059669",
  "#dc2626",
  "#d97706",
  "#0891b2",
  "#c026d3",
  "#ea580c",
  "#4f46e5",
  "#16a34a",
];

function buildSkillRate(data, topN = 15) {
  const totalJobs = data.length;
  const jobCount = {}; // count UNIQUE jobs per skill (not total mentions)

  for (const job of data) {
    const uniqueSkills = new Set(job.skills); // deduplicate within same job
    for (const skill of uniqueSkills) {
      jobCount[skill] = (jobCount[skill] || 0) + 1;
    }
  }

  const sorted = Object.entries(jobCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN);

  const mostInDemand = sorted.map(([skill, count]) => ({
    skill,
    count, // unique jobs
    rate: Math.round((count / totalJobs) * 1000) / 10, // e.g. 23.5%
  }));

  return { mostInDemand, totalJobs };
}

// ── Demand tier: Standard Deviation from Mean ──
// rate > mean + 1σ  → Hot
// rate > mean        → Rising
// rate ≤ mean        → Stable
function computeThresholds(skills) {
  const rates = skills.map((s) => s.rate);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const variance =
    rates.reduce((a, b) => a + (b - mean) ** 2, 0) / rates.length;
  const std = Math.sqrt(variance);
  return { mean, std };
}

// ── Trending Skills: TF-IDF Over Time ──
// Reference: Djumalieva, J. & Sleeman, C. (2018).
//   "An Open Source Skills Extraction Pipeline." Nesta.
//
// Method:
//   Each time period = a "document", each skill = a "term".
//   TF(skill, period)  = count(skill in period) / total_skill_mentions(period)
//   IDF(skill)         = log(N_periods / periods_containing_skill)
//   TF-IDF(skill, period) = TF × IDF
//
// A skill with rising TF-IDF in recent periods is "trending" — it is becoming
// disproportionately frequent relative to its historical presence.
//
// We rank skills by: Δ TF-IDF = TF-IDF(recent) − TF-IDF(earliest)
// Positive Δ = trending up, negative = declining.
//
const PERIOD_OPTIONS = [
  { key: "quarter", label: "Quarterly" },
  { key: "half", label: "Half-Year" },
  { key: "year", label: "Yearly" },
];

function buildTrendingSkills(data, periodType = "quarter", topN = 15) {
  // Group jobs into periods
  const periodJobs = {};
  for (const job of data) {
    const d = new Date(job.posted_date);
    const y = d.getFullYear();
    let key;
    if (periodType === "quarter") {
      const q = Math.ceil((d.getMonth() + 1) / 3);
      key = `${y} Q${q}`;
    } else if (periodType === "half") {
      const h = d.getMonth() < 6 ? "H1" : "H2";
      key = `${y} ${h}`;
    } else {
      key = `${y}`;
    }
    if (!periodJobs[key]) periodJobs[key] = [];
    periodJobs[key].push(job);
  }

  const periods = Object.keys(periodJobs).sort();
  const N = periods.length; // total number of "documents"

  // Step 1: Count skill mentions per period + total mentions per period
  const periodSkillCount = {}; // { period: { skill: count } }
  const periodTotalMentions = {}; // { period: total }
  for (const period of periods) {
    periodSkillCount[period] = {};
    let total = 0;
    for (const job of periodJobs[period]) {
      const unique = new Set(job.skills);
      for (const skill of unique) {
        periodSkillCount[period][skill] =
          (periodSkillCount[period][skill] || 0) + 1;
        total++;
      }
    }
    periodTotalMentions[period] = total;
  }

  // Step 2: IDF — how many periods contain each skill
  const allSkills = new Set();
  const skillPeriodPresence = {};
  for (const period of periods) {
    for (const skill of Object.keys(periodSkillCount[period])) {
      allSkills.add(skill);
      skillPeriodPresence[skill] = (skillPeriodPresence[skill] || 0) + 1;
    }
  }

  // Step 3: Compute TF-IDF for each skill in each period
  //   TF  = count(skill, period) / total_mentions(period)
  //   IDF = log(N / periods_with_skill)
  const tfidf = {}; // { skill: { period: score } }
  for (const skill of allSkills) {
    tfidf[skill] = {};
    const idf = Math.log(N / (skillPeriodPresence[skill] || 1));
    for (const period of periods) {
      const count = periodSkillCount[period][skill] || 0;
      const total = periodTotalMentions[period] || 1;
      const tf = count / total;
      tfidf[skill][period] = tf * idf;
    }
  }

  // Step 4: Rank by Δ TF-IDF (most recent period − earliest period)
  const firstPeriod = periods[0];
  const lastPeriod = periods[periods.length - 1];
  const results = [];

  for (const skill of allSkills) {
    const scoreFirst = tfidf[skill][firstPeriod] || 0;
    const scoreLast = tfidf[skill][lastPeriod] || 0;
    const delta = scoreLast - scoreFirst;

    // Filter: skill must appear in at least 2 periods and have recent presence
    if ((skillPeriodPresence[skill] || 0) < 2) continue;
    const lastCount = periodSkillCount[lastPeriod]?.[skill] || 0;
    const lastTotal = periodJobs[lastPeriod]?.length || 1;
    const lastRate = Math.round((lastCount / lastTotal) * 1000) / 10;
    if (lastRate < 2) continue; // must be ≥2% in most recent period

    // Collect TF-IDF time series for sparkline display
    const series = periods.map((p) => tfidf[skill][p] || 0);

    results.push({
      skill,
      tfidfFirst: Math.round(scoreFirst * 10000) / 10000,
      tfidfLast: Math.round(scoreLast * 10000) / 10000,
      delta: Math.round(delta * 10000) / 10000,
      lastRate,
      series,
    });
  }

  // Sort by delta descending — biggest TF-IDF increase = most trending
  results.sort((a, b) => b.delta - a.delta);
  const trending = results.slice(0, topN);

  return { trending, periods, firstPeriod, lastPeriod, totalJobs: data.length };
}

function trendLabel(delta) {
  if (delta > 0.002)
    return {
      label: "Surging",
      cls: "bg-emerald-50 text-emerald-600 border border-emerald-200",
    };
  if (delta > 0.0005)
    return {
      label: "Growing",
      cls: "bg-blue-50 text-blue-500 border border-blue-200",
    };
  if (delta > 0)
    return {
      label: "Gaining",
      cls: "bg-sky-50 text-sky-500 border border-sky-200",
    };
  if (delta > -0.0005)
    return {
      label: "Flat",
      cls: "bg-gray-50 text-gray-400 border border-gray-200",
    };
  return {
    label: "Declining",
    cls: "bg-red-50 text-red-400 border border-red-200",
  };
}

function demandTier(rate, mean, std) {
  if (rate > mean + std)
    return {
      label: "Hot",
      cls: "bg-red-50 text-red-500 border border-red-200",
    };
  if (rate > mean)
    return {
      label: "Rising",
      cls: "bg-orange-50 text-orange-400 border border-orange-200",
    };
  return {
    label: "Stable",
    cls: "bg-gray-50 text-gray-400 border border-gray-200",
  };
}

// ── Z-Score functions ──
function zToPercentile(z) {
  const a1 = 0.254829592,
    a2 = -0.284496736,
    a3 = 1.421413741;
  const a4 = -1.453152027,
    a5 = 1.061405429,
    p = 0.3275911;
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const erf =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return Math.round(0.5 * (1 + sign * erf) * 1000) / 10;
}

function buildZScoreBurst(data, windowQuarters = 2, topN = 15) {
  // Step 1: Assign each job to a quarter
  const quarterJobs = {};
  for (const job of data) {
    const d = new Date(job.posted_date);
    const q = Math.ceil((d.getMonth() + 1) / 3);
    const key = `${d.getFullYear()} Q${q}`;
    if (!quarterJobs[key]) quarterJobs[key] = [];
    quarterJobs[key].push(job);
  }

  const sortedQuarters = Object.keys(quarterJobs).sort();
  const totalQuarters = sortedQuarters.length;

  if (totalQuarters < 3)
    return {
      trending: [],
      emerging: [],
      declining: [],
      currentQuarters: [],
      baselineQuarters: [],
      totalJobs: data.length,
    };

  // Step 2: Split into BASELINE and CURRENT window
  // "half" means split total quarters 50/50 (All Time option)
  const effectiveWindow =
    windowQuarters === "half" ? Math.floor(totalQuarters / 2) : windowQuarters;
  const currentQuarters = sortedQuarters.slice(-effectiveWindow);
  const baselineQuarters = sortedQuarters.slice(0, -effectiveWindow);

  if (baselineQuarters.length < 2)
    return {
      trending: [],
      emerging: [],
      declining: [],
      currentQuarters,
      baselineQuarters,
      totalJobs: data.length,
    };

  // Step 3: Compute skill_rate per quarter
  // skill_rate(skill, quarter) = unique_jobs_with_skill / total_jobs_in_quarter
  const allSkills = new Set();
  const quarterRates = {}; // { quarter: { skill: rate } }

  for (const quarter of sortedQuarters) {
    const jobs = quarterJobs[quarter];
    const totalInQuarter = jobs.length;
    const skillCount = {};

    for (const job of jobs) {
      const unique = new Set(job.skills);
      for (const skill of unique) {
        allSkills.add(skill);
        skillCount[skill] = (skillCount[skill] || 0) + 1;
      }
    }

    quarterRates[quarter] = {};
    for (const [skill, count] of Object.entries(skillCount)) {
      quarterRates[quarter][skill] = (count / totalInQuarter) * 100; // percentage
    }
  }

  // Step 4: For each skill, compute baseline μ and σ, then Z-Score
  const results = [];

  for (const skill of allSkills) {
    // Baseline rates for this skill
    const baselineRates = baselineQuarters.map(
      (q) => quarterRates[q][skill] || 0,
    );

    // Baseline mean (μ)
    const μ = baselineRates.reduce((a, b) => a + b, 0) / baselineRates.length;

    // Baseline standard deviation (σ)
    const variance =
      baselineRates.reduce((a, b) => a + (b - μ) ** 2, 0) /
      baselineRates.length;
    const σ = Math.sqrt(variance);

    // Current window average rate
    const currentRates = currentQuarters.map(
      (q) => quarterRates[q][skill] || 0,
    );
    const fCurrent =
      currentRates.reduce((a, b) => a + b, 0) / currentRates.length;

    // Z-Score: z = (f_current - μ) / σ
    // If σ = 0 (skill had constant rate), use a small epsilon to avoid division by zero
    const z = σ > 0 ? (fCurrent - μ) / σ : fCurrent > μ ? 3.0 : 0;

    // Check if skill is EMERGING using a FIXED early anchor.
    // Anchor MUST be a subset of baseline (never overlap current) — otherwise a
    // skill could appear in only 1 current-window job yet still be classified
    // as "existed early" → contradicts Trending semantics.
    const ANCHOR_QUARTERS = Math.min(4, baselineQuarters.length);
    const anchorWindow = baselineQuarters.slice(0, ANCHOR_QUARTERS);
    const anchorRates = anchorWindow.map((q) => quarterRates[q][skill] || 0);
    const anchorAvg =
      anchorRates.reduce((a, b) => a + b, 0) / anchorRates.length;
    // Require BOTH: anchor presence AND non-zero baseline μ (skill must
    // exist in the baseline we're measuring growth against).
    const existedEarly = anchorAvg >= 1.0 && μ > 0;

    // Significance filter: |z| ≥ 2σ (95th percentile)
    if (Math.abs(z) < 2.0) continue;
    // Rising direction: require meaningful current presence
    if (z >= 2.0 && fCurrent < 1) continue;
    // Declining direction: require the skill was established (μ ≥ 1%)
    if (z <= -2.0 && μ < 1) continue;

    // Relative growth: (now - before) / before × 100
    // Guard against near-zero baselines (emerging skills) → 9999 = "NEW" sentinel
    const growth =
      μ > 0.1
        ? Math.round(((fCurrent - μ) / μ) * 100)
        : fCurrent > 0
          ? 9999
          : 0;

    results.push({
      skill,
      fCurrent: Math.round(fCurrent * 10) / 10,
      μ: Math.round(μ * 10) / 10,
      σ: Math.round(σ * 100) / 100,
      z: Math.round(z * 100) / 100,
      percentile: zToPercentile(z),
      existedEarly,
      growth,
    });
  }

  // Split by category
  //   Trending  — already existed, rising
  //   Emerging  — was absent, now appearing
  //   Declining — was established, now fading
  const trending = results
    .filter((r) => r.z >= 2.0 && r.existedEarly)
    .sort((a, b) => b.growth - a.growth)
    .slice(0, topN);
  const emerging = results
    .filter((r) => r.z >= 2.0 && !r.existedEarly)
    .sort((a, b) => b.growth - a.growth)
    .slice(0, topN);
  const declining = results
    .filter((r) => r.z <= -2.0)
    .sort((a, b) => a.growth - b.growth) // most negative growth first
    .slice(0, topN);

  return {
    trending,
    emerging,
    declining,
    currentQuarters,
    baselineQuarters,
    totalJobs: data.length,
  };
}
// ── end Z-Score functions ──

const ZSCORE_WINDOWS = [
  { key: "1q", label: "Last Quarter", quarters: 1 },
  { key: "2q", label: "Last 2 Quarters", quarters: 2 },
  { key: "1y", label: "Last Year", quarters: 4 },
  { key: "2y", label: "Last 2 Years", quarters: 8 },
  { key: "all", label: "All Time", quarters: "half" },
];

export default function TrendingSkillTab_v1() {
  const { mostInDemand, totalJobs } = useMemo(
    () => buildSkillRate(dummy_data, 15),
    [],
  );
  const { mean, std } = useMemo(
    () => computeThresholds(mostInDemand),
    [mostInDemand],
  );
  const maxRate = mostInDemand[0]?.rate || 1;
  const [periodType, setPeriodType] = useState("quarter");
  const {
    trending,
    firstPeriod,
    lastPeriod,
    totalJobs: trendTotalJobs,
  } = useMemo(
    () => buildTrendingSkills(dummy_data, periodType, 15),
    [periodType],
  );

  // ── Z-Score Burst Detection ──
  const [zWindow, setZWindow] = useState("2q");
  const zWindowQuarters =
    ZSCORE_WINDOWS.find((w) => w.key === zWindow)?.quarters || 2;
  const {
    trending: zTrending,
    emerging: zEmerging,
    currentQuarters: zCurrent,
    baselineQuarters: zBaseline,
  } = useMemo(
    () => buildZScoreBurst(dummy_data, zWindowQuarters, 10),
    [zWindowQuarters],
  );

  // Chart trajectories for the top-6 Z-Score-trending skills
  const zTrendingChartData = useMemo(() => {
    if (!zTrending || zTrending.length === 0)
      return { series: [], skills: [] };
    const topSkills = zTrending.slice(0, 6).map((r) => r.skill);
    const topSet = new Set(topSkills);
    const periodJobs = {};
    for (const job of dummy_data) {
      const d = new Date(job.posted_date);
      const q = Math.ceil((d.getMonth() + 1) / 3);
      const key = `${d.getFullYear()} Q${q}`;
      if (!periodJobs[key]) periodJobs[key] = [];
      periodJobs[key].push(job);
    }
    const periods = Object.keys(periodJobs).sort();
    const series = periods.map((period) => {
      const jobs = periodJobs[period];
      const total = jobs.length;
      const counts = {};
      for (const job of jobs) {
        for (const skill of new Set(job.skills)) {
          if (topSet.has(skill)) counts[skill] = (counts[skill] || 0) + 1;
        }
      }
      const row = { period };
      for (const skill of topSkills) {
        row[skill] = total
          ? Math.round(((counts[skill] || 0) / total) * 1000) / 10
          : 0;
      }
      return row;
    });
    return { series, skills: topSkills };
  }, [zTrending]);

  // Chart trajectories for the top-6 Z-Score-emerging skills
  const zEmergingChartData = useMemo(() => {
    if (!zEmerging || zEmerging.length === 0)
      return { series: [], skills: [] };
    const topSkills = zEmerging.slice(0, 6).map((r) => r.skill);
    const topSet = new Set(topSkills);
    const periodJobs = {};
    for (const job of dummy_data) {
      const d = new Date(job.posted_date);
      const q = Math.ceil((d.getMonth() + 1) / 3);
      const key = `${d.getFullYear()} Q${q}`;
      if (!periodJobs[key]) periodJobs[key] = [];
      periodJobs[key].push(job);
    }
    const periods = Object.keys(periodJobs).sort();
    const series = periods.map((period) => {
      const jobs = periodJobs[period];
      const total = jobs.length;
      const counts = {};
      for (const job of jobs) {
        for (const skill of new Set(job.skills)) {
          if (topSet.has(skill)) counts[skill] = (counts[skill] || 0) + 1;
        }
      }
      const row = { period };
      for (const skill of topSkills) {
        row[skill] = total
          ? Math.round(((counts[skill] || 0) / total) * 1000) / 10
          : 0;
      }
      return row;
    });
    return { series, skills: topSkills };
  }, [zEmerging]);

  return (
    <div>
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">
        Trending Skill
      </h1>

      <div className="grid grid-cols-2 gap-6 mb-8">
        {/* ── Most In-Demand Skills (Burning Glass: Skill Rate) ── */}
        <div>
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-gray-900">
              Most In-Demand Skills
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Skill Rate — % of {totalJobs.toLocaleString()} job postings
              requiring each skill
            </p>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
            {mostInDemand.map((item) => {
              const tier = demandTier(item.rate, mean, std);
              const barWidth = (item.rate / maxRate) * 100;
              return (
                <div key={item.skill} className="flex items-center gap-3">
                  <span className="w-44 text-sm text-gray-700 truncate shrink-0">
                    {item.skill}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-2">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        width: `${barWidth}%`,
                        backgroundColor: "#7c3aed",
                      }}
                    />
                  </div>
                  <span className="w-14 text-right text-xs font-semibold text-gray-700 shrink-0">
                    {item.rate}%
                  </span>
                  <span
                    className={`text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0 ${tier.cls}`}
                  >
                    {tier.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Standard Deviation Classification ── */}
        <div>
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-gray-900">
              Classification Method
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Standard Deviation from Mean — statistical thresholds
            </p>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-5">
            {/* Computed values */}
            <div>
              <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-3">
                Computed Values
              </p>
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg bg-gray-50 p-3 text-center">
                  <div className="text-lg font-bold text-gray-900 tabular-nums">
                    {mean.toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-gray-400 mt-0.5">
                    Mean (x&#772;)
                  </div>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 text-center">
                  <div className="text-lg font-bold text-gray-900 tabular-nums">
                    {std.toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-gray-400 mt-0.5">
                    Std Dev (&#963;)
                  </div>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 text-center">
                  <div className="text-lg font-bold text-gray-900 tabular-nums">
                    {(mean + std).toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-gray-400 mt-0.5">
                    x&#772; + 1&#963;
                  </div>
                </div>
              </div>
            </div>

            {/* Threshold rules */}
            <div>
              <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-3">
                Classification Rules
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-3 rounded-lg bg-red-50 border border-red-100 px-3 py-2">
                  <span className="text-[11px] font-medium text-red-500 bg-white border border-red-200 px-2 py-0.5 rounded-full">
                    Hot
                  </span>
                  <span className="text-xs text-gray-600">
                    rate &gt; {(mean + std).toFixed(1)}%
                  </span>
                  <span className="text-[11px] text-gray-400 ml-auto">
                    (x&#772; + 1&#963;)
                  </span>
                </div>
                <div className="flex items-center gap-3 rounded-lg bg-orange-50 border border-orange-100 px-3 py-2">
                  <span className="text-[11px] font-medium text-orange-400 bg-white border border-orange-200 px-2 py-0.5 rounded-full">
                    Rising
                  </span>
                  <span className="text-xs text-gray-600">
                    rate &gt; {mean.toFixed(1)}%
                  </span>
                  <span className="text-[11px] text-gray-400 ml-auto">
                    (x&#772;)
                  </span>
                </div>
                <div className="flex items-center gap-3 rounded-lg bg-gray-50 border border-gray-100 px-3 py-2">
                  <span className="text-[11px] font-medium text-gray-400 bg-white border border-gray-200 px-2 py-0.5 rounded-full">
                    Stable
                  </span>
                  <span className="text-xs text-gray-600">
                    rate &le; {mean.toFixed(1)}%
                  </span>
                  <span className="text-[11px] text-gray-400 ml-auto">
                    (&le; x&#772;)
                  </span>
                </div>
              </div>
            </div>

            {/* Distribution summary */}
            <div>
              <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-3">
                Distribution (Top {mostInDemand.length} Skills)
              </p>
              <div className="space-y-2">
                {[
                  {
                    label: "Hot",
                    cls: "bg-red-500",
                    count: mostInDemand.filter((s) => s.rate > mean + std)
                      .length,
                  },
                  {
                    label: "Rising",
                    cls: "bg-orange-400",
                    count: mostInDemand.filter(
                      (s) => s.rate > mean && s.rate <= mean + std,
                    ).length,
                  },
                  {
                    label: "Stable",
                    cls: "bg-gray-300",
                    count: mostInDemand.filter((s) => s.rate <= mean).length,
                  },
                ].map((tier) => (
                  <div key={tier.label} className="flex items-center gap-2">
                    <div className={`w-2.5 h-2.5 rounded-full ${tier.cls}`} />
                    <span className="text-xs text-gray-600 w-12">
                      {tier.label}
                    </span>
                    <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${tier.cls}`}
                        style={{
                          width: `${(tier.count / mostInDemand.length) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="text-xs font-medium text-gray-500 w-6 text-right">
                      {tier.count}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Formula reference */}
            <div className="rounded-lg bg-violet-50 border border-violet-100 px-3 py-2">
              <p className="text-[11px] text-violet-600 leading-relaxed">
                <span className="font-medium">Method:</span> Burning Glass
                Institute Skill Rate + Standard Deviation classification. Skills
                &gt;1&#963; above mean are statistical outliers in demand.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Top Trending Skills (TF-IDF) — COMMENTED OUT FOR NOW ── */}
      {false && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                Top Trending Skills
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                TF-IDF over time — {firstPeriod} to {lastPeriod} (
                {trendTotalJobs.toLocaleString()} jobs)
              </p>
            </div>
            <div className="flex bg-gray-100 rounded-lg p-0.5">
              {PERIOD_OPTIONS.map((p) => (
                <button
                  key={p.key}
                  onClick={() => setPeriodType(p.key)}
                  className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
                    periodType === p.key
                      ? "bg-white text-gray-900 shadow-sm font-medium"
                      : "text-gray-400 hover:text-gray-600"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-left">
                  <th className="pl-4 pr-2 py-2.5 text-xs font-medium text-gray-400 w-8">
                    #
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400">
                    Skill
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 text-right">
                    Current Rate
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 text-right">
                    TF-IDF ({firstPeriod})
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 text-right">
                    TF-IDF ({lastPeriod})
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 text-right">
                    &Delta; TF-IDF
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 w-28"></th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400">
                    Trend
                  </th>
                </tr>
              </thead>
              <tbody>
                {trending.map((item, i) => {
                  const tier = trendLabel(item.delta);
                  const maxDelta = trending[0]?.delta || 1;
                  const barPct =
                    maxDelta > 0
                      ? Math.min(
                          (Math.max(item.delta, 0) / maxDelta) * 100,
                          100,
                        )
                      : 0;
                  return (
                    <tr
                      key={item.skill}
                      className="border-b border-gray-50 hover:bg-gray-50/50"
                    >
                      <td className="pl-4 pr-2 py-2.5 text-xs text-gray-300">
                        {i + 1}
                      </td>
                      <td className="px-3 py-2.5 text-gray-700 font-medium">
                        {item.skill}
                      </td>
                      <td className="px-3 py-2.5 text-right text-gray-700 tabular-nums">
                        {item.lastRate}%
                      </td>
                      <td className="px-3 py-2.5 text-right text-gray-400 tabular-nums">
                        {item.tfidfFirst.toFixed(4)}
                      </td>
                      <td className="px-3 py-2.5 text-right text-gray-700 tabular-nums">
                        {item.tfidfLast.toFixed(4)}
                      </td>
                      <td
                        className="px-3 py-2.5 text-right font-semibold tabular-nums"
                        style={{
                          color: item.delta >= 0 ? "#059669" : "#dc2626",
                        }}
                      >
                        {item.delta >= 0 ? "+" : ""}
                        {item.delta.toFixed(4)}
                      </td>
                      <td className="px-3 py-2.5">
                        <div className="bg-gray-100 rounded-full h-1.5">
                          <div
                            className="h-1.5 rounded-full bg-emerald-400"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                      </td>
                      <td className="px-3 py-2.5">
                        <span
                          className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${tier.cls}`}
                        >
                          {tier.label}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-6 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <span className="font-medium">Reference:</span> Djumalieva, J.
              &amp; Sleeman, C. (2018). &quot;An Open Source Skills Extraction
              Pipeline.&quot; Nesta. Each time period is treated as a document.
              TF(skill, period) = skill_count / total_mentions. IDF(skill) =
              log(N_periods / periods_with_skill). Skills are ranked by
              &Delta;TF-IDF = TF-IDF(recent) &minus; TF-IDF(earliest). Positive
              &Delta; indicates the skill is becoming disproportionately more
              frequent.
            </p>
          </div>
        </div>
      )}

      {/* ── Trending Skill Section (Z-Score Burst Detection) ── */}
      <div className="mt-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">
              Trending Skill (Method: Z-Score Burst Detection)
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Shewhart 1931 / Mathioudakis &amp; Koudas 2010 ·{" "}
              {zTrending.length} skills · z ≥ 2σ (95th percentile) · early μ
              ≥ 1% (established) · baseline:{" "}
              {zBaseline[0] || "—"} → {zBaseline[zBaseline.length - 1] || "—"}{" "}
              · current: {zCurrent.join(", ") || "—"}
            </p>
          </div>
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            {ZSCORE_WINDOWS.map((w) => (
              <button
                key={w.key}
                onClick={() => setZWindow(w.key)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  zWindow === w.key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  📈 Fastest-Growing Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by relative growth %
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">z</span>{" "}
                = standard deviations above baseline mean.{" "}
                <span className="font-mono font-semibold text-emerald-700">
                  ↑X%
                </span>{" "}
                = relative growth — (current − μ) / μ × 100.
              </p>
            </div>
            {zTrending.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no trending skills for selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {zTrending.map((r, i) => {
                  const maxZ = zTrending[0]?.z || 1;
                  const barPct = Math.min(Math.round((r.z / maxZ) * 100), 100);
                  const tier =
                    r.growth >= 150
                      ? { label: "Surging", bar: "bg-red-500", chip: "bg-red-100 text-red-700" }
                      : r.growth >= 100
                        ? { label: "Growing", bar: "bg-amber-500", chip: "bg-amber-100 text-amber-700" }
                        : { label: "Gaining", bar: "bg-emerald-500", chip: "bg-emerald-100 text-emerald-700" };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="text-gray-300 w-5 shrink-0">{i + 1}.</span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 break-words leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          μ {r.μ}% → {r.fCurrent}%
                        </div>
                      </div>
                      <div className="w-12 bg-gray-100 rounded-full h-1.5 shrink-0">
                        <div className={`h-1.5 rounded-full ${tier.bar}`} style={{ width: `${barPct}%` }} />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.z.toFixed(2)}σ
                      </span>
                      <span className="tabular-nums font-semibold text-emerald-700 w-14 text-right shrink-0">
                        ↑{r.growth >= 9999 ? "NEW" : `${r.growth}%`}
                      </span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}>
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <span className="text-sm font-semibold text-gray-800">📊 Z-Score Trajectories</span>
              <span className="text-xs text-gray-400 ml-2">top 6 by Z · % rate per quarter</span>
            </div>
            <div className="p-3">
              {zTrendingChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">— no skills to plot —</p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={zTrendingChartData.series} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <XAxis dataKey="period" tick={{ fontSize: 10, fill: "#9ca3af" }} tickLine={false} axisLine={{ stroke: "#e5e7eb" }} interval={1} />
                    <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} tickLine={false} axisLine={false} width={32} unit="%" />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e5e7eb" }} formatter={(v) => `${v}%`} />
                    <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} iconType="circle" iconSize={6} />
                    {zTrendingChartData.skills.map((skill, i) => (
                      <Line key={skill} type="monotone" dataKey={skill} stroke={LINE_COLORS[i % LINE_COLORS.length]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Emerging Skill Section (Z-Score Burst Detection) ── */}
      <div className="mt-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">
              Emerging Skill (Method: Z-Score Burst Detection)
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Shewhart 1931 / Mathioudakis &amp; Koudas 2010 ·{" "}
              {zEmerging.length} skills · z ≥ 2σ · early μ &lt; 1% (newly
              appearing) · baseline:{" "}
              {zBaseline[0] || "—"} → {zBaseline[zBaseline.length - 1] || "—"}{" "}
              · current: {zCurrent.join(", ") || "—"}
            </p>
          </div>
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            {ZSCORE_WINDOWS.map((w) => (
              <button
                key={w.key}
                onClick={() => setZWindow(w.key)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  zWindow === w.key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  ✨ Newly-Emerging Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by relative growth %
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">z</span>{" "}
                = standard deviations above (near-zero) baseline mean.{" "}
                <span className="font-mono font-semibold text-violet-700">
                  ↑X%
                </span>{" "}
                = relative growth.{" "}
                <span className="font-mono font-semibold text-violet-700">NEW</span>{" "}
                = baseline was effectively zero.
              </p>
            </div>
            {zEmerging.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no emerging skills detected in selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {zEmerging.map((r, i) => {
                  const maxZ = zEmerging[0]?.z || 1;
                  const barPct = Math.min(Math.round((r.z / maxZ) * 100), 100);
                  const tier =
                    r.growth >= 500 || r.growth >= 9999
                      ? { label: "Breakout", bar: "bg-violet-500", chip: "bg-violet-100 text-violet-700" }
                      : r.growth >= 200
                        ? { label: "Rising", bar: "bg-blue-500", chip: "bg-blue-100 text-blue-700" }
                        : { label: "Appearing", bar: "bg-emerald-500", chip: "bg-emerald-100 text-emerald-700" };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="text-gray-300 w-5 shrink-0">{i + 1}.</span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 break-words leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          μ {r.μ}% → {r.fCurrent}%
                        </div>
                      </div>
                      <div className="w-12 bg-gray-100 rounded-full h-1.5 shrink-0">
                        <div className={`h-1.5 rounded-full ${tier.bar}`} style={{ width: `${barPct}%` }} />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.z.toFixed(2)}σ
                      </span>
                      <span className="tabular-nums font-semibold text-violet-700 w-14 text-right shrink-0">
                        {r.growth >= 9999 ? "NEW" : `↑${r.growth}%`}
                      </span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}>
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <span className="text-sm font-semibold text-gray-800">📊 Emergence Trajectories</span>
              <span className="text-xs text-gray-400 ml-2">top 6 · % rate per quarter</span>
            </div>
            <div className="p-3">
              {zEmergingChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">— no skills to plot —</p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={zEmergingChartData.series} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <XAxis dataKey="period" tick={{ fontSize: 10, fill: "#9ca3af" }} tickLine={false} axisLine={{ stroke: "#e5e7eb" }} interval={1} />
                    <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} tickLine={false} axisLine={false} width={32} unit="%" />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e5e7eb" }} formatter={(v) => `${v}%`} />
                    <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} iconType="circle" iconSize={6} />
                    {zEmergingChartData.skills.map((skill, i) => (
                      <Line key={skill} type="monotone" dataKey={skill} stroke={LINE_COLORS[i % LINE_COLORS.length]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Legacy Z-Score Trending (old table style, hidden) ── */}

      {/* ── Emerging Skills Section ── */}

      {/* ── Declining Skills Section ── */}

      {/* ── Kleinberg Burst Detection ── */}
    </div>
  );
}
