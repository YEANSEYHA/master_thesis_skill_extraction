"use client";

import { useMemo, useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { dummy_data } from "./dummy_data";
import { apiFetch } from "./ui";
// v2 — backed by synthetic dummy data (richer baseline for demo)

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

// ── Z-Score status badges (end-user friendly) ──

// ── Kleinberg Burst Detection (Kleinberg 2002) ──
function kleinbergBurst(timestamps, opts = {}) {
  const { s = 2, gamma = 1, minEvents = 5 } = opts;
  if (timestamps.length < minEvents) return [];
  const sorted = [...timestamps].sort((a, b) => a - b);
  const n = sorted.length;
  const gaps = [];
  for (let i = 1; i < n; i++)
    gaps.push(Math.max((sorted[i] - sorted[i - 1]) / 86400000, 0.001));
  const totalDays = (sorted[n - 1] - sorted[0]) / 86400000;
  if (totalDays <= 0) return [];
  const lambda0 = (n - 1) / totalDays;
  const lambda1 = lambda0 * s;
  const transCost = gamma * Math.log(n);
  const emit = (st, g) => {
    const λ = st === 0 ? lambda0 : lambda1;
    return -Math.log(λ) + λ * g;
  };
  const dp = [[0, Infinity]],
    back = [[null, null]];
  for (let i = 0; i < gaps.length; i++) {
    const cur = [Infinity, Infinity],
      bk = [null, null];
    for (let st = 0; st < 2; st++)
      for (let prev = 0; prev < 2; prev++) {
        const c =
          dp[i][prev] + (prev === st ? 0 : transCost) + emit(st, gaps[i]);
        if (c < cur[st]) {
          cur[st] = c;
          bk[st] = prev;
        }
      }
    dp.push(cur);
    back.push(bk);
  }
  let st = dp[gaps.length][0] < dp[gaps.length][1] ? 0 : 1;
  const states = new Array(gaps.length);
  for (let i = gaps.length; i > 0; i--) {
    states[i - 1] = st;
    st = back[i][st];
  }
  const bursts = [];
  let bs = null;
  for (let i = 0; i < states.length; i++) {
    if (states[i] === 1 && bs === null) bs = i + 1;
    else if (states[i] === 0 && bs !== null) {
      bursts.push({
        startTs: sorted[bs],
        endTs: sorted[i],
        nEvents: i - bs + 1,
        ongoing: false,
      });
      bs = null;
    }
  }
  if (bs !== null)
    bursts.push({
      startTs: sorted[bs],
      endTs: sorted[n - 1],
      nEvents: n - bs,
      ongoing: true,
    });
  return bursts;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx),
    hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function buildKleinbergBurst(data, opts = {}) {
  const { topN = 15, minOccurrences = 5, s = 2, gamma = 1 } = opts;

  const streams = {};
  for (const job of data) {
    const ts = new Date(job.posted_date).getTime();
    if (!Number.isFinite(ts)) continue;
    for (const skill of new Set(job.skills)) (streams[skill] ||= []).push(ts);
  }

  const raw = [];
  for (const [skill, timestamps] of Object.entries(streams)) {
    if (timestamps.length < minOccurrences) continue;
    const bursts = kleinbergBurst(timestamps, { s, gamma, minEvents: 5 });
    if (bursts.length === 0) continue;
    const headline = bursts.reduce((best, b) => {
      const bScore = b.nEvents * (b.ongoing ? 1.5 : 1);
      const bestScore = best
        ? best.nEvents * (best.ongoing ? 1.5 : 1)
        : -Infinity;
      return bScore > bestScore ? b : best;
    }, null);
    const span = (headline.endTs - headline.startTs) / 86400000;
    const totalEvents = timestamps.length;
    const preBurstCount = timestamps.filter((t) => t < headline.startTs).length;
    const firstAppearance = Math.min(...timestamps);
    const burstAge = (headline.startTs - firstAppearance) / 86400000;
    const preRate = burstAge > 0 ? preBurstCount / burstAge : 0;
    const burstRate = span > 0 ? headline.nEvents / span : headline.nEvents;
    raw.push({
      skill,
      totalEvents,
      headlineDays: Math.round(span),
      headlineEvents: headline.nEvents,
      headlineStartTs: headline.startTs,
      headlineEndTs: headline.endTs,
      ongoing: headline.ongoing,
      preRate,
      burstRate,
      intensity:
        span > 0
          ? Math.round((headline.nEvents / span) * 100) / 100
          : headline.nEvents,
    });
  }

  const sortedPreRates = [...raw.map((r) => r.preRate)].sort((a, b) => a - b);
  // EPSILON_ABS: P50 (median) instead of P25 — lets more skills pass the
  // "rare absolute" gate. P25 was overly strict for skill data where the
  // preRate distribution is dense near the lower end.
  const EPSILON_ABS = percentile(sortedPreRates, 0.5);
  // ALPHA: ratio threshold for "dormant before burst". preRate must be below
  // ALPHA × burstRate to qualify as Emerging. Cataldi et al. (2010) suggest
  // 0.4 for news/Twitter; we use 0.7 here because skill-demand series ramp
  // up more gradually than viral topics — a stricter 0.4 misclassifies real
  // newcomers (LLM, prompt engineering) as Trending once they accumulate
  // even modest pre-burst mentions.
  const ALPHA = 0.7;

  const results = raw.map((r) => {
    const isDormantRelative = r.preRate < ALPHA * r.burstRate;
    const isRareAbsolute = r.preRate < EPSILON_ABS;
    const classification =
      isDormantRelative && isRareAbsolute ? "emerging" : "trending";
    return { ...r, classification };
  });

  const maxIntensity =
    results.length > 0 ? Math.max(...results.map((r) => r.intensity)) : 1;
  for (const r of results)
    r.demandScore = Math.round((r.intensity / maxIntensity) * 100);
  results.sort((a, b) => {
    if (a.ongoing !== b.ongoing) return a.ongoing ? -1 : 1;
    return b.demandScore - a.demandScore;
  });
  return results.slice(0, topN);
}

// ── Mann-Kendall Trend Test (Mann 1945, Kendall 1975) ──
// Non-parametric test for monotonic trend in a time series. Walks every
// pair (i, j) with i < j and counts sign(x_j − x_i). Sum of those signs = S.
// Standardized to Z, then a two-tailed p-value via the normal CDF.
// Kendall's τ ∈ [-1, +1] expresses direction + strength.
//
// References:
//   Mann, H.B. (1945). "Nonparametric tests against trend." Econometrica.
//   Kendall, M.G. (1975). Rank Correlation Methods, 4th ed. Griffin.
//
// Why it fits skill demand: robust to outliers, no normality assumption,
// detects gradual monotonic growth (which Kleinberg cannot see). Used in
// hydrology, climate, and labor analytics for exactly this kind of series.
function normalCDF(z) {
  // Abramowitz & Stegun (1964), eq. 26.2.17 — same approximation as
  // zToPercentile but returns raw probability ∈ [0, 1] without rounding.
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
  return 0.5 * (1 + sign * erf);
}

function mannKendall(series) {
  const n = series.length;
  if (n < 4) return null;

  // Step 1: S statistic — sum of pairwise signs
  let S = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      S += Math.sign(series[j] - series[i]);
    }
  }

  // Step 2: variance with ties correction
  const tieGroups = {};
  for (const v of series) tieGroups[v] = (tieGroups[v] || 0) + 1;
  let tieAdj = 0;
  for (const cnt of Object.values(tieGroups)) {
    if (cnt > 1) tieAdj += cnt * (cnt - 1) * (2 * cnt + 5);
  }
  const variance = (n * (n - 1) * (2 * n + 5) - tieAdj) / 18;
  if (variance <= 0) return null;

  // Step 3: standardized Z (continuity correction: subtract/add 1)
  let Z;
  if (S > 0) Z = (S - 1) / Math.sqrt(variance);
  else if (S < 0) Z = (S + 1) / Math.sqrt(variance);
  else Z = 0;

  // Step 4: two-tailed p-value
  const pValue = 2 * (1 - normalCDF(Math.abs(Z)));

  // Step 5: Kendall's τ
  const tau = S / ((n * (n - 1)) / 2);

  return { S, Z, pValue, tau, n };
}

// Builds a per-skill quarterly %-rate series and runs Mann-Kendall on each.
// Filters: τ > 0 (rising), p < 0.05 (significant), mean rate ≥ 1% (real signal).
// Ranked by τ — strongest monotonic trend first.
function buildMannKendallTrending(data, topN = 10) {
  const periodJobs = {};
  for (const job of data) {
    const d = new Date(job.posted_date);
    const q = Math.ceil((d.getMonth() + 1) / 3);
    const key = `${d.getFullYear()} Q${q}`;
    if (!periodJobs[key]) periodJobs[key] = [];
    periodJobs[key].push(job);
  }
  const periods = Object.keys(periodJobs).sort();

  // Per-quarter %-rate for every skill (zero-filled when absent)
  const allSkills = new Set();
  const periodRates = {};
  for (const period of periods) {
    const jobs = periodJobs[period];
    const total = jobs.length;
    const counts = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        counts[skill] = (counts[skill] || 0) + 1;
        allSkills.add(skill);
      }
    }
    periodRates[period] = {};
    for (const [skill, count] of Object.entries(counts)) {
      periodRates[period][skill] = total ? (count / total) * 100 : 0;
    }
  }

  const results = [];
  for (const skill of allSkills) {
    const series = periods.map((p) => periodRates[p][skill] || 0);
    const meanRate = series.reduce((a, b) => a + b, 0) / series.length;
    if (meanRate < 1.0) continue;

    const mk = mannKendall(series);
    if (!mk) continue;
    if (mk.tau <= 0) continue;
    if (mk.pValue >= 0.05) continue;

    // Initial vs recent averages for human-readable context
    const k = Math.max(2, Math.floor(series.length / 8));
    const initialAvg = series.slice(0, k).reduce((a, b) => a + b, 0) / k;
    const recentAvg = series.slice(-k).reduce((a, b) => a + b, 0) / k;

    results.push({
      skill,
      tau: Math.round(mk.tau * 100) / 100,
      Z: Math.round(mk.Z * 100) / 100,
      pValue: mk.pValue,
      meanRate: Math.round(meanRate * 10) / 10,
      initialRate: Math.round(initialAvg * 10) / 10,
      recentRate: Math.round(recentAvg * 10) / 10,
      growth: Math.round((recentAvg - initialAvg) * 10) / 10,
      series,
    });
  }

  results.sort((a, b) => b.tau - a.tau);
  return {
    rows: results.slice(0, topN),
    periods,
    totalPeriods: periods.length,
  };
}

// ── Year-over-Year Skill Rate Growth (Lightcast / LinkedIn standard) ──
// "Growth Rate in Role" — the industry-standard headline metric. Compares
// the skill's share of postings in the current window vs the prior equivalent
// window. Simple, fast, and immediately interpretable ("went from 14% → 23%").
//
// Method:
//   current  = jobs in last W days
//   prior    = jobs in the W days before that  (same length, no overlap)
//   growth_abs = rate_current − rate_prior    (percentage points)
//   growth_rel = (rate_current − rate_prior) / rate_prior × 100   (% change)
//
// Filters:
//   • currentRate ≥ 2.0%   meaningful present-day demand
//   • priorRate   ≥ 1.0%   established baseline (excludes "from nothing"
//                          skills — those belong in Emerging)
//   • growth_abs > 0       strictly growing
//
// "all" window splits the full date range in half. Ranked by absolute growth.
//
// References:
//   Lightcast — Skill Projections methodology (kb.lightcast.io)
//   LinkedIn — Skills on the Rise 2025 methodology
//   Burning Glass Institute — Skills research methodology
const YOY_WINDOWS = { "90d": 90, "12mo": 365, "24mo": 730 };
const TRENDING_MIN_PREVIOUS_RATE = 1.0;
const TRENDING_MIN_PREVIOUS_COUNT = 5;
const TRENDING_MIN_CURRENT_COUNT = 5;
const TRENDING_MIN_DELTA_PP = 0.5;
const TRENDING_MIN_GROWTH_PERCENT = 25;
const EMERGING_MAX_PREVIOUS_RATE = 1.0;
const EMERGING_MAX_PREVIOUS_COUNT = 4;
const EMERGING_MIN_CURRENT_RATE = 1.0;
const EMERGING_MIN_CURRENT_COUNT = 5;
const EMERGING_MIN_DELTA_PP = 0.5;
const EMERGING_STRONG_CURRENT_RATE = 2.0;
const EMERGING_STRONG_DELTA_PP = 1.0;

// ── Wilson Score Interval for a single proportion ──
// Wilson, E. B. (1927). "Probable inference, the law of succession, and
// statistical inference." Journal of the American Statistical Association
// 22(158): 209-212. Recommended over Wald in Brown, Cai & DasGupta (2001).
function wilsonCI(count, n, z = 1.96) {
  if (n === 0) return [0, 0];
  const p = count / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const halfWidth =
    (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [
    Math.max(0, center - halfWidth),
    Math.min(1, center + halfWidth),
  ];
}

// ── Newcombe Hybrid Score CI on the Difference of Two Proportions ──
// Newcombe, R. G. (1998). "Interval Estimation for the Difference Between
// Independent Proportions: Comparison of Eleven Methods." Statistics in
// Medicine 17(8): 873-890.
// The paper compares 11 methods and recommends the hybrid score method
// (combining Wilson intervals for each proportion) for practical use across
// small and large samples. Returns the 95% CI on (p1 - p2).
function newcombeDiffCI(c1, n1, c2, n2, z = 1.96) {
  if (n1 === 0 || n2 === 0) return [0, 0];
  const p1 = c1 / n1;
  const p2 = c2 / n2;
  const [l1, u1] = wilsonCI(c1, n1, z);
  const [l2, u2] = wilsonCI(c2, n2, z);
  const delta = p1 - p2;
  const lower = delta - Math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2);
  const upper = delta + Math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2);
  return [lower, upper];
}

function buildYoYGrowth(data, windowKey = "12mo", topN = 10) {
  if (data.length === 0) return { rows: [], currentTotal: 0, priorTotal: 0 };

  const timestamps = data.map((j) => new Date(j.posted_date).getTime());
  const maxTs = Math.max(...timestamps);
  const minTs = Math.min(...timestamps);

  let currentStart, priorStart, priorEnd;
  if (windowKey === "all") {
    const midTs = (maxTs + minTs) / 2;
    currentStart = midTs;
    priorEnd = midTs;
    priorStart = minTs;
  } else {
    const days = YOY_WINDOWS[windowKey] ?? 365;
    currentStart = maxTs - days * 86400000;
    priorEnd = currentStart;
    priorStart = priorEnd - days * 86400000;
  }

  const currentJobs = [];
  const priorJobs = [];
  for (const job of data) {
    const t = new Date(job.posted_date).getTime();
    if (t >= currentStart && t <= maxTs) currentJobs.push(job);
    else if (t >= priorStart && t < priorEnd) priorJobs.push(job);
  }

  const computeRates = (jobs) => {
    const total = jobs.length;
    const counts = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        counts[skill] = (counts[skill] || 0) + 1;
      }
    }
    const rates = {};
    for (const [skill, count] of Object.entries(counts)) {
      rates[skill] = { count, rate: total ? (count / total) * 100 : 0 };
    }
    return { rates, total };
  };

  const cur = computeRates(currentJobs);
  const prior = computeRates(priorJobs);

  const results = [];
  for (const [skill, c] of Object.entries(cur.rates)) {
    const p = prior.rates[skill];
    const priorRate = p?.rate || 0;
    const priorCount = p?.count || 0;

    const absoluteGrowth = c.rate - priorRate;
    const relativeGrowth =
      priorRate > 0
        ? Math.round(((c.rate - priorRate) / priorRate) * 100)
        : null;

    if (priorRate < TRENDING_MIN_PREVIOUS_RATE) continue;
    if (priorCount < TRENDING_MIN_PREVIOUS_COUNT) continue;
    if (c.count < TRENDING_MIN_CURRENT_COUNT) continue;
    if (absoluteGrowth < TRENDING_MIN_DELTA_PP) continue;
    if (relativeGrowth === null || relativeGrowth < TRENDING_MIN_GROWTH_PERCENT)
      continue;
    if (c.rate <= priorRate) continue;

    // Newcombe (1998) hybrid score CI on the difference between two proportions.
    // Require the lower bound to be positive so every displayed Trending skill
    // has a statistically positive current-vs-previous increase.
    const [ciLow, ciHigh] = newcombeDiffCI(
      c.count,
      cur.total,
      priorCount,
      prior.total,
    );
    if (ciLow <= 0) continue;

    results.push({
      skill,
      currentRate: Math.round(c.rate * 10) / 10,
      priorRate: Math.round(priorRate * 10) / 10,
      absoluteGrowth: Math.round(absoluteGrowth * 10) / 10,
      relativeGrowth,
      currentCount: c.count,
      priorCount,
      ciLow: Math.round(ciLow * 1000) / 10, // as percentage points, 1 decimal
      ciHigh: Math.round(ciHigh * 1000) / 10,
    });
  }

  results.sort((a, b) => b.absoluteGrowth - a.absoluteGrowth);
  return {
    rows: results.slice(0, topN),
    currentTotal: cur.total,
    priorTotal: prior.total,
    windowKey,
  };
}

// ── Inverse-YoY for Emerging Skills ──
// Same statistical machinery as buildYoYGrowth, with priorCount < 5 instead
// of >= 5. Surfaces skills that didn't have an established baseline in the
// prior window but now appear meaningfully — the "from near-zero arrival"
// pattern. Symmetric with the Trending panel so the two are mutually
// exclusive: every skill that passes the Newcombe significance test is
// routed to exactly one panel based on its prior baseline strength.
//
// Cite: Newcombe (1998b) hybrid score CI on difference of proportions.
function buildYoYEmerging(data, windowKey = "12mo", topN = 10) {
  if (data.length === 0)
    return { rows: [], currentTotal: 0, priorTotal: 0, windowKey };

  const timestamps = data.map((j) => new Date(j.posted_date).getTime());
  const maxTs = Math.max(...timestamps);
  const minTs = Math.min(...timestamps);

  let currentStart, priorStart, priorEnd;
  if (windowKey === "all") {
    const midTs = (maxTs + minTs) / 2;
    currentStart = midTs;
    priorEnd = midTs;
    priorStart = minTs;
  } else {
    const days = YOY_WINDOWS[windowKey] ?? 365;
    currentStart = maxTs - days * 86400000;
    priorEnd = currentStart;
    priorStart = priorEnd - days * 86400000;
  }

  const currentJobs = [];
  const priorJobs = [];
  for (const job of data) {
    const t = new Date(job.posted_date).getTime();
    if (t >= currentStart && t <= maxTs) currentJobs.push(job);
    else if (t >= priorStart && t < priorEnd) priorJobs.push(job);
  }

  const computeRates = (jobs) => {
    const total = jobs.length;
    const counts = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        counts[skill] = (counts[skill] || 0) + 1;
      }
    }
    const rates = {};
    for (const [skill, count] of Object.entries(counts)) {
      rates[skill] = { count, rate: total ? (count / total) * 100 : 0 };
    }
    return { rates, total };
  };

  const cur = computeRates(currentJobs);
  const prior = computeRates(priorJobs);

  const results = [];
  for (const [skill, c] of Object.entries(cur.rates)) {
    const p = prior.rates[skill];
    const priorRate = p?.rate || 0;
    const priorCount = p?.count || 0;

    // EMERGING gate: skill must have near-zero prior presence but meaningful
    // current demand. This keeps Emerging distinct from Trending.
    if (priorRate >= EMERGING_MAX_PREVIOUS_RATE) continue;
    if (priorCount > EMERGING_MAX_PREVIOUS_COUNT) continue;
    if (c.rate < EMERGING_MIN_CURRENT_RATE) continue;
    if (c.count < EMERGING_MIN_CURRENT_COUNT) continue;

    const absoluteGrowth = c.rate - priorRate;
    if (absoluteGrowth < EMERGING_MIN_DELTA_PP) continue;

    // Statistical significance via Newcombe CI on difference.
    const [ciLow, ciHigh] = newcombeDiffCI(
      c.count,
      cur.total,
      priorCount,
      prior.total,
    );
    if (ciLow <= 0) continue;

    const relativeGrowth =
      priorRate > 0
        ? Math.round(((c.rate - priorRate) / priorRate) * 100)
        : null;

    results.push({
      skill,
      currentRate: Math.round(c.rate * 10) / 10,
      priorRate: Math.round(priorRate * 10) / 10,
      absoluteGrowth: Math.round(absoluteGrowth * 10) / 10,
      relativeGrowth,
      currentCount: c.count,
      priorCount,
      ciLow: Math.round(ciLow * 1000) / 10,
      ciHigh: Math.round(ciHigh * 1000) / 10,
    });
  }

  // Rank by current rate — emerging skills have near-zero priorRate,
  // so the absolute current presence is the most meaningful magnitude.
  results.sort((a, b) => b.currentRate - a.currentRate);

  return {
    rows: results.slice(0, topN),
    currentTotal: cur.total,
    priorTotal: prior.total,
    windowKey,
  };
}

// ── Google-Trends-style: Top + Rising ──
// Mirrors Google Trends' "Top queries" / "Rising queries" two-panel layout.
//   • Top  = highest absolute skill rate in the current window (volume).
//   • Rising = biggest % change vs the prior equivalent window (momentum).
//
// Both panels share a single global scale (max currentRate of the Top panel)
// so visual bar lengths are directly comparable between panels — a small bar
// in the Rising panel means "yes it's growing fast but still small overall."
//
// Breakouts: skills whose priorRate ≈ 0 are flagged "Breakout" (no % shown);
// matches Google Trends' own treatment of newly-emerging queries.
const GT_WINDOWS = { "90d": 90, "12mo": 365, "24mo": 730 };

function buildGoogleTrendsData(data, windowKey = "12mo", topN = 10) {
  if (data.length === 0)
    return {
      top: [],
      rising: [],
      currentTotal: 0,
      priorTotal: 0,
      globalMax: 1,
    };

  const timestamps = data.map((j) => new Date(j.posted_date).getTime());
  const maxTs = Math.max(...timestamps);
  const minTs = Math.min(...timestamps);

  let currentStart, priorStart, priorEnd;
  if (windowKey === "all") {
    const midTs = (maxTs + minTs) / 2;
    currentStart = midTs;
    priorEnd = midTs;
    priorStart = minTs;
  } else {
    const days = GT_WINDOWS[windowKey] ?? 365;
    currentStart = maxTs - days * 86400000;
    priorEnd = currentStart;
    priorStart = priorEnd - days * 86400000;
  }

  const currentJobs = [];
  const priorJobs = [];
  for (const job of data) {
    const t = new Date(job.posted_date).getTime();
    if (t >= currentStart && t <= maxTs) currentJobs.push(job);
    else if (t >= priorStart && t < priorEnd) priorJobs.push(job);
  }

  const computeRates = (jobs) => {
    const total = jobs.length;
    const counts = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        counts[skill] = (counts[skill] || 0) + 1;
      }
    }
    const rates = {};
    for (const [skill, count] of Object.entries(counts)) {
      rates[skill] = { count, rate: total ? (count / total) * 100 : 0 };
    }
    return { rates, total };
  };

  const cur = computeRates(currentJobs);
  const prior = computeRates(priorJobs);

  // Build a unified candidate list with current rate + signed change.
  const candidates = [];
  for (const [skill, c] of Object.entries(cur.rates)) {
    if (c.rate < 1.0) continue;
    const p = prior.rates[skill];
    const priorRate = p?.rate || 0;
    const change =
      priorRate > 0.5
        ? Math.round(((c.rate - priorRate) / priorRate) * 100)
        : null; // "Breakout" — no prior baseline
    candidates.push({
      skill,
      currentRate: Math.round(c.rate * 10) / 10,
      priorRate: Math.round(priorRate * 10) / 10,
      change,
    });
  }

  // Top: ranked by current rate (volume)
  const top = [...candidates]
    .sort((a, b) => b.currentRate - a.currentRate)
    .slice(0, topN);

  // Rising: positive change OR breakout, with meaningful current presence,
  // ranked by change% desc (breakouts surface first).
  const rising = candidates
    .filter((r) => r.currentRate >= 2.0)
    .filter((r) => r.change === null || r.change > 0)
    .sort((a, b) => {
      if (a.change === null && b.change !== null) return -1;
      if (b.change === null && a.change !== null) return 1;
      if (a.change === null && b.change === null)
        return b.currentRate - a.currentRate;
      return b.change - a.change;
    })
    .slice(0, topN);

  // Global scale: max currentRate from Top panel — shared by both panels.
  const globalMax = top[0]?.currentRate || 1;

  return {
    top,
    rising,
    currentTotal: cur.total,
    priorTotal: prior.total,
    globalMax,
    windowKey,
  };
}

// ── Linear Regression Slope + R² ──
// Fit a least-squares straight line through each skill's quarterly %-rate
// series. The slope (m) tells how many percentage points per quarter the
// skill is gaining/losing. R² tells how well a straight line explains the
// data (0 = pure noise, 1 = perfect fit). This is the most-readable trend
// method: two intuitive numbers, both grounded in basic regression.
//
// Method (least squares):
//   m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
//   b = (Σy − m·Σx) / n
//   ŷᵢ = m·xᵢ + b
//   R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
//
// Filters:
//   • slope    > 0      strictly rising
//   • R²       ≥ 0.5    line explains at least half the variance (decent fit)
//   • meanRate ≥ 1.0%   meaningful overall presence
//
// Ranked by slope descending (steepest rise first).
//
// References:
//   Standard ordinary least squares (Gauss 1809, Legendre 1805).
//   WEF Future of Jobs reports use this with fixed effects for skill demand.
function linearRegression(series) {
  const n = series.length;
  if (n < 4) return null;

  // x = [0, 1, 2, ..., n-1] — quarter index
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += series[i];
    sumXY += i * series[i];
    sumX2 += i * i;
  }
  const meanY = sumY / n;
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return null;

  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  // R² = 1 − SS_res / SS_tot
  let ssRes = 0,
    ssTot = 0;
  for (let i = 0; i < n; i++) {
    const yHat = slope * i + intercept;
    ssRes += (series[i] - yHat) ** 2;
    ssTot += (series[i] - meanY) ** 2;
  }
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  return { slope, intercept, r2, n };
}

const LRS_WINDOWS = { "90d": 90, "12mo": 365, "24mo": 730 };

function buildLRSTrending(data, windowKey = "all", topN = 10) {
  // Filter to the requested window (if not "all")
  let filtered = data;
  if (windowKey !== "all" && data.length > 0) {
    const maxTs = Math.max(
      ...data.map((j) => new Date(j.posted_date).getTime()),
    );
    const days = LRS_WINDOWS[windowKey] ?? 365;
    const cutoff = maxTs - days * 86400000;
    filtered = data.filter((j) => new Date(j.posted_date).getTime() >= cutoff);
  }

  const periodJobs = {};
  for (const job of filtered) {
    const d = new Date(job.posted_date);
    const q = Math.ceil((d.getMonth() + 1) / 3);
    const key = `${d.getFullYear()} Q${q}`;
    if (!periodJobs[key]) periodJobs[key] = [];
    periodJobs[key].push(job);
  }
  const periods = Object.keys(periodJobs).sort();

  // Per-quarter %-rate for every skill (zero-filled when absent)
  const allSkills = new Set();
  const periodRates = {};
  for (const period of periods) {
    const jobs = periodJobs[period];
    const total = jobs.length;
    const counts = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        counts[skill] = (counts[skill] || 0) + 1;
        allSkills.add(skill);
      }
    }
    periodRates[period] = {};
    for (const [skill, count] of Object.entries(counts)) {
      periodRates[period][skill] = total ? (count / total) * 100 : 0;
    }
  }

  const results = [];
  for (const skill of allSkills) {
    const series = periods.map((p) => periodRates[p][skill] || 0);
    const meanRate = series.reduce((a, b) => a + b, 0) / series.length;
    if (meanRate < 1.0) continue;

    const lrs = linearRegression(series);
    if (!lrs) continue;
    if (lrs.slope <= 0) continue;
    if (lrs.r2 < 0.5) continue;

    // Initial vs recent averages for human-readable context
    const k = Math.max(2, Math.floor(series.length / 8));
    const initialAvg = series.slice(0, k).reduce((a, b) => a + b, 0) / k;
    const recentAvg = series.slice(-k).reduce((a, b) => a + b, 0) / k;

    // Meaningful-presence floor: skill must currently appear in ≥ 3% of jobs.
    // Below 3% the rate is too noisy for the % growth number to mean anything
    // (3% of a 1,050-job window = only 31 jobs). Blocks tiny-base distortion.
    if (recentAvg < 3.0) continue;

    // Relative growth % since the start. Null when initialAvg is ~0
    // (skill emerged from nothing — "NEW", not a real percentage).
    const relativeGrowth =
      initialAvg > 0.1
        ? Math.round(((recentAvg - initialAvg) / initialAvg) * 100)
        : null;

    results.push({
      skill,
      slope: Math.round(lrs.slope * 100) / 100,
      r2: Math.round(lrs.r2 * 100) / 100,
      meanRate: Math.round(meanRate * 10) / 10,
      initialRate: Math.round(initialAvg * 10) / 10,
      recentRate: Math.round(recentAvg * 10) / 10,
      relativeGrowth,
      series,
    });
  }

  // Sort: NEW (null relativeGrowth) first — they're effectively infinite
  // growth; ranked among themselves by recentRate. Then established skills
  // by relativeGrowth desc — matches the visible ↑% column so users see
  // the list ordered by what they're actually reading.
  results.sort((a, b) => {
    const aNew = a.relativeGrowth === null;
    const bNew = b.relativeGrowth === null;
    if (aNew && !bNew) return -1;
    if (bNew && !aNew) return 1;
    if (aNew && bNew) return b.recentRate - a.recentRate;
    return b.relativeGrowth - a.relativeGrowth;
  });
  return {
    rows: results.slice(0, topN),
    periods,
    totalPeriods: periods.length,
  };
}

// ── Z-Score to percentile (%) ──
// Approximation of CDF using Abramowitz & Stegun (1964)
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

// ── Z-Score Burst Detection (Shewhart, 1931; Aggarwal, 2007) ──
// Compares current skill frequency against historical baseline.
// z(s) = (f_current - μ_baseline) / σ_baseline
// Skills with z > 1.0 = trending, absent from baseline = emerging.
//
const ZSCORE_WINDOWS = [
  { key: "1q", label: "Last Quarter", quarters: 1 },
  { key: "2q", label: "Last 2 Quarters", quarters: 2 },
  { key: "1y", label: "Last Year", quarters: 4 },
  { key: "2y", label: "Last 2 Years", quarters: 8 },
  { key: "all", label: "All Time", quarters: "half" }, // 50/50 split across full range
];

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
      results: [],
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
      results: [],
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

    // Check if skill is EMERGING using a FIXED early anchor of the full dataset
    // (window-independent: same verdict regardless of which current window the user picks)
    const ANCHOR_QUARTERS = 4;
    const anchorWindow = sortedQuarters.slice(0, ANCHOR_QUARTERS);
    const anchorRates = anchorWindow.map((q) => quarterRates[q][skill] || 0);
    const anchorAvg =
      anchorRates.reduce((a, b) => a + b, 0) / anchorRates.length;
    const existedEarly = anchorAvg >= 1.0;

    // Significance filter: |z| ≥ 2σ (95th percentile)
    if (Math.abs(z) < 2.0) continue;
    // Rising direction: require meaningful current presence
    if (z >= 2.0 && fCurrent < 1) continue;
    // Declining direction: require the skill was established (μ ≥ 1%)
    if (z <= -2.0 && μ < 1) continue;

    // Relative growth: (now - before) / before × 100
    // Guard against near-zero baselines (emerging skills)
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
      growth,
      existedEarly,
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

export default function TrendingSkillTab_v2() {
  // Data source. Only "scrape" (real Phase-3 postings) is exposed in the UI;
  // scraped data is fetched on mount.
  const [dataSource, setDataSource] = useState("scrape");
  const [scrapeData, setScrapeData] = useState(null);
  const [scrapeError, setScrapeError] = useState(null);

  useEffect(() => {
    if (dataSource !== "scrape" || scrapeData || scrapeError) return;
    let cancelled = false;
    apiFetch("/api/trends/scrape-postings")
      .then((d) => !cancelled && setScrapeData(d.postings || []))
      .catch((e) => !cancelled && setScrapeError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, [dataSource, scrapeData, scrapeError]);

  // Derived (avoids a synchronous setState in the effect).
  const scrapeLoading =
    dataSource === "scrape" && scrapeData === null && !scrapeError;

  // Active dataset: real scraped postings when "Scrape data" is selected, else
  // the synthetic dummy_data. The time-series panels (Trending/Emerging) use
  // only dated postings; Most Demanded uses all skill-bearing postings.
  const activeData = useMemo(
    () => (dataSource === "scrape" ? scrapeData || [] : dummy_data),
    [dataSource, scrapeData],
  );
  const activeTrendData = useMemo(
    () =>
      dataSource === "scrape"
        ? (scrapeData || []).filter((p) => p.posted_date)
        : dummy_data,
    [dataSource, scrapeData],
  );
  // ── In-Demand Skills: timeline filter ──
  // Windows chosen to match industry analytics platforms (Lightcast, LinkedIn
  // Economic Graph, Indeed Hiring Lab). 12mo is the default — it matches
  // university curriculum review cycles and gives ~1k jobs (±1% precision).
  // Uses the dataset's max timestamp as the reference point (not "today")
  // so the windows stay meaningful regardless of clock vs data drift.
  const [inDemandWindow, setInDemandWindow] = useState("12mo");
  const inDemandData = useMemo(() => {
    if (inDemandWindow === "all") return activeData;
    const maxTs = Math.max(
      ...activeData.map((j) => new Date(j.posted_date).getTime()),
    );
    const days = { "90d": 90, "12mo": 365, "24mo": 730 }[inDemandWindow];
    const cutoff = maxTs - days * 86400000;
    // Keep undated postings (scraped demand we can't timestamp) so the window
    // doesn't collapse the count; dummy data is all-dated so this is a no-op.
    return activeData.filter(
      (j) => !j.posted_date || new Date(j.posted_date).getTime() >= cutoff,
    );
  }, [inDemandWindow, activeData]);
  const { mostInDemand, totalJobs } = useMemo(
    () => buildSkillRate(inDemandData, 10),
    [inDemandData],
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

  // Chart data for TF-IDF Δ: %-rate per quarter of the top-6 ΔTF-IDF skills.
  // We plot %-rate (not raw TF-IDF) so the y-axis reads as "% of jobs",
  // matching the Mann-Kendall and YoY charts — easier to scan side-by-side.
  const tfidfChartData = useMemo(() => {
    if (trending.length === 0) return { series: [], skills: [] };
    const topSkills = trending.slice(0, 6).map((r) => r.skill);
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
  }, [trending]);

  // Z-Score Burst Detection
  const [zWindow] = useState("2q");
  const zWindowQuarters =
    ZSCORE_WINDOWS.find((w) => w.key === zWindow)?.quarters || 2;
  const {
    declining: zDeclining,
  } = useMemo(
    () => buildZScoreBurst(dummy_data, zWindowQuarters, 10),
    [zWindowQuarters],
  );

  // Chart trajectories for the top-6 Z-Score-trending skills.
  // Matches the YoY chart pattern — recompute %-rate per quarter for the
  // same skills the list ranks, so the chart visually tells the same story.

  // Chart trajectories for top-6 Z-Score-emerging skills.
  // Same pattern as zTrendingChartData but on the emerging classification.

  // Kleinberg Burst Detection
  // Run on the FULL dataset so the baseline (λ0) is computed across the entire
  // 4-year history — essential for the trending/emerging classification to work.
  // The time filter then narrows the *displayed* bursts to those whose headline
  // period falls within the selected window. Filtering the input instead of the
  // output strips Kleinberg of the historical baseline it needs.
  const [kTimeFilter, setKTimeFilter] = useState("all");

  const kAllResults = useMemo(
    () => buildKleinbergBurst(dummy_data, { topN: 500 }),
    [],
  );

  // Returns the candidate pool for the selected window — DO NOT slice to 10
  // here, because trending and emerging are ranked independently below.
  const kResults = useMemo(() => {
    if (kTimeFilter === "all") return kAllResults;

    const maxTs = Math.max(
      ...dummy_data.map((j) => new Date(j.posted_date).getTime()),
    );
    const days = { year: 365, month: 30, week: 7 }[kTimeFilter];
    const cutoff = maxTs - days * 86400000;

    // Count actual mentions per skill INSIDE the window.
    // Ongoing bursts (most of the AI-boom skills) all share endTs ≈ maxTs,
    // so filtering by endTs >= cutoff gives the same set for every window.
    // Counting events-in-window naturally differentiates: a skill with 200
    // year-mentions and only 3 month-mentions ranks below a fresher skill
    // when the window shrinks.
    const recentCount = {};
    for (const job of dummy_data) {
      const ts = new Date(job.posted_date).getTime();
      if (ts < cutoff) continue;
      for (const skill of new Set(job.skills)) {
        recentCount[skill] = (recentCount[skill] || 0) + 1;
      }
    }

    // Adaptive minimum: keep the threshold proportional to window length so
    // "Last week" isn't over-filtered, "Last year" isn't flooded with noise.
    const minEvents = days >= 365 ? 5 : days >= 30 ? 2 : 1;

    return kAllResults
      .filter((r) => (recentCount[r.skill] || 0) >= minEvents)
      .map((r) => ({ ...r, recentEvents: recentCount[r.skill] }))
      .sort((a, b) => b.recentEvents - a.recentEvents);
  }, [kTimeFilter, kAllResults]);

  // Rank trending and emerging independently — each panel gets its own top 10
  // instead of competing for slots in a shared top-10. Without this split,
  // trending skills (much more numerous) always crowd emerging down to 1–2 rows.
  const kTrending = kResults
    .filter((r) => r.classification === "trending")
    .slice(0, 10);
  const kEmerging = kResults
    .filter((r) => r.classification === "emerging")
    .slice(0, 10);

  // ── Emerging panel (Inverse-Newcombe) — symmetric with Trending YoY ──
  // Uses the same statistical framework as the Trending panel:
  //   - Newcombe (1998b) 95% CI on (currentRate − priorRate)
  //   - priorCount filter INVERTED: < 5 (was ≥ 5 for Trending)
  // Mutually exclusive with Trending — every skill that passes the Newcombe
  // significance test is routed to exactly one panel based on prior-baseline
  // strength. Coherent single-method framework for the methodology section.
  const [emWindow, setEmWindow] = useState("12mo");

  const emergingResults = useMemo(
    () => buildYoYEmerging(activeTrendData, emWindow, 10),
    [emWindow, activeTrendData],
  );

  // Chart data: quarterly %-rate for the top-6 emerging skills.
  // Same pattern as trending/yoy chart data for visual consistency.
  const emergingChartData = useMemo(() => {
    if (emergingResults.rows.length === 0) return { series: [], skills: [] };
    const topSkills = emergingResults.rows.slice(0, 6).map((r) => r.skill);
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
  }, [emergingResults]);

  // ── Chart data: % rate of the top trending skills, per quarter ──
  // Scoped to whatever kTrending currently identifies (responds to window
  // filter). Y-axis = % of jobs in that quarter that mention the skill.
  const trendingChartData = useMemo(() => {
    if (kTrending.length === 0) return { series: [], skills: [] };
    const topSkills = kTrending.slice(0, 6).map((t) => t.skill);

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
          counts[skill] = (counts[skill] || 0) + 1;
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
  }, [kTrending]);

  // ── Mann-Kendall Trending ──
  // Independent of the Kleinberg pipeline. Computes a quarterly %-rate series
  // per skill, then runs the Mann-Kendall monotonic-trend test. The chart
  // re-uses the same series — so the list and chart tell the same story.
  const mkTrending = useMemo(
    () => buildMannKendallTrending(dummy_data, 10),
    [],
  );

  // Chart data for Mann-Kendall: recompute %-rate per quarter for the top
  // skills, matching the pattern used by trendingChartData (the compiler
  // can't preserve memoization through indexed reads of r.series[i]).
  const mkChartData = useMemo(() => {
    if (mkTrending.rows.length === 0) return { series: [], skills: [] };
    const topSkills = mkTrending.rows.slice(0, 6).map((r) => r.skill);
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
  }, [mkTrending]);

  // ── Year-over-Year Skill Rate Growth ──
  // Lightcast/LinkedIn industry-standard method. Time filter switches the
  // comparison window: 90d, 12mo (default), 24mo, or full-range split.
  const [yoyWindow, setYoyWindow] = useState("12mo");

  const yoyResults = useMemo(
    () => buildYoYGrowth(activeTrendData, yoyWindow, 10),
    [yoyWindow, activeTrendData],
  );

  // Chart: quarterly %-rate trajectories for the top-6 YoY growers
  const yoyChartData = useMemo(() => {
    if (yoyResults.rows.length === 0) return { series: [], skills: [] };
    const topSkills = yoyResults.rows.slice(0, 6).map((r) => r.skill);
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
  }, [yoyResults]);

  // ── Linear Regression Slope + R² ──
  // Fits a least-squares line through each skill's quarterly %-rate series.
  // Two intuitive numbers: slope = pp/quarter, R² = goodness-of-fit (0-1).
  // Window filter scopes which quarters the line is fitted through.
  const [lrsWindow, setLrsWindow] = useState("all");

  const lrsTrending = useMemo(
    () => buildLRSTrending(dummy_data, lrsWindow, 10),
    [lrsWindow],
  );

  // Chart: quarterly %-rate trajectories for the top-6 by slope.
  // Chart range matches the LRS window so the visible lines match the fit.
  const lrsChartData = useMemo(() => {
    if (lrsTrending.rows.length === 0) return { series: [], skills: [] };
    const topSkills = lrsTrending.rows.slice(0, 6).map((r) => r.skill);
    const topSet = new Set(topSkills);

    let filtered = dummy_data;
    if (lrsWindow !== "all") {
      const maxTs = Math.max(
        ...dummy_data.map((j) => new Date(j.posted_date).getTime()),
      );
      const days = { "90d": 90, "12mo": 365, "24mo": 730 }[lrsWindow] ?? 365;
      const cutoff = maxTs - days * 86400000;
      filtered = dummy_data.filter(
        (j) => new Date(j.posted_date).getTime() >= cutoff,
      );
    }

    const periodJobs = {};
    for (const job of filtered) {
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
  }, [lrsTrending, lrsWindow]);

  // ── Google-Trends-style: Top + Rising ──
  // Single source of truth for both panels — they share window state and
  // bar scale so visual lengths are comparable between Top and Rising.
  const [gtWindow, setGtWindow] = useState("12mo");
  const gtData = useMemo(
    () => buildGoogleTrendsData(dummy_data, gtWindow, 10),
    [gtWindow],
  );

  // Raw count: total mentions across all jobs (top 20)

  return (
    <div>

      {/* ── Skill Trend Line Chart ── */}

      {/* ── Data-source toggle (UI only) ── */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium uppercase tracking-wide text-gray-400">
          Data source
        </span>
        <div className="flex w-fit items-center gap-1 rounded-lg bg-gray-100 p-1">
          <button
            onClick={() => setDataSource("scrape")}
            className={`flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              dataSource === "scrape"
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Scrape data
          </button>
        </div>
      </div>

      {/* ── Scrape-data status banner ── */}
      {dataSource === "scrape" && (
        <div
          className={`mb-6 rounded-lg border px-4 py-3 text-sm ${
            scrapeError
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-blue-100 bg-blue-50 text-blue-700"
          }`}
        >
          {scrapeLoading
            ? "Loading real scraped postings…"
            : scrapeError
              ? `Couldn't load scraped data: ${scrapeError}`
              : `Real scraped data — ${activeData.length.toLocaleString()} postings (${activeTrendData.length.toLocaleString()} with usable dates). Most Demanded uses all postings; Trending & Emerging use only dated postings, so they may be sparse.`}
        </div>
      )}

      {/* ── 2-column grid: Row1 = Most Demanded | Trending, Row2 = Emerging ── */}
      <div className="grid grid-cols-2 gap-6 items-stretch">
      <div className="min-w-0">
        {/* ── Most In-Demand Skills (Burning Glass: Skill Rate) ── */}
        <div className="flex h-full flex-col">
          <div className="flex flex-col gap-3 mb-4">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight text-gray-950">
                Most Demanded Skills
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Skill Rate — % of {totalJobs.toLocaleString()} job postings
                requiring each skill
              </p>
            </div>
            <div className="flex w-fit items-center gap-1 bg-gray-100 rounded-lg p-1 shrink-0">
              {[
                { key: "90d", label: "90d" },
                { key: "12mo", label: "12mo" },
                { key: "24mo", label: "24mo" },
                { key: "all", label: "All time" },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setInDemandWindow(key)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                    inDemandWindow === key
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm flex-1">
            <div className="flex items-center gap-3 border-b border-gray-100 bg-gray-50 px-4 py-3 text-[11px] font-semibold uppercase tracking-wide text-gray-400">
              <span className="w-7 shrink-0">#</span>
              <span className="w-32 shrink-0">Skill</span>
              <span className="flex-1">Skill Rate</span>
              <span className="w-12 shrink-0 text-right">Share</span>
            </div>
            <div className="divide-y divide-gray-100">
              {mostInDemand.map((item, index) => {
                const barWidth = (item.rate / maxRate) * 100;
                return (
                  <div
                    key={item.skill}
                    title={item.skill}
                    className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-gray-50"
                  >
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-gray-100 text-xs font-semibold tabular-nums text-gray-500">
                      {index + 1}
                    </span>
                    <span className="w-32 shrink-0 truncate text-sm font-medium text-gray-700">
                      {item.skill}
                    </span>
                    <div className="h-2.5 flex-1 rounded-full bg-gray-100">
                      <div
                        className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <span className="w-12 shrink-0 text-right text-sm font-semibold tabular-nums text-gray-700">
                      {item.rate}%
                    </span>
                  </div>
                );
              })}
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

          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
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

      {/* ── Trending Skills Section (Two-Proportion Growth Test) ── */}
      <div className="min-w-0">
        <div className="flex flex-col gap-3 mb-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-gray-950">
              Trending Skills
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Newcombe/Wilson two-rate comparison · {yoyResults.rows.length}{" "}
              skills · {yoyResults.currentTotal.toLocaleString()} current vs{" "}
              {yoyResults.priorTotal.toLocaleString()}{" "}
              previous jobs · filters: previous ≥ {TRENDING_MIN_PREVIOUS_RATE}% and ≥{" "}
              {TRENDING_MIN_PREVIOUS_COUNT} jobs, current count ≥{" "}
              {TRENDING_MIN_CURRENT_COUNT}, Δ ≥ {TRENDING_MIN_DELTA_PP}pp, growth
              ≥ {TRENDING_MIN_GROWTH_PERCENT}%
            </p>
          </div>
          <div className="flex w-fit items-center gap-1 bg-gray-100 rounded-lg p-1 shrink-0">
            {[
              { key: "90d", label: "90d" },
              { key: "12mo", label: "12mo" },
              { key: "24mo", label: "24mo" },
              { key: "all", label: "All Time" },
            ].map((w) => (
              <button
                key={w.key}
                onClick={() => setYoyWindow(w.key)}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  yoyWindow === w.key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* ── Left: ranked list by absolute growth ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Fastest-Growing Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by Δ percentage points
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">
                  pp
                </span>{" "}
                = <span className="font-medium">percentage points</span> —
                absolute change in share of job postings.{" "}
                <span className="font-mono font-semibold text-emerald-700">
                  ↑X%
                </span>{" "}
                = relative growth — (current − previous) / previous × 100 (e.g.
                4% → 7.5% ≈{" "}
                <span className="text-emerald-700 font-medium">↑144%</span>).
              </p>
            </div>
            {yoyResults.rows.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no trending skills for selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {yoyResults.rows.map((r, i) => {
                  const maxGrowth = yoyResults.rows[0]?.absoluteGrowth || 1;
                  const barPct = Math.min(
                    Math.round((r.absoluteGrowth / maxGrowth) * 100),
                    100,
                  );
                  const tier =
                    r.absoluteGrowth >= 5
                      ? {
                          label: "Surging",
                          bar: "bg-red-500",
                          chip: "bg-red-100 text-red-700",
                        }
                      : r.absoluteGrowth >= 2
                        ? {
                            label: "Growing",
                            bar: "bg-amber-500",
                            chip: "bg-amber-100 text-amber-700",
                          }
                        : {
                            label: "Gaining",
                            bar: "bg-emerald-500",
                            chip: "bg-emerald-100 text-emerald-700",
                          };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100 text-[11px] font-semibold tabular-nums text-gray-500">
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          {r.priorRate}% → {r.currentRate}% · {r.priorCount} →{" "}
                          {r.currentCount} jobs
                        </div>
                      </div>
                      <div className="w-16 bg-gray-100 rounded-full h-2.5 shrink-0">
                        <div
                          className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.absoluteGrowth}pp
                      </span>
                      <span className="tabular-nums font-semibold text-emerald-700 w-14 text-right shrink-0">
                        ↑{r.relativeGrowth}%
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}
                      >
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Right: Growth Trajectories chart — REMOVED (hidden) ── */}
          {false && (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Growth Trajectories
                </span>
                <span className="text-xs text-gray-400">
                  top 6 by Δpp · % rate per quarter
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Lines plot each skill&apos;s{" "}
                <span className="font-medium text-gray-700">% share</span> of
                jobs over time — the line that ends{" "}
                <span className="font-medium text-gray-700">highest</span> has
                the biggest{" "}
                <span className="font-medium text-gray-700">
                  current presence
                </span>
                , not necessarily the biggest{" "}
                <span className="font-medium text-gray-700">growth</span>.
              </p>
            </div>
            <div className="p-3">
              {yoyChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                  — no skills to plot —
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={yoyChartData.series}
                    margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="period"
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={{ stroke: "#e5e7eb" }}
                      interval={1}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={false}
                      width={32}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                        borderRadius: 8,
                        border: "1px solid #e5e7eb",
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                      iconType="circle"
                      iconSize={6}
                    />
                    {yoyChartData.skills.map((skill, i) => (
                      <Line
                        key={skill}
                        type="monotone"
                        dataKey={skill}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
          )}
        </div>
      </div>

      {/* ── Emerging Skills Section (Near-Zero Baseline + Newcombe CI) ── */}
      <div className="min-w-0">
        <div className="flex flex-col gap-3 mb-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-gray-950">
              Emerging Skills
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Newcombe/Wilson two-rate comparison · {emergingResults.rows.length}{" "}
              skills · {emergingResults.currentTotal.toLocaleString()} current vs{" "}
              {emergingResults.priorTotal.toLocaleString()}{" "}
              previous jobs · filters: previous &lt;{" "}
              {EMERGING_MAX_PREVIOUS_RATE}% and ≤ {EMERGING_MAX_PREVIOUS_COUNT}{" "}
              jobs, current ≥ {EMERGING_MIN_CURRENT_RATE}% and ≥{" "}
              {EMERGING_MIN_CURRENT_COUNT} jobs, Δ ≥ {EMERGING_MIN_DELTA_PP}pp
            </p>
          </div>
          <div className="flex w-fit items-center gap-1 bg-gray-100 rounded-lg p-1 shrink-0">
            {[
              { key: "90d", label: "90d" },
              { key: "12mo", label: "12mo" },
              { key: "24mo", label: "24mo" },
              { key: "all", label: "All Time" },
            ].map((w) => (
              <button
                key={w.key}
                onClick={() => setEmWindow(w.key)}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  emWindow === w.key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* ── Left: ranked list by current rate ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Newly-Emerging Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by current rate (prior baseline ≈ 0)
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">
                  pp
                </span>{" "}
                = percentage points. Emerging skills have near-zero prior
                presence, so Δpp is mainly current visibility.{" "}
                <span className="font-mono font-semibold text-violet-700">
                  NEW
                </span>{" "}
                = no previous-window mentions. Labels: Breakout = zero prior
                jobs; Emerging = current ≥ {EMERGING_STRONG_CURRENT_RATE}% and Δ
                ≥ {EMERGING_STRONG_DELTA_PP}pp; Appearing = passes minimum
                support.
              </p>
            </div>
            {emergingResults.rows.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no emerging skills detected in selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {emergingResults.rows.map((r, i) => {
                  const maxRate = emergingResults.rows[0]?.currentRate || 1;
                  const barPct = Math.min(
                    Math.round((r.currentRate / maxRate) * 100),
                    100,
                  );
                  const tier =
                    r.priorCount === 0
                      ? {
                          label: "Breakout",
                          bar: "bg-violet-500",
                          chip: "bg-violet-100 text-violet-700",
                        }
                      : r.currentRate >= EMERGING_STRONG_CURRENT_RATE &&
                          r.absoluteGrowth >= EMERGING_STRONG_DELTA_PP
                        ? {
                            label: "Emerging",
                            bar: "bg-blue-500",
                            chip: "bg-blue-100 text-blue-700",
                          }
                        : {
                            label: "Appearing",
                            bar: "bg-emerald-500",
                            chip: "bg-emerald-100 text-emerald-700",
                          };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100 text-[11px] font-semibold tabular-nums text-gray-500">
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          {r.priorRate}% → {r.currentRate}% · {r.priorCount} →{" "}
                          {r.currentCount} jobs
                        </div>
                      </div>
                      <div className="w-16 bg-gray-100 rounded-full h-2.5 shrink-0">
                        <div
                          className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.absoluteGrowth}pp
                      </span>
                      <span className="tabular-nums font-semibold text-violet-700 w-14 text-right shrink-0">
                        {r.priorCount === 0 ? "NEW" : `↑${r.relativeGrowth}%`}
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}
                      >
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Right: Emergence Trajectories chart — REMOVED (hidden) ── */}
          {false && (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Emergence Trajectories
                </span>
                <span className="text-xs text-gray-400">
                  top 6 · % rate per quarter
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Lines plot each emerging skill&apos;s{" "}
                <span className="font-medium text-gray-700">% share</span> of
                jobs over time — emerging skills start near zero and become
                meaningful in the current window.
              </p>
            </div>
            <div className="p-3">
              {emergingChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                  — no skills to plot —
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={emergingChartData.series}
                    margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="period"
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={{ stroke: "#e5e7eb" }}
                      interval={1}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={false}
                      width={32}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                        borderRadius: 8,
                        border: "1px solid #e5e7eb",
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                      iconType="circle"
                      iconSize={6}
                    />
                    {emergingChartData.skills.map((skill, i) => (
                      <Line
                        key={skill}
                        type="monotone"
                        dataKey={skill}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
          )}
        </div>
      </div>

      </div>
      {/* ── end 2-column grid (Most Demanded | Trending / Emerging) ── */}

      {/* ── Declining Skills Section (Z-Score) — HIDDEN ── */}
      {false && (
      <div className="mt-6">
        <div className="mb-3">
          <h3 className="text-sm font-semibold text-gray-900">
            📉 Declining Skills
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Established skills losing demand (early &mu; &ge; 1%, z &le;
            &minus;2&sigma;)
          </p>
        </div>

        {zDeclining.length === 0 ? (
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <p className="text-sm text-gray-500 text-center py-8">
              No declining skills for selected window
            </p>
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
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
                    Growth
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 text-right">
                    Z-Score
                  </th>
                  <th className="px-3 py-2.5 text-xs font-medium text-gray-400 w-32"></th>
                </tr>
              </thead>
              <tbody>
                {zDeclining.map((item, i) => {
                  const minZ = zDeclining[0]?.z || -1;
                  const barPct =
                    minZ < 0
                      ? Math.min((Math.abs(item.z) / Math.abs(minZ)) * 100, 100)
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
                      <td className="px-3 py-2.5 text-right tabular-nums font-semibold text-red-600">
                        {item.growth}%
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums font-semibold text-red-700">
                        {item.z.toFixed(2)}
                      </td>
                      <td className="px-3 py-2.5">
                        <div className="bg-gray-100 rounded-full h-1.5">
                          <div
                            className="h-1.5 rounded-full bg-red-400"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3 space-y-2">
          <p className="text-[11px] text-blue-600 leading-relaxed">
            <span className="font-medium">Formula:</span> z(s) = (f
            <sub>current</sub> &minus; &mu;<sub>baseline</sub>) / &sigma;
            <sub>baseline</sub>
          </p>
          <p className="text-[11px] text-blue-600 leading-relaxed">
            <span className="font-medium">Classification:</span> All categories
            require |z| &ge; 2&sigma; (95th percentile). <em>Trending</em> =
            already existed (early &mu; &ge; 1%), rising. <em>Emerging</em> =
            was absent (early &mu; &lt; 1%), now appearing. <em>Declining</em> =
            was established, now fading.
          </p>
          <p className="text-[11px] text-blue-600 leading-relaxed">
            <span className="font-medium">References:</span> Shewhart (1931),{" "}
            <em>Economic Control of Quality of Manufactured Product</em>;
            Aggarwal (2007), <em>Data Streams: Models and Algorithms</em>;
            Mathioudakis &amp; Koudas (2010), TwitterMonitor; Freedman, Pisani
            &amp; Purves (2007), <em>Statistics</em>, 4th ed. (empirical rule).
          </p>
        </div>
      </div>
      )}

      {/* ── Kleinberg Burst Detection (HIDDEN — keep only LRS visible) ── */}
      {false && (
        <div className="mt-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                Kleinberg Burst Detection
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                2-state HMM via Viterbi · {kTrending.length} trending ·{" "}
                {kEmerging.length} emerging · ε = P25(preRates) + α = 0.7
                (Cataldi et al. 2010, dynamic dual threshold; α tuned up from
                0.4 for gradual skill-demand series)
              </p>
            </div>
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
              {[
                { key: "all", label: "All time" },
                { key: "year", label: "Last year" },
                { key: "month", label: "Last month" },
                { key: "week", label: "Last week" },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setKTimeFilter(key)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    kTimeFilter === key
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {[
              {
                title: "🔥 Trending",
                subtitle: "established, accelerating (Klein Burg)",
                rows: kTrending,
              },
              // { title: "✨ Emerging", subtitle: "new, just appeared",        rows: kEmerging },
              // ↑ Hidden temporarily — Emerging panel will be rebuilt; Kleinberg classification logic kept intact
            ].map(({ title, subtitle, rows }) => (
              <div
                key={title}
                className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
              >
                <div className="px-4 py-3 border-b border-gray-100">
                  <span className="text-sm font-semibold text-gray-800">
                    {title}
                  </span>
                  <span className="text-xs text-gray-400 ml-2">{subtitle}</span>
                </div>
                {rows.length === 0 ? (
                  <p className="px-4 py-4 text-xs text-gray-400 italic">
                    — no skills detected —
                  </p>
                ) : (
                  <div className="divide-y divide-gray-50">
                    {rows.map((r, i) => (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-300 w-5">{i + 1}.</span>
                        <span className="flex-1 font-medium text-gray-800">
                          {r.skill}
                        </span>
                        <div className="w-16 bg-gray-100 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full ${r.demandScore >= 70 ? "bg-red-500" : r.demandScore >= 40 ? "bg-amber-500" : "bg-emerald-500"}`}
                            style={{ width: `${r.demandScore}%` }}
                          />
                        </div>
                        <span className="tabular-nums font-semibold text-gray-700 w-6 text-right">
                          {r.demandScore}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${r.demandScore >= 70 ? "bg-red-100 text-red-700" : r.demandScore >= 40 ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700"}`}
                        >
                          {r.demandScore >= 70
                            ? "Very Hot"
                            : r.demandScore >= 40
                              ? "Hot"
                              : "Rising"}
                        </span>
                        <span className="text-gray-400 tabular-nums">
                          {r.headlineDays < 30
                            ? `${r.headlineDays}d`
                            : `${Math.round(r.headlineDays / 30)}mo`}
                          {r.ongoing && (
                            <span className="ml-0.5 text-emerald-600">●</span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* ── Trending chart: top trending skills' %-rate over time ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📈 Trending Over Time
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  % of jobs mentioning each skill, by quarter
                </span>
              </div>
              <div className="p-3">
                {trendingChartData.series.length === 0 ? (
                  <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                    — no trending skills to plot —
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={trendingChartData.series}
                      margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                    >
                      <XAxis
                        dataKey="period"
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={{ stroke: "#e5e7eb" }}
                        interval={1}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={false}
                        width={32}
                        unit="%"
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: 11,
                          borderRadius: 8,
                          border: "1px solid #e5e7eb",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                        }}
                        formatter={(v) => `${v}%`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                        iconType="circle"
                        iconSize={6}
                      />
                      {trendingChartData.skills.map((skill, i) => (
                        <Line
                          key={skill}
                          type="monotone"
                          dataKey={skill}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TF-IDF Δ Over Time Section — HIDDEN ── */}
      {false && (
      <div className="mt-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                TF-IDF Δ Over Time
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Nesta &amp; Burning Glass methodology · {trending.length} skills
                · ranked by ΔTF-IDF from {firstPeriod} → {lastPeriod} (
                {trendTotalJobs.toLocaleString()} jobs)
              </p>
            </div>
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
              {PERIOD_OPTIONS.map((p) => (
                <button
                  key={p.key}
                  onClick={() => setPeriodType(p.key)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    periodType === p.key
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* ── Left: ranked list by ΔTF-IDF ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📚 Rising in Specificity
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  ranked by Δ TF-IDF (recent − earliest)
                </span>
              </div>
              {trending.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-400 italic">
                  — no skills with rising TF-IDF —
                </p>
              ) : (
                <div className="divide-y divide-gray-50">
                  {trending.slice(0, 10).map((r, i) => {
                    const maxDelta = trending[0]?.delta || 1;
                    const barPct =
                      maxDelta > 0
                        ? Math.min(
                            Math.round((Math.max(r.delta, 0) / maxDelta) * 100),
                            100,
                          )
                        : 0;
                    const tier = trendLabel(r.delta);
                    return (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-300 w-5">{i + 1}.</span>
                        <span className="flex-1 font-medium text-gray-800 truncate">
                          {r.skill}
                        </span>
                        <div className="w-12 bg-gray-100 rounded-full h-1.5">
                          <div
                            className="h-1.5 rounded-full bg-violet-500"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span
                          className="tabular-nums font-semibold w-14 text-right"
                          style={{
                            color: r.delta >= 0 ? "#059669" : "#dc2626",
                          }}
                        >
                          {r.delta >= 0 ? "+" : ""}
                          {r.delta.toFixed(4)}
                        </span>
                        <span
                          className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${tier.cls}`}
                        >
                          {tier.label}
                        </span>
                        <span className="text-gray-400 tabular-nums">
                          {r.lastRate}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* ── Right: %-rate trajectory chart of the same top skills ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📈 Rate Trajectories
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  top 6 by ΔTF-IDF · % rate per quarter
                </span>
              </div>
              <div className="p-3">
                {tfidfChartData.series.length === 0 ? (
                  <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                    — no skills to plot —
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={tfidfChartData.series}
                      margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                    >
                      <XAxis
                        dataKey="period"
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={{ stroke: "#e5e7eb" }}
                        interval={1}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={false}
                        width={32}
                        unit="%"
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: 11,
                          borderRadius: 8,
                          border: "1px solid #e5e7eb",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                        }}
                        formatter={(v) => `${v}%`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                        iconType="circle"
                        iconSize={6}
                      />
                      {tfidfChartData.skills.map((skill, i) => (
                        <Line
                          key={skill}
                          type="monotone"
                          dataKey={skill}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          {/* Method footer */}
          <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <span className="font-medium">Method:</span> Each time period is
              treated as a document. TF(skill, period) = mentions /
              total_mentions. IDF(skill) = log(N_periods / periods_with_skill).
              Skills are ranked by ΔTF-IDF = TF-IDF(recent) − TF-IDF(earliest).
              Positive Δ = becoming disproportionately frequent vs its own
              history. Burning Glass Institute weights this 70% in their
              official skill-trending formula.
            </p>
          </div>
        </div>
      )}

      {/* ── Mann-Kendall Trending Section (HIDDEN — keep only LRS visible) ── */}
      {false && (
        <div className="mt-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                Mann-Kendall Trend Test
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Non-parametric monotonic-trend detection ·{" "}
                {mkTrending.rows.length} skills · τ &gt; 0, p &lt; 0.05, mean
                rate ≥ 1% across {mkTrending.totalPeriods} quarters (Mann 1945,
                Kendall 1975)
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* ── Left: ranked list by Kendall's τ ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📊 Statistically Trending
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  ranked by Kendall&apos;s τ
                </span>
              </div>
              {mkTrending.rows.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-400 italic">
                  — no statistically significant upward trends —
                </p>
              ) : (
                <div className="divide-y divide-gray-50">
                  {mkTrending.rows.map((r, i) => {
                    const barPct = Math.round(r.tau * 100);
                    const tier =
                      r.tau >= 0.7
                        ? {
                            label: "Strong",
                            bar: "bg-red-500",
                            chip: "bg-red-100 text-red-700",
                          }
                        : r.tau >= 0.4
                          ? {
                              label: "Moderate",
                              bar: "bg-amber-500",
                              chip: "bg-amber-100 text-amber-700",
                            }
                          : {
                              label: "Weak",
                              bar: "bg-emerald-500",
                              chip: "bg-emerald-100 text-emerald-700",
                            };
                    return (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-300 w-5">{i + 1}.</span>
                        <span className="flex-1 font-medium text-gray-800 truncate">
                          {r.skill}
                        </span>
                        <div className="w-12 bg-gray-100 rounded-full h-1.5">
                          <div
                            className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span className="tabular-nums font-semibold text-gray-700 w-12 text-right">
                          τ={r.tau}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${tier.chip}`}
                        >
                          {tier.label}
                        </span>
                        <span className="text-gray-400 tabular-nums">
                          {r.initialRate}%→{r.recentRate}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* ── Right: trajectory chart of those same skills ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📈 Trend Trajectories
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  top 6 by τ · % rate per quarter
                </span>
              </div>
              <div className="p-3">
                {mkChartData.series.length === 0 ? (
                  <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                    — no skills to plot —
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={mkChartData.series}
                      margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                    >
                      <XAxis
                        dataKey="period"
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={{ stroke: "#e5e7eb" }}
                        interval={1}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={false}
                        width={32}
                        unit="%"
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: 11,
                          borderRadius: 8,
                          border: "1px solid #e5e7eb",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                        }}
                        formatter={(v) => `${v}%`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                        iconType="circle"
                        iconSize={6}
                      />
                      {mkChartData.skills.map((skill, i) => (
                        <Line
                          key={skill}
                          type="monotone"
                          dataKey={skill}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          {/* Method footer */}
          <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <span className="font-medium">Method:</span> For each skill, build
              a per-quarter %-rate series. Compute S = Σᵢ&lt;ⱼ sign(xⱼ − xᵢ),
              standardize to Z = (S ∓ 1) / √Var(S), derive p-value from the
              normal CDF, and report Kendall&apos;s τ = S / [n(n−1)/2]. Skills
              with τ &gt; 0 and p &lt; 0.05 have a statistically significant
              monotonic upward trend across the full {mkTrending.totalPeriods}
              -quarter history — robust to outliers and non-normal noise.
            </p>
          </div>
        </div>
      )}

      {/* ── Year-over-Year Skill Rate Growth Section — HIDDEN ── */}
      {false && (
      <div className="mt-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">
              Trending Skill (Method :Year-over-Year Skill Rate Growth)
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Lightcast / LinkedIn industry-standard · {yoyResults.rows.length}{" "}
              skills · {yoyResults.currentTotal.toLocaleString()} current vs{" "}
              {yoyResults.priorTotal.toLocaleString()}{" "}
              prior jobs · filter: Newcombe (1998) 95% CI on Δrate excludes 0 ·
              priorCount ≥ 5 for Trending vs Emerging split
            </p>
          </div>
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            {[
              { key: "90d", label: "90d" },
              { key: "12mo", label: "12mo" },
              { key: "24mo", label: "24mo" },
              { key: "all", label: "All time" },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setYoyWindow(key)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  yoyWindow === key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* ── Left: ranked list by absolute growth ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  📈 Fastest-Growing Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by Δ percentage points
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">
                  pp
                </span>{" "}
                = <span className="font-medium">percentage points</span> — the
                absolute change in a skill&apos;s share of all jobs (e.g.{" "}
                <span className="text-gray-700 font-medium">
                  14% → 19% = +5pp
                </span>
                ).
              </p>
            </div>
            {yoyResults.rows.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no skills growing in selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {yoyResults.rows.map((r, i) => {
                  const maxGrowth = yoyResults.rows[0]?.absoluteGrowth || 1;
                  const barPct = Math.round(
                    (r.absoluteGrowth / maxGrowth) * 100,
                  );
                  const tier =
                    r.absoluteGrowth >= 5
                      ? {
                          label: "Surging",
                          bar: "bg-red-500",
                          chip: "bg-red-100 text-red-700",
                        }
                      : r.absoluteGrowth >= 2
                        ? {
                            label: "Growing",
                            bar: "bg-amber-500",
                            chip: "bg-amber-100 text-amber-700",
                          }
                        : {
                            label: "Gaining",
                            bar: "bg-emerald-500",
                            chip: "bg-emerald-100 text-emerald-700",
                          };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100 text-[11px] font-semibold tabular-nums text-gray-500">
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          {r.priorRate}% → {r.currentRate}%
                        </div>
                      </div>
                      <div className="w-16 bg-gray-100 rounded-full h-2.5 shrink-0">
                        <div
                          className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.absoluteGrowth}pp
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}
                      >
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Right: trajectory chart of those same skills ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  📊 Growth Trajectories
                </span>
                <span className="text-xs text-gray-400">
                  top 6 by Δpp · % rate per quarter
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Lines plot each skill&apos;s{" "}
                <span className="font-medium text-gray-700">% share</span> of
                jobs over time — the line that ends{" "}
                <span className="font-medium text-gray-700">highest</span> has
                the biggest{" "}
                <span className="font-medium text-gray-700">
                  current presence
                </span>
                , not necessarily the biggest{" "}
                <span className="font-medium text-gray-700">growth</span>.
              </p>
            </div>
            <div className="p-3">
              {yoyChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                  — no skills to plot —
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={yoyChartData.series}
                    margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="period"
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={{ stroke: "#e5e7eb" }}
                      interval={1}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={false}
                      width={32}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                        borderRadius: 8,
                        border: "1px solid #e5e7eb",
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                      iconType="circle"
                      iconSize={6}
                    />
                    {yoyChartData.skills.map((skill, i) => (
                      <Line
                        key={skill}
                        type="monotone"
                        dataKey={skill}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Method footer */}
        <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
          <p className="text-[11px] text-blue-600 leading-relaxed">
            <span className="font-medium">Method:</span> Compare two adjacent
            equal-length windows. For each skill compute rate(window) =
            jobs_with_skill / total_jobs &times; 100. Growth = rate(current) −
            rate(prior) in percentage points. Ranked by Δpp.
          </p>
          <p className="text-[11px] text-blue-600 leading-relaxed mt-2">
            <span className="font-medium">Primary filter (statistically
              cited):</span>{" "}
            We compute the 95% Newcombe hybrid score CI on (p_current −
            p_prior) — combining Wilson score intervals for each proportion
            (Wilson 1927) using the method of Newcombe (1998b). A skill is
            retained only if the CI&apos;s lower bound is strictly greater
            than 0, i.e. the observed rise cannot be attributed to sampling
            noise at the 95% confidence level. This filter is fully
            sample-size-adaptive — small windows yield wide CIs that
            naturally fail, large windows yield tight CIs that tolerate
            smaller absolute effects.
          </p>
          <p className="text-[11px] text-blue-600 leading-relaxed mt-2">
            <span className="font-medium">Secondary filter (engineering
              split):</span>{" "}
            Among skills passing the statistical test, we route those with
            priorCount ≥ 5 to the Trending panel (established + rising) and
            priorCount &lt; 5 to the Emerging panel (rising from near-zero).
            The choice of 5 is our engineering decision, mirroring it on both
            sides for mutual exclusivity. The Newcombe paper does not
            prescribe a value for this distinction.
          </p>
          <p className="text-[11px] text-blue-500 leading-relaxed mt-1">
            <span className="font-medium">References:</span> Wilson, E. B.
            (1927), &quot;Probable inference, the law of succession, and
            statistical inference,&quot; <em>J. Amer. Stat. Assoc.</em>{" "}
            22(158): 209&ndash;212; Newcombe, R. G. (1998), &quot;Interval
            Estimation for the Difference Between Independent Proportions:
            Comparison of Eleven Methods,&quot; <em>Statistics in Medicine
            </em>{" "}
            17(8): 873&ndash;890; Brown, Cai &amp; DasGupta (2001),
            &quot;Interval Estimation for a Binomial Proportion,&quot;{" "}
            <em>Statistical Science</em> 16(2): 101&ndash;133.
          </p>
        </div>
      </div>
      )}

      {/* ── Emerging Skill Section (Method: Inverse-Newcombe) — HIDDEN ── */}
      {false && (
      <div className="mt-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">
              Emerging Skill (Method: Inverse Year-over-Year — Newcombe CI)
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Newcombe 1998b · same statistical machinery as Trending, with
              a rate-based &ldquo;near-zero prior&rdquo; gate (priorRate &lt; 1%) ·{" "}
              {emergingResults.rows.length} skills ·{" "}
              {emergingResults.currentTotal.toLocaleString()} current vs{" "}
              {emergingResults.priorTotal.toLocaleString()} prior jobs
            </p>
          </div>
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            {[
              { key: "90d", label: "90d" },
              { key: "12mo", label: "12mo" },
              { key: "24mo", label: "24mo" },
              { key: "all", label: "All time" },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setEmWindow(key)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  emWindow === key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* ── Left: ranked list by current rate ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  ✨ Newly-Emerging Skills
                </span>
                <span className="text-xs text-gray-400">
                  ranked by current rate (prior baseline ≈ 0)
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                <span className="font-mono font-semibold text-gray-700">
                  pp
                </span>{" "}
                = percentage points. For emerging skills, priorRate is near
                zero, so Δpp ≈ currentRate. All listed pass 95% Newcombe CI
                significance.
              </p>
            </div>
            {emergingResults.rows.length === 0 ? (
              <p className="px-4 py-4 text-xs text-gray-400 italic">
                — no emerging skills in selected window —
              </p>
            ) : (
              <div className="divide-y divide-gray-50">
                {emergingResults.rows.map((r, i) => {
                  const maxRate = emergingResults.rows[0]?.currentRate || 1;
                  const barPct = Math.round(
                    (r.currentRate / maxRate) * 100,
                  );
                  const tier =
                    r.absoluteGrowth >= 5
                      ? {
                          label: "Surging",
                          bar: "bg-red-500",
                          chip: "bg-red-100 text-red-700",
                        }
                      : r.absoluteGrowth >= 2
                        ? {
                            label: "Growing",
                            bar: "bg-amber-500",
                            chip: "bg-amber-100 text-amber-700",
                          }
                        : {
                            label: "Gaining",
                            bar: "bg-emerald-500",
                            chip: "bg-emerald-100 text-emerald-700",
                          };
                  return (
                    <div
                      key={r.skill}
                      className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      title={r.skill}
                    >
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100 text-[11px] font-semibold tabular-nums text-gray-500">
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-800 leading-tight">
                          {r.skill}
                        </div>
                        <div className="text-[10px] text-gray-400 tabular-nums mt-0.5">
                          {r.priorRate}% → {r.currentRate}% · prior n=
                          {r.priorCount} (&lt; 5)
                        </div>
                      </div>
                      <div className="w-16 bg-gray-100 rounded-full h-2.5 shrink-0">
                        <div
                          className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <span className="tabular-nums font-semibold text-gray-700 w-14 text-right shrink-0">
                        +{r.absoluteGrowth}pp
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${tier.chip}`}
                      >
                        {tier.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Right: trajectory chart of those same skills ── */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  📊 Emergence Trajectories
                </span>
                <span className="text-xs text-gray-400">
                  top 6 · % rate per quarter
                </span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Lines plot each emerging skill&apos;s{" "}
                <span className="font-medium text-gray-700">% share</span> of
                jobs over time — emerging skills start near 0% and rise after
                their burst-onset.
              </p>
            </div>
            <div className="p-3">
              {emergingChartData.series.length === 0 ? (
                <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                  — no skills to plot —
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={emergingChartData.series}
                    margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                  >
                    <XAxis
                      dataKey="period"
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={{ stroke: "#e5e7eb" }}
                      interval={1}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9ca3af" }}
                      tickLine={false}
                      axisLine={false}
                      width={32}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                        borderRadius: 8,
                        border: "1px solid #e5e7eb",
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                      iconType="circle"
                      iconSize={6}
                    />
                    {emergingChartData.skills.map((skill, i) => (
                      <Line
                        key={skill}
                        type="monotone"
                        dataKey={skill}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Method footer */}
        <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
          <p className="text-[11px] text-blue-600 leading-relaxed">
            <span className="font-medium">Method (symmetric with Trending):</span>{" "}
            For each skill, we compute the 95% Newcombe hybrid score
            confidence interval on (currentRate − priorRate), combining
            Wilson score intervals for each window&apos;s proportion
            (Newcombe 1998b). A skill enters the Emerging panel only if:
          </p>
          <p className="text-[11px] text-blue-600 leading-relaxed mt-1">
            <span className="font-mono">①</span>{" "}
            <span className="font-medium">ciLow &gt; 0</span> (statistically
            significant rise at 95% confidence — paper-cited inference) AND
            <span className="font-mono"> ②</span>{" "}
            <span className="font-medium">priorRate &lt; 1%</span> (near-zero
            prior presence — engineering choice). We use a rate-based gate
            rather than a count-based one so the rule is window-independent:
            &quot;less than 1 in 100 jobs mentioned it&quot; means the same
            thing whether the window has 200 or 5,000 jobs. Skills with prior
            rates above 1% pass the Newcombe test but route to Trending. The
            same statistical framework as the Trending panel — one method,
            one citation, two complementary panels.
          </p>
          <p className="text-[11px] text-blue-500 leading-relaxed mt-1">
            <span className="font-medium">References:</span> Wilson, E. B.
            (1927), &quot;Probable inference, the law of succession, and
            statistical inference,&quot; <em>J. Amer. Stat. Assoc.</em>{" "}
            22(158): 209&ndash;212; Newcombe, R. G. (1998), &quot;Interval
            Estimation for the Difference Between Independent Proportions:
            Comparison of Eleven Methods,&quot; <em>Statistics in Medicine
            </em>{" "}
            17(8): 873&ndash;890.
          </p>
        </div>
      </div>
      )}

      {/* ── Google-Trends-style: Top + Rising Skills ── */}
      {false && (
        <div className="mt-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                Top & Rising Skills
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Google Trends-style · {gtData.currentTotal.toLocaleString()}{" "}
                current vs {gtData.priorTotal.toLocaleString()} prior jobs ·
                shared bar scale (max = {gtData.globalMax.toFixed(1)}%) so
                volumes are directly comparable between panels
              </p>
            </div>
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
              {[
                { key: "90d", label: "90d" },
                { key: "12mo", label: "12mo" },
                { key: "24mo", label: "24mo" },
                { key: "all", label: "All time" },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setGtWindow(key)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    gtWindow === key
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* ── Top Skills (highest skill rate in window) ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Top Skills
                </span>
                <span className="text-[11px] text-gray-400 uppercase tracking-wider">
                  volume
                </span>
              </div>
              {gtData.top.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-400 italic">
                  — no skills in selected window —
                </p>
              ) : (
                <div className="divide-y divide-gray-50">
                  <div className="flex items-center gap-3 px-4 py-2 text-[10px] text-gray-400 uppercase tracking-wider">
                    <span className="w-5"></span>
                    <span className="flex-1">Skill</span>
                    <span className="w-20 text-right">Interest</span>
                    <span className="w-16 text-right">Change</span>
                  </div>
                  {gtData.top.map((r, i) => {
                    const barPct = Math.min(
                      Math.round((r.currentRate / gtData.globalMax) * 100),
                      100,
                    );
                    return (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-400 w-5 tabular-nums">
                          {i + 1}
                        </span>
                        <span className="flex-1 font-medium text-gray-800 truncate">
                          {r.skill}
                        </span>
                        <div className="w-20 bg-gray-100 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-blue-500"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span className="w-16 text-right tabular-nums font-medium">
                          {r.change === null ? (
                            <span className="text-violet-600">Breakout</span>
                          ) : r.change > 0 ? (
                            <span className="text-emerald-600">
                              ↑ +{r.change}%
                            </span>
                          ) : r.change < 0 ? (
                            <span className="text-red-500">↓ {r.change}%</span>
                          ) : (
                            <span className="text-gray-400">— 0%</span>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* ── Rising Skills (biggest % growth in window) ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">
                  Rising Skills
                </span>
                <span className="text-[11px] text-gray-400 uppercase tracking-wider">
                  momentum
                </span>
              </div>
              {gtData.rising.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-400 italic">
                  — no rising skills in selected window —
                </p>
              ) : (
                <div className="divide-y divide-gray-50">
                  <div className="flex items-center gap-3 px-4 py-2 text-[10px] text-gray-400 uppercase tracking-wider">
                    <span className="w-5"></span>
                    <span className="flex-1">Skill</span>
                    <span className="w-20 text-right">Interest</span>
                    <span className="w-16 text-right">Change</span>
                  </div>
                  {gtData.rising.map((r, i) => {
                    const barPct = Math.min(
                      Math.round((r.currentRate / gtData.globalMax) * 100),
                      100,
                    );
                    return (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-400 w-5 tabular-nums">
                          {i + 1}
                        </span>
                        <span className="flex-1 font-medium text-gray-800 truncate">
                          {r.skill}
                        </span>
                        <div className="w-20 bg-gray-100 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-blue-300"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span className="w-16 text-right tabular-nums font-medium">
                          {r.change === null ? (
                            <span className="text-violet-600">Breakout</span>
                          ) : (
                            <span className="text-emerald-600">
                              ↑ +{r.change}%
                            </span>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Method footer */}
          <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <span className="font-medium">Method:</span> Mirrors Google
              Trends&apos; Top + Rising layout.{" "}
              <span className="font-medium">Top</span> ranks by current-window
              skill rate (volume). <span className="font-medium">Rising</span>{" "}
              ranks by % change vs the prior equivalent window (momentum). Both
              panels share a single bar scale (max of Top) so a small bar in
              Rising means &quot;yes it&apos;s growing fast, but still small
              overall.&quot; Skills with no prior baseline (priorRate ≈ 0) show
              as <span className="text-violet-600 font-medium">Breakout</span>{" "}
              instead of a misleading infinity %.
            </p>
          </div>
        </div>
      )}

      {/* ── (Hidden) Linear Regression Slope + R² Section ── */}
      {false && (
        <div className="mt-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                Linear Regression Slope + R²
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Least-squares fit over {lrsTrending.totalPeriods} quarters ·{" "}
                {lrsTrending.rows.length} skills · slope &gt; 0, R² ≥ 0.5, mean
                rate ≥ 1% · two intuitive numbers (rate of change + fit quality)
              </p>
            </div>
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
              {[
                { key: "90d", label: "90d" },
                { key: "12mo", label: "12mo" },
                { key: "24mo", label: "24mo" },
                { key: "all", label: "All time" },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setLrsWindow(key)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    lrsWindow === key
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* ── Left: ranked list by slope ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📐 Steepest Risers
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  ranked by slope (pp / quarter)
                </span>
              </div>
              {lrsTrending.rows.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-400 italic">
                  — no skills with significant linear trend —
                </p>
              ) : (
                <div className="divide-y divide-gray-50">
                  {lrsTrending.rows.map((r, i) => {
                    const maxSlope = lrsTrending.rows[0]?.slope || 1;
                    const barPct = Math.min(
                      Math.round((r.slope / maxSlope) * 100),
                      100,
                    );
                    const tier =
                      r.slope >= 1.0
                        ? {
                            label: "Strong",
                            bar: "bg-red-500",
                            chip: "bg-red-100 text-red-700",
                          }
                        : r.slope >= 0.5
                          ? {
                              label: "Moderate",
                              bar: "bg-amber-500",
                              chip: "bg-amber-100 text-amber-700",
                            }
                          : {
                              label: "Weak",
                              bar: "bg-emerald-500",
                              chip: "bg-emerald-100 text-emerald-700",
                            };
                    // R² → star count: drops jargon, keeps the "is this trend
                    // trustworthy" signal in a universally readable form.
                    const stars =
                      r.r2 >= 0.75 ? "⭐⭐⭐" : r.r2 >= 0.6 ? "⭐⭐" : "⭐";
                    return (
                      <div
                        key={r.skill}
                        className="flex items-center gap-3 px-4 py-2.5 text-xs"
                      >
                        <span className="text-gray-300 w-5">{i + 1}.</span>
                        <span className="flex-1 font-medium text-gray-800 truncate">
                          {r.skill}
                        </span>
                        <div className="w-12 bg-gray-100 rounded-full h-1.5">
                          <div
                            className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span
                          className={`tabular-nums font-semibold w-16 text-right ${
                            r.relativeGrowth == null
                              ? "text-violet-600"
                              : "text-emerald-700"
                          }`}
                        >
                          {r.relativeGrowth == null
                            ? "NEW"
                            : `↑${r.relativeGrowth}%`}
                        </span>
                        <span className="text-gray-400 tabular-nums w-24 text-right">
                          {r.initialRate}% → {r.recentRate}%
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${tier.chip}`}
                        >
                          {tier.label} {stars}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* ── Right: trajectory chart with fitted lines ── */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <span className="text-sm font-semibold text-gray-800">
                  📊 Rate Trajectories
                </span>
                <span className="text-xs text-gray-400 ml-2">
                  top 6 by slope · % rate per quarter
                </span>
              </div>
              <div className="p-3">
                {lrsChartData.series.length === 0 ? (
                  <p className="px-2 py-8 text-xs text-gray-400 italic text-center">
                    — no skills to plot —
                  </p>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={lrsChartData.series}
                      margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                    >
                      <XAxis
                        dataKey="period"
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={{ stroke: "#e5e7eb" }}
                        interval={1}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#9ca3af" }}
                        tickLine={false}
                        axisLine={false}
                        width={32}
                        unit="%"
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: 11,
                          borderRadius: 8,
                          border: "1px solid #e5e7eb",
                          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                        }}
                        formatter={(v) => `${v}%`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                        iconType="circle"
                        iconSize={6}
                      />
                      {lrsChartData.skills.map((skill, i) => (
                        <Line
                          key={skill}
                          type="monotone"
                          dataKey={skill}
                          stroke={LINE_COLORS[i % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          {/* Method footer */}
          <div className="mt-3 rounded-xl bg-blue-50 border border-blue-100 px-4 py-3">
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <span className="font-medium">Method:</span> For each skill, fit a
              least-squares line through its quarterly %-rate series.
              <span className="font-mono">
                {" "}
                slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
              </span>
              ;{" "}
              <span className="font-mono">
                R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
              </span>
              . <span className="font-medium">Slope</span> = pp gained per
              quarter; <span className="font-medium">R²</span> = how well a
              straight line fits (1 = perfect, 0 = pure noise). Filters: slope
              &gt; 0, R² ≥ 0.5 — keeps only real, consistent risers.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
