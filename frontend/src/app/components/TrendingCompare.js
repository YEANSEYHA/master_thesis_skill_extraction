"use client";

import { useMemo, useState } from "react";
import { dummy_data } from "./dummy_data";

// =====================================================================
// Z-Score Burst Detection (Shewhart 1931; Aggarwal 2007)
//   z(s) = (f_current − μ_baseline) / σ_baseline
//   Splits the timeline into baseline (older) and current (recent)
//   quarters, then flags skills with |z| ≥ 2 as significantly trending.
// =====================================================================

const ZSCORE_WINDOWS = [
  { key: "1q", label: "Last Quarter", quarters: 1 },
  { key: "2q", label: "Last 2 Quarters", quarters: 2 },
  { key: "1y", label: "Last Year", quarters: 4 },
  { key: "all", label: "All Time", quarters: "half" },
];

function buildZScoreBurst(data, windowQuarters = 2, topN = 15) {
  const quarterJobs = {};
  for (const job of data) {
    const d = new Date(job.posted_date);
    const q = Math.ceil((d.getMonth() + 1) / 3);
    const key = `${d.getFullYear()} Q${q}`;
    (quarterJobs[key] ||= []).push(job);
  }

  const sortedQuarters = Object.keys(quarterJobs).sort();
  const totalQuarters = sortedQuarters.length;
  if (totalQuarters < 3) return { trending: [], emerging: [], declining: [] };

  const eff = windowQuarters === "half" ? Math.floor(totalQuarters / 2) : windowQuarters;
  const currentQuarters = sortedQuarters.slice(-eff);
  const baselineQuarters = sortedQuarters.slice(0, -eff);
  if (baselineQuarters.length < 2) return { trending: [], emerging: [], declining: [] };

  const allSkills = new Set();
  const quarterRates = {};
  for (const quarter of sortedQuarters) {
    const jobs = quarterJobs[quarter];
    const total = jobs.length;
    const skillCount = {};
    for (const job of jobs) {
      for (const skill of new Set(job.skills)) {
        allSkills.add(skill);
        skillCount[skill] = (skillCount[skill] || 0) + 1;
      }
    }
    quarterRates[quarter] = {};
    for (const [skill, count] of Object.entries(skillCount)) {
      quarterRates[quarter][skill] = (count / total) * 100;
    }
  }

  const results = [];
  for (const skill of allSkills) {
    const baseRates = baselineQuarters.map((q) => quarterRates[q][skill] || 0);
    const μ = baseRates.reduce((a, b) => a + b, 0) / baseRates.length;
    const variance = baseRates.reduce((a, b) => a + (b - μ) ** 2, 0) / baseRates.length;
    const σ = Math.sqrt(variance);
    const curRates = currentQuarters.map((q) => quarterRates[q][skill] || 0);
    const fCurrent = curRates.reduce((a, b) => a + b, 0) / curRates.length;
    const z = σ > 0 ? (fCurrent - μ) / σ : fCurrent > μ ? 3.0 : 0;

    const earlyBase = baselineQuarters.slice(0, Math.ceil(baselineQuarters.length / 2));
    const earlyRates = earlyBase.map((q) => quarterRates[q][skill] || 0);
    const earlyAvg = earlyRates.reduce((a, b) => a + b, 0) / earlyRates.length;
    const existedEarly = earlyAvg >= 1.0;

    if (Math.abs(z) < 2.0) continue;
    if (z >= 2.0 && fCurrent < 1) continue;
    if (z <= -2.0 && μ < 1) continue;

    results.push({
      skill,
      fCurrent: Math.round(fCurrent * 10) / 10,
      μ: Math.round(μ * 10) / 10,
      σ: Math.round(σ * 100) / 100,
      z: Math.round(z * 100) / 100,
      existedEarly,
    });
  }

  return {
    trending: results.filter((r) => r.z >= 2 && r.existedEarly).sort((a, b) => b.z - a.z).slice(0, topN),
    emerging: results.filter((r) => r.z >= 2 && !r.existedEarly).sort((a, b) => b.z - a.z).slice(0, topN),
    declining: results.filter((r) => r.z <= -2).sort((a, b) => a.z - b.z).slice(0, topN),
    baselineQuarters,
    currentQuarters,
  };
}

// =====================================================================
// Kleinberg Burst Detection (Kleinberg 2002)
//   2-state HMM via Viterbi. State 0 = base rate λ₀, State 1 = burst
//   rate s·λ₀. Transition cost γ·log(n) prevents single-event flips.
//   Returns burst intervals per skill (start_ts, end_ts, n_events).
// =====================================================================

function kleinbergBurst(timestamps, opts = {}) {
  const { s = 2, gamma = 1, minEvents = 5 } = opts;
  if (timestamps.length < minEvents) return [];

  const sorted = [...timestamps].sort((a, b) => a - b);
  const n = sorted.length;

  // Gaps between consecutive events (in days)
  const gaps = [];
  for (let i = 1; i < n; i++) {
    const dt = (sorted[i] - sorted[i - 1]) / 86400000; // ms → days
    gaps.push(Math.max(dt, 0.001)); // avoid log(0)
  }

  const totalDays = (sorted[n - 1] - sorted[0]) / 86400000;
  if (totalDays <= 0) return [];

  const lambda0 = (n - 1) / totalDays;        // base rate (events/day)
  const lambda1 = lambda0 * s;                // burst rate
  const transCost = gamma * Math.log(n);

  // emit cost: -log(λ * exp(-λg)) = -log(λ) + λg
  const emit = (state, g) => {
    const λ = state === 0 ? lambda0 : lambda1;
    return -Math.log(λ) + λ * g;
  };

  // Viterbi: dp[i][state] = min cost to reach gap i in state
  const dp = [[0, Infinity]];
  const back = [[null, null]];
  for (let i = 0; i < gaps.length; i++) {
    const cur = [Infinity, Infinity];
    const bk = [null, null];
    for (let st = 0; st < 2; st++) {
      for (let prev = 0; prev < 2; prev++) {
        const c = dp[i][prev] + (prev === st ? 0 : transCost) + emit(st, gaps[i]);
        if (c < cur[st]) { cur[st] = c; bk[st] = prev; }
      }
    }
    dp.push(cur);
    back.push(bk);
  }

  // Backtrace
  let st = dp[gaps.length][0] < dp[gaps.length][1] ? 0 : 1;
  const states = new Array(gaps.length);
  for (let i = gaps.length; i > 0; i--) {
    states[i - 1] = st;
    st = back[i][st];
  }

  // Find burst intervals (continuous state-1 runs)
  const bursts = [];
  let bs = null;
  for (let i = 0; i < states.length; i++) {
    if (states[i] === 1 && bs === null) bs = i + 1;     // event index in sorted[]
    else if (states[i] === 0 && bs !== null) {
      bursts.push({ startTs: sorted[bs], endTs: sorted[i], nEvents: i - bs + 1, ongoing: false });
      bs = null;
    }
  }
  if (bs !== null) {
    bursts.push({ startTs: sorted[bs], endTs: sorted[n - 1], nEvents: n - bs, ongoing: true });
  }

  return bursts;
}

function buildKleinbergBurst(data, opts = {}) {
  const { topN = 15, minOccurrences = 10, s = 2, gamma = 1 } = opts;

  // Per-skill timestamp streams
  const streams = {};
  for (const job of data) {
    const ts = new Date(job.posted_date).getTime();
    if (!Number.isFinite(ts)) continue;
    for (const skill of new Set(job.skills)) {
      (streams[skill] ||= []).push(ts);
    }
  }

  // Pass 1: collect raw burst metrics for every qualifying skill
  const raw = [];
  for (const [skill, timestamps] of Object.entries(streams)) {
    if (timestamps.length < minOccurrences) continue;
    const bursts = kleinbergBurst(timestamps, { s, gamma, minEvents: 5 });
    if (bursts.length === 0) continue;

    const headline = bursts.reduce((best, b) => {
      const bScore = b.nEvents * (b.ongoing ? 1.5 : 1);
      const bestScore = best ? best.nEvents * (best.ongoing ? 1.5 : 1) : -Infinity;
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
      skill, totalEvents, bursts, headline, preRate, burstRate,
      headlineStart: new Date(headline.startTs).toISOString().slice(0, 10),
      headlineEnd: new Date(headline.endTs).toISOString().slice(0, 10),
      headlineDays: Math.round(span),
      headlineEvents: headline.nEvents,
      ongoing: headline.ongoing,
      intensity: span > 0 ? Math.round((headline.nEvents / span) * 100) / 100 : headline.nEvents,
      preBurstCount, burstAge: Math.round(burstAge),
    });
  }

  // Pass 2: derive EPSILON_ABS from the P25 of observed preRates
  // This makes the threshold adapt to dataset size and time range automatically.
  const sortedPreRates = [...raw.map((r) => r.preRate)].sort((a, b) => a - b);
  const p = 0.25 * (sortedPreRates.length - 1);
  const EPSILON_ABS = sortedPreRates[Math.floor(p)] +
    (sortedPreRates[Math.ceil(p)] - sortedPreRates[Math.floor(p)]) * (p - Math.floor(p));
  const ALPHA = 0.4; // Cataldi et al. (2010)

  // Pass 3: classify with data-driven threshold
  const results = raw.map((r) => {
    const isDormantRelative = r.preRate < ALPHA * r.burstRate;
    const isRareAbsolute    = r.preRate < EPSILON_ABS;
    return { ...r, classification: isDormantRelative && isRareAbsolute ? "emerging" : "trending" };
  });

  // Normalize intensity to 0-100 demand score
  const maxIntensity = results.length > 0 ? Math.max(...results.map((r) => r.intensity)) : 1;
  for (const r of results) {
    r.demandScore = Math.round((r.intensity / maxIntensity) * 100);
  }

  // Rank: ongoing first, then by demand score (intensity-based)
  results.sort((a, b) => {
    if (a.ongoing !== b.ongoing) return a.ongoing ? -1 : 1;
    return b.demandScore - a.demandScore;
  });

  return results.slice(0, topN);
}

// =====================================================================
// UI
// =====================================================================

export default function TrendingCompare() {
  const [zWindow, setZWindow] = useState("2q");
  const [klS, setKlS] = useState(2);
  const [klGamma, setKlGamma] = useState(1);
  const [topN, setTopN] = useState(15);

  const zWindowQuarters = ZSCORE_WINDOWS.find((w) => w.key === zWindow)?.quarters || 2;
  const zResults = useMemo(
    () => buildZScoreBurst(dummy_data, zWindowQuarters, topN),
    [zWindowQuarters, topN]
  );

  const kResults = useMemo(
    () => buildKleinbergBurst(dummy_data, { topN, s: klS, gamma: klGamma }),
    [topN, klS, klGamma]
  );

  const kTrending = kResults.filter((r) => r.classification === "trending");
  const kEmerging = kResults.filter((r) => r.classification === "emerging");

  // Compute overlap (skills detected by both methods)
  const zSkills = new Set(
    [...zResults.trending, ...zResults.emerging].map((r) => r.skill)
  );
  const kSkills = new Set(kResults.map((r) => r.skill));
  const overlapSkills = [...zSkills].filter((s) => kSkills.has(s));

  return (
    <div>
      <h1 className="text-2xl font-semibold text-gray-900 mb-2">
        Trending Skill — Method Comparison
      </h1>
      <p className="text-sm text-gray-500 mb-8">
        Z-Score (statistical control chart) vs Kleinberg (burst-detection HMM)
        running on the same {dummy_data.length} job postings.
      </p>

      {/* Controls */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Z-Score window
          </label>
          <select
            value={zWindow}
            onChange={(e) => setZWindow(e.target.value)}
            className="mt-1 w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          >
            {ZSCORE_WINDOWS.map((w) => (
              <option key={w.key} value={w.key}>{w.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Kleinberg s (burst rate ×)
          </label>
          <input
            type="number" min="1.5" max="5" step="0.5"
            value={klS} onChange={(e) => setKlS(Number(e.target.value))}
            className="mt-1 w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Kleinberg γ (transition cost)
          </label>
          <input
            type="number" min="0.1" max="5" step="0.1"
            value={klGamma} onChange={(e) => setKlGamma(Number(e.target.value))}
            className="mt-1 w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Top N
          </label>
          <input
            type="number" min="5" max="30" step="1"
            value={topN} onChange={(e) => setTopN(Number(e.target.value))}
            className="mt-1 w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          />
        </div>
      </div>

      {/* Overlap summary */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-amber-800">
              Overlap — skills detected by BOTH methods
            </div>
            <div className="text-2xl font-bold text-amber-900 mt-0.5">
              {overlapSkills.length}
              <span className="text-sm font-normal text-amber-700 ml-2">
                of Z={zSkills.size} · K={kSkills.size}
              </span>
            </div>
          </div>
          <div className="text-xs text-amber-800 max-w-md text-right">
            High overlap = high confidence in detected trends.
            Skills in only one method are weaker signals.
          </div>
        </div>
        {overlapSkills.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {overlapSkills.map((s) => (
              <span
                key={s}
                className="px-2 py-0.5 rounded-full bg-amber-200 text-amber-900 text-xs font-medium"
              >
                {s}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Z-Score column ── */}
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="bg-purple-50 border-b border-purple-200 px-5 py-3">
            <div className="text-xs uppercase tracking-wider text-purple-600 font-semibold">
              Method 1
            </div>
            <div className="text-base font-semibold text-purple-900 mt-0.5">
              Z-Score Burst Detection
            </div>
            <div className="text-xs text-purple-700 mt-1">
              Shewhart (1931) · Aggarwal (2007) · |z| ≥ 2 = 95th percentile
            </div>
          </div>

          <div className="px-5 py-4 text-xs text-gray-500 border-b border-gray-100">
            Baseline: {zResults.baselineQuarters?.length || 0} quarters ·
            Current: {zResults.currentQuarters?.length || 0} quarters
          </div>

          {/* Trending */}
          <Section title="🔥 Trending (existed before, now rising)" rows={zResults.trending} method="z" />
          <Section title="✨ Emerging (was absent, now appearing)" rows={zResults.emerging} method="z" />
          <Section title="📉 Declining" rows={zResults.declining} method="z" />
        </div>

        {/* ── Kleinberg column ── */}
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="bg-emerald-50 border-b border-emerald-200 px-5 py-3">
            <div className="text-xs uppercase tracking-wider text-emerald-600 font-semibold">
              Method 2
            </div>
            <div className="text-base font-semibold text-emerald-900 mt-0.5">
              Kleinberg Burst Detection
            </div>
            <div className="text-xs text-emerald-700 mt-1">
              Kleinberg (2002) · 2-state HMM via Viterbi · s={klS}, γ={klGamma}
            </div>
          </div>

          <div className="px-5 py-4 text-xs text-gray-500 border-b border-gray-100">
            {kTrending.length} trending · {kEmerging.length} emerging · ε = P25(preRates) + α = 0.4 (Cataldi et al. 2010, dynamic dual threshold)
          </div>

          <Section title="🔥 Trending (established, accelerating)" rows={kTrending} method="kleinberg" />
          <Section title="✨ Emerging (new, just appeared)"        rows={kEmerging} method="kleinberg" />
        </div>
      </div>
    </div>
  );
}

function Section({ title, rows, method }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="px-5 py-4 border-t border-gray-100">
        <div className="text-sm font-medium text-gray-700 mb-1">{title}</div>
        <div className="text-xs text-gray-400 italic">— no skills detected —</div>
      </div>
    );
  }
  return (
    <div className="px-5 py-4 border-t border-gray-100">
      <div className="text-sm font-medium text-gray-700 mb-3">{title}</div>
      <div className="space-y-1.5">
        {rows.map((r, i) => (
          <div key={r.skill} className="flex items-center text-xs">
            <span className="text-gray-400 w-6">{i + 1}.</span>
            <span className="flex-1 font-medium text-gray-800">{r.skill}</span>
            {method === "z" ? (
              <span className="text-gray-500 tabular-nums">
                z={r.z} · μ={r.μ} → {r.fCurrent}%
              </span>
            ) : (
              <div className="flex items-center gap-2">
                <div className="w-16 bg-gray-100 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${
                      r.demandScore >= 70 ? "bg-red-500" :
                      r.demandScore >= 40 ? "bg-amber-500" : "bg-emerald-500"
                    }`}
                    style={{ width: `${r.demandScore}%` }}
                  />
                </div>
                <span className="tabular-nums font-semibold text-gray-700 w-6 text-right">
                  {r.demandScore}
                </span>
                <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                  r.demandScore >= 70 ? "bg-red-100 text-red-700" :
                  r.demandScore >= 40 ? "bg-amber-100 text-amber-700" :
                  "bg-green-100 text-green-700"
                }`}>
                  {r.demandScore >= 70 ? "Very Hot" : r.demandScore >= 40 ? "Hot" : "Rising"}
                </span>
                <span className="text-gray-400 tabular-nums">
                  {r.headlineDays < 30 ? `${r.headlineDays}d` : `${Math.round(r.headlineDays / 30)}mo`}
                  {r.ongoing && <span className="ml-0.5 text-emerald-600">●</span>}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
