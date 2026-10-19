"use client";

export const API_BASE = "http://localhost:8000";

// fetch + JSON parse that tells "backend unreachable" apart from "backend
// answered with an error". Parsing the body inside the caller's try meant a
// 500 with a non-JSON body threw too, and every failure looked like a
// connection failure — including CORS rejections, which are unreachable-shaped
// to JS but really mean the origin isn't allowlisted on the API.
export async function apiFetch(path, options) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error(
      `Could not reach the API at ${API_BASE}. Is the backend running on port 8000?`
    );
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    // Non-JSON body (empty 500, proxy error page) — status decides below.
  }

  if (!res.ok) {
    throw new Error(data?.detail || `API error ${res.status} ${res.statusText}`);
  }
  return data;
}

// Convert country name to flag emoji using regional indicator symbols
export function countryFlag(countryName) {
  const COUNTRY_CODES = {
    cambodia: "KH", england: "GB", australia: "AU", india: "IN", canada: "CA",
    philippines: "PH", spain: "ES", brazil: "BR", vietnam: "VN", malaysia: "MY",
    "south korea": "KR", france: "FR", "united states": "US", indonesia: "ID",
    japan: "JP", singapore: "SG", thailand: "TH", germany: "DE", china: "CN",
    "united kingdom": "GB", myanmar: "MM", laos: "LA", taiwan: "TW", italy: "IT",
    netherlands: "NL", sweden: "SE", norway: "NO", denmark: "DK", finland: "FI",
    switzerland: "CH", austria: "AT", belgium: "BE", portugal: "PT", ireland: "IE",
    "new zealand": "NZ", mexico: "MX", argentina: "AR", chile: "CL", colombia: "CO",
    peru: "PE", nigeria: "NG", "south africa": "ZA", kenya: "KE", egypt: "EG",
    morocco: "MA", turkey: "TR", "saudi arabia": "SA", uae: "AE",
    "united arab emirates": "AE", pakistan: "PK", bangladesh: "BD", nepal: "NP",
    "sri lanka": "LK", poland: "PL", ukraine: "UA", romania: "RO", czechia: "CZ",
    hungary: "HU", greece: "GR", israel: "IL", russia: "RU",
  };
  const code = COUNTRY_CODES[countryName.toLowerCase().trim()];
  if (!code) return "";
  return String.fromCodePoint(...[...code].map(c => 0x1F1E6 + c.charCodeAt(0) - 65));
}

// Small inline spinner for buttons / loading panels.
// Inherits text color via `currentColor`; size via the `size` prop (px).
export function Spinner({ size = 16, className = "" }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

export function ConfidenceBadge({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 90 ? "text-emerald-600 bg-emerald-50 border-emerald-200"
    : pct >= 70 ? "text-amber-600 bg-amber-50 border-amber-200"
    : "text-gray-500 bg-gray-50 border-gray-200";
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${color}`}>
      {pct}%
    </span>
  );
}

export function SiteTypeBadge({ type }) {
  return (
    <span className="text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
      {type.replace(/_/g, " ")}
    </span>
  );
}

export function Terminal({ logRef, logs, label, isLive, colorize }) {
  const defaultColorize = (line) => {
    if (line.includes("Starting crawl") || line.includes("===") || line.includes("CRAWL SUMMARY")) return "text-white font-medium";
    if (line.includes("STEP")) return "text-white font-medium";
    if (line.includes("Success") || line.includes("success") || line.includes("pages saved") || line.includes("Confirmed") || line.includes("confirmed")) return "text-emerald-400";
    if (line.includes("STOP")) return "text-amber-400 font-medium";
    if (line.includes("Blocked") || line.includes("blocked") || line.includes("BLOCKED")) return "text-amber-400";
    if (line.includes("Failed") || line.includes("Error") || line.includes("error") || line.includes("Rejected") || line.includes("rejected")) return "text-red-400";
    if (line.includes("Progress") || line.includes("[CRAWL]")) return "text-blue-400";
    if (line.includes("[NAV]") || line.includes("[SITEMAP]")) return "text-purple-400";
    if (line.includes("[LINKS]")) return "text-cyan-400";
    return "text-gray-400";
  };

  const getColor = colorize || defaultColorize;

  return (
    <div className="rounded-xl overflow-hidden border border-gray-800">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-[#0f1117] border-b border-gray-800">
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
        </div>
        <span className="text-[11px] text-gray-500 ml-2 font-mono">{label}</span>
        {isLive && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px] text-green-400">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            live
          </span>
        )}
      </div>
      <div
        ref={logRef}
        className="px-4 py-3 bg-[#0f1117] font-mono text-[12px] leading-5
                   max-h-72 overflow-y-auto text-gray-400"
        style={{ scrollbarWidth: "thin", scrollbarColor: "#374151 transparent" }}
      >
        {logs.map((line, i) => (
          <div
            key={i}
            className={`${getColor(line)} animate-fade-in-up`}
            style={{ animationDuration: "0.25s", animationFillMode: "both" }}
          >
            {line}
          </div>
        ))}
        {isLive && (
          <div className="flex items-center gap-1 mt-1 text-gray-600">
            <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-pulse" />
            <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-pulse" style={{ animationDelay: "0.2s" }} />
            <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-pulse" style={{ animationDelay: "0.4s" }} />
          </div>
        )}
      </div>
    </div>
  );
}
