"use client";

import { useState, useCallback } from "react";
import { API_BASE } from "./components/ui";
import DiscoveryTab from "./components/DiscoveryTab";
import PipelineTab from "./components/PipelineTab";
import AnalyticsTab from "./components/AnalyticsTab";
import EscoTab from "./components/EscoTab";
import RecommendationV3Tab from "./components/RecommendationV3Tab";
import RecommendationV8Tab from "./components/RecommendationV8Tab";
import RecommendationESCOTab from "./components/RecommendationESCOTab";
import TrendingSkillTab_v1 from "./components/TrendingSkillTab_v1";
import TrendingSkillTab_v2 from "./components/TrendingSkillTab_v2";
import TrendingCompare from "./components/TrendingCompare";
import DashboardTab from "./components/DashboardTab";

// Tabs shown in the nav bar. Earlier methodology tabs (keys R3, R3_HELLO, R5,
// R6, R4, F1, F3) still have live render branches below but no button — add a
// row here to bring one back.
const TABS = [
  { key: "A", label: "Discovery" },
  { key: "B", label: "Pipeline" },
  { key: "C", label: "Analytics" },
  { key: "E", label: "Analytics (ESCO Exact Match)" },
  { key: "R8", label: "Curriculum–Market Gap Analysis" },
  { key: "F2", label: "Skill Demand & Trends" },
  // { key: "DASH", label: "Dashboard" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("F2");
  const [countries, setCountries] = useState([]);
  const [countriesLoading, setCountriesLoading] = useState(false);

  const fetchCountries = useCallback(async () => {
    setCountriesLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sites/countries`);
      const data = await res.json();
      setCountries(data.countries || []);
    } catch {
      setCountries([]);
    } finally {
      setCountriesLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen">
      {/* Top nav bar */}
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-6 flex items-center justify-between h-14">
          <span className="text-sm font-semibold tracking-tight text-gray-900">
            Skill Extraction
          </span>
          <nav className="flex items-center gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-colors
                  ${
                    activeTab === tab.key
                      ? "text-gray-900 bg-gray-100"
                      : "text-gray-400 hover:text-gray-600 hover:bg-gray-50"
                  }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main
        className={`${
          activeTab === "R8" ? "max-w-[1800px]" : "max-w-4xl"
        } mx-auto px-6 py-12`}
      >
        {activeTab === "A" && (
          <DiscoveryTab
            countries={countries}
            countriesLoading={countriesLoading}
            fetchCountries={fetchCountries}
          />
        )}

        {activeTab === "B" && (
          <PipelineTab
            countries={countries}
            countriesLoading={countriesLoading}
            fetchCountries={fetchCountries}
            onViewAnalytics={() => setActiveTab("C")}
          />
        )}

        {activeTab === "C" && <AnalyticsTab />}

        {activeTab === "E" && <EscoTab />}

        {activeTab === "R3" && <RecommendationV3Tab />}
        {activeTab === "R3_HELLO" && <RecommendationV3Tab variant="independent" />}
        {activeTab === "R5" && <RecommendationV3Tab variant="verified" />}
        {activeTab === "R6" && <RecommendationV3Tab variant="verified_raw" />}
        {activeTab === "R4" && <RecommendationESCOTab />}

        {activeTab === "R8" && <RecommendationV8Tab />}

        {activeTab === "F1" && <TrendingSkillTab_v1 />}
        {activeTab === "F2" && <TrendingSkillTab_v2 />}
        {activeTab === "F3" && <TrendingCompare />}
        {activeTab === "DASH" && <DashboardTab />}
      </main>
    </div>
  );
}
