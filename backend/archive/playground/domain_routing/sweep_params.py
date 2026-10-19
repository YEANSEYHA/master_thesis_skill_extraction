"""Parameter sweep for Stage 2 (retrieve). Embeds titles + anchors ONCE, then
re-runs the floor/margin/cap gating for several parameter sets on the cached
cosine matrix (cheap). Reports quality metrics per config so we can compare.

Quality signals (Stage 2 wants HIGH RECALL, decent balance — precision is the
LLM's job in Stage 3):
  kept        total titles sorted into cohorts (recall proxy)
  starved     domains with < 10 titles (too few to build a profile)
  capped      domains hitting MAX_TITLES (may be over-broad / diluted)
  lex_match%  rough PRECISION proxy: % of cohort titles sharing a meaningful
              token with the domain name / a small synonym set. Imperfect
              (misses "site reliability engineer" for cloud_devops) but flags
              over-broad cohorts well.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent / "output"   # generated results live here

DOMAINS = [
    "artificial_intelligence", "data_science", "software_engineering",
    "cybersecurity", "cloud_devops", "networking", "data_engineering",
    "data_analytics", "qa_testing", "business_analysis", "web_frontend",
    "mobile_development", "embedded_systems", "database_administration",
    "it_operations", "it_support", "project_product_management",
    "blockchain", "ui_ux_design", "game_development",
]

# tokens that count as "on-topic" for the lexical precision proxy
SYN = {
    "artificial_intelligence": {"ai", "ml", "machine", "learning", "artificial", "intelligence"},
    "data_science": {"data", "scientist", "science"},
    "software_engineering": {"software", "developer", "engineer", "programmer", "sde"},
    "cybersecurity": {"security", "cyber", "soc", "infosec", "penetration", "pentest"},
    "cloud_devops": {"cloud", "devops", "sre", "aws", "azure", "kubernetes", "infrastructure"},
    "networking": {"network", "networking", "cisco", "noc"},
    "data_engineering": {"data", "engineer", "etl", "pipeline"},
    "data_analytics": {"analytics", "analyst", "bi", "intelligence", "data"},
    "qa_testing": {"qa", "test", "tester", "quality", "sdet", "automation"},
    "business_analysis": {"business", "analyst", "analysis", "systems"},
    "web_frontend": {"frontend", "front", "web", "react", "javascript", "ui"},
    "mobile_development": {"mobile", "android", "ios", "app", "flutter"},
    "embedded_systems": {"embedded", "firmware", "hardware", "systems"},
    "database_administration": {"database", "dba", "sql", "administration"},
    "it_operations": {"it", "operations", "ops", "systems"},
    "it_support": {"it", "support", "helpdesk", "desktop", "technician"},
    "project_product_management": {"project", "product", "manager", "management", "scrum", "pm"},
    "blockchain": {"blockchain", "crypto", "web3", "solidity", "smart"},
    "ui_ux_design": {"ui", "ux", "design", "designer", "product"},
    "game_development": {"game", "gaming", "unity", "unreal", "games"},
}

CONFIGS = [
    {"name": "baseline",       "floor": 0.45, "margin": 0.05, "cap": 500},
    {"name": "strict-floor",   "floor": 0.55, "margin": 0.05, "cap": 500},
    {"name": "strict-margin",  "floor": 0.45, "margin": 0.10, "cap": 500},
    {"name": "strict-both",    "floor": 0.52, "margin": 0.08, "cap": 300},
    {"name": "loose",          "floor": 0.40, "margin": 0.03, "cap": 500},
]


def gate(sims, titles, floor, margin, cap):
    order = np.argsort(-sims, axis=1)
    best = order[:, 0]
    s1 = sims[np.arange(len(titles)), best]
    s2 = sims[np.arange(len(titles)), order[:, 1]]
    cand = {d: [] for d in DOMAINS}
    for i, t in enumerate(titles):
        if s1[i] >= floor and (s1[i] - s2[i]) >= margin:
            cand[DOMAINS[best[i]]].append((float(s1[i]), t))
    cohorts = {}
    for d in DOMAINS:
        cand[d].sort(reverse=True)
        cohorts[d] = [t for _, t in cand[d][:cap]]
    return cohorts


def lex_match_rate(cohorts):
    """% of cohort titles that share a token with their domain's synonym set."""
    hit = tot = 0
    for d, ts in cohorts.items():
        syn = SYN[d]
        for t in ts:
            tot += 1
            if syn & set(t.replace("/", " ").replace("-", " ").split()):
                hit += 1
    return 100.0 * hit / tot if tot else 0.0


def main():
    titles = json.loads((OUT / "it_slice.json").read_text())["titles"]
    print(f"embedding {len(titles):,} titles once...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    anchors = np.asarray(model.encode([d.replace("_", " ") for d in DOMAINS],
                                       normalize_embeddings=True))
    T = np.asarray(model.encode(titles, normalize_embeddings=True,
                                batch_size=512, show_progress_bar=False))
    sims = T @ anchors.T
    print("done embedding. sweeping configs...\n", flush=True)

    print(f"{'config':<15}{'floor':>6}{'margin':>7}{'cap':>5}"
          f"{'kept':>7}{'starved':>8}{'capped':>7}{'lex%':>7}")
    print("-" * 62)
    results = []
    for c in CONFIGS:
        coh = gate(sims, titles, c["floor"], c["margin"], c["cap"])
        kept = sum(len(v) for v in coh.values())
        starved = sum(1 for v in coh.values() if len(v) < 10)
        capped = sum(1 for v in coh.values() if len(v) >= c["cap"])
        lex = lex_match_rate(coh)
        results.append((c, coh, kept, starved, capped, lex))
        print(f"{c['name']:<15}{c['floor']:>6}{c['margin']:>7}{c['cap']:>5}"
              f"{kept:>7,}{starved:>8}{capped:>7}{lex:>6.1f}")

    # per-domain sizes for baseline vs strict-both (the two most interesting)
    print("\nper-domain cohort sizes (baseline vs strict-both):")
    print(f"{'domain':<28}{'baseline':>10}{'strict-both':>13}")
    base = results[0][1]
    strict = results[3][1]
    for d in DOMAINS:
        print(f"{d:<28}{len(base[d]):>10}{len(strict[d]):>13}")


if __name__ == "__main__":
    main()
