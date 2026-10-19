"""Bigger verify-task comparison: qwen3:14b vs llama3.1:8b vs qwen3:8b.

~120 single-annotator labeled titles across 12 domains. Each domain has clear
MEMBERS (belong) + hard IMPOSTORS (share a strong word / adjacent field but a
different job). Genuinely-ambiguous cases (robotics↔AI, graphic↔ux) are omitted
so the labels are defensible.

NOTE: single-annotator "silver" labels — a pilot for the real 2-annotator κ
study, not a substitute for it. Enough to make a switch decision, not a paper claim.
"""

import json
import time

import ollama

MODELS = ["qwen3:14b", "llama3.1:8b", "qwen3:8b"]

SCHEMA = {"type": "object",
          "properties": {"reject": {"type": "array", "items": {"type": "integer"}}},
          "required": ["reject"]}

SYS = """You audit job titles proposed for ONE career domain.
A title belongs only if that job's PRIMARY day-to-day work is in the domain.
REJECT titles that:
- belong to a different field (a "display builder" builds physical displays, not UI/UX),
- merely share a word with the domain name ("intelligence analyst" is NOT
  artificial intelligence; "mobile veterinarian" is NOT mobile development;
  "mainframe developer" is NOT game development),
- are too generic to imply the domain (a bare "developer" or "analyst").
When unsure, REJECT (the cohort must stay high-precision).
Return JSON: {"reject": [numbers of the titles that do NOT belong]}."""

M, X = True, False   # M = member (keep), X = impostor (reject)
GOLD = {
    "artificial_intelligence": [
        ("machine learning engineer", M), ("ai research scientist", M),
        ("nlp engineer", M), ("computer vision engineer", M),
        ("deep learning engineer", M),
        ("intelligence analyst", X), ("business intelligence developer", X),
        ("ai sales specialist", X), ("machine operator", X),
        ("emotional intelligence coach", X),
    ],
    "data_science": [
        ("data scientist", M), ("applied scientist", M),
        ("quantitative researcher", M), ("machine learning scientist", M),
        ("research data scientist", M),
        ("data entry clerk", X), ("data center technician", X),
        ("data cabling technician", X), ("census data collector", X),
        ("master data steward", X),
    ],
    "software_engineering": [
        ("software engineer", M), ("backend developer", M),
        ("full stack developer", M), ("senior software engineer", M),
        ("application developer", M),
        ("sales engineer", X), ("field service engineer", X),
        ("civil engineer", X), ("sound engineer", X),
        ("stationary engineer", X),
    ],
    "cybersecurity": [
        ("penetration tester", M), ("soc analyst", M),
        ("security operations engineer", M), ("incident response analyst", M),
        ("threat intelligence analyst", M),
        ("security guard", X), ("physical security officer", X),
        ("financial securities analyst", X), ("loss prevention officer", X),
        ("national security advisor", X),
    ],
    "cloud_devops": [
        ("devops engineer", M), ("site reliability engineer", M),
        ("cloud engineer", M), ("platform engineer", M),
        ("kubernetes administrator", M),
        ("cloud kitchen manager", X), ("meteorologist", X),
        ("warehouse associate", X), ("retail cashier", X),
        ("cloud 9 spa receptionist", X),
    ],
    "networking": [
        ("network engineer", M), ("network administrator", M),
        ("noc engineer", M), ("wireless network engineer", M),
        ("network operations technician", M),
        ("professional networking coach", X), ("event networking coordinator", X),
        ("television network producer", X), ("social media manager", X),
        ("business networking consultant", X),
    ],
    "mobile_development": [
        ("android developer", M), ("ios engineer", M),
        ("react native developer", M), ("mobile app developer", M),
        ("flutter developer", M),
        ("mobile veterinarian", X), ("mobile marketing manager", X),
        ("t-mobile retail associate", X), ("mobile crane operator", X),
        ("mobile phlebotomist", X),
    ],
    "game_development": [
        ("game developer", M), ("unity developer", M),
        ("gameplay programmer", M), ("unreal engine developer", M),
        ("game engine programmer", M),
        ("game warden", X), ("board game cafe manager", X),
        ("casino games dealer", X), ("hunting guide", X),
        ("learning & development consultant", X),
    ],
    "web_frontend": [
        ("frontend developer", M), ("react developer", M),
        ("ui developer", M), ("web developer", M),
        ("javascript engineer", M),
        ("web content writer", X), ("front desk receptionist", X),
        ("storefront retail manager", X), ("web press operator", X),
        ("spider control technician", X),
    ],
    "database_administration": [
        ("database administrator", M), ("oracle dba", M),
        ("sql database administrator", M), ("database engineer", M),
        ("postgresql administrator", M),
        ("data entry operator", X), ("records management clerk", X),
        ("filing clerk", X), ("warehouse inventory clerk", X),
        ("library circulation clerk", X),
    ],
    "qa_testing": [
        ("qa engineer", M), ("test automation engineer", M),
        ("sdet", M), ("quality assurance analyst", M),
        ("software test engineer", M),
        ("quality control inspector", X), ("food safety tester", X),
        ("laboratory testing technician", X), ("covid testing coordinator", X),
        ("product quality auditor", X),
    ],
    "ui_ux_design": [
        ("ux designer", M), ("ui designer", M),
        ("product designer", M), ("interaction designer", M),
        ("ux researcher", M),
        ("interior designer", X), ("fashion designer", X),
        ("floral designer", X), ("display builder", X),
        ("landscape designer", X),
    ],
}


def run_model(model):
    tp = fp = fn = tn = 0
    errors = {"kept_impostors": [], "dropped_members": []}
    t0 = time.time()
    for domain, rows in GOLD.items():
        titles = [t for t, _ in rows]
        user = (f"Career domain: {domain.replace('_', ' ')}\n\nJob titles:\n" +
                "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles)))
        try:
            resp = ollama.chat(model=model, think=False,
                               options={"temperature": 0, "num_ctx": 8192},
                               format=SCHEMA,
                               messages=[{"role": "system", "content": SYS},
                                         {"role": "user", "content": user}])
            rej = set(json.loads(resp.message.content).get("reject", []))
        except Exception as e:
            print(f"  {model} failed on {domain}: {e}")
            rej = set()
        for i, (title, belongs) in enumerate(rows, start=1):
            rejected = i in rej
            if not belongs and rejected:       tp += 1
            elif not belongs and not rejected:
                fn += 1; errors["kept_impostors"].append(f"{domain}: {title}")
            elif belongs and rejected:
                fp += 1; errors["dropped_members"].append(f"{domain}: {title}")
            else:                              tn += 1
    return tp, fp, fn, tn, time.time() - t0, errors


def main():
    n_imp = sum(1 for rows in GOLD.values() for _, b in rows if not b)
    n_mem = sum(1 for rows in GOLD.values() for _, b in rows if b)
    print(f"gold: {n_mem} members + {n_imp} impostors across {len(GOLD)} domains\n")
    print(f"{'model':<14}{'imp-recall':>11}{'mem-keep':>10}{'F1':>7}{'secs':>7}")
    print("-" * 49)
    stash = {}
    for m in MODELS:
        tp, fp, fn, tn, secs, errors = run_model(m)
        imp_recall = tp / (tp + fn) if (tp + fn) else 0
        mem_keep = tn / (tn + fp) if (tn + fp) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * imp_recall / (prec + imp_recall) if (prec + imp_recall) else 0
        stash[m] = errors
        print(f"{m:<14}{imp_recall:>10.0%}{mem_keep:>10.0%}{f1:>7.2f}{secs:>7.1f}")

    for m in MODELS:
        e = stash[m]
        print(f"\n=== {m} errors ===")
        print(f"  impostors KEPT ({len(e['kept_impostors'])}): "
              + ("; ".join(e["kept_impostors"]) or "none"))
        print(f"  members DROPPED ({len(e['dropped_members'])}): "
              + ("; ".join(e["dropped_members"]) or "none"))


if __name__ == "__main__":
    main()
