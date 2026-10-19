"""Compare open-source local models on the Stage-3 VERIFY task.

We hand-label a small set of clear MEMBERS (belong) and clear IMPOSTORS (share a
word but a different job) per domain, then ask each model to reject. A good
verifier: rejects impostors (high impostor-recall) WITHOUT rejecting real
members (high member-keep). We report both + latency.

Metric that matters most: F1 over the "reject impostor" decision — but member-keep
is weighted heavily because a false reject silently deletes a real cohort title.
"""

import json
import time

import ollama

MODELS = ["qwen3:14b", "qwen3:8b", "llama3.1:8b"]

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

# GOLD: (title, belongs?)  True = member (keep),  False = impostor (reject)
# Mix of easy cases + HARD ones: impostors that share a STRONG word with the
# domain (the cases that separate a strong verifier from a weak one).
GOLD = {
    "artificial_intelligence": [
        ("machine learning engineer", True),
        ("artificial intelligence engineer", True),
        ("computer vision engineer", True),
        ("robotics technician", False),
        ("intelligence analyst", False),
        ("machine operator", False),            # HARD: shares "machine"
        ("ai sales specialist", False),         # HARD: sells AI, not builds it
        ("learning experience designer", False),# HARD: shares "learning"
    ],
    "mobile_development": [
        ("senior android engineer", True),
        ("ios developer", True),
        ("react native developer", True),
        ("mobile veterinarian", False),
        ("mobile marketing manager", False),    # HARD: shares "mobile"
        ("t-mobile retail associate", False),   # HARD: brand name "mobile"
        ("automotive mobility engineer", False),# HARD: mobility ≠ mobile dev
    ],
    "game_development": [
        ("game developer", True),
        ("unity developer", True),
        ("gameplay programmer", True),
        ("mainframe developer", False),
        ("game warden", False),                 # HARD: shares "game"
        ("board game cafe manager", False),     # HARD: shares "game"
        ("learning & development consultant", False),
    ],
    "cybersecurity": [
        ("penetration tester", True),
        ("soc analyst", True),
        ("security operations engineer", True),
        ("physical security officer", False),
        ("security guard", False),              # HARD: shares "security"
        ("financial securities analyst", False),# HARD: "securities" ≠ security
    ],
}


def run_model(model):
    """Return (tp, fp, fn, tn, secs) aggregated over all domains.
    positive class = 'reject impostor'.
      tp = impostor correctly rejected     fn = impostor wrongly kept
      fp = member wrongly rejected          tn = member correctly kept
    """
    tp = fp = fn = tn = 0
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
            if not belongs and rejected:      tp += 1
            elif not belongs and not rejected: fn += 1
            elif belongs and rejected:         fp += 1
            else:                              tn += 1
    return tp, fp, fn, tn, time.time() - t0


def main():
    n_imp = sum(1 for rows in GOLD.values() for _, b in rows if not b)
    n_mem = sum(1 for rows in GOLD.values() for _, b in rows if b)
    print(f"gold: {n_mem} members (keep) + {n_imp} impostors (reject)\n")
    print(f"{'model':<14}{'imp-recall':>11}{'mem-keep':>10}{'F1':>7}{'secs':>7}")
    print("-" * 49)
    for m in MODELS:
        tp, fp, fn, tn, secs = run_model(m)
        imp_recall = tp / (tp + fn) if (tp + fn) else 0   # impostors caught
        mem_keep = tn / (tn + fp) if (tn + fp) else 0     # members preserved
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * imp_recall / (prec + imp_recall) if (prec + imp_recall) else 0
        print(f"{m:<14}{imp_recall:>10.0%}{mem_keep:>10.0%}{f1:>7.2f}{secs:>7.1f}")


if __name__ == "__main__":
    main()
