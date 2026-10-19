import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"   # generated results live here
OUT.mkdir(exist_ok=True)

MODEL = "llama3.1:8b"
CHUNK = 80              # titles judged per LLM call (keeps the prompt small)

# force qwen3 to answer in a fixed JSON shape: a list of indices to reject
SCHEMA = {
    "type": "object",
    "properties": {"reject": {"type": "array", "items": {"type": "integer"}}},
    "required": ["reject"],
}

SYS = """
You audit job titles proposed for ONE career domain.
A title belongs only if that job's PRIMARY day-to-day work is in the domain.
REJECT titles that:
- belong to a different field (a "display builder" builds physical displays, not UI/UX),
- merely share a word with the domain name ("intelligence analyst" is NOT artificial intelligence;
"mobile veterinarian" is NOT mobile development; "mainframe developer" is NOT game development),
- are too generic to imply the domain (a bare "developer" or "analyst").
When unsure, REJECT (the cohort must stay high-precision).
Return JSON: {"reject": [numbers of the titles that do NOT belong]}

"""

def verify_cohort(domain, titles):
    """Return the set of titles qwen3 rejects for this domain."""
    rejected = set()
    for lo in range(0, len(titles), CHUNK):
        chunk = titles[lo:lo + CHUNK]
        user = (f"Career domain: {domain.replace('_', ' ')}\n\nJob titles:\n" +
                "\n".join(f"{i+1}. {t}" for i, t in enumerate(chunk)))
        try:
            resp = ollama.chat(
                model=MODEL, think=False,
                options={"temperature": 0, "num_ctx": 8192},
                format=SCHEMA,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
            )
            for i in json.loads(resp.message.content).get("reject", []):
                if 1 <= i <= len(chunk):
                    rejected.add(chunk[i - 1])
        except Exception as e:
            print(f"  verify failed for {domain} chunk@{lo} ({e}) — keeping chunk")
    return rejected


# cohorts.json  ──llama3.1:8b judges each title──▶  verified.json
# (noisy: has                                        (clean: impostors
#  impostors)                                         removed)

if __name__ == "__main__":
    cohorts = json.loads((OUT / "cohorts.json").read_text())

    # DEMO: verify just a few interesting domains first (fast). Set to None
    # later to audit ALL 20 (slower — a few minutes).
    ONLY = ["artificial_intelligence", "mobile_development", "game_development"]

    verified, rejected_all = {}, {}
    for d, titles in cohorts.items():
        if ONLY and d not in ONLY:
            continue
        rej = verify_cohort(d, titles)
        verified[d] = [t for t in titles if t not in rej]
        rejected_all[d] = sorted(rej)
        print(f"{d:<28} kept {len(verified[d]):3}/{len(titles):3}  "
              f"rejected {len(rej)}")
        for t in sorted(rej)[:6]:
            print(f"     ✗ {t}")

    (OUT / "verified.json").write_text(json.dumps(verified, indent=1))
    (OUT / "rejected.json").write_text(json.dumps(rejected_all, indent=1))
    print("\nsaved output/verified.json + output/rejected.json")