"""Stage 3 — VERIFY: qwen3 audits every retrieved cohort title (precision).

Same retrieve→verify pattern as the V6 matcher: the embedding proposes (stage 2),
the LLM decides. Kills the zero-shot name-anchor failure modes:
"intelligence analyst" ≠ artificial intelligence, "display builder" ≠ ui/ux
design, "mainframe developer" ≠ game development, "mobile veterinarian" ≠
mobile development. Degrades gracefully to the embedding-only cohort if Ollama
is down.
"""

import json
import os
import sys

# Verify-only model override. Defaults to llama3.1:8b: on a 120-title labeled
# pilot (playground/domain_routing/bigger_compare.py) it beat qwen3:14b on the
# cohort-verify task — perfect member/impostor separation (F1 1.00 vs 0.90) at
# ~2/3 the latency, and deterministic. qwen3:14b over-rejects here, degenerately
# dropping real members (react developer, kubernetes administrator, ML scientist).
# Scoped to VERIFY only via its own env var so stage5 routing stays on
# PHASE5_LLM_MODEL (qwen3:14b). Revert by setting PHASE5_VERIFY_MODEL=qwen3:14b.
_LLM = os.getenv("PHASE5_VERIFY_MODEL", "llama3.1:8b")
_VERIFY_CHUNK = 80                                   # titles judged per LLM call

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"reject": {"type": "array", "items": {"type": "integer"}}},
    "required": ["reject"],
}

_VERIFY_SYS = """You audit job titles proposed for ONE career domain.
A title belongs only if that job's PRIMARY day-to-day work is in the domain.
REJECT titles that:
- belong to a different field (a "display builder" builds physical displays, not UI/UX),
- merely share a word with the domain name ("intelligence analyst" is NOT
  artificial intelligence; "mobile veterinarian" is NOT mobile development;
  "mainframe developer" is NOT game development),
- are too generic to imply the domain (a bare "developer" or "analyst").
When unsure, REJECT (the cohort must stay high-precision).
Return JSON: {"reject": [numbers of the titles that do NOT belong]}."""


def verify_cohorts(ordered: dict[str, list[str]]) -> dict[str, set]:
    """{domain: set(rejected titles)}. On any Ollama failure the affected chunk
    is kept as-is (embedding-only fallback), so the build never dies here."""
    rejected: dict[str, set] = {d: set() for d in ordered}
    try:
        import ollama  # noqa: F401
    except Exception:
        print("[v3:profiles] ollama unavailable — skipping LLM verify",
              file=sys.stderr, flush=True)
        return rejected
    import ollama
    for d, titles in ordered.items():
        for lo in range(0, len(titles), _VERIFY_CHUNK):
            chunk = titles[lo:lo + _VERIFY_CHUNK]
            user = (f"Career domain: {d.replace('_', ' ')}\n\nJob titles:\n" +
                    "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk)))
            try:
                resp = ollama.chat(
                    model=_LLM, think=False,
                    options={"temperature": 0, "num_ctx": 8192},
                    format=_VERIFY_SCHEMA,
                    messages=[{"role": "system", "content": _VERIFY_SYS},
                              {"role": "user", "content": user}],
                )
                for i in json.loads(resp.message.content).get("reject", []):
                    if 1 <= i <= len(chunk):
                        rejected[d].add(chunk[i - 1])
            except Exception as e:
                print(f"[v3:profiles] verify failed for {d} chunk@{lo} ({e}) — "
                      f"keeping chunk", file=sys.stderr, flush=True)
        print(f"  verify {d}: rejected {len(rejected[d])}/{len(titles)} titles",
              file=sys.stderr, flush=True)
    return rejected
