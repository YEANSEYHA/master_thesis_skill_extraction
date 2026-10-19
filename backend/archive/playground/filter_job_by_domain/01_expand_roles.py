"""A2 Step 1 — HyDE: user's domain name → 5-8 job titles to search with."""

import json
from pathlib import Path

import ollama

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

DOMAINS = ["Data Science", "Network", "Software Engineering"]   # the user inputs


def expand_roles(domain):
    """qwen3: field name → job TITLES commonly posted for it. temp=0 = stable."""
    resp = ollama.chat(
        model="qwen3:14b", think=False,
        options={"temperature": 0, "num_ctx": 4096},
        format={"type": "object", "properties": {
            "roles": {"type": "array", "items": {"type": "string"},
                      "minItems": 5, "maxItems": 8}},
            "required": ["roles"]},
        messages=[
            {"role": "system", "content":
             "Given a field, domain, or course name, list 5-8 JOB TITLES "
             "commonly posted for that field on job boards. "
             'Return JSON: {"roles": [...]}'},
            {"role": "user", "content": domain}],
    )
    return json.loads(resp.message.content)["roles"]


#    ┌─────────────────────────────────────────────┐
#    │  DOMAINS = ["Data Science", "Network",      │ ◄── the user inputs
#    │             "Software Engineering"]         │     (later: from the UI)
#    └──────────────────────┬──────────────────────┘
#                           │  one domain at a time
#                           ▼
#    ╔═════════════════════════════════════════════╗
#    ║  expand_roles(domain)      — the HyDE call  ║
#    ╚═════════════════════════════════════════════╝
#    ┌─────────────────────────────────────────────┐
#    │  ollama.chat( model="qwen3:14b" )           │
#    │                                             │
#    │   system: "Given a field, list 5-8 JOB      │
#    │            TITLES posted on job boards"     │
#    │   user:   "Data Science"                    │
#    │                                             │
#    │   options:                                  │
#    │     temperature=0  ◄── same answer each run │
#    │     think=False    ◄── no chain-of-thought  │
#    │   format: JSON schema                       │
#    │     {"roles": [str x 5..8]}  ◄── forces     │
#    │                        clean, parseable out │
#    └──────────────────────┬──────────────────────┘
#                           │  resp.message.content  =  '{"roles": [...]}'
#                           ▼
#    ┌─────────────────────────────────────────────┐
#    │  json.loads(...)["roles"]                   │
#    │  → ["data scientist", "ml engineer",        │
#    │     "data analyst", ...]                    │
#    └──────────────────────┬──────────────────────┘
#                           │  collected per domain
#                           ▼
#    ┌─────────────────────────────────────────────┐
#    │  roles = { "Data Science":  [8 titles],     │
#    │            "Network":       [8 titles],     │
#    │            "Software Eng.": [8 titles] }    │
#    └──────────┬──────────────────────┬───────────┘
#               │                      │
#               ▼                      ▼
#    ┌────────────────────┐   ┌────────────────────────┐
#    │  print to terminal │   │  output/roles.json     │
#    │  (you eyeball the  │   │  (Step 3 retrieves     │
#    │   title quality)   │   │   with these titles)   │
#    └────────────────────┘   └────────────────────────┘


if __name__ == "__main__":
    roles = {}
    for d in DOMAINS:
        roles[d] = expand_roles(d)
        print(f"■ {d}")
        for r in roles[d]:
            print("   -", r)
        print()
    (OUT / "roles.json").write_text(json.dumps(roles, indent=2))
    print("saved output/roles.json   (Step 3 searches with these)")
