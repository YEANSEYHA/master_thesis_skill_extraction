"""Domain routing — the career-domain pipeline, one stage per file, in order.

    stage1_corpus.py    CATEGORY_DOMAINS taxonomy + corpus loading (IT slice …)
    stage2_retrieve.py  anchor = embed the domain NAME → pass-1 title cohorts
    stage3_verify.py    qwen3 audits each cohort (retrieve → verify)
    stage4_profiles.py  aggregate cohort skills → domain_profiles.json  [BUILD CLI]
    stage5_route.py     runtime routing: centroids + cosine + route_hybrid [SERVED]

Build (offline):   python stage4_profiles.py [category ...]
Runtime (FastAPI): api.py path-loads stage5_route.py

Data files stay in the parent folder (recommendation_v3/domain_profiles.json) so
build_inventory.py and friends keep reading them unchanged.
"""

import sys
from pathlib import Path

# stage files import each other flat (`import stage1_corpus`), matching the repo
# idiom (recommend.py / coverage.py) — make that work from any entry point.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
