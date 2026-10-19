#!/usr/bin/env python3
"""Export the frozen 250-pair sheet into a BLINDED payload for the web app.

The web application must never receive the cosine score, the band, the
stratum (census / boundary / control) or the system's covered/borderline/gap
decision.  Anything shipped to the browser can be read by a rater, so the
blinding is enforced here, at the export boundary, rather than in the UI.

Reads   backend/phase5_curriculum_alignment/05_evaluation/annotation/sheet_250_master.json
        backend/phase5_curriculum_alignment/05_evaluation/annotation/build_sheet_250.py (GLOSSARY)
Writes  web_annotate/backend/data/pairs.json

Run again only if the frozen sheet itself is rebuilt.  The sheet_hash is
carried through so the ratings can always be tied back to the exact sheet.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANNOT = REPO / "backend" / "phase5_curriculum_alignment" / "05_evaluation" / "annotation"
MASTER = ANNOT / "sheet_250_master.json"
BUILDER = ANNOT / "build_sheet_250.py"
OUT = HERE.parent / "data" / "pairs.json"

# fields that must NOT leave the master file
BLIND = {"cos", "band", "set"}

# The sheet stores course names title-cased from the filename, which mangles
# roman numerals and initialisms ("Networks Ii", "It Project Management").
# Purely orthographic: the pair itself is keyed on `row`, unchanged.
SMALL = {"and", "to", "of", "for", "in", "the", "with"}
FIXED = {"ii": "II", "iii": "III", "it": "IT", "devops": "DevOps",
         "ai": "AI", "sql": "SQL"}


def pretty_course(name: str) -> str:
    out = []
    for i, word in enumerate(name.split()):
        low = word.lower()
        if low in FIXED:
            out.append(FIXED[low])
        elif i and low in SMALL:
            out.append(low)
        else:
            out.append(word)
    return " ".join(out)


def glossary():
    """Pull the GLOSSARY literal out of build_sheet_250.py without importing
    it (importing drags in the whole alignment pipeline: torch, numpy...)."""
    tree = ast.parse(BUILDER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "GLOSSARY" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("GLOSSARY not found in build_sheet_250.py")


def main():
    master = json.loads(MASTER.read_text())
    gloss = glossary()

    pairs = []
    for p in master["pairs"]:
        pairs.append({
            "row": p["row"],
            "skill": p["skill"],
            "jobs": p["jobs"],
            "course": pretty_course(p["course"]),
            "clo": p["clo"],
            "cohort": p["cohort"],
            "anchor": bool(p["anchor"]),
            "glossary": gloss.get(p["skill"]),
        })
    assert len(pairs) == 250, f"expected 250 pairs, got {len(pairs)}"
    assert not (BLIND & set(pairs[0])), "blinded field leaked into payload"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "sheet_hash": master["sheet_hash"],
        "locked_run": master["locked_run"],
        "config_hash": master["config_hash"],
        "input_hash": master["input_hash"],
        "seed": master["seed"],
        "scoring": master["scoring"],
        "counts": master["counts"],
        "pairs": pairs,
    }, indent=1))
    n_anchor = sum(1 for p in pairs if p["anchor"])
    n_gloss = sum(1 for p in pairs if p["glossary"])
    print(f"wrote {OUT.relative_to(REPO)}: {len(pairs)} pairs "
          f"({n_anchor} anchors, {n_gloss} with a glossary entry)")
    print(f"sheet_hash {master['sheet_hash'][:16]}  blinded: {sorted(BLIND)}")


if __name__ == "__main__":
    main()
