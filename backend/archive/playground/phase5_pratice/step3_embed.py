"""Step 3 — embed both sides.

Production stage 3 of 13.  The market side is embedded inside
`build_market()` (cached as `skill_emb`); the curriculum side is these
lines in `gen_report_data.main()`:

    clo_emb = model.encode(clo_texts, normalize_embeddings=True)

Two things to understand here.

(a) normalize_embeddings=True makes every vector length 1. For unit
    vectors  cos(a, b) = a . b , so the whole similarity step in step 4
    collapses into ONE matrix multiply — no norms, no division.

(b) `clo_owner` is collected alongside `clo_texts`. After the matrix
    multiply the 226 columns are anonymous; clo_owner remembers which
    course each one came from, which is what lets step 5 collapse them
    into 8 course columns.

Both sides go through the SAME encoder. That is what makes a market skill
and a course outcome comparable at all — two different models would put
them in two different vector spaces.

Run:  python3 step3_embed.py
"""

import numpy as np

from phase5_pratice import (banner, curriculum_units, engine, market, model,
                            read_curriculum)


def embed_curriculum(mdl, clo_texts):
    """Encode the course outcomes to UNIT vectors (dot product = cosine)."""
    return mdl.encode(clo_texts, normalize_embeddings=True,
                      show_progress_bar=False)


def main() -> None:
    banner("STEP 3", "embed both sides",
           "unit vectors for every market skill and every course outcome")

    p5 = engine()
    mkt = market()
    mdl = model()

    # --- market side: already embedded and cached by build_market()
    skill_emb = mkt["skill_emb"]
    base = mkt["skill_base"]

    # --- curriculum side: encode now
    courses = read_curriculum()
    course_names, clo_texts, clo_owner = curriculum_units(courses)
    clo_emb = embed_curriculum(mdl, clo_texts)

    print(f"encoder     : {p5.MODEL_NAME}")
    print(f"market side : {skill_emb.shape[0]} skills      -> "
          f"{skill_emb.shape[1]}-dim vectors   (cached by build_market)")
    print(f"school side : {clo_emb.shape[0]} CLO skills -> "
          f"{clo_emb.shape[1]}-dim vectors   (encoded just now)")
    print(f"courses     : {len(course_names)}")
    print(f"unit length check (should be 1.0): "
          f"{np.linalg.norm(clo_emb[0]):.4f}\n")

    print("CLO skills per course (clo_owner is what remembers this):")
    for name in course_names:
        n = sum(1 for o in clo_owner if o == name)
        print(f"   {n:4d}  {name}")

    # a preview of step 4, done one row at a time: for a few market skills,
    # which single course outcome is closest?
    print("\nclosest course outcome for a few market skills")
    print("(cosine = dot product, because both sides are unit vectors):")
    idx = {s: i for i, s in enumerate(base)}
    for skill in ("database management", "docker containerization",
                  "java programming", "code reviews"):
        if skill not in idx:
            continue
        sims = clo_emb @ skill_emb[idx[skill]]      # one row of step 4
        j = int(np.argmax(sims))
        print(f"   {skill:26s} {sims[j]:.2f}  \"{clo_texts[j]}\""
              f"  [{clo_owner[j]}]")

    print("\n   ^ 'java programming' matches a CLO called \"Python\": the")
    print("     encoder reads both as 'a programming language'. Same failure")
    print("     the 0.80 merge made in step 1 — it is a property of the")
    print("     encoder, not of any one stage. Step 7 is what catches it.")
    print("\n   ^ 'code reviews' has no real match anywhere, yet still scores")
    print("     ~0.67. That is the encoder's NOISE FLOOR, and it is why the")
    print("     gap threshold sits at 0.69 rather than anywhere near zero.")


if __name__ == "__main__":
    main()
