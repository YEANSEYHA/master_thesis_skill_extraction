"""Step 1 — build the market vocabulary.

Production stage 1 of 13.  Mirrors the body of `pipeline.build_market()`,
calling the engine's own functions at every stage — nothing is
re-implemented here, so the funnel below is the pipeline's real funnel.

    254 postings
        -> clean_jobs()        lowercase, drop ad boilerplate
        -> count               1 vote per posting (set() per job)
        -> normalize_skills()  merge near-duplicates, cosine >= 0.80
        -> re-count            through the merge mapping
        -> frequency floor     keep skills seen in >= 2 postings
    == the market vocabulary + a demand count for every skill

Run:  python3 step1_build_market.py
"""

from collections import Counter

from phase5_pratice import JOB_MARKET, banner, engine, model as load_encoder


def build_market_vocab(p5, model, path=JOB_MARKET):
    """Return every intermediate of build_market() for inspection.

    Same sequence as the production function, but each stage is handed
    back so the practice can print what changed.
    """
    # -- clean: lowercase, length cap, SKILL_NOISE_RE (degrees, shifts, ...)
    jobs, stats = p5.clean_jobs(path)

    # our inputs/job_market.json is ALREADY the IT segment, so the
    # classify + bucket filter production runs here is a no-op; we only
    # drop postings whose skill list came out empty
    segment = [j for j in jobs if j["skills"]]

    # -- count: one vote per posting, so a verbose ad cannot inflate a skill
    raw = Counter(s for j in segment for s in set(j["skills"]))

    # -- merge: greedy embedding clustering, most-frequent-first, so the
    #    winner of a cluster is always its most common spelling
    mapping = p5.normalize_skills(model, raw)

    # -- re-count THROUGH the mapping (still one vote per posting)
    counts = Counter()
    for j in segment:
        counts.update({mapping.get(s, s) for s in j["skills"]})

    # -- floor: scales with corpus size; 254 // 500 == 0, so it stays at 2
    min_freq = max(p5.MIN_SKILL_FREQ, len(segment) // 500)
    skill_base = sorted(s for s, c in counts.items() if c >= min_freq)

    return {"stats": stats, "segment": segment, "raw": raw,
            "mapping": mapping, "counts": counts, "skill_base": skill_base,
            "min_freq": min_freq}


def main() -> None:
    banner("STEP 1", "build the market vocabulary",
           "a unique skill list + how many postings ask for each")

    p5 = engine()
    print(f"engine   : {p5.__file__}")
    print(f"encoder  : {p5.MODEL_NAME}")
    print(f"merge at : cosine >= {p5.SKILL_MERGE_THRESHOLD}")
    print("loading encoder ...\n")
    m = build_market_vocab(p5, load_encoder())
    seg, raw, counts, base = (m["segment"], m["raw"], m["counts"],
                              m["skill_base"])

    print(f"postings kept              : {len(seg)}")
    print(f"raw skill strings          : {len(raw)}")
    print(f"after merging near-dupes   : {len(counts)}")
    print(f"after floor (>= {m['min_freq']} postings) : {len(base)}"
          f"   <- the market vocabulary\n")

    print("top 15 by demand:")
    for skill, c in Counter({s: counts[s] for s in base}).most_common(15):
        print(f"   {c:4d} jobs  {skill}")

    absorbed = [(v, k) for k, v in m["mapping"].items() if k != v]
    print(f"\n{len(absorbed)} strings were absorbed into a canonical form:")
    for canon, variant in absorbed[:8]:
        print(f'   "{variant}"  ->  "{canon}"')
    print("\nnote: merging is imperfect at 0.80 — sibling technologies can")
    print("collapse together. Watch for it again in step 4.")


if __name__ == "__main__":
    main()
