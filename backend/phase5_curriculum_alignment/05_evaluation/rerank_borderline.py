# Cross-encoder verification of the pairs that decide the dashboard
# (thesis task #5, "similar != taught"). Stage 1 (BGE bi-encoder, cached)
# retrieves each top-demanded skill's best 3 CLOs by cosine; stage 2
# re-reads those pairs jointly with BAAI/bge-reranker-v2-m3 and scores
# real relevance. Retrieve-then-rerank (Nogueira & Cho 2019).
#
# Reranker scores are a THIRD scale — never compare them with BGE or
# MiniLM cosines. Verdict bands below are PROVISIONAL until the task #1
# annotation calibrates them.
#
# Run from the project root (first run downloads the reranker, ~2.3 GB):
#   python3 05_evaluation/rerank_borderline.py

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_engine"))
from pipeline import (
    BASE_DIR, MODEL_NAME, best_device, build_market, is_soft_skill,
    load_model,
)

RERANKER = "BAAI/bge-reranker-v2-m3"
CLO_DIR = BASE_DIR / "02_input_curriculum" / "clo_skills"
OUT = Path(__file__).resolve().parent / "rerank_results.json"
TOP_N = 25          # demanded skills to verify (same as the dashboard)
TOP_K = 3           # CLO candidates per skill to re-read

# PROVISIONAL bands on the reranker's sigmoid score (task #2 recalibrates)
SUPPORTED_AT = 0.50
WEAK_AT = 0.10


def verdict(score: float) -> str:
    if score >= SUPPORTED_AT:
        return "SUPPORTED"
    if score >= WEAK_AT:
        return "WEAK"
    return "NOT SUPPORTED"


def clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    print(f"Stage 1 — retrieval with '{MODEL_NAME}' (cached market) ...")
    model = load_model(MODEL_NAME)
    market = build_market(model)

    clo_texts, clo_owner = [], []
    for f in sorted(CLO_DIR.glob("*.json")):
        name = f.stem.split("_", 1)[1].replace("_", " ").title()
        for item in json.loads(f.read_text()):
            clo_texts.append(item["skill"])
            clo_owner.append(name)
    print(f"  curriculum: {len(set(clo_owner))} courses, "
          f"{len(clo_texts)} CLO skills")

    clo_emb = model.encode(clo_texts, normalize_embeddings=True,
                           show_progress_bar=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = market["skill_emb"] @ clo_emb.T

    base, counts = market["skill_base"], market["counts"]
    hard = [i for i, s in enumerate(base) if not is_soft_skill(s)]
    top = sorted(hard, key=lambda i: -counts[base[i]])[:TOP_N]

    # candidate pairs: each top skill x its best TOP_K CLOs by cosine
    records, pairs = [], []
    for i in top:
        cand = np.argsort(-sim[i])[:TOP_K]
        records.append({"row": i, "skill": base[i],
                        "jobs": int(counts[base[i]]),
                        "cand_idx": [int(j) for j in cand]})
        pairs.extend((base[i], clo_texts[j]) for j in cand)
    print(f"  candidates: {len(records)} skills x {TOP_K} CLOs "
          f"= {len(pairs)} pairs (of {sim.size:,} in the full matrix)")

    print(f"\nStage 2 — verification with '{RERANKER}' "
          f"(device={best_device() or 'cpu'}) ...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANKER, device=best_device(), max_length=256)
    scores = np.asarray(reranker.predict(pairs, batch_size=32,
                                         show_progress_bar=False),
                        dtype=float)
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-scores))   # raw logits -> sigmoid
        print("  (raw logits detected -> sigmoid applied)")

    # assemble per-skill results
    results, k = [], 0
    for rec in records:
        i, cands = rec.pop("row"), []
        for j in rec.pop("cand_idx"):
            cands.append({"clo": clo_texts[j], "course": clo_owner[j],
                          "cosine": round(float(sim[i, j]), 2),
                          "rerank": round(float(scores[k]), 2)})
            k += 1
        by_rerank = sorted(cands, key=lambda c: -c["rerank"])
        rec["candidates"] = by_rerank
        rec["cosine_best"] = cands[0]           # argsort kept cosine order
        rec["rerank_best"] = by_rerank[0]
        rec["agree"] = by_rerank[0] is cands[0]
        rec["verdict"] = verdict(by_rerank[0]["rerank"])
        results.append(rec)

    print(f"\n{'skill':<24} {'jobs':>7}  "
          f"{'cosine-best CLO [course]':<52} {'cos':>5} {'rerank':>6}  verdict")
    for r in results:
        cb, rb = r["cosine_best"], r["rerank_best"]
        print(f"{clip(r['skill'], 24):<24} {r['jobs']:>7,}  "
              f"{clip(cb['clo'] + ' [' + cb['course'] + ']', 52):<52} "
              f"{cb['cosine']:>5.2f} {cb['rerank']:>6.2f}  {r['verdict']}")
        if not r["agree"]:
            print(f"{'':<33} ↳ reranker prefers: "
                  f"{clip(rb['clo'] + ' [' + rb['course'] + ']', 52)} "
                  f"({rb['rerank']:.2f})")

    n = len(results)
    agree = sum(r["agree"] for r in results)
    tally = {v: sum(r["verdict"] == v for r in results)
             for v in ("SUPPORTED", "WEAK", "NOT SUPPORTED")}
    demoted = [r["skill"] for r in results
               if r["cosine_best"]["rerank"] < WEAK_AT]
    print(f"\n=== SUMMARY (verdict bands PROVISIONAL: "
          f"supported >= {SUPPORTED_AT}, weak >= {WEAK_AT}) ===")
    print(f"reranker agrees with cosine-best CLO : {agree}/{n}")
    print(f"verdicts on rerank-best CLO          : "
          f"{tally['SUPPORTED']} supported, {tally['WEAK']} weak, "
          f"{tally['NOT SUPPORTED']} not supported")
    print(f"cosine-best CLOs exposed as noise    : "
          f"{', '.join(demoted) if demoted else 'none'}")

    OUT.write_text(json.dumps({
        "bi_encoder": MODEL_NAME, "reranker": RERANKER,
        "top_n": TOP_N, "top_k": TOP_K,
        "bands": {"supported": SUPPORTED_AT, "weak": WEAK_AT,
                  "note": "PROVISIONAL until task #1 annotation"},
        "results": results,
    }, indent=1))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
