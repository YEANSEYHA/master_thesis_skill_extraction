job_markets=[
  "tensorflow",
  "pytorch",
  "deep learning",
  "machine learning",
  "python",
  "model deployment",
  "computer vision",
  "model evaluation",
  "data preprocessing",
  "model training"
]


skills_from_currirulums=[
  {
    "competency": "Supervised Learning",
    "evidence": ["Supervised Learning", "Linear Regression", "Logistic Regression",
                 "Decision Trees", "K-Nearest Neighbor", "Gradient Boosting",
                 "Gradient Descent", "Algorithm Selection for Datasets", "Algorithm Evaluation"]
  },
  {
    "competency": "Unsupervised Learning",
    "evidence": ["Unsupervised Learning", "Clustering", "K-Means", "Gaussian Mixture"]
  },
  {
    "competency": "Python Programming Tools",
    "evidence": ["python", "pandas", "jupyter"]
  },
  {
    "competency": "Data Exploration and Analysis (EDA)",
    "evidence": ["Data Exploration and Analysis (EDA)", "Data Preparation"]
  },
  {
    "competency": "AI Application Development",
    "evidence": ["AI Application Development", "AI Sub-Fields"]
  },
  {
    "competency": "Model Evaluation and Validation",
    "evidence": ["Cross-Validation"]
  }
]


import numpy as np

# candidate models — not yet decided; pass any to compare(model_name=, backend=)
#   ollama:  "dengcao/Qwen3-Embedding-0.6B:Q8_0"      (current)
#   st:      "sentence-transformers/all-MiniLM-L6-v2"
#   st:      "TechWolf/JobBERT-v2"
#   st:      "BAAI/bge-large-en-v1.5"
#   st:      "sentence-transformers/all-mpnet-base-v2"


def _encoder(model_name, backend, trust_remote_code):
    """Return enc(texts) -> L2-normalized embeddings, for the chosen backend."""
    if backend == "ollama":
        import ollama

        def enc(xs):
            v = np.asarray(ollama.embed(model=model_name, input=list(xs)).embeddings,
                           dtype=np.float64)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return enc

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    return lambda xs: np.asarray(model.encode(xs, normalize_embeddings=True),
                                 dtype=np.float64)


def _parts(competencies):
    """Each competency → its parts (label + evidence) + the owner index, so a
    market skill can match the strongest part instead of a diluted blob."""
    parts, owner, labels = [], [], []
    for j, c in enumerate(competencies):
        labels.append(c["competency"])
        for p in [c["competency"], *c.get("evidence", [])]:
            parts.append(p)
            owner.append(j)
    return parts, owner, labels


def compare(model_name="dengcao/Qwen3-Embedding-0.6B:Q8_0", backend="ollama",
            market=job_markets, competencies=skills_from_currirulums,
            threshold=0.60, trust_remote_code=False):
    """Embed the two inputs with `model_name` and compare.

    For each market skill, returns its best-matching course competency
    (max cosine over that competency's parts: label + evidence).
    Model + backend are parameters — swap them to compare embedders.
    """
    enc = _encoder(model_name, backend, trust_remote_code)

    M = enc(market)                                  # (n_market, d)
    parts, owner, labels = _parts(competencies)
    P = enc(parts)                                   # (n_parts, d)
    simP = M @ P.T                                   # (n_market, n_parts)

    sim = np.full((len(market), len(labels)), -1.0)  # parts → competency (max-pool)
    for k, j in enumerate(owner):
        np.maximum(sim[:, j], simP[:, k], out=sim[:, j])

    rows = []
    for i, skill in enumerate(market):
        j = int(sim[i].argmax())
        score = float(sim[i][j])
        rows.append({"market": skill, "best_competency": labels[j],
                     "score": round(score, 3), "covered": score >= threshold})
    return rows


def retrieve(model_name="dengcao/Qwen3-Embedding-0.6B:Q8_0", backend="ollama",
             market=job_markets, competencies=skills_from_currirulums,
             top_n=2, floor=0.55, trust_remote_code=False):
    """STAGE A — RETRIEVE (embedding, high recall).

    For each market skill, shortlist the top-N course competencies by max-pooled
    cosine (≥ floor). This does NOT decide coverage — it only proposes candidates
    for the verifier. Keep `floor` low: recall matters here, precision comes next.
    """
    enc = _encoder(model_name, backend, trust_remote_code)
    M = enc(market)
    parts, owner, labels = _parts(competencies)
    P = enc(parts)
    simP = M @ P.T

    sim = np.full((len(market), len(labels)), -1.0)        # parts → competency (max)
    for k, j in enumerate(owner):
        np.maximum(sim[:, j], simP[:, k], out=sim[:, j])

    shortlist = []
    for i, skill in enumerate(market):
        order = np.argsort(-sim[i])
        cands = [{"competency": labels[j], "score": round(float(sim[i][j]), 3)}
                 for j in order if sim[i][j] >= floor][:top_n]
        shortlist.append({"market": skill, "candidates": cands})
    return shortlist


def verify(shortlist, competencies=skills_from_currirulums, model="qwen3:14b"):
    """STAGE B — VERIFY (LLM yes/no, high precision).

    For each shortlisted (market skill, competency) pair, ask the LLM whether the
    competency genuinely teaches the skill. A skill is COVERED if any candidate is
    a yes; else GAP. Reading both sides jointly fixes the embedding's false
    positives (e.g. tensorflow ↔ Python Tools → no).
    """
    
    import ollama
    ev = {c["competency"]: c.get("evidence", []) for c in competencies}
    sys_p = ("You judge whether a university course COMPETENCY actually teaches a "
             "labour-market SKILL. Answer ONLY 'yes' or 'no'. Say yes only if the "
             "competency genuinely covers that skill — not merely related.")

    def ask(skill, comp):
        u = (f"Market skill: {skill}\n"
             f"Course competency: {comp}\n"
             f"Topics taught under it: {', '.join(ev.get(comp, [])) or comp}\n"
             f"Does this competency teach '{skill}'? yes or no.")
        r = ollama.chat(model=model, think=False,
                        options={"temperature": 0, "num_predict": 3},
                        messages=[{"role": "system", "content": sys_p},
                                  {"role": "user", "content": u}])
        return r.message.content.strip().lower().startswith("y")

    rows = []
    for item in shortlist:
        skill, matched = item["market"], None
        for cand in item["candidates"]:
            if ask(skill, cand["competency"]):
                matched = cand["competency"]
                break
        rows.append({"market": skill, "covered": matched is not None,
                     "matched": matched,
                     "candidates": [c["competency"] for c in item["candidates"]]})
    return rows


if __name__ == "__main__":
    MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"   # Stage A embedder
    BACKEND = "ollama"                            # "ollama" | "st"

    print(f"STAGE A — retrieve  (embedder: {MODEL})\n")
    shortlist = retrieve(MODEL, backend=BACKEND, top_n=2, floor=0.55)
    for s in shortlist:
        cs = ", ".join(f"{c['competency']} {c['score']:.2f}" for c in s["candidates"])
        print(f"  {s['market']:20s} → {cs or '(none)'}")

    print(f"\nSTAGE B — verify  (LLM: qwen3:14b)\n")
    rows = verify(shortlist)
    for r in rows:
        tick = "✓" if r["covered"] else "✗"
        print(f"  {tick} {r['market']:20s} ← {r['matched'] or '—'}")
    hits = sum(r["covered"] for r in rows)
    total = len(rows)
    print(f"\n  total hits: {hits}/{total}  ({100 * hits / total:.1f}%)")
