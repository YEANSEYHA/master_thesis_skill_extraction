"""
EMBEDDING PLAYGROUND — one step at a time.

Step 1: a phrase becomes a vector (an "embedding").
Run:  python embedder_playground.py
"""

from embedder import Embedder   # baseline's own Qwen3 embedder


# ── STEP 1 — text → vector ────────────────────────────────────────────────────
def step1():
    e = Embedder()
    v = e.embed(["machine learning"])     # give it a list of phrases
    print("phrase : 'machine learning'")
    print("shape  :", v.shape)            # (1, 1024) → 1 phrase, 1024 numbers
    print("first5 :", v[0][:5])           # the first 5 of the 1024 numbers


# ── YOUR STEP 1 EXERCISE — write your code below ──────────────────────────────
def my_turn():
    # GOAL: prove you understand "text → vector".
    #
    #  1) make an Embedder
    #  2) embed a LIST of TWO phrases of your choice (any 2 skills)
    #  3) print the shape       → guess it first: what is it for 2 phrases?
    #  4) print the L2 norm (length) of the FIRST vector
    #       hint: import numpy as np   →   np.linalg.norm(<vector>)
    #
    # write your code here:

    e = Embedder()
    v = e.embed(["Automata","Deep Learning"])
    print(v.shape)


# ── STEP 2 — cosine similarity (compare two vectors) ──────────────────────────
def step2():
    e = Embedder()
    a, b = "machine learning", "deep learning"
    va, vb = e.embed([a, b])              # two phrases → two vectors
    score = float(va @ vb)                # @ = dot product = cosine (unit vectors)
    print(f"'{a}' vs '{b}'  ->  cosine = {round(score, 3)}   (high = similar)")


# ── YOUR STEP 2 EXERCISE — write your code below ──────────────────────────────
def my_turn2():
    # GOAL: prove you understand "cosine = how similar two meanings are".
    #
    #  1) make an Embedder

    #  2) embed TWO phrases at once → va, vb = e.embed([a, b])
    #  3) print the cosine:  float(va @ vb)
    #  4) try THREE pairs and PREDICT high/med/low before you run:
    #       - a synonym pair   (expect HIGH)   e.g. "K-Means" vs "k-means clustering"
    #       - a related pair   (expect MED)    e.g. "Decision Trees" vs "machine learning"
    #       - an unrelated pair(expect LOW)    e.g. "Turing Machines" vs "sales"
    #
    # write your code here:
    e = Embedder()
    a, b = "Mobile Developer", "mobile dev"
    va, vb = e.embed([a,b])
    print("Va:",va)
    print("Vb",vb)

    score = float(va @ vb)                # @ = dot product = cosine (unit vectors)
    print(f"'{a}' vs '{b}'  ->  cosine = {round(score, 3)}   (high = similar)")

    

    pass   # ← delete this when done


# ── STEP 3 — one course skill vs MANY job skills → nearest ────────────────────
def step3():
    e = Embedder()
    course = "Decision Trees"
    jobs   = ["machine learning", "deep learning", "sql", "communication", "data mining"]
    tau    = 0.70

    cv = e.embed([course])[0]      # course skill → 1 vector  (1024,)
    jv = e.embed(jobs)             # job skills   → N vectors (N, 1024)
    sims = jv @ cv                 # cosine of course vs EACH job → (N,)

    for j, s in sorted(zip(jobs, sims), key=lambda x: -x[1]):   # high → low
        print(f"   {float(s):.3f}  {j}")
    best = jobs[int(sims.argmax())]          # the single nearest job skill
    top  = float(sims.max())
    print(f"   => nearest '{best}' ({top:.3f})  {'✅ in demand' if top>=tau else '❌ not sought'}")


# ── YOUR STEP 3 EXERCISE — write your code below ──────────────────────────────
def my_turn3():
    e = Embedder()
    course = "HTML"
    jobs = ["css","java scripts"]
    tau = 0.65
    cv = e.embed([course])[0]
    jv = e.embed(jobs)
    sims = jv @ cv

    for j, s in sorted(zip(jobs, sims), key=lambda x: -x[1]):   # high → low
        print(f"   {float(s):.3f}  {j}")
    best = jobs[int(sims.argmax())]          # the single nearest job skill
    top  = float(sims.max())
    print(f"   => nearest '{best}' ({top:.3f})  {'✅ in demand' if top>=tau else '❌ not sought'}")
    # GOAL: find the NEAREST job skill for one course skill (argmax + threshold).
    #
    #  1) pick a course skill + a LIST of 4-5 job skills (mix relevant + noise)
    #  2) embed course → cv (1 vector);  embed jobs → jv (N vectors)
    #  3) sims = jv @ cv        # cosine to each job
    #  4) nearest = jobs[ sims.argmax() ]   # the winner
    #  5) PREDICT the winner before running. Does it pass tau=0.70?
    #
    # write your code here:

    pass   # ← delete this when done


# ── STEP 4 — MANY course skills × MANY job skills = the cosine MATRIX ──────────
def step4():
    e = Embedder()
    course = ["K-Means", "Decision Trees", "Neural Networks"]                # C skills
    jobs   = ["machine learning", "deep learning", "sql", "communication"]   # J skills
    tau    = 0.70

    C = e.embed(course)        # (3, 1024)  — 3 course vectors
    J = e.embed(jobs)          # (4, 1024)  — 4 job vectors
    sim = C @ J.T              # (3, 4)  — EVERY course×job cosine at once
    #          ^^^ J.T flips (4,1024)->(1024,4) so the inner 1024 line up

    print("            " + "  ".join(f"{j[:9]:>9}" for j in jobs))
    for i, c in enumerate(course):
        print(f"{c[:12]:<12}" + "  ".join(f"{sim[i, k]:9.2f}" for k in range(len(jobs))))

    print("nearest per course skill (max of each ROW):")
    for i, c in enumerate(course):
        k = int(sim[i].argmax()); s = float(sim[i].max())
        print(f"  {c:<16} -> {jobs[k]:<18} {s:.2f}  {'✅' if s >= tau else '❌'}")


# ── YOUR STEP 4 EXERCISE — write your code below ──────────────────────────────
def my_turn4():

    e = Embedder()
    course = ["Image Classification", "Face Detection", "Neural Networks"]                # C skills
    jobs   = ["software engineer", "deep learning", "sqlite", "api"]   # J skills
    tau    = 0.70

    C = e.embed(course)
    J = e.embed(jobs)
    sim = C @ J.T

    print("            " + "  ".join(f"{j[:9]:>9}" for j in jobs))
    for i, c in enumerate(course):
        print(f"{c[:12]:<12}" + "  ".join(f"{sim[i, k]:9.2f}" for k in range(len(jobs))))

    print("nearest per course skill (max of each ROW):")
    for i, c in enumerate(course):
        k = int(sim[i].argmax()); s = float(sim[i].max())
        print(f"  {c:<16} -> {jobs[k]:<18} {s:.2f}  {'✅' if s >= tau else '❌'}")

    # GOAL: build the matrix for YOUR skills (this IS market-value, small).
    #
    #  1) course = [3 of your course skills]   jobs = [4-5 job skills]
    #  2) C = e.embed(course)        # (3, 1024)
    #     J = e.embed(jobs)          # (5, 1024)
    #  3) sim = C @ J.T              # (3, 5) matrix  ← note the .T (transpose)!
    #  4) for each ROW i: sim[i].argmax() = nearest job for course skill i
    #
    # write your code here:

    pass   # ← delete this when done


# ── STEP 5 — count ≥ τ → % in demand → HIGH/MODERATE/LOW (the summary card) ────
def step5():
    e = Embedder()
    course = ["K-Means", "Decision Trees", "Neural Networks", "Turing Machines"]
    jobs   = ["machine learning", "deep learning", "data mining", "sql", "communication"]
    tau    = 0.70

    C = e.embed(course)
    J = e.embed(jobs)
    sim  = C @ J.T                    # (4, 5) matrix
    best = sim.max(axis=1)            # best job PER course skill → (4,)  (max of each row)
    hits = int((best >= tau).sum())   # how many course skills clear τ
    pct  = round(100 * hits / len(course), 1)
    label = "HIGH" if pct >= 66 else "MODERATE" if pct >= 33 else "LOW"

    print(f"best score per course skill: {[round(float(x),2) for x in best]}")
    print(f"in demand: {hits}/{len(course)} = {pct}%   ->   {label}")


# ── YOUR STEP 5 EXERCISE — write your code below ──────────────────────────────
def my_turn5():
    # GOAL: turn the matrix into the SUMMARY (this is the colored card).
    #
    #  1) reuse your Step-4 course + jobs (add a clearly OFF-topic course skill
    #     so it won't match — to see the % drop)
    #  2) sim  = C @ J.T
    #     best = sim.max(axis=1)         # best per row, in ONE call
    #     hits = (best >= tau).sum()
    #     pct  = 100*hits/len(course)
    #  3) label = HIGH(>=66) / MODERATE(>=33) / LOW
    #
    # write your code here:

    pass   # ← delete this when done


# ── STEP 6 (CAPSTONE) — REAL job vocab → the actual market-value ──────────────
def step6():
    import json, pathlib
    e = Embedder()

    # 1) load REAL IT job skills (+ demand) from the LinkedIn inventory
    here = pathlib.Path(__file__).resolve().parent
    inv_path = next(here.parent.glob("**/linkedin_inventory.json"))
    inv = json.load(open(inv_path))
    it = inv["categories"]["it"]                                   # {skill: demand}
    items = sorted(it.items(), key=lambda kv: kv[1], reverse=True)[:300]  # top-300
    jobs  = [s for s, _ in items]
    freqs = [f for _, f in items]

    # 2) YOUR real course skills + threshold
    course = ["K-Means", "Decision Trees", "Neural Networks",
              "Gradient Boosting", "Logistic Regression"]
    tau = 0.70

    # 3) the matrix (course × real job vocab) — same 5 steps, real data
    C = e.embed(course)            # (5, 1024)
    J = e.embed(jobs)             # (300, 1024)
    sim = C @ J.T                  # (5, 300)
    best_idx = sim.argmax(axis=1)  # nearest job index per course skill
    best     = sim.max(axis=1)     # its score

    # 4) per-skill + summary card
    for i, c in enumerate(course):
        j = int(best_idx[i]); s = float(best[i])
        print(f"  {c:<20} -> {jobs[j]:<22} {s:.2f}  demand={freqs[j]:<6} {'✅' if s>=tau else '❌'}")
    hits = int((best >= tau).sum()); pct = round(100 * hits / len(course), 1)
    label = "HIGH" if pct >= 66 else "MODERATE" if pct >= 33 else "LOW"
    print(f"  => in demand: {hits}/{len(course)} = {pct}%  ->  {label}")


if __name__ == "__main__":
    step1()
    my_turn()
    print()
    step2()
    my_turn2()
    print()
    step3()
    my_turn3()
    print()
    step4()
    my_turn4()
    print()
    step5()
    print()
    step6()
