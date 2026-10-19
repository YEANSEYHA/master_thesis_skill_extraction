# Curriculum-alignment rating study — web app

Collects the **human reference standard** for Phase 5: 0–10 judgements of the
250 frozen skill × course pairs, from ~20 raters (lecturers, students,
industry). Replaces the paper sheet
(`backend/phase5_curriculum_alignment/05_evaluation/annotation/curriculum_annotation_250_score.pdf`)
with something people will actually finish.

```
web_annotate/
├── backend/            Express + node-postgres  →  Neon
│   ├── src/            index.js (API) · db.js (schema) · auth.js (session)
│   ├── scripts/        build_pairs.py · seed.js · export.js
│   └── data/pairs.json the 250 pairs, BLINDED
└── frontend/           Next.js 16 (app router, JS, no CSS framework)
    └── app/            / login · /start briefing · /rate · /done
```

## What a rater sees

Log in with an **access code + their name** (plus an optional student/staff
ID). Then a one-screen briefing — scale anchors, a worked example drawn from
outside computing so it cannot prime any of the 30 skills, role, familiarity,
consent — then 250 screens of one pair each: the market skill on the left, the
course topic on the right, a 0–10 ramp, and an optional note.

Scoring never advances the page on its own. The chosen number is read back in
words ("You chose 7 out of 10 — teaches part of the skill") so a mis-key is
visible before the rater moves on with `Next`. There is no "unable to assess"
button: a rater who cannot judge a pair leaves it unscored and moves on — a
skip, treated as missing data in the analysis (never imputed).

Keys `0`–`9` score, `=` scores 10, `N` jumps to the note, `←`/`→` move.
Every answer is saved the moment it is given; logging back in with the same
name resumes at the first unanswered pair.

Typing `ADMIN_CODE` instead of the rater code in the same login form opens
`/admin` — live per-rater progress and the CSV download — rather than the
rating flow.

## The two design decisions that matter

**Blinding is enforced at the export boundary, not in the UI.**
`scripts/build_pairs.py` strips `cos`, `band` and `set` when it converts
`sheet_250_master.json` into `data/pairs.json`, and `seed.js` refuses to load
a file that still contains them. Nothing the browser can read — payload,
DOM, network tab — carries the system's score or its covered/gap decision.

**Order: 30 anchors, then a per-rater shuffle.**
Every rater rates the same 30 anchor pairs first, in the same order (that
block is deliberately borderline-heavy, and is what inter-rater agreement is
computed on). The remaining 220 are shuffled per rater at signup and stored,
so a rater who stops early still leaves an *unbiased random subset* of the
sheet rather than "the first N rows".

## Running it locally

```bash
# 1. backend
cd web_annotate/backend
cp .env.example .env          # fill in DATABASE_URL (or PG_*), ACCESS_CODE and SESSION_SECRET
npm install
npm run pairs                 # sheet_250_master.json -> data/pairs.json
npm run seed                  # create tables + load the 250 pairs
npm start                     # :8787

# 2. frontend
cd ../frontend
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8787
npm install
npm run dev                   # :3001
```

`npm run seed` is idempotent, and refuses to swap in a different sheet once
ratings exist against the current one.

Generate a session secret with `node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"`
and set a private `ACCESS_CODE` in the backend environment. Give the access
code directly to invited raters; never put it in a `NEXT_PUBLIC_*` variable.
The API requires both values and has no shared credential defaults.

## Deploying

**Frontend → Vercel.** Import the repo, set **Root Directory** to
`web_annotate/frontend`, add `NEXT_PUBLIC_API_URL` = the deployed API origin.

**Backend → any node host** (Render, Railway, Fly). Root `web_annotate/backend`,
build `npm install`, start `npm start`. Set `DATABASE_URL`, `ACCESS_CODE`,
`SESSION_SECRET`, `ADMIN_KEY`, and `CORS_ORIGIN` = the Vercel URL. Run
`npm run seed` once against the production database.

Neon is the same project as the Phase 1 discovery tables; everything here is
prefixed `annot_` (`annot_pairs`, `annot_raters`, `annot_ratings`,
`annot_meta`).

## Watching progress / getting the data out

```bash
curl "$API/api/admin/summary?key=$ADMIN_KEY"      # per-rater progress, coverage
curl "$API/api/admin/export.csv?key=$ADMIN_KEY" -o ratings.csv
npm run export                                    # -> phase5_curriculum_alignment/05_evaluation/annotation/
```

`npm run export` writes `human_ratings_250.csv` + `.json` (long format, one
row per rater × pair) into the annotation folder, **still blinded**. The
analysis script joins them back to `sheet_250_master.json` on `row` — that
join is the first and only place the human score and the cosine meet.

## Where this lands in the thesis

The full analysis plan — reliability statistic, threshold-selection method,
rater-exclusion rules, role weighting, coverage stop rule — is pre-registered
in `backend/phase5_curriculum_alignment/05_evaluation/annotation/PREREGISTRATION.md`, declared
before any participant data was collected. Headline: the threshold fitted
from the continuous 0–10 rater mean (Youden's J, covered ⇔ mean ≥ 5.0) is
adopted whatever value it takes; the provisional 0.73 bar becomes a
sensitivity analysis. The categorical mapping (0–3 No / 4–6 Partly / 7–10
Yes) is only used to compare against the completed 150-pair LLM silver study.

Sheet `78dfd3fe9b637a36…`, locked run `20260727T105055Z-475eb3`,
config `475eb38e…`, input `a7c8b24a…`.
