// ITC curriculum-alignment human reference study — API.
//
// One job: collect 0-10 human ratings for the 250 frozen skill x course
// pairs in data/pairs.json, from ~20 raters, without ever revealing the
// system's own score or decision.
import "dotenv/config";
import express from "express";
import cors from "cors";
import crypto from "node:crypto";

import { q, initSchema, pool } from "./db.js";
import { sign, requireRater } from "./auth.js";

const PORT = process.env.PORT || 8787;
const ACCESS_CODE = (process.env.ACCESS_CODE || "").trim();
if (!ACCESS_CODE) throw new Error("Set ACCESS_CODE before starting the rating API");
const ADMIN_KEY = process.env.ADMIN_KEY || "";
// typed into the normal login form instead of the rater access code; lands
// on the /admin progress dashboard rather than the rating flow
const ADMIN_CODE = (process.env.ADMIN_CODE || "").trim();
const TOTAL = 250;

const app = express();
app.use(express.json({ limit: "256kb" }));
app.use(
  cors({
    origin: process.env.CORS_ORIGIN
      ? process.env.CORS_ORIGIN.split(",").map((s) => s.trim())
      : true,
  }),
);

const nameKey = (s) => s.trim().toLowerCase().replace(/\s+/g, " ");

// Anchor block first (the 30 shared pairs everyone rates, in sheet order),
// then the remaining 220 shuffled per rater. Randomising the main pool is
// what makes a partially finished sheet an unbiased random subset rather
// than "the first N rows of the sheet".
function raterOrder(anchorRows, otherRows) {
  const rest = [...otherRows];
  for (let i = rest.length - 1; i > 0; i--) {
    const j = crypto.randomInt(i + 1);
    [rest[i], rest[j]] = [rest[j], rest[i]];
  }
  return [...anchorRows, ...rest];
}

/* ---------------------------------------------------------------- health */

app.get("/api/health", async (_req, res) => {
  try {
    const { rows } = await q("select count(*)::int n from annot_pairs");
    res.json({ ok: true, pairs: rows[0].n });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/* ----------------------------------------------------------------- login */

app.post("/api/login", async (req, res) => {
  const code = String(req.body?.code || "").trim();
  const name = String(req.body?.name || "").trim().slice(0, 60);
  const extId = String(req.body?.extId || "").trim().slice(0, 40) || null;

  // the admin code needs no name — it identifies the researcher, not a rater
  if (ADMIN_CODE && code.toUpperCase() === ADMIN_CODE.toUpperCase()) {
    return res.json({ admin: true });
  }

  if (code.toUpperCase() !== ACCESS_CODE.toUpperCase()) {
    return res.status(403).json({ error: "That access code is not valid." });
  }
  if (name.length < 2) {
    return res.status(400).json({ error: "Please enter your name." });
  }

  const key = nameKey(name);
  const existing = await q("select * from annot_raters where name_key = $1", [key]);

  let rater;
  if (existing.rowCount) {
    rater = existing.rows[0];
    // a freshly typed ID wins; logging in without one keeps what we have
    await q(
      "update annot_raters set last_seen = now(), ext_id = coalesce($2, ext_id) where id = $1",
      [rater.id, extId],
    );
  } else {
    const anchors = await q(
      "select row from annot_pairs where anchor order by row",
    );
    const others = await q(
      "select row from annot_pairs where not anchor order by row",
    );
    const order = raterOrder(
      anchors.rows.map((r) => r.row),
      others.rows.map((r) => r.row),
    );
    const ins = await q(
      `insert into annot_raters (id, name, name_key, ext_id, order_rows, user_agent)
       values ($1, $2, $3, $4, $5, $6) returning *`,
      [
        crypto.randomUUID(),
        name,
        key,
        extId,
        JSON.stringify(order),
        (req.get("user-agent") || "").slice(0, 300),
      ],
    );
    rater = ins.rows[0];
  }

  const done = await q(
    "select count(*)::int n from annot_ratings where rater_id = $1",
    [rater.id],
  );

  res.json({
    token: sign(rater.id),
    rater: {
      name: rater.name,
      role: rater.role,
      familiarity: rater.familiarity,
      consent: rater.consent,
      done: done.rows[0].n,
      total: TOTAL,
      returning: existing.rowCount > 0,
    },
  });
});

/* --------------------------------------------------------------- profile */

app.post("/api/profile", requireRater, async (req, res) => {
  const role = String(req.body?.role || "").slice(0, 40) || null;
  const familiarity = String(req.body?.familiarity || "").slice(0, 40) || null;
  const consent = Boolean(req.body?.consent);
  await q(
    `update annot_raters
        set role = $2, familiarity = $3, consent = $4, last_seen = now()
      where id = $1`,
    [req.raterId, role, familiarity, consent],
  );
  res.json({ ok: true });
});

/* ----------------------------------------------------------------- tasks */

// The whole ordered sheet plus this rater's existing answers, in one call.
// 250 blinded pairs is a small payload and it makes every subsequent screen
// instant — which matters a lot across a 250-item sitting.
app.get("/api/tasks", requireRater, async (req, res) => {
  const rater = await q("select * from annot_raters where id = $1", [req.raterId]);
  if (!rater.rowCount) return res.status(401).json({ error: "unknown rater" });

  const order = rater.rows[0].order_rows;
  const pairs = await q("select * from annot_pairs");
  const byRow = new Map(pairs.rows.map((p) => [p.row, p]));

  const ratings = await q(
    "select row, score, unable, note from annot_ratings where rater_id = $1",
    [req.raterId],
  );
  const answers = Object.fromEntries(
    ratings.rows.map((r) => [
      r.row,
      { score: r.score, unable: r.unable, note: r.note },
    ]),
  );

  res.json({
    rater: {
      name: rater.rows[0].name,
      role: rater.rows[0].role,
      familiarity: rater.rows[0].familiarity,
      consent: rater.rows[0].consent,
    },
    total: order.length,
    anchorCount: pairs.rows.filter((p) => p.anchor).length,
    // deliberately blinded: no cosine, no band, no stratum, no system label —
    // and no job-ad frequency, which is a demand signal that could sway the
    // rating (it stays in the DB and the master sheet for analysis)
    tasks: order.map((row, i) => {
      const p = byRow.get(row);
      return {
        position: i + 1,
        row,
        skill: p.skill,
        course: p.course,
        clo: p.clo,
        cohort: p.cohort,
        glossary: p.glossary,
      };
    }),
    answers,
  });
});

/* ---------------------------------------------------------------- rating */

app.post("/api/rate", requireRater, async (req, res) => {
  const row = Number(req.body?.row);
  const unable = Boolean(req.body?.unable);
  const score = unable ? null : Number(req.body?.score);
  const note = (req.body?.note ? String(req.body.note) : "").slice(0, 500) || null;
  const ms = Number.isFinite(Number(req.body?.ms)) ? Number(req.body.ms) : null;
  const position = Number.isFinite(Number(req.body?.position))
    ? Number(req.body.position)
    : null;

  if (!Number.isInteger(row)) {
    return res.status(400).json({ error: "row must be an integer" });
  }
  if (!unable && (!Number.isInteger(score) || score < 0 || score > 10)) {
    return res.status(400).json({ error: "score must be an integer 0-10" });
  }

  await q(
    `insert into annot_ratings (rater_id, row, score, unable, note, ms, position)
     values ($1, $2, $3, $4, $5, $6, $7)
     on conflict (rater_id, row) do update
       set score = excluded.score,
           unable = excluded.unable,
           note = excluded.note,
           ms = excluded.ms,
           position = excluded.position,
           updated_at = now()`,
    [req.raterId, row, score, unable, note, ms, position],
  );
  await q("update annot_raters set last_seen = now() where id = $1", [req.raterId]);

  const done = await q(
    "select count(*)::int n from annot_ratings where rater_id = $1",
    [req.raterId],
  );
  res.json({ ok: true, done: done.rows[0].n, total: TOTAL });
});

// "Start over": a rater who wants a clean sheet drops their own answers only.
// Their stored pair order is kept, so the sheet they restart is the same
// instrument they were given — only the answers go.
app.delete("/api/ratings", requireRater, async (req, res) => {
  const gone = await q("delete from annot_ratings where rater_id = $1", [
    req.raterId,
  ]);
  await q("update annot_raters set last_seen = now() where id = $1", [req.raterId]);
  res.json({ ok: true, cleared: gone.rowCount });
});

/* ----------------------------------------------------------------- admin */

function requireAdmin(req, res, next) {
  const key = String(req.query.key || req.get("x-admin-key") || "");
  const ok =
    (ADMIN_KEY && key === ADMIN_KEY) ||
    (ADMIN_CODE && key.toUpperCase() === ADMIN_CODE.toUpperCase());
  if (!ok) {
    return res.status(403).json({ error: "forbidden" });
  }
  next();
}

app.get("/api/admin/summary", requireAdmin, async (_req, res) => {
  const raters = await q(
    `select r.name, r.ext_id, r.role, r.familiarity, r.created_at, r.last_seen,
            count(t.id)::int  done,
            count(t.id) filter (where t.unable)::int unable,
            round(avg(t.ms) filter (where t.ms is not null))::int med_ms
       from annot_raters r
       left join annot_ratings t on t.rater_id = r.id
      group by r.id
      order by done desc`,
  );
  const coverage = await q(
    `select count(*)::int rated_pairs,
            min(n)::int min_raters, max(n)::int max_raters,
            round(avg(n), 2) avg_raters
       from (select row, count(*)::int n from annot_ratings group by row) s`,
  );
  res.json({ raters: raters.rows, coverage: coverage.rows[0] });
});

// Long-format CSV, one row per (rater, pair) — ready for the agreement and
// threshold-calibration analysis. Still blinded: join to sheet_250_master.json
// on `row` to bring the cosine back in at analysis time.
app.get("/api/admin/export.csv", requireAdmin, async (_req, res) => {
  const { rows } = await q(
    `select r.name rater, r.ext_id, r.role, r.familiarity,
            t.row, p.skill, p.clo, p.course, p.cohort, p.anchor,
            t.score, t.unable, t.note, t.ms, t.position, t.updated_at
       from annot_ratings t
       join annot_raters r on r.id = t.rater_id
       join annot_pairs  p on p.row = t.row
      order by r.name, t.position`,
  );
  const cols = [
    "rater", "ext_id", "role", "familiarity", "row", "skill", "clo", "course",
    "cohort", "anchor", "score", "unable", "note", "ms", "position",
    "updated_at",
  ];
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = v instanceof Date ? v.toISOString() : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [
    cols.join(","),
    ...rows.map((r) => cols.map((c) => esc(r[c])).join(",")),
  ].join("\n");
  res.type("text/csv").attachment("annot_ratings.csv").send(csv);
});

/* ------------------------------------------------------------------ boot */

const server = await (async () => {
  await initSchema();
  const { rows } = await q("select count(*)::int n from annot_pairs");
  if (rows[0].n === 0) {
    console.warn("annot_pairs is empty — run `npm run seed` before raters arrive");
  }
  return app.listen(PORT, () =>
    console.log(`annotate API on :${PORT}  (${rows[0].n} pairs loaded)`),
  );
})();

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => server.close(() => pool.end()));
}

export default app;
