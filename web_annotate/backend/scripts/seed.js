// Load the frozen 250-pair sheet into Neon. Idempotent: re-running updates
// the pair text in place and never touches ratings.
//
//   npm run pairs   # regenerate data/pairs.json from the frozen sheet
//   npm run seed
import "dotenv/config";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { q, initSchema, pool } from "../src/db.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PAIRS = join(HERE, "..", "data", "pairs.json");

const sheet = JSON.parse(await readFile(PAIRS, "utf8"));

if (sheet.pairs.length !== 250) {
  throw new Error(`expected 250 pairs, found ${sheet.pairs.length}`);
}
for (const f of ["cos", "band", "set"]) {
  if (f in sheet.pairs[0]) {
    throw new Error(`ABORT: blinded field "${f}" present in pairs.json`);
  }
}

await initSchema();

// If ratings already exist, the sheet must not change underneath them.
const existing = await q("select value from annot_meta where key = 'sheet'");
const rated = await q("select count(*)::int n from annot_ratings");
if (existing.rowCount && rated.rows[0].n > 0) {
  const before = existing.rows[0].value.sheet_hash;
  if (before !== sheet.sheet_hash) {
    console.error(
      `ABORT: ${rated.rows[0].n} ratings already exist against sheet ` +
        `${before.slice(0, 16)}, refusing to swap in ${sheet.sheet_hash.slice(0, 16)}`,
    );
    process.exit(1);
  }
}

for (const p of sheet.pairs) {
  await q(
    `insert into annot_pairs (row, skill, jobs, course, clo, cohort, anchor, glossary)
     values ($1, $2, $3, $4, $5, $6, $7, $8)
     on conflict (row) do update
       set skill = excluded.skill, jobs = excluded.jobs,
           course = excluded.course, clo = excluded.clo,
           cohort = excluded.cohort, anchor = excluded.anchor,
           glossary = excluded.glossary`,
    [p.row, p.skill, p.jobs, p.course, p.clo, p.cohort, p.anchor, p.glossary],
  );
}

await q(
  `insert into annot_meta (key, value) values ('sheet', $1)
   on conflict (key) do update set value = excluded.value`,
  [
    JSON.stringify({
      sheet_hash: sheet.sheet_hash,
      locked_run: sheet.locked_run,
      config_hash: sheet.config_hash,
      input_hash: sheet.input_hash,
      seed: sheet.seed,
      counts: sheet.counts,
      seeded_at: new Date().toISOString(),
    }),
  ],
);

const n = await q("select count(*)::int n, count(*) filter (where anchor)::int a from annot_pairs");
console.log(
  `seeded ${n.rows[0].n} pairs (${n.rows[0].a} anchors) ` +
    `from sheet ${sheet.sheet_hash.slice(0, 16)} / run ${sheet.locked_run}`,
);
await pool.end();
