// Pull the collected ratings out of Neon for analysis.
//
//   npm run export                 -> ../../backend/phase5_curriculum_alignment/05_evaluation/annotation/
//                                     human_ratings_250.csv + .json
//
// Writes the long format (one row per rater x pair) still blinded; the
// analysis script joins it back to sheet_250_master.json on `row` to bring
// the cosine in. That join is where the cosine and the human score meet for
// the first time — keeping it out of the collection app is the whole point.
import "dotenv/config";
import { writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { q, pool } from "../src/db.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR =
  process.env.EXPORT_DIR ||
  join(HERE, "..", "..", "..", "backend", "phase5_curriculum_alignment", "05_evaluation", "annotation");

const { rows } = await q(
  `select r.name rater, r.ext_id, r.role, r.familiarity,
          t.row, p.skill, p.clo, p.course, p.cohort, p.anchor,
          t.score, t.unable, t.note, t.ms, t.position, t.updated_at
     from annot_ratings t
     join annot_raters r on r.id = t.rater_id
     join annot_pairs  p on p.row = t.row
    order by r.name, t.position`,
);

const meta = await q("select value from annot_meta where key = 'sheet'");
const raters = await q(
  `select r.name, r.ext_id, r.role, r.familiarity, r.consent, r.created_at,
          r.last_seen, count(t.id)::int done
     from annot_raters r left join annot_ratings t on t.rater_id = r.id
    group by r.id order by r.name`,
);

const cols = [
  "rater", "ext_id", "role", "familiarity", "row", "skill", "clo", "course",
  "cohort", "anchor", "score", "unable", "note", "ms", "position", "updated_at",
];
const esc = (v) => {
  if (v === null || v === undefined) return "";
  const s = v instanceof Date ? v.toISOString() : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

await mkdir(OUT_DIR, { recursive: true });
await writeFile(
  join(OUT_DIR, "human_ratings_250.csv"),
  [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n"),
);
await writeFile(
  join(OUT_DIR, "human_ratings_250.json"),
  JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      sheet: meta.rows[0]?.value ?? null,
      raters: raters.rows,
      ratings: rows,
    },
    null,
    1,
  ),
);

const complete = raters.rows.filter((r) => r.done >= 250).length;
console.log(
  `exported ${rows.length} ratings from ${raters.rowCount} raters ` +
    `(${complete} complete) -> ${OUT_DIR}`,
);
await pool.end();
