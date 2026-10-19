// Neon PostgreSQL connection + schema.
//
// Tables are prefixed `annot_` because this study shares the Neon project
// with the Phase 1 discovery tables (job_sites, discovery_log).
import pg from "pg";

const { Pool } = pg;

function connectionString() {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const { PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_NAME } = process.env;
  if (!PG_USER || !PG_HOST || !PG_NAME) {
    throw new Error(
      "No database configured. Set DATABASE_URL (Neon connection string) " +
        "or PG_USER / PG_PASS / PG_HOST / PG_PORT / PG_NAME in .env",
    );
  }
  const port = PG_PORT || 5432;
  const auth = `${encodeURIComponent(PG_USER)}:${encodeURIComponent(PG_PASS || "")}`;
  // no sslmode in the URL — TLS is configured through the `ssl` option below,
  // and passing both makes node-postgres emit a deprecation warning
  return `postgresql://${auth}@${PG_HOST}:${port}/${PG_NAME}`;
}

export const pool = new Pool({
  connectionString: connectionString(),
  ssl: { rejectUnauthorized: false },
  max: 8,
  idleTimeoutMillis: 30_000,
});

export const q = (text, params) => pool.query(text, params);

// One statement per array entry: node-postgres cannot run multi-statement
// strings through the extended (parameterised) protocol.
const SCHEMA = [
  `create table if not exists annot_pairs (
     row       integer primary key,
     skill     text    not null,
     jobs      integer not null,
     course    text    not null,
     clo       text    not null,
     cohort    text    not null,
     anchor    boolean not null default false,
     glossary  text
   )`,

  `create table if not exists annot_raters (
     id          text        primary key,
     name        text        not null,
     name_key    text        not null unique,
     role        text,
     familiarity text,
     consent     boolean     not null default false,
     order_rows  jsonb       not null,
     created_at  timestamptz not null default now(),
     last_seen   timestamptz not null default now(),
     user_agent  text
   )`,

  `create table if not exists annot_ratings (
     id         bigserial   primary key,
     rater_id   text        not null references annot_raters(id) on delete cascade,
     row        integer     not null references annot_pairs(row),
     score      integer     check (score is null or (score >= 0 and score <= 10)),
     unable     boolean     not null default false,
     note       text,
     ms         integer,
     position   integer,
     created_at timestamptz not null default now(),
     updated_at timestamptz not null default now(),
     unique (rater_id, row),
     check (unable or score is not null)
   )`,

  // optional student/staff ID, added 2026-07-28 — hence alter, not create
  `alter table annot_raters add column if not exists ext_id text`,

  `create index if not exists annot_ratings_rater on annot_ratings (rater_id)`,
  `create index if not exists annot_ratings_row on annot_ratings (row)`,

  // provenance of the sheet the ratings belong to — one row, key 'sheet'
  `create table if not exists annot_meta (
     key   text primary key,
     value jsonb not null
   )`,
];

export async function initSchema() {
  for (const stmt of SCHEMA) await q(stmt);
}
