"use client";

// Researcher-only progress dashboard. Reached by typing the admin code
// (ADMIN_CODE in backend/.env) into the normal login form; the code is kept
// in sessionStorage and passed as ?key= to the guarded /api/admin routes.
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8787";
const TOTAL = 250;
const STORE = "itc_annotate_admin";

const when = (iso) =>
  new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));

export default function AdminPage() {
  const router = useRouter();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [updated, setUpdated] = useState(null);
  const [key, setKey] = useState(null);

  const load = useCallback(async () => {
    const k = sessionStorage.getItem(STORE);
    if (!k) {
      router.replace("/");
      return;
    }
    setKey(k);
    try {
      const res = await fetch(
        `${BASE}/api/admin/summary?key=${encodeURIComponent(k)}`,
      );
      if (!res.ok) {
        throw new Error(
          res.status === 403
            ? "That admin code was not accepted."
            : `request failed (${res.status})`,
        );
      }
      setData(await res.json());
      setUpdated(new Date());
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }, [router]);

  // refresh on a timer — this page is left open while raters work
  useEffect(() => {
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, [load]);

  if (!data && !error) {
    return (
      <main className="center">
        <span className="muted">Loading progress…</span>
      </main>
    );
  }

  const raters = data?.raters ?? [];
  const cov = data?.coverage ?? {};
  const totalDone = raters.reduce((s, r) => s + r.done, 0);

  return (
    <main className="shell" style={{ paddingTop: "3rem" }}>
      <div className="stack">
        <div className="spread">
          <div className="stack-sm">
            <span className="eyebrow">Study dashboard</span>
            <h1>Rater progress</h1>
          </div>
          <div className="row">
            {key && (
              <a
                className="btn"
                href={`${BASE}/api/admin/export.csv?key=${encodeURIComponent(key)}`}
              >
                Download CSV
              </a>
            )}
            <button type="button" className="btn" onClick={load}>
              Refresh
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                sessionStorage.removeItem(STORE);
                router.push("/");
              }}
            >
              Sign out
            </button>
          </div>
        </div>

        {error && <span className="error">{error}</span>}

        {data && (
          <>
            <div className="card">
              <div className="row" style={{ gap: "2rem" }}>
                <div className="stack-sm">
                  <span className="eyebrow">Raters</span>
                  <span className="term">{raters.length}</span>
                </div>
                <div className="stack-sm">
                  <span className="eyebrow">Ratings collected</span>
                  <span className="term">{totalDone}</span>
                </div>
                <div className="stack-sm">
                  <span className="eyebrow">Pairs rated ≥ once</span>
                  <span className="term">
                    {cov.rated_pairs ?? 0} / {TOTAL}
                  </span>
                </div>
                <div className="stack-sm">
                  <span className="eyebrow">Raters per pair</span>
                  <span className="term">
                    {cov.min_raters ?? 0}–{cov.max_raters ?? 0}
                    <span className="muted"> (avg {cov.avg_raters ?? 0})</span>
                  </span>
                </div>
              </div>
            </div>

            {raters.length === 0 && (
              <p className="muted">No raters have signed in yet.</p>
            )}

            {raters.map((r) => {
              const pct = Math.round((r.done / TOTAL) * 100);
              return (
                <div className="card stack-sm" key={r.name}>
                  <div className="spread">
                    <span className="row" style={{ gap: "0.5rem" }}>
                      <strong>{r.name}</strong>
                      {r.ext_id && <span className="tag">{r.ext_id}</span>}
                      <span className="muted">
                        {[r.role, r.familiarity].filter(Boolean).join(" · ")}
                      </span>
                    </span>
                    <span className="fine">
                      {r.done}/{TOTAL} ({pct}%)
                      {r.med_ms != null &&
                        ` · ~${Math.round(r.med_ms / 1000)}s per pair`}
                      {" · last seen "}
                      {when(r.last_seen)}
                    </span>
                  </div>
                  <div className="bar">
                    <i style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}

            <p className="fine">
              Auto-refreshes every 30 seconds
              {updated && ` · last updated ${when(updated)}`}.
            </p>
          </>
        )}
      </div>
    </main>
  );
}
