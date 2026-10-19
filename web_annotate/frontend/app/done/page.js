"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken, clearToken } from "@/lib/api";

export default function DonePage() {
  const router = useRouter();
  const [state, setState] = useState(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/");
      return;
    }
    api("/api/tasks").then((d) =>
      setState({
        name: d.rater.name,
        done: Object.keys(d.answers).length,
        total: d.total,
      }),
    );
  }, [router]);

  if (!state) {
    return (
      <main className="center">
        <span className="muted">…</span>
      </main>
    );
  }

  const complete = state.done >= state.total;

  return (
    <main className="center">
      <div className="stack" style={{ width: "min(30rem, 100%)" }}>
        <div className="stack-sm">
          <span className="eyebrow">
            {complete ? "Complete" : "Progress saved"}
          </span>
          <h1>
            {complete
              ? "Thank you — all 250 pairs rated."
              : `${state.done} of ${state.total} pairs rated.`}
          </h1>
          <p className="muted">
            {complete
              ? "Your ratings will help measure how well ITC courses match the skills asked for in Cambodian IT jobs. Thank you for your time."
              : "Everything you answered is saved. Come back whenever you have time and log in with the same name — you will continue exactly where you stopped."}
          </p>
        </div>

        <div className="row">
          {!complete && (
            <button
              className="btn btn-primary"
              onClick={() => router.push("/rate")}
            >
              Continue rating
            </button>
          )}
          <button
            className="btn"
            onClick={() => {
              clearToken();
              router.push("/");
            }}
          >
            Sign out
          </button>
        </div>

        <p className="fine">
          For questions about the study, contact the study organiser.
        </p>
      </div>
    </main>
  );
}
