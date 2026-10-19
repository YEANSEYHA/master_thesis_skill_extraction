"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [name, setName] = useState("");
  const [extId, setExtId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // already signed in on this device -> straight back to where they stopped
  useEffect(() => {
    if (!getToken()) return;
    api("/api/tasks")
      .then((d) => router.replace(d.rater.consent ? "/rate" : "/start"))
      .catch(() => {});
  }, [router]);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const data = await api("/api/login", {
        method: "POST",
        body: { code, name, extId },
      });
      if (data.admin) {
        sessionStorage.setItem("itc_annotate_admin", code.trim());
        router.push("/admin");
        return;
      }
      setToken(data.token);
      router.push(data.rater.consent ? "/rate" : "/start");
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <main className="center">
      <div className="stack" style={{ width: "min(24rem, 100%)" }}>
        <div className="stack-sm">
          <span className="eyebrow">Institute of Technology of Cambodia</span>
          <h1>Do ITC courses teach the skills employers ask for?</h1>
          <p className="muted">
            We will show you two things side by side: a <strong>skill from
            real Cambodian IT job ads</strong>, and a <strong>topic taught in
            an ITC course</strong>. You give a score from 0 to 10 — how well
            does that topic teach that skill? There are 250 pairs, about
            25&nbsp;minutes. Answers save as you go, so you can stop and come
            back any time.
          </p>
        </div>

        <form className="card stack" onSubmit={submit}>
          <div className="field">
            <label htmlFor="code">Access code</label>
            <input
              id="code"
              type="text"
              value={code}
              autoCapitalize="characters"
              autoComplete="off"
              placeholder="given to you with the link"
              onChange={(e) => setCode(e.target.value)}
              required
            />
            <span className="fine">
              Enter the access code provided by the study organiser.
            </span>
          </div>

          <div className="field">
            <label htmlFor="name">
              Your name <span className="fine">required</span>
            </label>
            <input
              id="name"
              type="text"
              value={name}
              autoComplete="name"
              placeholder="e.g. Sok Dara"
              onChange={(e) => setName(e.target.value)}
            />
            <span className="fine">
              Used only to keep your progress together and to check agreement
              between raters. Type it the same way each time you come back.
            </span>
          </div>

          <div className="field">
            <label htmlFor="extId">
              ID <span className="fine">optional</span>
            </label>
            <input
              id="extId"
              type="text"
              value={extId}
              autoComplete="off"
              placeholder="student or staff ID, if you have one"
              onChange={(e) => setExtId(e.target.value)}
            />
          </div>

          {error && <span className="error">{error}</span>}

          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? "Checking…" : "Start rating"}
          </button>
        </form>
      </div>
    </main>
  );
}
