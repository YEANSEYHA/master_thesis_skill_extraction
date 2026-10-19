"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken } from "@/lib/api";

const ROLES = [
  ["lecturer", "Lecturer / faculty member", "I teach at ITC or another university"],
  ["assistant", "Teaching assistant", "I assist in teaching or lab sessions"],
  ["student", "Student", "Final-year or postgraduate student"],
  ["industry", "Industry practitioner", "I work in the Cambodian IT industry"],
];

const FAMILIARITY = [
  ["teach", "I teach on this programme"],
  ["studied", "I studied on this programme"],
  ["general", "I know it in general terms"],
  ["none", "I am not familiar with it"],
];

// Deliberately drawn from outside computing so that reading the examples
// cannot prime a rater towards any of the 30 skills in the study.
const EXAMPLES = [
  ["Spreadsheet Formulas and Pivot Tables", 9, "teaches the skill directly"],
  ["Introduction to Business Mathematics", 4, "useful foundation, but not this skill"],
  ["History of Southeast Asian Art", 0, "unrelated"],
];

export default function StartPage() {
  const router = useRouter();
  const [role, setRole] = useState(null);
  const [familiarity, setFamiliarity] = useState(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!getToken()) router.replace("/");
  }, [router]);

  async function begin() {
    setBusy(true);
    await api("/api/profile", {
      method: "POST",
      body: { role, familiarity, consent: true },
    });
    router.push("/rate");
  }

  return (
    <main className="shell" style={{ paddingTop: "3rem" }}>
      <div className="stack">
        <div className="stack-sm">
          <span className="eyebrow">Before you begin</span>
          <h1>How to score</h1>
        </div>

        <div className="card stack">
          <p>
            Each screen shows one <strong>skill from Cambodian IT job
            ads</strong> next to one <strong>topic taught in an ITC
            course</strong>. You answer one simple question:{" "}
            <em>does this course topic teach this skill?</em>
          </p>

          <div className="stack-sm">
            <div className="row">
              <span className="tag">0</span>
              <span>
                <strong>No</strong> — the course does not teach this skill.
              </span>
            </div>
            <div className="row">
              <span className="tag">5</span>
              <span>
                <strong>Halfway</strong> — it teaches the basics behind the
                skill, but not the skill itself.
              </span>
            </div>
            <div className="row">
              <span className="tag">10</span>
              <span>
                <strong>Yes</strong> — it fully teaches this exact skill.
              </span>
            </div>
          </div>

          <p className="muted">
            Use any number from 0 to 10. The middle numbers are for
            &ldquo;partly&rdquo; — and most pairs are partly, so do not worry
            about being exact. There is no wrong answer.
          </p>
        </div>

        <div className="card stack">
          <h2>Example</h2>
          <p className="muted">
            Imagine the job ad asks for <strong>&ldquo;bookkeeping in
            Excel&rdquo;</strong>. How would you score these course topics?
          </p>
          <div className="stack-sm">
            {EXAMPLES.map(([topic, score, why]) => (
              <div className="row" key={topic}>
                <span className="tag">{score}</span>
                <span>
                  <strong>{topic}</strong> <span className="muted">— {why}</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="card stack">
          <h2>A few practical things</h2>
          <ul className="muted" style={{ margin: 0, paddingLeft: "1.1rem" }}>
            <li>
              Many pairs have nothing to do with each other. That is normal —
              give a 0 and move on.
            </li>
            <li>
              If you are not sure, just give your best honest guess from the
              words on the screen. Your opinion is exactly what we are asking
              for — there is no correct answer to look up.
            </li>
            <li>
              If you truly cannot judge a pair — a course you know nothing
              about — leave it unscored and press Next. A short note saying why
              helps us.
            </li>
            <li>
              Every answer saves right away. You can close the tab any time,
              and log in again with the same name to continue where you
              stopped.
            </li>
          </ul>
        </div>

        <div className="card stack">
          <h2>About you</h2>
          <p className="muted">
            One question — so we can compare how lecturers, students and
            industry people rate.
          </p>

          <div className="stack-sm">
            {ROLES.map(([id, label, hint]) => (
              <button
                key={id}
                type="button"
                className="choice"
                aria-pressed={role === id}
                onClick={() => setRole(id)}
              >
                <strong>{label}</strong>
                <span className="fine">{hint}</span>
              </button>
            ))}
          </div>

          <div className="stack-sm" style={{ marginTop: "0.5rem" }}>
            <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>
              How familiar are you with the ITC computer-science programme?
            </label>
            <div className="row">
              {FAMILIARITY.map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className="choice"
                  style={{ flex: "1 1 12rem" }}
                  aria-pressed={familiarity === id}
                  onClick={() => setFamiliarity(id)}
                >
                  <strong>{label}</strong>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="card stack">
          <label className="row" style={{ alignItems: "flex-start" }}>
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              style={{ marginTop: "0.35rem" }}
            />
            <span className="muted" style={{ flex: 1 }}>
              I agree that my ratings may be used, anonymised and in aggregate,
              in a Master&rsquo;s thesis and any resulting publication. The
              study records the name and optional ID I entered, my role and
              familiarity answers, my ratings and notes, and how I answer (time
              per pair, browser type). My name and ID are used only so I can
              log back in and continue — they will never appear in the thesis
              or any publication. I may stop at any point.
            </span>
          </label>

          <button
            className="btn btn-primary"
            disabled={!consent || !role || !familiarity || busy}
            onClick={begin}
          >
            {busy ? "Loading…" : "Begin — 250 pairs"}
          </button>
          {(!role || !familiarity) && (
            <span className="fine">
              Choose your role and familiarity to continue.
            </span>
          )}
        </div>
      </div>
    </main>
  );
}
