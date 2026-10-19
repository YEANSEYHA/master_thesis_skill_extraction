"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken, clearToken } from "@/lib/api";

const SCORES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Read back to the rater in words what the number they just pressed means,
// so a mis-key is obvious before the page moves on. Plain words: many raters
// read English as a second language.
const MEANING = [
  "the course does not teach this skill",
  "the skill is barely mentioned",
  "the skill is barely mentioned",
  "background only — not the skill itself",
  "background only — not the skill itself",
  "teaches the basics behind the skill",
  "teaches part of the skill",
  "teaches part of the skill",
  "teaches most of the skill",
  "teaches most of the skill",
  "teaches the skill fully",
];

// The button ramp is the scale: neutral at 0, full accent at 10. It carries
// the meaning of the number, so a rater sees their own answer at a glance.
function tone(n) {
  const pct = Math.round((n / 10) * 100);
  return {
    "--tone": `color-mix(in oklab, var(--accent) ${pct}%, var(--zero))`,
    // measured crossover: against the light --zero the tone only gets dark
    // enough to need white ink at 9 (dark 3.5:1 vs white 5.0:1 there)
    "--tone-ink": n >= 9 ? "var(--accent-ink)" : "var(--ink)",
  };
}

export default function RatePage() {
  const router = useRouter();
  const [tasks, setTasks] = useState(null);
  const [answers, setAnswers] = useState({});
  const [idx, setIdx] = useState(0);
  const [note, setNote] = useState("");
  const [failed, setFailed] = useState({});
  const [name, setName] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);
  const shownAt = useRef(Date.now());
  const noteRef = useRef(null);

  const task = tasks?.[idx];
  const answer = task ? answers[task.row] : undefined;
  const done = useMemo(() => Object.keys(answers).length, [answers]);

  /* ------------------------------------------------------------- loading */

  useEffect(() => {
    if (!getToken()) {
      router.replace("/");
      return;
    }
    api("/api/tasks").then((d) => {
      if (!d.rater.consent) {
        router.replace("/start");
        return;
      }
      setName(d.rater.name);
      setTasks(d.tasks);
      setAnswers(d.answers);
      const first = d.tasks.findIndex((t) => !(t.row in d.answers));
      setIdx(first === -1 ? d.tasks.length - 1 : first);
    });
  }, [router]);

  useEffect(() => {
    shownAt.current = Date.now();
    setNote(task ? (answers[task.row]?.note ?? "") : "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, tasks]);

  /* --------------------------------------------------------------- save */

  const save = useCallback(async (payload) => {
    try {
      await api("/api/rate", { method: "POST", body: payload });
      setFailed((f) => {
        if (!(payload.row in f)) return f;
        const next = { ...f };
        delete next[payload.row];
        return next;
      });
    } catch {
      setFailed((f) => ({ ...f, [payload.row]: payload }));
    }
  }, []);

  // anything that failed (flaky wifi in a lecture hall) is retried quietly
  useEffect(() => {
    const t = setInterval(() => {
      Object.values(failed).forEach(save);
    }, 10_000);
    return () => clearInterval(t);
  }, [failed, save]);

  const record = useCallback(
    (score) => {
      if (!task) return;
      // `unable` stays in the payload so the API contract is unchanged; the
      // UI no longer offers it, so it is always false from here on.
      const entry = { score, unable: false, note: note.trim() || null };
      setAnswers((a) => ({ ...a, [task.row]: entry }));
      save({
        row: task.row,
        position: task.position,
        ms: Date.now() - shownAt.current,
        ...entry,
      });
      // scoring never moves the page — the rater advances with Next / → / Enter
    },
    [task, note, save],
  );

  // keep an edited note attached to an already-scored pair
  const saveNote = useCallback(() => {
    if (!task) return;
    const prev = answers[task.row];
    if (!prev) return;
    const trimmed = note.trim() || null;
    if ((prev.note ?? null) === trimmed) return;
    const entry = { ...prev, note: trimmed };
    setAnswers((a) => ({ ...a, [task.row]: entry }));
    save({ row: task.row, position: task.position, ms: null, ...entry });
  }, [task, note, answers, save]);

  /* ------------------------------------------------------ session actions */

  function signOut() {
    clearToken();
    router.push("/");
  }

  async function startOver() {
    await api("/api/ratings", { method: "DELETE" });
    setAnswers({});
    setFailed({});
    setNote("");
    setConfirmReset(false);
    setIdx(0);
  }

  /* ----------------------------------------------------------- keyboard */

  useEffect(() => {
    function onKey(e) {
      const typing = ["INPUT", "TEXTAREA"].includes(
        document.activeElement?.tagName,
      );
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key >= "0" && e.key <= "9") {
        record(Number(e.key));
      } else if (e.key === "=" || e.key === "+") {
        record(10);
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        setIdx((i) => Math.min(i + 1, (tasks?.length ?? 1) - 1));
      } else if (e.key === "ArrowLeft") {
        setIdx((i) => Math.max(i - 1, 0));
      } else if (e.key.toLowerCase() === "n") {
        e.preventDefault();
        noteRef.current?.focus();
      } else {
        return;
      }
      e.preventDefault();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [record, tasks]);

  /* ------------------------------------------------------------- render */

  if (!tasks) {
    return (
      <main className="center">
        <span className="muted">Loading your sheet…</span>
      </main>
    );
  }

  const remaining = tasks.length - done;
  const failedCount = Object.keys(failed).length;

  return (
    <main className="shell">
      <div className="progress">
        <div className="stack-sm">
          <div className="topbar">
            <span className="fine">
              Signed in as <strong>{name}</strong>
            </span>
            {confirmReset ? (
              <span className="row" style={{ gap: "0.5rem" }}>
                <span className="fine">
                  Delete all {done} of your answers and start again?
                </span>
                <button type="button" className="linkish danger" onClick={startOver}>
                  Yes, clear them
                </button>
                <button
                  type="button"
                  className="linkish"
                  onClick={() => setConfirmReset(false)}
                >
                  Cancel
                </button>
              </span>
            ) : (
              <span className="row" style={{ gap: "1rem" }}>
                <button
                  type="button"
                  className="linkish"
                  disabled={done === 0}
                  onClick={() => setConfirmReset(true)}
                >
                  Start over
                </button>
                <button type="button" className="linkish" onClick={signOut}>
                  Sign out
                </button>
              </span>
            )}
          </div>

          <div className="spread">
            <span className="eyebrow">
              Pair {task.position} of {tasks.length}
            </span>
            <span className="fine">
              {done} rated · {remaining} left
              {failedCount > 0 && (
                <span className="error"> · {failedCount} not saved yet</span>
              )}
            </span>
          </div>
          <div className="bar">
            <i style={{ width: `${(done / tasks.length) * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="stack">
        <div className="pair">
          <div>
            <span className="eyebrow">The skill — job ads ask for</span>
            <span className="term">{task.skill}</span>
            {task.glossary && <span className="gloss">{task.glossary}</span>}
            <span className="tag">from Cambodian IT job ads</span>
          </div>
          <div>
            <span className="eyebrow">The course topic — ITC teaches</span>
            <span className="term">{task.clo}</span>
            <span className="gloss">{task.course}</span>
            <span className="tag">Year {task.cohort.slice(1)}</span>
          </div>
        </div>

        <p className="ask">
          Does this course topic teach this skill? Pick a number.
        </p>

        <div className="stack-sm">
          <div className="scale">
            {SCORES.map((n) => (
              <button
                key={n}
                type="button"
                style={tone(n)}
                aria-pressed={answer?.score === n}
                aria-label={`score ${n}`}
                onClick={() => record(n)}
              >
                {n}
              </button>
            ))}
          </div>
          <div className="legend">
            <span>0 — no, not taught</span>
            <span>5 — the basics only</span>
            <span>10 — yes, fully taught</span>
          </div>
        </div>

        <div
          className={`answer${answer ? " answer-set" : ""}`}
          aria-live="polite"
        >
          {answer ? (
            <>
              <span className="answer-badge" style={tone(answer.score)}>
                {answer.score}
              </span>
              <span>
                You chose <strong>{answer.score} out of 10</strong> —{" "}
                {MEANING[answer.score]}
              </span>
              <span className="saved">Saved ✓</span>
            </>
          ) : (
            <span className="muted">
              Click a number: 0 = not taught, 10 = fully taught.
            </span>
          )}
        </div>

        <div className="field">
          <label htmlFor="note">
            Note <span className="fine">optional — why, or what is missing</span>
          </label>
          <textarea
            id="note"
            ref={noteRef}
            rows={2}
            value={note}
            placeholder="e.g. taught only as theory, no practical work"
            onChange={(e) => setNote(e.target.value)}
            onBlur={saveNote}
          />
        </div>

        <div className="spread">
          <button
            type="button"
            className="btn"
            disabled={idx === 0}
            onClick={() => {
              saveNote();
              setIdx((i) => Math.max(i - 1, 0));
            }}
          >
            ← Previous
          </button>

          {idx === tasks.length - 1 ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                saveNote();
                router.push("/done");
              }}
            >
              Finish
            </button>
          ) : (
            <button
              type="button"
              /* once this pair is answered, Next is the obvious next move */
              className={`btn${answer ? " btn-primary" : ""}`}
              onClick={() => {
                saveNote();
                setIdx((i) => Math.min(i + 1, tasks.length - 1));
              }}
            >
              Next →
            </button>
          )}
        </div>

        <p className="fine" style={{ textAlign: "center" }}>
          Keys <kbd>0</kbd>–<kbd>9</kbd> score, <kbd>=</kbd> scores 10,{" "}
          <kbd>N</kbd> note, <kbd>←</kbd> <kbd>→</kbd> move. Your answers save as
          you go — you can close this and come back. A pair you cannot judge
          may be left unscored — just press Next.
        </p>

        {remaining === 0 && (
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => router.push("/done")}
          >
            All 250 rated — finish
          </button>
        )}
      </div>
    </main>
  );
}
