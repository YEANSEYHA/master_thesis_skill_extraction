"""
Stage 2 Side-by-Side Debugger
Left  = INPUT  : cleaned text that was fed to the LLM
Right = OUTPUT : what the LLM extracted from it

Run:
    cd visualization
    streamlit run app_stage2.py
"""

import json
import os
import sys
import streamlit as st

# Allow importing clean_html_trafilatura from stage2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03_extract"))
from stage2_skill_extraction import clean_html_trafilatura

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stage 2 Debugger",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Stage 2 — Input vs Output Debugger")
st.caption("Left: cleaned text sent to the LLM · Right: what the LLM extracted")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Config")

stage2_json = st.sidebar.text_input(
    "stage2_output.json path",
    value="../03_extract/stage2_output.json",
)
html_folder = st.sidebar.text_input(
    "HTML source folder",
    value="../data_collection/data/100_html_job_detail",
)

# ── Load stage2 output ────────────────────────────────────────────────────────
if not os.path.exists(stage2_json):
    st.sidebar.error(f"File not found:\n{stage2_json}")
    st.stop()

with open(stage2_json, encoding="utf-8") as f:
    raw_data = json.load(f)

# Support both old format (flat list of dicts) and new format ({input, output})
if raw_data and isinstance(raw_data[0], dict) and "output" in raw_data[0]:
    all_jobs = raw_data   # new format — keep as-is, access via job["output"]
    NEW_FORMAT = True
else:
    # old format — wrap to match new structure so the rest of the code is uniform
    all_jobs = [{"input": None, "output": j} for j in raw_data]
    NEW_FORMAT = False

# ── Quality badge helper ──────────────────────────────────────────────────────
FIELDS = ["job_title", "company_name", "salary", "location", "posted_date", "job_category", "skills"]

def quality_score(job: dict) -> int:
    """Count how many key fields are filled."""
    out = job["output"]
    score = 0
    for f in FIELDS:
        v = out.get(f)
        if f == "skills":
            score += 1 if isinstance(v, list) and len(v) > 0 else 0
        else:
            score += 1 if v else 0
    return score

def quality_color(score: int) -> str:
    pct = score / len(FIELDS)
    if pct >= 0.8:
        return "🟢"
    elif pct >= 0.5:
        return "🟡"
    return "🔴"

# ── Sidebar: job list ─────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader(f"Jobs ({len(all_jobs)})")

search = st.sidebar.text_input("Search", placeholder="title, company, skill...")

def matches(job, q):
    q = q.lower()
    out = job["output"]
    return (
        q in (out.get("job_title") or "").lower()
        or q in (out.get("company_name") or "").lower()
        or any(q in s.lower() for s in (out.get("skills") or []))
        or q in (out.get("source_file") or "").lower()
    )

filtered = [j for j in all_jobs if not search or matches(j, search)]

if not filtered:
    st.sidebar.warning("No jobs match search.")
    st.stop()

job_labels = [
    f"{quality_color(quality_score(j))} {j['output'].get('job_title') or 'Unknown'[:40]} — {j['output'].get('source_file', '')}"
    for j in filtered
]

selected_idx = st.sidebar.radio(
    "Select job",
    range(len(filtered)),
    format_func=lambda i: job_labels[i],
)

job = filtered[selected_idx]
source_file = job["output"].get("source_file", "")
html_path = os.path.join(html_folder, source_file)

# ── Quality indicator bar ─────────────────────────────────────────────────────
score = quality_score(job)
total_fields = len(FIELDS)
pct = int(score / total_fields * 100)

col_q1, col_q2 = st.columns([3, 1])
with col_q1:
    st.progress(score / total_fields, text=f"Extraction quality: {score}/{total_fields} fields filled ({pct}%)")
with col_q2:
    out = job["output"]
    missing = [f for f in FIELDS if not out.get(f) or (f == "skills" and not out.get("skills"))]
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
    else:
        st.success("All fields extracted!")

st.divider()

# ── Side by side ──────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

# ── LEFT: INPUT ───────────────────────────────────────────────────────────────
with left:
    st.subheader("INPUT — Cleaned Text (fed to LLM)")
    st.caption(f"Source file: `{source_file}`")

    # Use saved input from new format, fallback to re-cleaning the HTML
    if job.get("input"):
        cleaned = job["input"]
    elif os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            raw_html = f.read()
        cleaned = clean_html_trafilatura(raw_html)
    else:
        cleaned = None

    if cleaned is None:
        st.error(f"HTML file not found:\n`{html_path}`\n\nCheck the HTML folder path in the sidebar.")
    else:
        char_count = len(cleaned)
        line_count = cleaned.count("\n")

        st.caption(f"{char_count:,} chars · {line_count:,} lines")

        # Highlight search term in cleaned text
        highlight = st.text_input("Highlight keyword in text", placeholder="e.g. salary, flutter", key="hl")

        if highlight:
            # Show only lines containing the keyword
            lines = cleaned.splitlines()
            matched = [l for l in lines if highlight.lower() in l.lower()]
            if matched:
                st.info(f"{len(matched)} lines containing '{highlight}'")
                st.text("\n".join(matched))
            else:
                st.warning(f"'{highlight}' not found in cleaned text")
        else:
            st.text_area(
                label="",
                value=cleaned,
                height=600,
                disabled=True,
                key="input_text",
            )

# ── RIGHT: OUTPUT ─────────────────────────────────────────────────────────────
with right:
    st.subheader("OUTPUT — LLM Extracted Fields")
    st.caption(f"Model output from stage2")

    def field_row(label, value, expected_type="text"):
        is_null = value is None or (isinstance(value, list) and len(value) == 0)
        icon = "🔴" if is_null else "🟢"
        st.markdown(f"**{icon} {label}**")
        if is_null:
            st.caption("_null / empty_")
        elif expected_type == "skills":
            cols = st.columns(3)
            for i, skill in enumerate(value):
                cols[i % 3].markdown(
                    f"<span style='background:#1a2e4a;color:#7dd3fc;padding:3px 10px;"
                    f"border-radius:12px;font-size:0.82rem;display:inline-block'>{skill}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.text_area("", value=str(value), height=80, disabled=True, key=f"out_{label}")

    field_row("Job Title", out.get("job_title"))
    field_row("Company", out.get("company_name"))

    c1, c2 = st.columns(2)
    with c1:
        field_row("Salary", out.get("salary"))
        field_row("Posted Date", out.get("posted_date"))
    with c2:
        field_row("Location", out.get("location"))
        field_row("Category", out.get("job_category"))

    if out.get("full_url"):
        st.markdown(f"**🔗 URL:** [{out['full_url']}]({out['full_url']})")

    st.divider()
    field_row("Skills", out.get("skills"), expected_type="skills")

    st.divider()
    with st.expander("Job Description"):
        st.write(out.get("job_description") or "_null_")
    with st.expander("Job Requirements"):
        st.write(out.get("job_requirement") or "_null_")

    st.divider()
    with st.expander("Raw JSON (output only)"):
        st.json({k: v for k, v in out.items() if k not in ("job_description", "job_requirement")})
