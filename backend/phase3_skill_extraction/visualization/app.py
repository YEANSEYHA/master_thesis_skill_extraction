"""
Job Extraction Explorer — Streamlit App
Supports stage2, stage3, stage4, stage5 JSON outputs.

Run:
    cd visualization
    streamlit run app.py
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Extraction Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
STAGE_FILES = {
    "stage2_output.json": "../03_extract/stage2_output.json",
    "stage3_output.json": "../03_extract/stage3_output.json",
    "stage4_output.json": "../03_extract/stage4_output.json",
    "stage5_output.json": "../03_extract/stage5_output.json",
}


@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def skill_list(val):
    return val if isinstance(val, list) else []


def field_coverage(df: pd.DataFrame) -> pd.DataFrame:
    fields = ["job_title", "job_description", "job_requirement",
              "skills", "salary", "location", "posted_date",
              "company_name", "job_category"]
    rows = []
    for col in fields:
        if col not in df.columns:
            continue
        if col == "skills":
            filled = df[col].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
        else:
            filled = df[col].notna().sum()
        pct = round(filled / len(df) * 100, 1)
        rows.append({"Field": col, "Filled": filled, "Missing": len(df) - filled, "% Filled": pct})
    return pd.DataFrame(rows)


# ── Sidebar — file picker ──────────────────────────────────────────────────
st.sidebar.title("🔍 Job Extraction Explorer")

source = st.sidebar.radio("Data source", ["Preset file", "Upload JSON"])

data = None
stage = None

if source == "Preset file":
    available = {k: v for k, v in STAGE_FILES.items() if os.path.exists(v)}
    if not available:
        st.sidebar.error("No output JSON files found.")
        st.stop()
    chosen = st.sidebar.selectbox("Select output file", list(available.keys()))
    data = load_json(available[chosen])
    stage = chosen.split("_")[0]  # "stage2", "stage3", etc.

else:
    uploaded = st.sidebar.file_uploader("Upload JSON", type="json")
    if uploaded:
        data = json.load(uploaded)
        stage = "uploaded"
    else:
        st.info("Upload a stage JSON file to get started.")
        st.stop()

if data is None:
    st.stop()

# ── Detect stage5 format (nested dict) ───────────────────────────────────
is_stage5 = isinstance(data, dict) and "all_skills" in data
is_stage4 = isinstance(data, dict) and "results" in data

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 VIEW
# ══════════════════════════════════════════════════════════════════════════════
if is_stage5:
    cfg = data.get("config", {})
    all_skills = data.get("all_skills", [])
    emerging = data.get("emerging_skills", [])

    st.title("Stage 5 — Trending & Emerging Skills")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total jobs", cfg.get("total_jobs", "—"))
    c2.metric("Skill clusters", cfg.get("total_clusters", len(all_skills)))
    c3.metric("ESCO-covered", cfg.get("esco_covered_count", "—"))
    c4.metric("Emerging", cfg.get("emerging_count", len(emerging)))

    sk_df = pd.DataFrame(all_skills)

    tab1, tab2 = st.tabs(["All Skills", "Emerging Only"])

    with tab1:
        top_n = st.slider("Show top N", 10, len(sk_df), min(50, len(sk_df)), key="top_all")
        fig = px.bar(
            sk_df.head(top_n), x="job_count", y="skill",
            orientation="h", color="is_emerging",
            color_discrete_map={True: "#e53e3e", False: "#48bb78"},
            title=f"Top {top_n} Trending Skills",
            labels={"job_count": "Jobs mentioning skill", "skill": "", "is_emerging": "Emerging"},
            height=max(400, top_n * 22),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sk_df.head(top_n), use_container_width=True)

    with tab2:
        em_df = pd.DataFrame(emerging)
        if em_df.empty:
            st.info("No emerging skills found at current threshold.")
        else:
            top_e = st.slider("Show top N emerging", 10, len(em_df), min(30, len(em_df)), key="top_em")
            fig2 = px.bar(
                em_df.head(top_e), x="job_count", y="skill",
                orientation="h",
                title=f"Top {top_e} Emerging Skills (not in ESCO)",
                color_discrete_sequence=["#e53e3e"],
                height=max(400, top_e * 22),
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(em_df.head(top_e), use_container_width=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 VIEW
# ══════════════════════════════════════════════════════════════════════════════
if is_stage4:
    cfg = data.get("config", {})
    results = data.get("results", [])

    st.title("Stage 4 — Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs evaluated", cfg.get("total_jobs", len(results)))
    c2.metric("Avg Hard F1", f"{cfg.get('avg_hard_f1', 0)*100:.1f}%")
    c3.metric("Avg Semantic F1", f"{cfg.get('avg_semantic_f1', 0)*100:.1f}%")
    c4.metric("Model", cfg.get("similarity_model", "—").split("/")[-1])

    res_df = pd.DataFrame(results)

    tab1, tab2 = st.tabs(["Score Chart", "Per-Job Table"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Hard F1", x=res_df["source_file"], y=res_df["hard_f1"],
                             marker_color="#6366f1"))
        fig.add_trace(go.Bar(name="Semantic F1", x=res_df["source_file"], y=res_df["semantic_f1"],
                             marker_color="#48bb78"))
        fig.update_layout(barmode="group", xaxis_tickangle=-45, title="F1 Scores per Job",
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(res_df, use_container_width=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 / 3 VIEW  (list of job dicts)
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(data)
df["skill_count"] = df["skills"].apply(lambda x: len(skill_list(x)))

st.title(f"Job Extraction Explorer — {stage}")

# ── Top metrics ───────────────────────────────────────────────────────────
total = len(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total jobs", total)
c2.metric("Avg skills / job", f"{df['skill_count'].mean():.1f}")
c3.metric("Has salary", f"{df['salary'].notna().sum()}/{total}")
c4.metric("Has location", f"{df['location'].notna().sum()}/{total}")
c5.metric("Has company", f"{df['company_name'].notna().sum()}/{total}")

st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────
st.sidebar.subheader("Filters")

# Category filter
if "job_category" in df.columns:
    cats = ["All"] + sorted(df["job_category"].dropna().unique().tolist())
    sel_cat = st.sidebar.selectbox("Category", cats)
else:
    sel_cat = "All"

# Location filter
if "location" in df.columns:
    locs = ["All"] + sorted(df["location"].dropna().unique().tolist())
    sel_loc = st.sidebar.selectbox("Location", locs)
else:
    sel_loc = "All"

# Salary toggle
show_missing_salary = st.sidebar.checkbox("Only jobs with missing salary", value=False)

# Skill search
skill_query = st.sidebar.text_input("Search skill", placeholder="e.g. flutter")

# Min skill count
min_skills = st.sidebar.slider("Min skills extracted", 0, int(df["skill_count"].max()), 0)

# Apply filters
fdf = df.copy()
if sel_cat != "All":
    fdf = fdf[fdf["job_category"] == sel_cat]
if sel_loc != "All":
    fdf = fdf[fdf["location"] == sel_loc]
if show_missing_salary:
    fdf = fdf[fdf["salary"].isna()]
if skill_query:
    q = skill_query.lower()
    fdf = fdf[fdf["skills"].apply(lambda s: any(q in sk.lower() for sk in skill_list(s)))]
fdf = fdf[fdf["skill_count"] >= min_skills]

st.caption(f"Showing {len(fdf)} of {total} jobs")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_list, tab_detail, tab_quality, tab_skills, tab_charts = st.tabs([
    "Job List", "Job Detail", "Data Quality", "Top Skills", "Charts"
])

# ── Tab: Job List ─────────────────────────────────────────────────────────
with tab_list:
    cols_to_show = [c for c in ["job_title", "company_name", "location", "salary",
                                 "job_category", "posted_date", "skill_count", "source_file"]
                    if c in fdf.columns]
    st.dataframe(
        fdf[cols_to_show].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

# ── Tab: Job Detail ───────────────────────────────────────────────────────
with tab_detail:
    if fdf.empty:
        st.info("No jobs match current filters.")
    else:
        titles = [f"[{i}] {row.get('job_title') or 'Unknown'} — {row.get('company_name') or '—'}"
                  for i, row in fdf.iterrows()]
        selected_label = st.selectbox("Select a job", titles)
        sel_idx = int(selected_label.split("]")[0].strip("["))
        job = fdf.loc[sel_idx].to_dict()

        # Header
        st.subheader(job.get("job_title") or "Unknown Title")
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Company:** {job.get('company_name') or '—'}")
        meta_cols[1].markdown(f"**Location:** {job.get('location') or '—'}")
        meta_cols[2].markdown(f"**Salary:** {job.get('salary') or '—'}")
        meta_cols[3].markdown(f"**Posted:** {job.get('posted_date') or '—'}")

        if job.get("full_url"):
            st.markdown(f"**URL:** {job['full_url']}")

        st.divider()

        d_col, r_col = st.columns(2)
        with d_col:
            st.markdown("**Job Description**")
            st.text_area("", value=job.get("job_description") or "—", height=250, disabled=True, key="desc")
        with r_col:
            st.markdown("**Job Requirements**")
            st.text_area("", value=job.get("job_requirement") or "—", height=250, disabled=True, key="req")

        st.markdown(f"**Skills ({job.get('skill_count', 0)})**")
        skills = skill_list(job.get("skills"))
        if skills:
            st.write(" · ".join(f"`{s}`" for s in skills))
        else:
            st.caption("No skills extracted")

        # Stage3 extras
        if "esco_ground_truth" in job:
            st.markdown(f"**ESCO Ground Truth ({len(skill_list(job.get('esco_ground_truth')))})**")
            gt = skill_list(job.get("esco_ground_truth"))
            if gt:
                st.write(" · ".join(f"`{s}`" for s in gt))

# ── Tab: Data Quality ─────────────────────────────────────────────────────
with tab_quality:
    cov = field_coverage(df)
    fig = px.bar(
        cov, x="Field", y="% Filled",
        color="% Filled",
        color_continuous_scale=["#e53e3e", "#ed8936", "#48bb78"],
        range_color=[0, 100],
        text="% Filled",
        title="Field Coverage — % of jobs with non-null value",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 115], coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Jobs with missing fields")
    for field in ["job_title", "salary", "location", "company_name"]:
        if field not in df.columns:
            continue
        missing = df[df[field].isna()]
        if not missing.empty:
            with st.expander(f"Missing '{field}' — {len(missing)} jobs"):
                show_cols = [c for c in ["source_file", "job_title", "company_name", "location"] if c in missing.columns]
                st.dataframe(missing[show_cols], use_container_width=True)

# ── Tab: Top Skills ───────────────────────────────────────────────────────
with tab_skills:
    top_n = st.slider("Top N skills", 10, 50, 25)
    all_skills_flat = [s.lower().strip() for row in fdf["skills"] for s in skill_list(row)]
    counts = Counter(all_skills_flat).most_common(top_n)

    if counts:
        sk_df = pd.DataFrame(counts, columns=["Skill", "Count"])
        fig = px.bar(
            sk_df.sort_values("Count"), x="Count", y="Skill",
            orientation="h",
            title=f"Top {top_n} Skills (filtered view)",
            color="Count",
            color_continuous_scale="Blues",
            text="Count",
            height=max(400, top_n * 24),
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sk_df, use_container_width=True)
    else:
        st.info("No skills found with current filters.")

# ── Tab: Charts ───────────────────────────────────────────────────────────
with tab_charts:
    col_a, col_b = st.columns(2)

    with col_a:
        # Skills per job histogram
        fig = px.histogram(
            fdf, x="skill_count",
            title="Skills extracted per job",
            nbins=15,
            color_discrete_sequence=["#6366f1"],
            labels={"skill_count": "Skill count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Category donut
        if "job_category" in fdf.columns:
            cat_df = fdf["job_category"].fillna("Unknown").value_counts().reset_index()
            cat_df.columns = ["Category", "Count"]
            fig2 = px.pie(cat_df, names="Category", values="Count",
                          title="Jobs by Category", hole=0.35)
            st.plotly_chart(fig2, use_container_width=True)

    # Location bar
    if "location" in fdf.columns:
        loc_df = fdf["location"].fillna("Unknown").value_counts().head(15).reset_index()
        loc_df.columns = ["Location", "Count"]
        fig3 = px.bar(
            loc_df, x="Count", y="Location", orientation="h",
            title="Top 15 Locations", color="Count",
            color_continuous_scale="Teal", text="Count",
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Stage 3 only: LLM vs ESCO count per job
    if "esco_ground_truth" in df.columns:
        st.subheader("LLM Skills vs ESCO Ground Truth per Job")
        df["gt_count"] = df["esco_ground_truth"].apply(lambda x: len(skill_list(x)))
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="LLM Skills", x=df["source_file"],
                              y=df["skill_count"], marker_color="#6366f1"))
        fig4.add_trace(go.Bar(name="ESCO Ground Truth", x=df["source_file"],
                              y=df["gt_count"], marker_color="#48bb78"))
        fig4.update_layout(barmode="group", xaxis_tickangle=-45, yaxis_range=[0, None])
        st.plotly_chart(fig4, use_container_width=True)
