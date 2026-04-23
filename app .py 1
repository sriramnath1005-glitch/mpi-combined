import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="India MPI Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #0d47a1 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center;
    }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; }
    .insight-box {
        background: #e3f2fd; border-left: 4px solid #1565c0; border-radius: 6px;
        padding: 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    mpi = pd.read_excel("data/mpi_project_data_sources.xlsx", sheet_name="NITI_MPI_States")
    hdi = pd.read_excel("data/mpi_project_data_sources.xlsx", sheet_name="GDL_SHDI_States")
    hdi = hdi[hdi["state_ut"] != "Total"].copy()

    hdi_long = hdi.melt(
        id_vars=["state_ut", "source", "source_url"],
        value_vars=["2019", "2020", "2021", "2022", "2023"],
        var_name="year", value_name="hdi_value"
    )
    hdi_long["year"] = hdi_long["year"].astype(int)

    def categorize(v):
        if v >= 30:   return "High Poverty"
        elif v >= 15: return "Moderate Poverty"
        elif v >= 5:  return "Low Poverty"
        else:         return "Very Low Poverty"

    mpi["poverty_category"] = mpi["headcount_2019_21_pct"].apply(categorize)
    mpi["improvement"]      = mpi["change_pct_points"].abs()
    mpi["improvement_rate"] = (
        (mpi["headcount_2015_16_pct"] - mpi["headcount_2019_21_pct"])
        / mpi["headcount_2015_16_pct"] * 100
    ).round(1)

    # ---- Merged dataset for ML tabs ----
    merged = pd.merge(
        mpi[["state_ut", "headcount_2019_21_pct", "change_pct_points"]],
        hdi[["state_ut", "2019", "2023"]].rename(columns={"2019": "hdi_2019", "2023": "hdi_2023"}),
        on="state_ut", how="inner"
    )

    # K-Means clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ["hdi_2019", "hdi_2023", "change_pct_points"]
    X = merged[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    merged["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = merged.groupby("Cluster")["headcount_2019_21_pct"].mean()
    risk_map   = cluster_summary.rank().astype(int).to_dict()
    label_map  = {k: ["Low Risk", "Medium Risk", "High Risk"][v - 1] for k, v in risk_map.items()}
    merged["RiskSegment"] = merged["Cluster"].map(label_map)

    # TOPSIS
    matrix   = merged[["hdi_2023", "headcount_2019_21_pct", "change_pct_points"]].values.astype(float)
    norm     = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weights  = np.array([0.40, 0.35, 0.25])
    weighted = norm * weights
    benefit  = [True, False, False]
    ideal_b  = np.where(benefit, weighted.max(0), weighted.min(0))
    ideal_w  = np.where(benefit, weighted.min(0), weighted.max(0))
    d_b      = np.sqrt(((weighted - ideal_b) ** 2).sum(axis=1))
    d_w      = np.sqrt(((weighted - ideal_w) ** 2).sum(axis=1))
    merged["TOPSIS_Score"] = (d_w / (d_b + d_w)).round(4)

    return mpi, hdi, hdi_long, merged


mpi_df, hdi_df, hdi_long, merged_df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
        width=100
    )
    st.markdown("## 🗂 Navigate")
    page = st.radio("", [
        "📊 Overview",
        "🗺 State Explorer",
        "📈 HDI Trends",
        "🔬 ML Analysis",
        "🎯 MPI Predictor",
        "📋 Data Table"
    ])
    st.markdown("---")
    categories = mpi_df["poverty_category"].unique().tolist()
    sel_cats = st.multiselect("Poverty Categories", categories, default=categories)
    rng = st.slider(
        "Poverty Rate Range (%)",
        float(mpi_df["headcount_2019_21_pct"].min()),
        float(mpi_df["headcount_2019_21_pct"].max()),
        (float(mpi_df["headcount_2019_21_pct"].min()), float(mpi_df["headcount_2019_21_pct"].max()))
    )

fmpi = mpi_df[
    mpi_df["poverty_category"].isin(sel_cats) &
    mpi_df["headcount_2019_21_pct"].between(*rng)
]

# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown(
        '''<div class="main-header">
        <h1>🇮🇳 India MPI Dashboard</h1>
        <p>NITI Aayog MPI 2023 · Applied Business Analytics Final Project</p>
        </div>''',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States/UTs", len(mpi_df))
    c2.metric(
        "Avg MPI 2019-21",
        f"{mpi_df['headcount_2019_21_pct'].mean():.1f}%",
        delta=f"{mpi_df['headcount_2019_21_pct'].mean() - mpi_df['headcount_2015_16_pct'].mean():.1f}%"
    )
    best = mpi_df.loc[mpi_df["change_pct_points"].idxmin()]
    c3.metric("Best Improvement", best["state_ut"], delta=f"{best['change_pct_points']:.1f} pp")
    c4.metric("High Poverty States", int((mpi_df["headcount_2019_21_pct"] >= 30).sum()))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 Poverty States (2019-21)")
        top10 = fmpi.nlargest(10, "headcount_2019_21_pct")
        fig = px.bar(
            top10, x="headcount_2019_21_pct", y="state_ut", orientation="h",
            color="headcount_2019_21_pct", color_continuous_scale="Reds",
            text="headcount_2019_21_pct"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Poverty Reduction (2016 → 2021)")
        top_imp = fmpi.nlargest(10, "improvement")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="2015-16", x=top_imp["state_ut"],
                              y=top_imp["headcount_2015_16_pct"], marker_color="#ef5350"))
        fig2.add_trace(go.Bar(name="2019-21", x=top_imp["state_ut"],
                              y=top_imp["headcount_2019_21_pct"], marker_color="#42a5f5"))
        fig2.update_layout(barmode="group", height=400, plot_bgcolor="white", xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Poverty Category Distribution")
    cat_counts = fmpi["poverty_category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    fig3 = px.pie(cat_counts, values="count", names="category",
                  color_discrete_sequence=["#c8e6c9", "#fff9c4", "#ffe0b2", "#ffcdd2"],
                  hole=0.4)
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: STATE EXPLORER ──────────────────────────────────────────────────────
elif page == "🗺 State Explorer":
    st.title("🗺 State Explorer")
    sel = st.selectbox("Select State/UT", sorted(mpi_df["state_ut"].tolist()))
    row = mpi_df[mpi_df["state_ut"] == sel].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("MPI 2019-21", f"{row['headcount_2019_21_pct']:.2f}%")
    c2.metric("MPI 2015-16", f"{row['headcount_2015_16_pct']:.2f}%")
    c3.metric("Change", f"{row['change_pct_points']:.2f} pp")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=row["headcount_2019_21_pct"],
        delta={"reference": row["headcount_2015_16_pct"]},
        title={"text": f"MPI Headcount — {sel}"},
        gauge={
            "axis": {"range": [0, 60]},
            "bar":  {"color": "#1565c0"},
            "steps": [
                {"range": [0,  5],  "color": "#c8e6c9"},
                {"range": [5,  15], "color": "#fff9c4"},
                {"range": [15, 30], "color": "#ffe0b2"},
                {"range": [30, 60], "color": "#ffcdd2"},
            ]
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How this state compares")
    rank = int(mpi_df["headcount_2019_21_pct"].rank(ascending=False)[mpi_df["state_ut"] == sel].values[0])
    st.info(f"**{sel}** ranks **#{rank}** out of {len(mpi_df)} states/UTs by MPI headcount (2019-21).")

# ── PAGE: HDI TRENDS ─────────────────────────────────────────────────────────
elif page == "📈 HDI Trends":
    st.title("📈 Subnational HDI Trends (2019–2023)")
    states_hdi = st.multiselect(
        "Select States",
        sorted(hdi_long["state_ut"].unique()),
        default=["Bihar", "Kerala", "Uttar Pradesh", "Tamil Nadu"]
    )
    fhdi = hdi_long[hdi_long["state_ut"].isin(states_hdi)]
    fig = px.line(fhdi, x="year", y="hdi_value", color="state_ut", markers=True,
                  title="Subnational HDI Comparison (2019–2023)")
    fig.update_layout(height=450, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE: ML ANALYSIS ────────────────────────────────────────────────────────
elif page == "🔬 ML Analysis":
    st.title("🔬 ML Analysis — Clustering & TOPSIS")

    tab1, tab2 = st.tabs(["K-Means Clustering", "TOPSIS Ranking"])

    with tab1:
        st.markdown("#### State Development Segments (K-Means, k=3)")
        fig = px.scatter(
            merged_df, x="hdi_2019", y="headcount_2019_21_pct",
            color="RiskSegment",
            color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"},
            hover_name="state_ut", size="headcount_2019_21_pct",
            title="K-Means Clusters — HDI vs MPI Headcount"
        )
        fig.update_layout(height=480, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            merged_df[["state_ut", "headcount_2019_21_pct", "hdi_2019", "RiskSegment"]]
            .sort_values("headcount_2019_21_pct", ascending=False),
            use_container_width=True
        )

    with tab2:
        st.markdown("#### TOPSIS Development Ranking")
        topsis_sorted = merged_df.sort_values("TOPSIS_Score", ascending=True)
        fig2 = px.bar(
            topsis_sorted, x="TOPSIS_Score", y="state_ut", orientation="h",
            color="TOPSIS_Score",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            title="TOPSIS Score by State/UT (Higher = Better Developed)"
        )
        fig2.add_vline(x=0.5, line_dash="dash", line_color="black")
        fig2.update_layout(height=550, showlegend=False, plot_bgcolor="white",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE: MPI PREDICTOR ───────────────────────────────────────────────────────
elif page == "🎯 MPI Predictor":
    st.title("🎯 MPI Score Estimator")
    st.latex(r"MPI = H \times A")
    st.caption("Adjust deprivation percentages across all 10 indicators to estimate MPI.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Education**")
        sy = st.slider("Years of Schooling Dep%",    0, 100, 40)
        sa = st.slider("School Attendance Dep%",     0, 100, 30)
    with c2:
        st.markdown("**💊 Health**")
        nu = st.slider("Nutrition Dep%",             0, 100, 35)
        cm = st.slider("Child Mortality Dep%",       0, 100, 20)
    with c3:
        st.markdown("**🏠 Living Standards**")
        cf = st.slider("Cooking Fuel Dep%",          0, 100, 50)
        sn = st.slider("Sanitation Dep%",            0, 100, 45)
        dw = st.slider("Drinking Water Dep%",        0, 100, 25)
        el = st.slider("Electricity Dep%",           0, 100, 20)
        ho = st.slider("Housing Dep%",               0, 100, 30)
        as_ = st.slider("Assets Dep%",               0, 100, 35)

    w  = {"sy": 1/6, "sa": 1/6, "nu": 1/6, "cm": 1/6,
          "cf": 1/18, "sn": 1/18, "dw": 1/18, "el": 1/18, "ho": 1/18, "as_": 1/18}
    v  = {"sy": sy/100, "sa": sa/100, "nu": nu/100, "cm": cm/100,
          "cf": cf/100, "sn": sn/100, "dw": dw/100, "el": el/100, "ho": ho/100, "as_": as_/100}
    c_score = sum(w[k] * v[k] for k in w)
    H       = min(1.0, c_score / 0.33)
    MPI     = round(H * c_score, 4)

    r1, r2, r3 = st.columns(3)
    r1.metric("Deprivation Score (A)", f"{c_score:.3f}")
    r2.metric("Headcount Ratio (H)",   f"{H:.3f}")
    r3.metric("🎯 MPI Score",          f"{MPI:.4f}")

    if   MPI < 0.01: st.success("✅ Very Low Poverty")
    elif MPI < 0.05: st.info("ℹ️ Low Poverty")
    elif MPI < 0.15: st.warning("⚠️ Moderate Poverty")
    else:            st.error("🚨 High Poverty — Urgent intervention needed")

# ── PAGE: DATA TABLE ──────────────────────────────────────────────────────────
elif page == "📋 Data Table":
    st.title("📋 Full Dataset")
    tab1, tab2, tab3 = st.tabs(["MPI Data", "HDI Data", "ML Results"])

    with tab1:
        st.dataframe(
            fmpi[["state_ut", "headcount_2019_21_pct", "headcount_2015_16_pct",
                  "change_pct_points", "improvement_rate", "poverty_category"]],
            use_container_width=True
        )
        st.download_button("⬇️ Download MPI CSV",
                           fmpi.to_csv(index=False).encode(), "mpi_data.csv", "text/csv")

    with tab2:
        st.dataframe(hdi_long, use_container_width=True)
        st.download_button("⬇️ Download HDI CSV",
                           hdi_long.to_csv(index=False).encode(), "hdi_data.csv", "text/csv")

    with tab3:
        st.dataframe(
            merged_df[["state_ut", "headcount_2019_21_pct", "hdi_2019", "hdi_2023",
                       "RiskSegment", "TOPSIS_Score"]].sort_values("TOPSIS_Score", ascending=False),
            use_container_width=True
        )
        st.download_button("⬇️ Download ML Results CSV",
                           merged_df.to_csv(index=False).encode(), "ml_results.csv", "text/csv")

st.markdown("---")
st.caption("📌 Sources: NITI Aayog National MPI 2023 · Global Data Lab SHDI | ABA Final Project")
