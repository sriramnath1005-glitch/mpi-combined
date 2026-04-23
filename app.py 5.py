import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
import pickle
import os
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India MPI Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.banner {
    background: linear-gradient(120deg, #0d47a1, #1976d2, #42a5f5);
    padding: 2.2rem 2rem;
    border-radius: 14px;
    color: #ffffff !important;
    text-align: center;
    margin-bottom: 1.8rem;
}
.banner h1 { font-size: 1.9rem; margin: 0 0 0.4rem; font-weight: 700; color: #ffffff !important; }
.banner p  { font-size: 0.95rem; opacity: 0.88; margin: 0; color: #ffffff !important; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: white;
}
.metric-card .label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #e2e8f0; }
.metric-card .delta { font-size: 0.8rem; margin-top: 4px; }

.sec-head {
    font-size: 1rem;
    font-weight: 700;
    color: #4fc3f7 !important;
    border-left: 4px solid #42a5f5;
    padding-left: 0.6rem;
    margin: 1.4rem 0 0.6rem;
}

.section-header {
    background: linear-gradient(90deg, #e74c3c, #f39c12);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.info-card {
    background: rgba(30, 136, 229, 0.15);
    border-left: 4px solid #42a5f5;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #90caf9 !important;
    margin-top: 1rem;
}

.insight-box {
    background: #0f172a;
    border-left: 4px solid #e74c3c;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.6;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 0.7rem 1rem !important;
}

h1 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
PLT_DARK = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e3a5f",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#334155",
}

def apply_dark():
    plt.rcParams.update(PLT_DARK)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "High Poverty":     "#e53935",
    "Moderate Poverty": "#fb8c00",
    "Low Poverty":      "#43a047",
    "Very Low Poverty": "#1e88e5",
}

def banner(title, subtitle=""):
    st.markdown(
        f'<div class="banner"><h1>{title}</h1>'
        + (f"<p>{subtitle}</p>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )

def sec(label):
    st.markdown(f'<div class="sec-head">{label}</div>', unsafe_allow_html=True)

def card(text):
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data_file(uploaded_file):
    """Load from an uploaded file object (app_py_4 style)."""
    xl = pd.ExcelFile(uploaded_file)
    mpi = xl.parse("NITI_MPI_States")
    shdi = xl.parse("GDL_SHDI_States")
    mpi = mpi.rename(columns={
        "headcount_2019_21_pct": "headcount_2019_21",
        "headcount_2015_16_pct": "headcount_2015_16",
        "change_pct_points":     "change_pct",
    })
    return mpi, shdi

@st.cache_data
def load_data_path():
    """Load from fixed path (original app.py style for Streamlit Cloud)."""
    data_path = "mpi_project_data_sources.xlsx"
    if not os.path.exists(data_path):
        data_path = "data/mpi_project_data_sources.xlsx"
    mpi = pd.read_excel(data_path, sheet_name="NITI_MPI_States")
    hdi = pd.read_excel(data_path, sheet_name="GDL_SHDI_States")

    hdi_clean = hdi[hdi["state_ut"] != "Total"].copy().reset_index(drop=True)
    hdi_long = hdi_clean.melt(
        id_vars=["state_ut"],
        value_vars=["2019", "2020", "2021", "2022", "2023"],
        var_name="year",
        value_name="hdi_value",
    )
    hdi_long["year"]      = hdi_long["year"].astype(int)
    hdi_long["hdi_value"] = pd.to_numeric(hdi_long["hdi_value"], errors="coerce")

    def cat(v):
        if v >= 30:   return "High Poverty"
        elif v >= 15: return "Moderate Poverty"
        elif v >= 5:  return "Low Poverty"
        else:         return "Very Low Poverty"

    mpi["category"]         = mpi["headcount_2019_21_pct"].apply(cat)
    mpi["improvement_abs"]  = mpi["change_pct_points"].abs()
    mpi["improvement_rate"] = (
        (mpi["headcount_2015_16_pct"] - mpi["headcount_2019_21_pct"])
        / mpi["headcount_2015_16_pct"] * 100
    ).round(1)

    return mpi, hdi_clean, hdi_long

@st.cache_data
def build_merged(mpi_df, shdi_df):
    shdi_states = shdi_df[shdi_df["state_ut"] != "Total"].copy()
    merged = pd.merge(
        mpi_df[["state_ut", "headcount_2019_21", "headcount_2015_16", "change_pct"]],
        shdi_states[["state_ut", "2019", "2020", "2021", "2022", "2023"]].rename(
            columns={"2019": "hdi_2019", "2020": "hdi_2020",
                     "2021": "hdi_2021", "2022": "hdi_2022", "2023": "hdi_2023"}),
        on="state_ut", how="inner",
    )
    return merged, shdi_states

@st.cache_data
def run_ml(merged_df):
    df = merged_df.copy()
    median_pov = df["headcount_2019_21"].median()
    df["high_poverty"] = (df["headcount_2019_21"] > median_pov).astype(int)

    features = ["hdi_2019", "hdi_2023", "change_pct"]
    X = df[features].values
    y = df["high_poverty"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred,
                                   target_names=["Low Poverty", "High Poverty"],
                                   output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_summary = df.groupby("Cluster")[
        ["headcount_2019_21", "change_pct", "hdi_2019", "hdi_2023"]].mean().round(3)

    risk_map = cluster_summary["headcount_2019_21"].rank().astype(int).to_dict()
    label_map = {k: ["Low Risk", "Medium Risk", "High Risk"][v - 1] for k, v in risk_map.items()}
    df["RiskSegment"] = df["Cluster"].map(label_map)

    # TOPSIS
    topsis_df = df[["hdi_2023", "headcount_2019_21", "change_pct"]].copy()
    matrix = topsis_df.values.astype(float)
    norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weights = np.array([0.40, 0.35, 0.25])
    weighted = norm * weights
    benefit = [True, False, False]
    ideal_best  = np.where(benefit, weighted.max(0), weighted.min(0))
    ideal_worst = np.where(benefit, weighted.min(0), weighted.max(0))
    d_best  = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    df["TOPSIS_Score"] = d_worst / (d_best + d_worst)

    return df, lr, scaler, kmeans, X_scaled, X_test, y_test, report, auc, cluster_summary, median_pov


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇮🇳 India MPI\n**Dashboard & Analysis**")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload `mpi_project_data_sources.xlsx`",
        type=["xlsx"],
        help="Must contain sheets: NITI_MPI_States and GDL_SHDI_States",
    )

    st.markdown("---")
    st.markdown("**Navigate**")
    page = st.radio("", [
        "📊 Overview",
        "🗺️ State Explorer",
        "📈 HDI Trends",
        "🔬 MPI Predictor",
        "📉 Poverty Distribution",
        "🔝 State Rankings",
        "🔗 MPI vs HDI",
        "🤖 ML Models",
        "🏆 TOPSIS Ranking",
        "📋 Key Findings",
        "📋 Data Table",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.caption("Sources: NITI Aayog MPI 2023\nGlobal Data Lab SHDI")


# ─────────────────────────────────────────────────────────────────────────────
# DATA RESOLUTION — uploaded file takes priority; fall back to fixed path
# ─────────────────────────────────────────────────────────────────────────────
USE_UPLOAD = uploaded is not None

if USE_UPLOAD:
    mpi_raw, shdi_raw = load_data_file(uploaded)
    # Build hdi_long from shdi_raw for pages that need it
    shdi_clean = shdi_raw[shdi_raw["state_ut"] != "Total"].copy().reset_index(drop=True)
    hdi_long = shdi_clean.melt(
        id_vars=["state_ut"],
        value_vars=["2019", "2020", "2021", "2022", "2023"],
        var_name="year",
        value_name="hdi_value",
    )
    hdi_long["year"]      = hdi_long["year"].astype(int)
    hdi_long["hdi_value"] = pd.to_numeric(hdi_long["hdi_value"], errors="coerce")

    def cat(v):
        if v >= 30:   return "High Poverty"
        elif v >= 15: return "Moderate Poverty"
        elif v >= 5:  return "Low Poverty"
        else:         return "Very Low Poverty"

    mpi_raw["category"]         = mpi_raw["headcount_2019_21"].apply(cat)
    mpi_raw["improvement_abs"]  = mpi_raw["change_pct"].abs()
    mpi_raw["improvement_rate"] = (
        (mpi_raw["headcount_2015_16"] - mpi_raw["headcount_2019_21"])
        / mpi_raw["headcount_2015_16"] * 100
    ).round(1)

    # Unified aliases
    mpi_df  = mpi_raw.copy()
    hdi_df  = shdi_clean.copy()

    # Columns used by plotly pages expect _pct suffix — add aliases
    mpi_df["headcount_2019_21_pct"] = mpi_df["headcount_2019_21"]
    mpi_df["headcount_2015_16_pct"] = mpi_df["headcount_2015_16"]
    mpi_df["change_pct_points"]     = mpi_df["change_pct"]

    merged, shdi_states = build_merged(mpi_raw, shdi_raw)
    model_df, lr_model, scaler, kmeans, X_scaled, X_test, y_test, report, auc, cluster_summary, median_pov = run_ml(merged)
    apply_dark()

else:
    # Try loading from fixed path (Streamlit Cloud deployment)
    data_path_exists = (
        os.path.exists("mpi_project_data_sources.xlsx") or
        os.path.exists("data/mpi_project_data_sources.xlsx")
    )

    if not data_path_exists:
        st.info("👈 Upload your Excel file from the sidebar to begin.")
        st.markdown("""
        **What this dashboard covers:**
        - Interactive overview with Plotly charts
        - State explorer with gauge & national comparison
        - HDI trend lines and MPI vs HDI scatter
        - MPI Score Estimator (OPHI methodology)
        - Poverty distribution histograms
        - State rankings (top/bottom + reduction)
        - MPI vs HDI correlation & heatmap
        - Logistic Regression & K-Means ML models
        - TOPSIS multi-criteria development ranking
        - Key findings summary & model downloads
        """)
        st.stop()

    mpi_df, hdi_df, hdi_long = load_data_path()

    # Add upload-style aliases for ML pages
    mpi_df["headcount_2019_21"] = mpi_df["headcount_2019_21_pct"]
    mpi_df["headcount_2015_16"] = mpi_df["headcount_2015_16_pct"]
    mpi_df["change_pct"]        = mpi_df["change_pct_points"]

    shdi_raw  = hdi_df.copy()
    # Re-add year columns for build_merged
    for yr in ["2019","2020","2021","2022","2023"]:
        hdi_wide = hdi_long[hdi_long["year"] == int(yr)][["state_ut","hdi_value"]].rename(columns={"hdi_value": yr})
        shdi_raw = shdi_raw.merge(hdi_wide, on="state_ut", how="left") if yr not in shdi_raw.columns else shdi_raw

    merged, shdi_states = build_merged(mpi_df, shdi_raw)
    model_df, lr_model, scaler, kmeans, X_scaled, X_test, y_test, report, auc, cluster_summary, median_pov = run_ml(merged)
    apply_dark()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS (for Overview / State Explorer pages)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**Filters** *(Overview)*")
    all_cats  = sorted(mpi_df["category"].unique())
    sel_cats  = st.multiselect("Poverty Category", all_cats, default=all_cats)
    lo        = float(mpi_df["headcount_2019_21_pct"].min())
    hi        = float(mpi_df["headcount_2019_21_pct"].max())
    pov_range = st.slider("Poverty Rate Range (%)", lo, hi, (lo, hi), 0.1)

filtered = mpi_df[
    mpi_df["category"].isin(sel_cats) &
    mpi_df["headcount_2019_21_pct"].between(*pov_range)
].copy()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW (Plotly interactive)
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    banner("🇮🇳 India MPI Dashboard",
           "NITI Aayog MPI 2023 · Applied Business Analytics Final Project")

    avg_now  = mpi_df["headcount_2019_21_pct"].mean()
    avg_then = mpi_df["headcount_2015_16_pct"].mean()
    best_row = mpi_df.loc[mpi_df["improvement_abs"].idxmax()]
    high_n   = int((mpi_df["headcount_2019_21_pct"] >= 30).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("States / UTs", len(mpi_df))
    k2.metric("Avg Headcount 2019-21", f"{avg_now:.1f}%", f"{avg_now - avg_then:+.1f}%")
    k3.metric("Most Improved", best_row["state_ut"], f"{best_row['change_pct_points']:.1f} pp")
    k4.metric("High Poverty States (≥30%)", high_n)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("Top 10 States by MPI Headcount 2019-21")
        top10 = filtered.nlargest(10, "headcount_2019_21_pct")
        fig = px.bar(
            top10, x="headcount_2019_21_pct", y="state_ut",
            orientation="h", color="category",
            color_discrete_map=COLORS, text="headcount_2019_21_pct",
            labels={"headcount_2019_21_pct": "Headcount (%)", "state_ut": "", "category": "Category"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=400, plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.18, x=0),
            margin=dict(l=5, r=45, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Before vs After: Top 10 Most Improved States")
        top_imp = filtered.nlargest(10, "improvement_abs")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="2015-16", x=top_imp["state_ut"],
            y=top_imp["headcount_2015_16_pct"], marker_color="#ef9a9a",
        ))
        fig2.add_trace(go.Bar(
            name="2019-21", x=top_imp["state_ut"],
            y=top_imp["headcount_2019_21_pct"], marker_color="#42a5f5",
        ))
        fig2.update_layout(
            barmode="group", height=400, plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-35,
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=5, r=5, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        sec("Category Distribution (2021)")
        counts = filtered["category"].value_counts()
        fig3 = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=COLORS, hole=0.45,
        )
        fig3.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("Improvement Rate vs Current Poverty Level")
        fig4 = px.scatter(
            filtered, x="headcount_2019_21_pct", y="improvement_rate",
            size="improvement_abs", color="category",
            color_discrete_map=COLORS, hover_name="state_ut",
            labels={
                "headcount_2019_21_pct": "Current Poverty (%)",
                "improvement_rate": "Improvement Rate (%)",
                "category": "Category",
            },
        )
        fig4.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5, r=5, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    card("💡 <b>Key Insight:</b> Bihar leads in absolute reduction (−18.13 pp), while Uttar Pradesh "
         "and Madhya Pradesh show the fastest proportional improvement — high-poverty states can "
         "achieve rapid gains with targeted multi-sector interventions.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STATE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ State Explorer":
    banner("🗺️ State Explorer", "Drill into any State or Union Territory")

    states = sorted(mpi_df["state_ut"].tolist())
    sel    = st.selectbox("Select a State / UT", states)
    row    = mpi_df[mpi_df["state_ut"] == sel].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MPI Headcount 2019-21", f"{row['headcount_2019_21_pct']:.2f}%")
    m2.metric("MPI Headcount 2015-16", f"{row['headcount_2015_16_pct']:.2f}%")
    m3.metric("Change", f"{row['change_pct_points']:.2f} pp")
    m4.metric("Category", row["category"])

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(row["headcount_2019_21_pct"]),
        delta={"reference": float(row["headcount_2015_16_pct"]), "valueformat": ".1f"},
        title={"text": f"MPI Headcount — {sel}"},
        gauge={
            "axis": {"range": [0, 60]},
            "bar":  {"color": "#1565c0"},
            "steps": [
                {"range": [0,  5],  "color": "#c8e6c9"},
                {"range": [5,  15], "color": "#fff9c4"},
                {"range": [15, 30], "color": "#ffe0b2"},
                {"range": [30, 60], "color": "#ffcdd2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "value": float(row["headcount_2015_16_pct"]),
            },
        },
    ))
    fig_g.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig_g, use_container_width=True)

    ranked = mpi_df.sort_values("headcount_2019_21_pct", ascending=False).reset_index(drop=True)
    rank   = int(ranked[ranked["state_ut"] == sel].index[0]) + 1
    st.info(f"📌 **{sel}** ranks **#{rank}** out of {len(mpi_df)} States/UTs (1 = highest poverty)")

    sec("National Comparison (2019-21)")
    fig_cmp = px.bar(
        ranked, x="state_ut", y="headcount_2019_21_pct",
        color=ranked["state_ut"].apply(lambda x: "Selected" if x == sel else "Other"),
        color_discrete_map={"Selected": "#e53935", "Other": "#90caf9"},
        labels={"headcount_2019_21_pct": "MPI Headcount (%)", "state_ut": ""},
    )
    fig_cmp.update_layout(
        height=380, showlegend=False, xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5, r=5, t=10, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    hdi_row = hdi_df[hdi_df["state_ut"].str.strip().str.lower() == sel.strip().lower()]
    if not hdi_row.empty:
        sec(f"HDI Trend for {sel} (2019–2023)")
        years = [2019, 2020, 2021, 2022, 2023]
        vals  = [float(hdi_row.iloc[0][str(y)]) for y in years]
        fig_h = px.line(x=years, y=vals, markers=True,
                        labels={"x": "Year", "y": "HDI Value"})
        fig_h.update_traces(line_color="#1565c0", line_width=2.5, marker_size=9)
        fig_h.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=300,
                            margin=dict(l=5, r=5, t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info(f"HDI data not available for {sel}.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HDI TRENDS (Plotly)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 HDI Trends":
    banner("📈 HDI Trends", "Subnational Human Development Index 2019–2023")

    all_hdi  = sorted(hdi_long["state_ut"].unique())
    defaults = [s for s in ["Bihar", "Kerala", "Uttar Pradesh", "Tamil Nadu"] if s in all_hdi]

    c1, c2 = st.columns(2)
    with c1:
        sel_states = st.multiselect("Select States", all_hdi, default=defaults)
    with c2:
        yr = st.select_slider("Year Range", options=[2019, 2020, 2021, 2022, 2023],
                              value=(2019, 2023))

    if sel_states:
        fh = hdi_long[hdi_long["state_ut"].isin(sel_states) & hdi_long["year"].between(*yr)]
        fig_t = px.line(fh, x="year", y="hdi_value", color="state_ut", markers=True,
                        labels={"hdi_value": "HDI Value", "year": "Year", "state_ut": "State"},
                        title="Subnational HDI Comparison")
        fig_t.update_traces(line_width=2.5, marker_size=8)
        fig_t.update_layout(height=440, plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=5, r=5, t=40, b=10))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.warning("Please select at least one state.")

    sec("HDI vs MPI Relationship (2021)")
    hdi_2021 = hdi_long[hdi_long["year"] == 2021][["state_ut", "hdi_value"]].copy()
    merged_hdi = pd.merge(mpi_df, hdi_2021, on="state_ut").dropna(
        subset=["hdi_value", "headcount_2019_21_pct"])

    if not merged_hdi.empty:
        x_arr  = merged_hdi["hdi_value"].values.astype(float)
        y_arr  = merged_hdi["headcount_2019_21_pct"].values.astype(float)
        m_fit, b_fit = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        y_line = m_fit * x_line + b_fit

        fig_s = px.scatter(
            merged_hdi, x="hdi_value", y="headcount_2019_21_pct",
            color="category", color_discrete_map=COLORS, hover_name="state_ut",
            labels={"hdi_value": "HDI (2021)",
                    "headcount_2019_21_pct": "MPI Headcount % (2019-21)",
                    "category": "Category"},
            title="Higher HDI correlates with Lower Poverty",
        )
        fig_s.add_scatter(x=x_line, y=y_line, mode="lines",
                          line=dict(color="#1565c0", width=2, dash="dash"),
                          name="Trend (OLS)")
        fig_s.update_layout(height=440, plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=5, r=5, t=40, b=10))
        st.plotly_chart(fig_s, use_container_width=True)

        corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
        card(f"📊 <b>Pearson Correlation (HDI vs MPI):</b> {corr:.3f} — Strong negative "
             "relationship: higher human development consistently accompanies lower "
             "multidimensional poverty.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MPI PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 MPI Predictor":
    banner("🔬 MPI Score Estimator",
           "Estimate a custom MPI score using the OPHI methodology")

    st.markdown("### 📐 Formula")
    st.latex(r"MPI = H \times A")
    st.markdown("> **H** = Headcount ratio &nbsp;|&nbsp; **A** = Average intensity of deprivation")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🏫 Education (1/3)**")
        school_yrs = st.slider("Years of Schooling (%)",  0, 100, 40)
        school_att = st.slider("School Attendance (%)",   0, 100, 30)

    with c2:
        st.markdown("**🏥 Health (1/3)**")
        nutrition  = st.slider("Nutrition (%)",           0, 100, 35)
        child_mort = st.slider("Child Mortality (%)",     0, 100, 20)

    with c3:
        st.markdown("**🏠 Living Standards (1/3)**")
        cooking     = st.slider("Cooking Fuel (%)",   0, 100, 50)
        sanitation  = st.slider("Sanitation (%)",     0, 100, 45)
        water       = st.slider("Drinking Water (%)", 0, 100, 25)
        electricity = st.slider("Electricity (%)",    0, 100, 20)
        housing     = st.slider("Housing (%)",        0, 100, 30)
        assets      = st.slider("Assets (%)",         0, 100, 35)

    k = st.slider("Poverty Threshold (k)", 0.10, 1.00, 0.33, 0.01)

    W = {"school_yrs": 1/6, "school_att": 1/6, "nutrition": 1/6, "child_mort": 1/6,
         "cooking": 1/18, "sanitation": 1/18, "water": 1/18,
         "electricity": 1/18, "housing": 1/18, "assets": 1/18}
    V = {"school_yrs": school_yrs/100, "school_att": school_att/100,
         "nutrition": nutrition/100, "child_mort": child_mort/100,
         "cooking": cooking/100, "sanitation": sanitation/100,
         "water": water/100, "electricity": electricity/100,
         "housing": housing/100, "assets": assets/100}

    c_score = sum(W[d] * V[d] for d in W)
    H_val   = min(1.0, c_score / k) if k > 0 else 0.0
    A_val   = c_score
    MPI_val = round(H_val * A_val, 4)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Weighted Deprivation Score", f"{c_score:.3f}")
    r2.metric("Headcount Ratio (H)", f"{H_val:.3f}")
    r3.metric("🎯 Estimated MPI", f"{MPI_val:.4f}")

    if   MPI_val < 0.01: st.success("✅ Very Low Poverty — strong performance across all dimensions.")
    elif MPI_val < 0.05: st.info("🟡 Low Poverty — some gaps; targeted interventions advised.")
    elif MPI_val < 0.15: st.warning("🟠 Moderate Poverty — significant multi-dimensional gaps.")
    else:                st.error("🔴 High Poverty — urgent multi-sector intervention required.")

    sec("Dimensional Contribution Chart")
    dim_w = [W[d] * V[d] for d in W]
    fig_d = px.bar(x=list(W.keys()), y=dim_w, color=dim_w,
                   color_continuous_scale="OrRd",
                   labels={"x": "Dimension", "y": "Weighted Deprivation", "color": "Score"})
    fig_d.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=370, coloraxis_showscale=False,
                        xaxis_tickangle=-30, margin=dict(l=5, r=5, t=10, b=60))
    st.plotly_chart(fig_d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — POVERTY DISTRIBUTION (Matplotlib)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📉 Poverty Distribution":
    st.markdown('<div class="section-header">Distribution of Poverty Headcount — 2015-16 vs 2019-21</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(mpi_df["headcount_2015_16"], color="#e74c3c", kde=True, alpha=0.55, label="2015-16", bins=15, ax=ax)
    sns.histplot(mpi_df["headcount_2019_21"], color="#2ecc71", kde=True, alpha=0.55, label="2019-21", bins=15, ax=ax)
    ax.set_title("Distribution of State-level Poverty Headcount Rates", fontsize=13)
    ax.set_xlabel("Poverty Headcount (%)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    col1.metric("Mean 2015-16", f"{mpi_df['headcount_2015_16'].mean():.2f}%")
    col1.metric("Median 2015-16", f"{mpi_df['headcount_2015_16'].median():.2f}%")
    col2.metric("Mean 2019-21", f"{mpi_df['headcount_2019_21'].mean():.2f}%",
                delta=f"{mpi_df['headcount_2019_21'].mean() - mpi_df['headcount_2015_16'].mean():.2f}%")
    col2.metric("Median 2019-21", f"{mpi_df['headcount_2019_21'].median():.2f}%",
                delta=f"{mpi_df['headcount_2019_21'].median() - mpi_df['headcount_2015_16'].median():.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — STATE RANKINGS (Matplotlib)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔝 State Rankings":
    tab1, tab2 = st.tabs(["Top/Bottom States (2019-21)", "Poverty Reduction"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        top10 = mpi_df.nlargest(10, "headcount_2019_21").sort_values("headcount_2019_21")
        bot10 = mpi_df.nsmallest(10, "headcount_2019_21").sort_values("headcount_2019_21", ascending=False)

        axes[0].barh(top10["state_ut"], top10["headcount_2019_21"], color="#e74c3c", edgecolor="#0f172a")
        axes[0].set_title("🔴 Highest MPI Headcount (2019-21)", fontsize=12)
        axes[0].set_xlabel("Poverty Headcount (%)")

        axes[1].barh(bot10["state_ut"], bot10["headcount_2019_21"], color="#2ecc71", edgecolor="#0f172a")
        axes[1].set_title("🟢 Lowest MPI Headcount (2019-21)", fontsize=12)
        axes[1].set_xlabel("Poverty Headcount (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        mpi_sorted = mpi_df.sort_values("change_pct")
        colors_bar = ["#e74c3c" if x < -10 else "#f39c12" if x < -5 else "#2ecc71"
                      for x in mpi_sorted["change_pct"]]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(mpi_sorted["state_ut"], mpi_sorted["change_pct"],
                color=colors_bar, edgecolor="#0f172a", height=0.7)
        ax.axvline(0, color="#e2e8f0", linewidth=0.8)
        ax.set_title("Change in MPI Headcount: 2015-16 → 2019-21\n(Negative = Improvement)", fontsize=13)
        ax.set_xlabel("Change (Percentage Points)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best  = mpi_df.loc[mpi_df["change_pct"].idxmin()]
        worst = mpi_df.loc[mpi_df["change_pct"].idxmax()]
        st.success(f"🏆 Best performer: **{best['state_ut']}** ({best['change_pct']:.2f} pp)")
        st.warning(f"⚠️ Worst performer: **{worst['state_ut']}** (+{worst['change_pct']:.2f} pp)")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MPI vs HDI (Matplotlib)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔗 MPI vs HDI":
    st.markdown('<div class="section-header">Correlation: MPI Headcount vs HDI (2019)</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(merged["hdi_2019"], merged["headcount_2019_21"],
                         c=merged["headcount_2019_21"], cmap="RdYlGn_r",
                         s=merged["headcount_2019_21"] * 6 + 30, alpha=0.85, edgecolors="#0f172a", linewidth=0.5)
    for _, row in merged.iterrows():
        ax.annotate(row["state_ut"],
                    (row["hdi_2019"], row["headcount_2019_21"]),
                    fontsize=6, ha="left", va="bottom", color="#94a3b8")
    plt.colorbar(scatter, ax=ax, label="MPI Headcount (%)")
    ax.set_title("MPI Headcount (2019-21) vs Subnational HDI (2019)", fontsize=13)
    ax.set_xlabel("HDI 2019"); ax.set_ylabel("MPI Headcount (%)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    corr = merged["hdi_2019"].corr(merged["headcount_2019_21"])
    st.metric("Pearson Correlation (HDI vs MPI Headcount)", f"{corr:.3f}")
    st.markdown(f"""<div class="insight-box">
    A strong negative correlation of <b>{corr:.3f}</b> confirms that higher HDI is strongly associated with lower poverty rates.
    States like Kerala and Goa cluster at high HDI / low poverty, while Bihar and Jharkhand occupy the opposite end.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Correlation Heatmap")
    num_df = model_df[["headcount_2019_21", "change_pct", "hdi_2019", "hdi_2023"]].copy()
    num_df.columns = ["MPI_2019_21", "MPI_Change", "HDI_2019", "HDI_2023"]
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap — MPI & HDI Indicators")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ML MODELS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    tab1, tab2 = st.tabs(["Logistic Regression", "K-Means Clustering"])

    with tab1:
        st.markdown('<div class="section-header">High vs Low Poverty Classification</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC-ROC", f"{auc:.4f}")
        col2.metric("Precision (High Poverty)", f"{report['High Poverty']['precision']:.2f}")
        col3.metric("Recall (High Poverty)",    f"{report['High Poverty']['recall']:.2f}")

        st.markdown(f"""<div class="insight-box">
        Median poverty threshold: <b>{median_pov:.2f}%</b><br>
        Features used: HDI 2019, HDI 2023, Change in MPI headcount
        </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(
            lr_model, X_test, y_test,
            display_labels=["Low Poverty", "High Poverty"],
            cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix — Logistic Regression")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Classification Report")
        rdf = pd.DataFrame(report).T.round(3)
        st.dataframe(rdf.style.background_gradient(cmap="RdYlGn", subset=["precision", "recall", "f1-score"]))

    with tab2:
        st.markdown('<div class="section-header">K-Means State Development Segments</div>', unsafe_allow_html=True)

        colors_k = ["#2ecc71", "#f39c12", "#e74c3c"]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i in range(3):
            sub = model_df[model_df["Cluster"] == i]
            ax.scatter(sub["hdi_2019"], sub["headcount_2019_21"],
                       label=f"Cluster {i}", alpha=0.85, s=80, color=colors_k[i], edgecolors="#0f172a")
            for _, row in sub.iterrows():
                ax.annotate(row["state_ut"],
                            (row["hdi_2019"], row["headcount_2019_21"]),
                            fontsize=6.5, ha="left", color="#cbd5e1")
        ax.set_xlabel("HDI 2019"); ax.set_ylabel("MPI Headcount 2019-21 (%)")
        ax.set_title("State Development Segments (K-Means, k=3)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Cluster Summary")
        st.dataframe(cluster_summary.style.background_gradient(cmap="RdYlGn_r", subset=["headcount_2019_21"]))

        st.markdown("#### Risk Segmentation")
        risk_df = model_df[["state_ut", "headcount_2019_21", "hdi_2019", "RiskSegment"]]\
                    .sort_values("headcount_2019_21", ascending=False)
        st.dataframe(
            risk_df.style.applymap(
                lambda v: "background-color:#7f1d1d; color:white" if v == "High Risk"
                          else ("background-color:#78350f; color:white" if v == "Medium Risk"
                                else "background-color:#14532d; color:white"),
                subset=["RiskSegment"]
            )
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 9 — TOPSIS RANKING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆 TOPSIS Ranking":
    st.markdown('<div class="section-header">TOPSIS — Multi-Criteria Development Ranking</div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    Weights: HDI 2023 (40%) | MPI Headcount (35%) | Change in MPI (25%)<br>
    Higher TOPSIS score = better overall development.
    </div>""", unsafe_allow_html=True)

    topsis_sorted = model_df.sort_values("TOPSIS_Score", ascending=True)
    bar_colors = ["#2ecc71" if v >= 0.6 else "#f39c12" if v >= 0.4 else "#e74c3c"
                  for v in topsis_sorted["TOPSIS_Score"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(topsis_sorted["state_ut"], topsis_sorted["TOPSIS_Score"],
            color=bar_colors, edgecolor="#0f172a", height=0.7)
    ax.axvline(0.5, color="#e2e8f0", linestyle="--", linewidth=0.8, label="Midpoint 0.5")
    ax.set_title("TOPSIS Development Score by State/UT\n(Higher = Better Developed)", fontsize=13)
    ax.set_xlabel("TOPSIS Score")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Top 10 Best Developed")
        top10 = model_df.nlargest(10, "TOPSIS_Score")[
            ["state_ut", "hdi_2023", "headcount_2019_21", "TOPSIS_Score"]].round(3)
        st.dataframe(top10, hide_index=True)
    with col2:
        st.markdown("#### 🔻 Top 10 Least Developed")
        bot10 = model_df.nsmallest(10, "TOPSIS_Score")[
            ["state_ut", "hdi_2023", "headcount_2019_21", "TOPSIS_Score"]].round(3)
        st.dataframe(bot10, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 10 — KEY FINDINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Key Findings":
    st.markdown('<div class="section-header">Key Findings Summary</div>', unsafe_allow_html=True)

    corr_val = merged["hdi_2019"].corr(merged["headcount_2019_21"])
    best_f   = mpi_df.loc[mpi_df["change_pct"].idxmin()]
    worst_f  = mpi_df.loc[mpi_df["change_pct"].idxmax()]

    findings = [
        ("1. National Average", f"Average MPI headcount (2019-21): **{mpi_df['headcount_2019_21'].mean():.2f}%**"),
        ("2. Most Impoverished (Top 3)",
         " | ".join(f"{r['state_ut']} ({r['headcount_2019_21']:.1f}%)"
                    for _, r in mpi_df.nlargest(3, "headcount_2019_21").iterrows())),
        ("3. Least Impoverished (Top 3)",
         " | ".join(f"{r['state_ut']} ({r['headcount_2019_21']:.1f}%)"
                    for _, r in mpi_df.nsmallest(3, "headcount_2019_21").iterrows())),
        ("4. Best Poverty Reducer", f"**{best_f['state_ut']}** — {best_f['change_pct']:.2f} pp"),
        ("5. Worst Performer", f"**{worst_f['state_ut']}** — +{worst_f['change_pct']:.2f} pp"),
        ("6. HDI–MPI Correlation", f"**{corr_val:.3f}** — strong negative (higher HDI → lower poverty)"),
        ("7. ML Model AUC-ROC", f"Logistic Regression AUC: **{auc:.4f}**"),
        ("8. Risk Segment Distribution",
         " | ".join(f"{seg}: {cnt}" for seg, cnt in model_df["RiskSegment"].value_counts().items())),
        ("9. Top TOPSIS State",
         f"**{model_df.loc[model_df['TOPSIS_Score'].idxmax(), 'state_ut']}** "
         f"(score: {model_df['TOPSIS_Score'].max():.3f})"),
        ("10. Lowest TOPSIS State",
         f"**{model_df.loc[model_df['TOPSIS_Score'].idxmin(), 'state_ut']}** "
         f"(score: {model_df['TOPSIS_Score'].min():.3f})"),
    ]

    for title, body in findings:
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")
    st.markdown("#### 💾 Download Trained Models")
    col1, col2, col3 = st.columns(3)

    for col, obj, fname in [
        (col1, lr_model, "model_lr_mpi.pkl"),
        (col2, scaler,   "scaler_mpi.pkl"),
        (col3, kmeans,   "model_kmeans_mpi.pkl"),
    ]:
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        buf.seek(0)
        col.download_button(f"⬇ {fname}", buf, file_name=fname, mime="application/octet-stream")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 11 — DATA TABLE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Table":
    banner("📋 Data Table", "Browse and download the underlying datasets")

    tab1, tab2 = st.tabs(["📌 NITI Aayog MPI Data", "📌 GDL Subnational HDI Data"])

    with tab1:
        st.markdown(f"**{len(filtered)} records** shown after sidebar filters")
        cols = ["state_ut", "headcount_2019_21_pct", "headcount_2015_16_pct",
                "change_pct_points", "improvement_rate", "category"]
        st.dataframe(filtered[cols].reset_index(drop=True),
                     use_container_width=True, height=480)
        st.download_button("⬇️ Download MPI Data (CSV)",
                           filtered[cols].to_csv(index=False).encode(),
                           "mpi_data.csv", "text/csv")

    with tab2:
        st.markdown(f"**{len(hdi_long)} records** (all states · all years)")
        st.dataframe(hdi_long.reset_index(drop=True),
                     use_container_width=True, height=480)
        st.download_button("⬇️ Download HDI Data (CSV)",
                           hdi_long.to_csv(index=False).encode(),
                           "hdi_data.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;font-size:0.78rem;'>"
    "Sources: NITI Aayog National MPI 2023 · Global Data Lab SHDI India · "
    "Federal Bank TSM Centre of Excellence &nbsp;|&nbsp; ABA Final Project</p>",
    unsafe_allow_html=True,
)
