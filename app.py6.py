import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
import pickle
import os

# ──────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────
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
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; }
    .insight-box {
        background: #e3f2fd; border-left: 4px solid #1565c0; border-radius: 6px;
        padding: 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────
DATA_PATH = "data/mpi_project_data_sources.xlsx"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "mpi_project_data_sources.xlsx"   # fallback: same directory

@st.cache_data
def load_data():
    mpi = pd.read_excel(DATA_PATH, sheet_name="NITI_MPI_States")
    mpi = mpi.rename(columns={
        "headcount_2019_21_pct": "headcount_2019_21",
        "headcount_2015_16_pct": "headcount_2015_16",
        "change_pct_points":     "change_pct",
    })

    hdi = pd.read_excel(DATA_PATH, sheet_name="GDL_SHDI_States")
    hdi = hdi[hdi["state_ut"] != "Total"].copy()

    hdi_long = hdi.melt(
        id_vars=["state_ut"],
        value_vars=["2019", "2020", "2021", "2022", "2023"],
        var_name="year", value_name="hdi_value",
    )
    hdi_long["year"] = hdi_long["year"].astype(int)

    # Poverty category
    bins   = [0, 5, 15, 30, 100]
    labels = ["Very Low (<5%)", "Low (5–15%)", "Moderate (15–30%)", "High (>30%)"]
    mpi["poverty_category"] = pd.cut(mpi["headcount_2019_21"], bins=bins, labels=labels)
    mpi["improvement"]      = mpi["change_pct"].abs()
    mpi["improvement_rate"] = (
        (mpi["headcount_2015_16"] - mpi["headcount_2019_21"])
        / mpi["headcount_2015_16"] * 100
    ).round(1)

    return mpi, hdi, hdi_long


@st.cache_data
def build_model_df(_mpi, _hdi):
    hdi_states = _hdi[_hdi["state_ut"] != "Total"].copy()
    merged = pd.merge(
        _mpi[["state_ut", "headcount_2019_21", "change_pct"]],
        hdi_states[["state_ut", "2019", "2023"]].rename(
            columns={"2019": "hdi_2019", "2023": "hdi_2023"}
        ),
        on="state_ut", how="inner",
    )

    # ── Logistic Regression ──
    median_pov = merged["headcount_2019_21"].median()
    merged["high_poverty"] = (merged["headcount_2019_21"] > median_pov).astype(int)

    features = ["hdi_2019", "hdi_2023", "change_pct"]
    X = merged[features].values
    y = merged["high_poverty"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Low Poverty", "High Poverty"],
                                   output_dict=True)

    # ── K-Means Clustering ──
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    merged["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = merged.groupby("Cluster")[
        ["headcount_2019_21", "change_pct", "hdi_2019", "hdi_2023"]
    ].mean().round(3)

    risk_map   = cluster_summary["headcount_2019_21"].rank().astype(int).to_dict()
    label_map  = {k: ["Low Risk", "Medium Risk", "High Risk"][v - 1] for k, v in risk_map.items()}
    merged["RiskSegment"] = merged["Cluster"].map(label_map)

    # ── TOPSIS ──
    topsis_data = merged[["hdi_2023", "headcount_2019_21", "change_pct"]].values.astype(float)
    norm        = topsis_data / np.sqrt((topsis_data ** 2).sum(axis=0))
    weights     = np.array([0.40, 0.35, 0.25])
    weighted    = norm * weights
    benefit     = [True, False, False]

    ideal_best  = np.where(benefit, weighted.max(0), weighted.min(0))
    ideal_worst = np.where(benefit, weighted.min(0), weighted.max(0))
    d_best      = np.sqrt(((weighted - ideal_best)  ** 2).sum(axis=1))
    d_worst     = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    merged["TOPSIS_Score"] = (d_worst / (d_best + d_worst)).round(4)

    return merged, lr, scaler, auc, cm, report, cluster_summary, median_pov


mpi_df, hdi_df, hdi_long = load_data()
model_df, lr_model, scaler, auc_score, conf_matrix, clf_report, cluster_summary, median_pov = build_model_df(mpi_df, hdi_df)

# ──────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
        width=100,
    )
    st.markdown("## 🗂️ Navigate")
    page = st.radio("", [
        "📊 Overview",
        "🗺️ State Explorer",
        "📈 HDI Trends",
        "🤖 ML Analysis",
        "🏅 TOPSIS Ranking",
        "🔮 MPI Predictor",
        "📋 Data Table",
    ])
    st.markdown("---")
    categories = list(mpi_df["poverty_category"].cat.categories)
    sel_cats   = st.multiselect("Poverty Categories", categories, default=categories)
    rng = st.slider(
        "Poverty Rate Range (%)",
        float(mpi_df["headcount_2019_21"].min()),
        float(mpi_df["headcount_2019_21"].max()),
        (float(mpi_df["headcount_2019_21"].min()), float(mpi_df["headcount_2019_21"].max())),
    )

fmpi = mpi_df[
    mpi_df["poverty_category"].isin(sel_cats) &
    mpi_df["headcount_2019_21"].between(*rng)
]

# ──────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇳 India MPI Dashboard</h1>
        <p>NITI Aayog MPI 2023 · Applied Business Analytics Final Project</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States / UTs", len(mpi_df))
    avg_now  = mpi_df["headcount_2019_21"].mean()
    avg_then = mpi_df["headcount_2015_16"].mean()
    c2.metric("Avg MPI 2019-21", f"{avg_now:.1f}%", delta=f"{avg_now - avg_then:.1f}%")
    best = mpi_df.loc[mpi_df["change_pct"].idxmin()]
    c3.metric("Best Improvement", best["state_ut"], delta=f"{best['change_pct']:.1f} pp")
    c4.metric("High Poverty States", int((mpi_df["headcount_2019_21"] >= 30).sum()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 Poverty States (2019-21)")
        top10 = fmpi.nlargest(10, "headcount_2019_21")
        fig = px.bar(
            top10, x="headcount_2019_21", y="state_ut", orientation="h",
            color="headcount_2019_21", color_continuous_scale="Reds",
            text="headcount_2019_21",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Poverty Reduction (2015-16 → 2019-21)")
        top_imp = fmpi.nlargest(10, "improvement")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="2015-16", x=top_imp["state_ut"],
                              y=top_imp["headcount_2015_16"], marker_color="#ef5350"))
        fig2.add_trace(go.Bar(name="2019-21", x=top_imp["state_ut"],
                              y=top_imp["headcount_2019_21"], marker_color="#42a5f5"))
        fig2.update_layout(barmode="group", height=400,
                           plot_bgcolor="white", xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Poverty Category Distribution")
    cat_counts = fmpi["poverty_category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig3 = px.pie(cat_counts, names="Category", values="Count",
                  color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────
# PAGE: STATE EXPLORER
# ──────────────────────────────────────────────────
elif page == "🗺️ State Explorer":
    st.title("🗺️ State Explorer")
    sel = st.selectbox("Select State / UT", sorted(mpi_df["state_ut"].tolist()))
    row = mpi_df[mpi_df["state_ut"] == sel].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("MPI 2019-21", f"{row['headcount_2019_21']:.2f}%")
    c2.metric("MPI 2015-16", f"{row['headcount_2015_16']:.2f}%")
    c3.metric("Change", f"{row['change_pct']:.2f} pp")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=row["headcount_2019_21"],
        delta={"reference": row["headcount_2015_16"]},
        title={"text": f"MPI Headcount – {sel}"},
        gauge={
            "axis": {"range": [0, 60]},
            "bar":  {"color": "#1565c0"},
            "steps": [
                {"range": [0,  5],  "color": "#c8e6c9"},
                {"range": [5,  15], "color": "#fff9c4"},
                {"range": [15, 30], "color": "#ffe0b2"},
                {"range": [30, 60], "color": "#ffcdd2"},
            ],
        },
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Risk segment & TOPSIS from model_df
    if sel in model_df["state_ut"].values:
        mrow = model_df[model_df["state_ut"] == sel].iloc[0]
        st.markdown(f"""
        <div class='insight-box'>
            📌 <b>Risk Segment:</b> {mrow['RiskSegment']} &nbsp;|&nbsp;
            🏅 <b>TOPSIS Score:</b> {mrow['TOPSIS_Score']:.4f} &nbsp;|&nbsp;
            🧩 <b>Cluster:</b> {int(mrow['Cluster'])}
        </div>""", unsafe_allow_html=True)

    # State vs national average
    fig4 = go.Figure()
    years_labels = ["2015-16", "2019-21"]
    fig4.add_trace(go.Scatter(x=years_labels,
                              y=[row["headcount_2015_16"], row["headcount_2019_21"]],
                              name=sel, mode="lines+markers", line=dict(color="#1565c0", width=3)))
    fig4.add_trace(go.Scatter(x=years_labels,
                              y=[mpi_df["headcount_2015_16"].mean(), mpi_df["headcount_2019_21"].mean()],
                              name="National Avg", mode="lines+markers",
                              line=dict(color="#e74c3c", width=2, dash="dash")))
    fig4.update_layout(title=f"{sel} vs National Average", height=350, plot_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────────
# PAGE: HDI TRENDS
# ──────────────────────────────────────────────────
elif page == "📈 HDI Trends":
    st.title("📈 HDI Trends 2019–2023")

    default_states = ["Bihar", "Kerala", "Uttar Pradesh", "Tamil Nadu", "Jharkhand", "Goa"]
    default_states = [s for s in default_states if s in hdi_long["state_ut"].unique()]
    states_sel = st.multiselect("Select States",
                                sorted(hdi_long["state_ut"].unique()),
                                default=default_states)
    fhdi = hdi_long[hdi_long["state_ut"].isin(states_sel)]
    fig = px.line(fhdi, x="year", y="hdi_value", color="state_ut", markers=True,
                  title="Subnational HDI Comparison (2019–2023)")
    fig.update_layout(height=450, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # MPI vs HDI scatter
    st.markdown("### MPI Headcount vs HDI 2019 — All States")
    if not model_df.empty:
        fig2 = px.scatter(model_df, x="hdi_2019", y="headcount_2019_21",
                          text="state_ut", color="headcount_2019_21",
                          color_continuous_scale="RdYlGn_r", size="headcount_2019_21",
                          size_max=30, title="MPI vs HDI (2019)")
        fig2.update_traces(textposition="top center", textfont_size=8)
        fig2.update_layout(height=500, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        corr = model_df["hdi_2019"].corr(model_df["headcount_2019_21"])
        st.markdown(f"<div class='insight-box'>📉 Pearson Correlation (HDI vs MPI): <b>{corr:.3f}</b> — strong negative relationship</div>",
                    unsafe_allow_html=True)

# ──────────────────────────────────────────────────
# PAGE: ML ANALYSIS
# ──────────────────────────────────────────────────
elif page == "🤖 ML Analysis":
    st.title("🤖 ML Analysis")

    tab1, tab2, tab3 = st.tabs(["Logistic Regression", "K-Means Clustering", "Correlation Heatmap"])

    with tab1:
        st.markdown("### Binary Classification — High vs Low Poverty")
        st.markdown(f"""
        <div class='insight-box'>
            🎯 AUC-ROC Score: <b>{auc_score:.4f}</b> &nbsp;|&nbsp;
            Threshold: headcount &gt; <b>{median_pov:.2f}%</b> = High Poverty
        </div>""", unsafe_allow_html=True)

        # Confusion matrix as heatmap
        cm_labels = ["Low Poverty", "High Poverty"]
        fig_cm = px.imshow(conf_matrix, text_auto=True,
                           x=cm_labels, y=cm_labels,
                           color_continuous_scale="Blues",
                           title="Confusion Matrix — Logistic Regression")
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report table
        cr_df = pd.DataFrame(clf_report).T.round(2).drop("support", axis=1, errors="ignore")
        st.dataframe(cr_df, use_container_width=True)

        # Coefficients
        features = ["HDI 2019", "HDI 2023", "MPI Change"]
        coef_df  = pd.DataFrame({"Feature": features, "Coefficient": lr_model.coef_[0]})
        fig_coef = px.bar(coef_df, x="Feature", y="Coefficient",
                          color="Coefficient", color_continuous_scale="RdYlGn",
                          title="Logistic Regression — Feature Coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)

    with tab2:
        st.markdown("### K-Means Clustering — State Development Segments")
        colors = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
        fig_cl = px.scatter(model_df, x="hdi_2019", y="headcount_2019_21",
                            color="RiskSegment", text="state_ut",
                            color_discrete_map=colors, size_max=20,
                            title="State Segments by Risk (K-Means, k=3)")
        fig_cl.update_traces(textposition="top center", textfont_size=7)
        fig_cl.update_layout(height=500, plot_bgcolor="white")
        st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown("#### Cluster Summary")
        st.dataframe(cluster_summary.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### Risk Segment Distribution")
        seg_counts = model_df["RiskSegment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(seg_counts, names="Segment", values="Count",
                         color="Segment", color_discrete_map=colors)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.markdown("### Correlation Heatmap — MPI & HDI Indicators")
        num_cols = model_df[["headcount_2019_21", "change_pct", "hdi_2019", "hdi_2023"]].copy()
        num_cols.columns = ["MPI 2019-21", "MPI Change", "HDI 2019", "HDI 2023"]
        corr_matrix = num_cols.corr().round(2)
        fig_heat = px.imshow(corr_matrix, text_auto=True,
                             color_continuous_scale="RdYlGn",
                             title="Correlation Heatmap")
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

# ──────────────────────────────────────────────────
# PAGE: TOPSIS RANKING
# ──────────────────────────────────────────────────
elif page == "🏅 TOPSIS Ranking":
    st.title("🏅 TOPSIS Development Ranking")
    st.markdown("""
    <div class='insight-box'>
        TOPSIS weights: <b>HDI 2023 (40%)</b> · <b>MPI Headcount (35%)</b> · <b>MPI Change (25%)</b><br>
        Higher score = Better developed state.
    </div>""", unsafe_allow_html=True)

    topsis_sorted = model_df.sort_values("TOPSIS_Score")
    bar_colors = [
        "#2ecc71" if v >= 0.6 else "#f39c12" if v >= 0.4 else "#e74c3c"
        for v in topsis_sorted["TOPSIS_Score"]
    ]
    fig = go.Figure(go.Bar(
        x=topsis_sorted["TOPSIS_Score"],
        y=topsis_sorted["state_ut"],
        orientation="h",
        marker_color=bar_colors,
        text=topsis_sorted["TOPSIS_Score"].round(3),
        textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="black")
    fig.update_layout(title="TOPSIS Score by State/UT",
                      height=600, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Top 5 Developed")
        st.dataframe(
            model_df.nlargest(5, "TOPSIS_Score")
            [["state_ut", "hdi_2023", "headcount_2019_21", "TOPSIS_Score"]]
            .rename(columns={"state_ut": "State", "hdi_2023": "HDI 2023",
                              "headcount_2019_21": "MPI %", "TOPSIS_Score": "Score"})
            .reset_index(drop=True),
            use_container_width=True,
        )
    with col2:
        st.markdown("#### ⚠️ Bottom 5 Developed")
        st.dataframe(
            model_df.nsmallest(5, "TOPSIS_Score")
            [["state_ut", "hdi_2023", "headcount_2019_21", "TOPSIS_Score"]]
            .rename(columns={"state_ut": "State", "hdi_2023": "HDI 2023",
                              "headcount_2019_21": "MPI %", "TOPSIS_Score": "Score"})
            .reset_index(drop=True),
            use_container_width=True,
        )

# ──────────────────────────────────────────────────
# PAGE: MPI PREDICTOR
# ──────────────────────────────────────────────────
elif page == "🔮 MPI Predictor":
    st.title("🔮 MPI Score Estimator")
    st.latex(r"MPI = H \times A")
    st.markdown("Adjust the deprivation levels across the 10 OPHI indicators to estimate an MPI score.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Education**")
        sy = st.slider("Years of Schooling Dep %", 0, 100, 40)
        sa = st.slider("School Attendance Dep %",  0, 100, 30)
    with c2:
        st.markdown("**🏥 Health**")
        nu = st.slider("Nutrition Dep %",        0, 100, 35)
        cm = st.slider("Child Mortality Dep %",  0, 100, 20)
    with c3:
        st.markdown("**🏠 Living Standards**")
        cf  = st.slider("Cooking Fuel Dep %",    0, 100, 50)
        san = st.slider("Sanitation Dep %",      0, 100, 45)
        dw  = st.slider("Drinking Water Dep %",  0, 100, 25)
        el  = st.slider("Electricity Dep %",     0, 100, 20)
        ho  = st.slider("Housing Dep %",         0, 100, 30)
        ast = st.slider("Assets Dep %",          0, 100, 35)

    weights = {"sy": 1/6, "sa": 1/6, "nu": 1/6, "cm": 1/6,
               "cf": 1/18, "san": 1/18, "dw": 1/18, "el": 1/18, "ho": 1/18, "ast": 1/18}
    vals    = {"sy": sy/100, "sa": sa/100, "nu": nu/100, "cm": cm/100,
               "cf": cf/100, "san": san/100, "dw": dw/100, "el": el/100, "ho": ho/100, "ast": ast/100}

    c_score = sum(weights[k] * vals[k] for k in weights)
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

    # Radar chart
    labels_radar = ["Schooling", "Attendance", "Nutrition", "Child Mort.",
                    "Cooking Fuel", "Sanitation", "Water", "Electricity", "Housing", "Assets"]
    values_radar = [sy, sa, nu, cm, cf, san, dw, el, ho, ast]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=labels_radar + [labels_radar[0]],
        fill="toself", fillcolor="rgba(21,101,192,0.2)",
        line=dict(color="#1565c0"),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Deprivation Profile",
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ──────────────────────────────────────────────────
# PAGE: DATA TABLE
# ──────────────────────────────────────────────────
elif page == "📋 Data Table":
    st.title("📋 Full Dataset")
    tab1, tab2, tab3 = st.tabs(["MPI Data", "HDI Data", "Model Output"])

    with tab1:
        show_cols = ["state_ut", "headcount_2019_21", "headcount_2015_16",
                     "change_pct", "improvement_rate", "poverty_category"]
        st.dataframe(fmpi[show_cols], use_container_width=True)
        st.download_button("⬇️ Download MPI CSV",
                           fmpi.to_csv(index=False).encode(), "mpi_data.csv", "text/csv")

    with tab2:
        st.dataframe(hdi_long, use_container_width=True)
        st.download_button("⬇️ Download HDI CSV",
                           hdi_long.to_csv(index=False).encode(), "hdi_data.csv", "text/csv")

    with tab3:
        out_cols = ["state_ut", "headcount_2019_21", "hdi_2019", "hdi_2023",
                    "change_pct", "RiskSegment", "TOPSIS_Score", "high_poverty"]
        st.dataframe(model_df[out_cols], use_container_width=True)
        st.download_button("⬇️ Download Model Output CSV",
                           model_df[out_cols].to_csv(index=False).encode(),
                           "model_output.csv", "text/csv")

# ──────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Sources: NITI Aayog MPI 2023 · Global Data Lab SHDI  |  "
    "ABA Final Project · Federal Bank TSM Centre of Excellence"
)
