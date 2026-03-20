import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import get_diagnostics, clt_ci, bootstrap_ci, compare_methods, get_ai_insights

# UI Settings
st.set_page_config(page_title="Inference Arena", layout="wide")
COLOR_DATA, COLOR_CLT, COLOR_BOOT = "#4C78A8", "#72B7B2", "#F58518"

st.title("⚔️ Inference Arena: CLT vs Bootstrap")
st.caption("Universal Statistical Showdown: Formula-based vs. Computational Inference")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    stat_label = st.selectbox("Estimate Population Statistic:", ["Mean", "Median", "Std Dev", "75th Percentile"])
    stat_map = {"Mean": np.mean, "Median": np.median, "Std Dev": np.std, 
                "75th Percentile": lambda x, axis=None: np.percentile(x, 75, axis=axis)}
    
    st.divider()
    unit = st.text_input("Unit (e.g., kg, meters)", "units")
    confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])
    ci_label = f"{int(confidence*100)}% CI"

# Data Logic
if "df" not in st.session_state: st.session_state.df = None
c1, c2 = st.columns([1, 4])
with c1: 
    if st.button("🧪 Sample Data"): st.session_state.df = pd.DataFrame({"value": np.random.gamma(2, 10, 100)})
with c2: 
    up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if up: st.session_state.df = pd.read_csv(up)

if st.session_state.df is not None:
    df = st.session_state.df
    target_col = "value" if "value" in df.columns else df.columns[0]
    data = df[target_col].dropna().values
    diag = get_diagnostics(data)

    # 1. Diagnostics Display
    st.write("## 📊 Data Diagnostics")
    if abs(diag["skewness"]) > 1: st.warning(f"⚠️ High Skewness ({diag['skewness']:.2f}) detected.")
    elif diag["n"] < 30: st.info(f"ℹ️ Small sample size (n={diag['n']}).")

    m = st.columns(5)
    m[0].metric("n", diag["n"]); m[1].metric("Mean", f"{diag['mean']:.2f}"); m[2].metric("Median", f"{diag['median']:.2f}")
    m[3].metric("Skewness", f"{diag['skewness']:.2f}"); m[4].metric("Kurtosis", f"{diag['kurtosis']:.2f}")

    # 2. Plot Raw Data
    fig_raw = px.histogram(data, nbins=25, title="Raw Data Distribution", color_discrete_sequence=[COLOR_DATA])
    fig_raw.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_raw, use_container_width=True)

    # 3. Run Calculations
    boot_res, boot_samples = bootstrap_ci(data, stat_map[stat_label], confidence, n_bootstrap)
    boot_width = boot_res[1] - boot_res[0]

    # 4. The Arena
    st.write(f"## ⚔️ {ci_label} Battle for {stat_label}")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("CLT (Theoretical)")
        if stat_label == "Mean":
            clt_res = clt_ci(data, confidence)
            clt_width = clt_res[1] - clt_res[0]
            clt_sim = np.random.normal(diag["mean"], diag["std"]/np.sqrt(diag["n"]), 5000)
            
            fig_clt = px.histogram(clt_sim, nbins=25, color_discrete_sequence=[COLOR_CLT])
            fig_clt.add_vline(x=clt_res[0], line_dash="dash", line_color="red", annotation_text="Lower")
            fig_clt.add_vline(x=clt_res[1], line_dash="dash", line_color="red", annotation_text="Upper")
            st.plotly_chart(fig_clt, use_container_width=True)
            st.info(f"**{ci_label}:** {clt_res[0]:.2f} to {clt_res[1]:.2f} {unit}")
            st.write(f"**Width:** {clt_width:.4f}")
        else:
            st.error(f"❌ CLT not supported for {stat_label}")

    with col_right:
        st.subheader("Bootstrap (Empirical)")
        fig_boot = px.histogram(boot_samples, nbins=25, color_discrete_sequence=[COLOR_BOOT])
        fig_boot.add_vline(x=boot_res[0], line_dash="dash", line_color="black", annotation_text="Lower")
        fig_boot.add_vline(x=boot_res[1], line_dash="dash", line_color="black", annotation_text="Upper")
        st.plotly_chart(fig_boot, use_container_width=True)
        st.success(f"**{ci_label}:** {boot_res[0]:.2f} to {boot_res[1]:.2f} {unit}")
        st.write(f"**Width:** {boot_width:.4f}")

    # 5. AI Referee
    st.divider()
    st.write("### 🧠 AI Referee Insights")
    if st.button("Generate AI Insights"):
        # Improved Summary Payload
        summary = {**diag, "unit": unit, "stat": stat_label, "boot_ci": boot_res, "boot_width": boot_width}
        if stat_label == "Mean": 
            summary["clt_ci"] = clt_res
            summary["clt_width"] = clt_width
            
        with st.spinner("Analyzing..."):
            st.markdown(get_ai_insights(summary))
