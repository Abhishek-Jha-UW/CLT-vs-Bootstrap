import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from model import (
    get_diagnostics,
    clt_ci,
    bootstrap_ci,
    compare_methods,
    get_ai_insights
)

# -----------------------------
# Setup & Visual Identity
# -----------------------------
st.set_page_config(page_title="Inference Arena", layout="wide")

COLOR_DATA = "#4C78A8"       # Blue
COLOR_CLT = "#72B7B2"        # Teal
COLOR_BOOTSTRAP = "#F58518"  # Orange

st.title("⚔️ Inference Arena: CLT vs Bootstrap")
st.caption("Universal Statistical Showdown")

# -----------------------------
# Sidebar Configuration
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Statistic Selector
    stat_label = st.selectbox("What do you want to estimate?", 
                              ["Mean", "Median", "Std Dev", "75th Percentile"])
    
    stat_map = {
        "Mean": np.mean,
        "Median": np.median,
        "Std Dev": np.std,
        "75th Percentile": lambda x, axis=None: np.percentile(x, 75, axis=axis)
    }
    selected_func = stat_map[stat_label]

    st.divider()
    unit = st.text_input("Unit (e.g., kg, meters)", "units")
    confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])
    ci_label = f"{int(confidence*100)}% CI"

# -----------------------------
# Data Loading
# -----------------------------
if "df" not in st.session_state: st.session_state.df = None

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🧪 Sample Data"):
        st.session_state.df = pd.DataFrame({"value": np.random.gamma(2, 10, 100)})
with c2:
    up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if up: st.session_state.df = pd.read_csv(up)

# -----------------------------
# Execution Arena
# -----------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    data = df.iloc[:, 0].dropna().values # Take first column automatically
    diag = get_diagnostics(data)

    st.write("## 📊 Data Diagnostics")
    # Alert Logic
    if abs(diag["skewness"]) > 1:
        st.warning(f"⚠️ High Skewness ({diag['skewness']:.2f}). Bootstrap is preferred.")
    elif diag["n"] < 30:
        st.info(f"ℹ️ Small Sample (n={diag['n']}). CLT may be unreliable.")

    m = st.columns(5)
    m[0].metric("n", diag["n"])
    m[1].metric("Mean", f"{diag['mean']:.2f}")
    m[2].metric("Median", f"{diag['median']:.2f}")
    m[3].metric("Skewness", f"{diag['skewness']:.2f}")
    m[4].metric("Std Dev", f"{diag['std']:.2f}")

    # Calculations
    boot_res, boot_samples = bootstrap_ci(data, selected_func, confidence, n_bootstrap)
    boot_width = boot_res[1] - boot_res[0]

    # Display Arena
    st.write(f"## ⚔️ {ci_label} Battle for Population {stat_label}")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"CLT (Theoretical {stat_label})")
        if stat_label == "Mean":
            clt_res = clt_ci(data, confidence)
            clt_width = clt_res[1] - clt_res[0]
            
            # Simulate CLT Sampling Distribution
            clt_sim = np.random.normal(diag["mean"], diag["std"]/np.sqrt(diag["n"]), 5000)
            fig_clt = px.histogram(clt_sim, nbins=25, color_discrete_sequence=[COLOR_CLT])
            fig_clt.add_vline(x=clt_res[0], line_dash="dash", line_color="red")
            fig_clt.add_vline(x=clt_res[1], line_dash="dash", line_color="red")
            st.plotly_chart(fig_clt, use_container_width=True)
            st.info(f"**{ci_label}:** {clt_res[0]:.2f} to {clt_res[1]:.2f} {unit}")
            st.write(f"**Interval Width:** {clt_width:.4f}")
        else:
            st.error(f"❌ CFF formula not supported for '{stat_label}'.")
            st.write("Theoretical formulas for non-mean statistics require specific population assumptions (e.g. Chi-Square for Std Dev).")

    with col_right:
        st.subheader(f"Bootstrap (Empirical {stat_label})")
        fig_boot = px.histogram(boot_samples, nbins=25, color_discrete_sequence=[COLOR_BOOTSTRAP])
        fig_boot.add_vline(x=boot_res[0], line_dash="dash", line_color="black")
        fig_boot.add_vline(x=boot_res[1], line_dash="dash", line_color="black")
        st.plotly_chart(fig_boot, use_container_width=True)
        st.success(f"**{ci_label}:** {boot_res[0]:.2f} to {boot_res[1]:.2f} {unit}")
        st.write(f"**Interval Width:** {boot_width:.4f}")

    # AI Verdict
    st.divider()
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        st.write("### 🏁 Verdict")
        st.write(compare_methods(data, stat_label))
    with v_col2:
        if st.button("Generate AI Insights"):
            summary = {**diag, "unit": unit, "stat": stat_label, "boot_ci": boot_res}
            if stat_label == "Mean": summary["clt_ci"] = clt_res
            st.markdown(get_ai_insights(summary))
