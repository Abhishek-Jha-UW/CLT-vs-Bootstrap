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
# Configuration & Theme
# -----------------------------
st.set_page_config(page_title="Inference Arena", layout="wide")

COLOR_DATA = "#4C78A8"       # Blue
COLOR_CLT = "#72B7B2"        # Teal
COLOR_BOOTSTRAP = "#F58518"  # Orange

st.title("⚔️ Inference Arena: CLT vs Bootstrap")
st.caption("Universal Statistical Showdown: Classical Frequentist vs. Empirical Resampling")

# -----------------------------
# Sidebar Configuration
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # NEW: Statistic Selector
    stat_label = st.selectbox("Estimate Population Statistic:", 
                              ["Mean", "Median", "Std Dev", "75th Percentile"])
    
    stat_map = {
        "Mean": np.mean,
        "Median": np.median,
        "Std Dev": np.std,
        "75th Percentile": lambda x, axis=None: np.percentile(x, 75, axis=axis)
    }
    selected_func = stat_map[stat_label]

    st.divider()
    
    def get_template_csv():
        df = pd.DataFrame({"value": [10, 12, 15, 18, 20, 22, 25, 30, 10, 15, 45, 50]})
        return df.to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download Template", data=get_template_csv(), file_name="template.csv", mime="text/csv")
    
    st.divider()
    unit = st.text_input("Unit (e.g., kg, meters)", "units")
    confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])
    ci_label = f"{int(confidence*100)}% CI"

# -----------------------------
# Data Ingestion
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🧪 Use Sample Data"):
        st.session_state.df = pd.DataFrame({"value": np.random.gamma(shape=2, scale=10, size=100)})
with col_btn2:
    uploaded_file = st.file_uploader("Or Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

# -----------------------------
# Analysis Arena
# -----------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    # Ensure we have a column to work with
    target_col = "value" if "value" in df.columns else df.columns[0]
    data = df[target_col].dropna().values
    
    diag = get_diagnostics(data)

    # 1. Diagnostics Header
    st.write("## 📊 Data Diagnostics")
    
    if abs(diag["skewness"]) > 1:
        st.warning(f"⚠️ High Skewness ({diag['skewness']:.2f}) detected. Bootstrap is preferred.")
    elif diag["n"] < 30:
        st.info(f"ℹ️ Small sample size (n={diag['n']}). CLT may be unreliable.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("n", diag["n"])
    m2.metric("Mean", f"{diag['mean']:.2f}")
    m3.metric("Median", f"{diag['median']:.2f}")
    m4.metric("Skewness", f"{diag['skewness']:.2f}")
    m5.metric("Std Dev", f"{diag['std']:.2f}")

    fig_raw = px.histogram(data, nbins=25, title="Raw Data Distribution", color_discrete_sequence=[COLOR_DATA])
    fig_raw.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_raw, use_container_width=True)

    # 2. Computations
    # Bootstrap works for everything
    boot_res, boot_samples = bootstrap_ci(data, selected_func, confidence, n_bootstrap)
    boot_width = boot_res[1] - boot_res[0]

    # 3. The Battle Arena Layout
    st.write(f"## ⚔️ {ci_label} Battle for Population {stat_label}")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"CLT (Theoretical {stat_label})")
        if stat_label == "Mean":
            clt_res = clt_ci(data, confidence)
            clt_width = clt_res[1] - clt_res[0]
            
            # Simulated Sampling Distribution
            clt_sim = np.random.normal(diag["mean"], diag["std"]/np.sqrt(diag["n"]), 5000)
            
            fig_clt = px.histogram(clt_sim, nbins=25, color_discrete_sequence=[COLOR_CLT])
            fig_clt.add_vline(x=clt_res[0], line_dash="dash", line_color="red", annotation_text="Lower")
            fig_clt.add_vline(x=clt_res[1], line_dash="dash", line_color="red", annotation_text="Upper")
            fig_clt.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_clt, use_container_width=True)
            
            st.info(f"**{ci_label} (CLT):** {clt_res[0]:.2f} to {clt_res[1]:.2f} {unit}")
            st.write(f"**Interval Width:** {clt_width:.4f}")
        else:
            st.error(f"❌ Theoretical formula for '{stat_label}' is not supported in this arena.")
            st.write("CFF/CLT formulas are mathematically restrictive. This highlights why Bootstrapping is so powerful for non-mean statistics!")

    with col_right:
        st.subheader(f"Bootstrap (Empirical {stat_label})")
        fig_boot = px.histogram(boot_samples, nbins=25, color_discrete_sequence=[COLOR_BOOTSTRAP])
        fig_boot.add_vline(x=boot_res[0], line_dash="dash", line_color="black", annotation_text="Lower")
        fig_boot.add_vline(x=boot_res[1], line_dash="dash", line_color="black", annotation_text="Upper")
        fig_boot.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_boot, use_container_width=True)
        
        st.success(f"**{ci_label} (Bootstrap):** {boot_res[0]:.2f} to {boot_res[1]:.2f} {unit}")
        st.write(f"**Interval Width:** {boot_width:.4f}")

    # 4. Final Verdict & AI
    st.divider()
    v_col1, v_col2 = st.columns([1, 2])
    
    with v_col1:
        st.write("### 🏁 Statistical Verdict")
        verdict_text = compare_methods(data, stat_label)
        st.success(verdict_text)

    with v_col2:
        st.write("### 🧠 AI Referee Insights")
        if st.button("Generate AI Insights"):
            summary = {
                **diag, 
                "unit": unit, 
                "stat_targeted": stat_label,
                "bootstrap_ci": boot_res,
                "confidence_level": ci_label,
                "bootstrap_width": boot_width
            }
            if stat_label == "Mean":
                summary["clt_ci"] = clt_res
                summary["clt_width"] = clt_width
                
            with st.spinner("Analyzing stats..."):
                insights = get_ai_insights(summary)
                st.markdown(f"> {insights}")

else:
    st.info("Please upload a CSV file or use sample data to begin the battle.")
