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

COLOR_DATA = "#4C78A8"       
COLOR_CLT = "#72B7B2"        
COLOR_BOOTSTRAP = "#F58518"  

st.title("⚔️ Inference Arena: CLT vs Bootstrap")
st.caption("Compare Classical vs Empirical Inference with AI-powered insights")

# -----------------------------
# Sidebar & Template
# -----------------------------
with st.sidebar:
    st.header("Project Settings")
    
    def get_template_csv():
        # Added a slightly skewed default to make the battle interesting
        df = pd.DataFrame({"value": [10, 12, 15, 18, 20, 22, 25, 30, 10, 15, 45, 50]})
        return df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Template",
        data=get_template_csv(),
        file_name="template.csv",
        mime="text/csv"
    )
    
    st.divider()
    unit = st.text_input("Enter Unit (e.g., kg, meters)", "units")
    confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])
    
    ci_label = f"{int(confidence*100)}% CI"

# -----------------------------
# Data Ingestion Logic
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🧪 Use Sample Data"):
        st.session_state.df = pd.DataFrame({
            "value": np.random.gamma(shape=2, scale=10, size=100)
        })
with col_btn2:
    uploaded_file = st.file_uploader("Or Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

# -----------------------------
# Main Analysis Arena
# -----------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    if "value" not in df.columns:
        st.error("CSV must contain a column named 'value'")
        st.stop()

    data = df["value"].dropna().values
    diag = get_diagnostics(data)

    # 1. Diagnostics & Logic Alerts
    st.write("## 📊 Data Diagnostics")
    
    if abs(diag["skewness"]) > 1:
        st.warning(f"⚠️ High Skewness ({diag['skewness']:.2f}) detected. Bootstrap is likely more robust.")
    elif diag["n"] < 30:
        st.info(f"ℹ️ Small sample size (n={diag['n']}). CLT assumptions might be weak.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("n", diag["n"])
    m2.metric("Mean", f"{diag['mean']:.2f}")
    m3.metric("Median", f"{diag['median']:.2f}")
    m4.metric("Skewness", f"{diag['skewness']:.2f}")
    m5.metric("Std Dev", f"{diag['std']:.2f}")

    fig_raw = px.histogram(data, nbins=25, title="Raw Data Distribution", color_discrete_sequence=[COLOR_DATA])
    fig_raw.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_raw, use_container_width=True)

    # 2. Run Calculations
    clt_result = clt_ci(data, confidence)
    boot_result, boot_samples = bootstrap_ci(data, np.mean, confidence, n_bootstrap)
    
    clt_width = clt_result[1] - clt_result[0]
    boot_width = boot_result[1] - boot_result[0]

    # 3. Simulated CLT Distribution
    clt_samples = np.random.normal(
        loc=np.mean(data),
        scale=np.std(data, ddof=1) / np.sqrt(len(data)),
        size=5000
    )

    # 4. The Battle Arena
    st.write(f"## ⚔️ {ci_label} Battle: CLT vs Bootstrap")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("CLT (Theoretical)")
        fig_clt = px.histogram(clt_samples, nbins=25, color_discrete_sequence=[COLOR_CLT])
        # Add Vertical Lines for CI
        fig_clt.add_vline(x=clt_result[0], line_dash="dash", line_color="red", annotation_text="Lower")
        fig_clt.add_vline(x=clt_result[1], line_dash="dash", line_color="red", annotation_text="Upper")
        fig_clt.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_clt, use_container_width=True)
        st.info(f"**{ci_label} (CLT):** {clt_result[0]:.2f} to {clt_result[1]:.2f} {unit}")

    with col_right:
        st.subheader("Bootstrap (Empirical)")
        fig_boot = px.histogram(boot_samples, nbins=25, color_discrete_sequence=[COLOR_BOOTSTRAP])
        # Add Vertical Lines for CI
        fig_boot.add_vline(x=boot_result[0], line_dash="dash", line_color="black", annotation_text="Lower")
        fig_boot.add_vline(x=boot_result[1], line_dash="dash", line_color="black", annotation_text="Upper")
        fig_boot.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_boot, use_container_width=True)
        st.success(f"**{ci_label} (Bootstrap):** {boot_result[0]:.2f} to {boot_result[1]:.2f} {unit}")

    # 5. Comparison Metrics
    st.write("### 📏 Uncertainty (CI Width) Comparison")
    cw1, cw2 = st.columns(2)
    cw1.write(f"**CLT Width:** {clt_width:.4f} {unit}")
    cw2.write(f"**Bootstrap Width:** {boot_width:.4f} {unit}")

    # 6. AI Insights
    st.divider()
    st.write("### 🧠 AI Referee Insights")
    
    summary = {
        "sample_size": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(diag["median"]),
        "clt_ci": clt_result,
        "bootstrap_ci": boot_result,
        "clt_width": float(clt_width),
        "bootstrap_width": float(boot_width),
        "skewness": float(diag["skewness"]),
        "unit": unit,
        "confidence_level": ci_label
    }

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing intervals..."):
            insights = get_ai_insights(summary)
            st.markdown(f"> {insights}")

else:
    st.info("Upload data to begin the inference battle.")
