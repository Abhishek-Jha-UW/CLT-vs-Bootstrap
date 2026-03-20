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

st.set_page_config(page_title="Inference Arena", layout="wide")

st.title("⚔️ Inference Arena: CLT vs Bootstrap")
st.caption("Compare Classical vs Empirical Inference with AI-powered insights")

# -----------------------------
# Template Download
# -----------------------------
def get_template_csv():
    df = pd.DataFrame({"value": [10, 12, 15, 18, 20]})
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Template",
    data=get_template_csv(),
    file_name="template.csv",
    mime="text/csv"
)

# -----------------------------
# Sample Data
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if st.button("🧪 Use Sample Data"):
    st.session_state.df = pd.DataFrame({
        "value": np.random.gamma(shape=2, scale=10, size=100)
    })
    st.success("Sample data loaded!")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

# -----------------------------
# Proceed if Data Exists
# -----------------------------
if st.session_state.df is not None:

    df = st.session_state.df

    st.write("### Data Preview")
    st.dataframe(df.head())

    if "value" not in df.columns:
        st.error("CSV must contain a column named 'value'")
        st.stop()

    data = df["value"].dropna().values

    # -----------------------------
    # Controls
    # -----------------------------
    unit = st.text_input("Enter Unit (optional)", "")
    confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])

    # -----------------------------
    # Diagnostics
    # -----------------------------
    st.write("## 📊 Diagnostics")
    diag = get_diagnostics(data)

    st.json(diag)

    fig = px.histogram(data, nbins=30, title="Data Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Inference
    # -----------------------------
    clt_result = clt_ci(data, confidence)
    boot_result, boot_samples = bootstrap_ci(
        data, np.mean, confidence, n_bootstrap
    )

    # -----------------------------
    # Comparison
    # -----------------------------
    st.write("## ⚔️ CLT vs Bootstrap")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CLT CI")
        st.write(f"{clt_result[0]:.2f} to {clt_result[1]:.2f} {unit}")

    with col2:
        st.subheader("Bootstrap CI")
        st.write(f"{boot_result[0]:.2f} to {boot_result[1]:.2f} {unit}")

    # Bootstrap Distribution
    fig2 = px.histogram(boot_samples, nbins=30, title="Bootstrap Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Verdict
    # -----------------------------
    st.write("## 🤖 Verdict")

    verdict = compare_methods(data)
    st.success(verdict)

    # -----------------------------
    # AI Insights
    # -----------------------------
    st.write("## 🧠 AI Insights")

    summary = {
        "sample_size": int(len(data)),
        "mean": float(np.mean(data)),
        "clt_ci": clt_result,
        "bootstrap_ci": boot_result,
        "skewness": float(diag["skewness"]),
        "unit": unit
    }

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing..."):
            insights = get_ai_insights(summary)
            st.write(insights)
