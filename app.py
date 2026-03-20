import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from model import (
    get_diagnostics,
    clt_ci,
    bootstrap_ci,
    compare_methods
)

st.set_page_config(page_title="Inference Arena", layout="wide")

st.title("⚔️ Inference Arena: CLT vs Bootstrap")

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    column = st.selectbox("Select Column", df.columns)

    data = df[column].dropna().values

    # -----------------------------
    # Controls
    # -----------------------------
    confidence = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap Samples", [1000, 5000, 10000])

    # -----------------------------
    # Diagnostics
    # -----------------------------
    st.write("## 📊 Diagnostics")
    diag = get_diagnostics(data)
    st.write(diag)

    fig = px.histogram(data, nbins=30, title="Data Distribution")
    st.plotly_chart(fig)

    # -----------------------------
    # CLT
    # -----------------------------
    clt_result = clt_ci(data, confidence)

    # -----------------------------
    # Bootstrap
    # -----------------------------
    boot_result, boot_samples = bootstrap_ci(
        data, np.mean, confidence, n_bootstrap
    )

    # -----------------------------
    # Comparison
    # -----------------------------
    st.write("## ⚔️ Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CLT CI")
        st.write(clt_result)

    with col2:
        st.subheader("Bootstrap CI")
        st.write(boot_result)

    # Bootstrap distribution
    fig2 = px.histogram(boot_samples, nbins=30, title="Bootstrap Distribution")
    st.plotly_chart(fig2)

    # -----------------------------
    # Verdict
    # -----------------------------
    st.write("## 🤖 Verdict")

    verdict = compare_methods(data)
    st.success(verdict)
