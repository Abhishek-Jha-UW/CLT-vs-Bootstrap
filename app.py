from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from inference import (
    bootstrap_ci,
    clt_mean_t_ci,
    compare_methods,
    get_ai_insights,
    get_diagnostics,
    is_binary_01,
    wilson_proportion_ci,
)

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inference Arena", layout="wide", page_icon="⚔️")

COLOR_DATA = "#4C78A8"
COLOR_CLT = "#72B7B2"
COLOR_BOOTSTRAP = "#F58518"

st.markdown(
    """
<style>
div.stButton > button[kind="primary"] {
  background-color: #72B7B2;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 700;
}
div.stButton > button[kind="primary"]:hover {
  background-color: #F58518;
  color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Inference Arena: CLT vs bootstrap")
st.caption(
    "Portfolio-grade comparison: classical pivots where they are standard (mean, binomial proportion) "
    "versus percentile bootstrap for the same estimands."
)

with st.expander("Methodology (read me)", expanded=False):
    st.markdown(
        """
- **Mean (classical)**: Student *t* interval for \\(\\mu\\) using the standard error of the mean.
- **Proportion (classical)**: **Wilson score** interval for a Bernoulli mean (more stable than a naive Wald interval).
- **Bootstrap**: **Percentile bootstrap** on the chosen statistic with a user-controlled RNG seed.
- **Other estimands (median, SD, percentiles)**: bootstrap is shown; classical paths are omitted on purpose.

**LLM section**: optional narration via OpenAI. Configure `OPENAI_API_KEY` in Streamlit secrets; optionally set
`OPENAI_MODEL` (defaults to `gpt-4o-mini` for cost control).
"""
    )

# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "ai_markdown" not in st.session_state:
    st.session_state.ai_markdown = ""

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    stat_label = st.selectbox(
        "Estimand",
        ["Mean", "Median", "Std Dev", "75th Percentile", "Proportion (0/1 column)"],
    )

    stat_map: dict[str, object] = {
        "Mean": np.mean,
        "Median": np.median,
        "Std Dev": lambda x, axis=None: np.std(x, axis=axis, ddof=1),
        "75th Percentile": lambda x, axis=None: np.percentile(x, 75, axis=axis),
        "Proportion (0/1 column)": np.mean,
    }
    selected_func = stat_map[stat_label]

    st.divider()
    unit = st.text_input("Unit / label for the axis", "units")

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95)
    n_bootstrap = st.selectbox("Bootstrap replicates", [2000, 5000, 10000, 20000])

    seed = st.number_input("Random seed (reproducibility)", min_value=0, value=42, step=1)
    rng = np.random.default_rng(int(seed))

    st.divider()

    def template_csv() -> bytes:
        skewed = pd.DataFrame({"value": [10, 12, 15, 18, 20, 22, 25, 30, 10, 15, 45, 50, 60, 12, 14]})
        return skewed.to_csv(index=False).encode("utf-8")

    st.download_button("Download skewed template CSV", data=template_csv(), file_name="template_skewed.csv")

    def template_binary_csv() -> bytes:
        rng_local = np.random.default_rng(7)
        p = 0.22
        bits = (rng_local.random(60) < p).astype(int)
        return pd.DataFrame({"converted": bits}).to_csv(index=False).encode("utf-8")

    st.download_button("Download 0/1 template CSV", data=template_binary_csv(), file_name="template_binary.csv")

    ci_label = f"{int(confidence * 100)}% CI"

# -----------------------------------------------------------------------------
# Data ingest
# -----------------------------------------------------------------------------
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Load skewed sample", use_container_width=True):
        st.session_state.df = pd.DataFrame({"value": rng.gamma(shape=2.0, scale=10.0, size=120)})
with c2:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        st.session_state.df = pd.read_csv(uploaded)

if st.session_state.df is None:
    st.info("Upload a CSV or load the skewed sample to begin.")
    st.stop()

df = st.session_state.df
num_cols = list(df.select_dtypes(include=[np.number]).columns)
if not num_cols:
    st.error("No numeric columns found. Provide at least one numeric column.")
    st.stop()

default_col = "value" if "value" in num_cols else num_cols[0]
target_col = st.selectbox("Numeric column", num_cols, index=num_cols.index(default_col) if default_col in num_cols else 0)
data = df[target_col].dropna().to_numpy(dtype=float)

if data.size == 0:
    st.error("Selected column is empty after dropping NaNs.")
    st.stop()

is_prop_mode = stat_label == "Proportion (0/1 column)"
if is_prop_mode and not is_binary_01(data):
    st.error(
        "Proportion mode requires a **0/1** numeric column. "
        "Encode successes as 1 and failures as 0 (integers or floats)."
    )
    st.stop()

diag = get_diagnostics(data)

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
st.subheader("Diagnostics")
alerts: list[str] = []
if diag["n"] < 30:
    alerts.append(f"Small sample (n={diag['n']}): treat asymptotic arguments cautiously.")
if abs(diag["skewness"]) > 1.0 and not is_prop_mode:
    alerts.append("High skew: bootstrap is often the more honest frequentist summary here.")
if diag["kurtosis"] > 3.5 and not is_prop_mode:
    alerts.append("Heavy tails / outliers: check robust estimands and data quality.")

for msg in alerts:
    st.warning(msg)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("n", diag["n"])
m2.metric("Mean", f"{diag['mean']:.3f}")
m3.metric("Median", f"{diag['median']:.3f}")
m4.metric("Skewness", f"{diag['skewness']:.3f}")
m5.metric("Excess kurtosis", f"{diag['kurtosis']:.3f}")

left, right = st.columns(2)
with left:
    fig_hist = px.histogram(
        data,
        nbins=min(40, max(10, int(np.sqrt(diag["n"])) * 2)),
        title="Empirical distribution",
        color_discrete_sequence=[COLOR_DATA],
    )
    fig_hist.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=unit, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    if is_prop_mode:
        st.caption("Normal QQ is less informative for pure binary data; see the interval comparison below.")
        counts = pd.Series(data).value_counts().sort_index()
        fig_bar = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": "value", "y": "count"})
        fig_bar.update_layout(height=360, title="Class frequencies", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        osm, osr = stats.probplot(data, dist="norm", fit=False)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data", marker=dict(color=COLOR_DATA)))
        lims = float(min(osm.min(), osr.min())), float(max(osm.max(), osr.max()))
        fig_qq.add_trace(
            go.Scatter(x=lims, y=lims, mode="lines", name="y = x", line=dict(color="#999", dash="dash"))
        )
        fig_qq.update_layout(
            title="Normal QQ plot (raw observations)",
            height=360,
            xaxis_title="Theoretical quantiles",
            yaxis_title="Ordered values",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
boot_res, boot_samples = bootstrap_ci(
    data,
    stat_func=selected_func,  # type: ignore[arg-type]
    confidence=float(confidence),
    n_bootstrap=int(n_bootstrap),
    rng=rng,
)
boot_lo, boot_hi = boot_res
boot_width = boot_hi - boot_lo

clt_lo = clt_hi = None
clt_label = ""
if stat_label == "Mean":
    clt_lo, clt_hi = clt_mean_t_ci(data, confidence=float(confidence))
    clt_label = "Classical (Student t for the mean)"
elif is_prop_mode:
    clt_lo, clt_hi = wilson_proportion_ci(data, confidence=float(confidence))
    clt_label = "Classical (Wilson interval for a proportion)"

st.subheader(f"{ci_label}: classical vs bootstrap")

col_left, col_right = st.columns(2)

with col_left:
    if clt_lo is None:
        st.markdown("**Classical shortcut**")
        st.info(
            "No default classical interval is shown for this estimand. "
            "That is intentional: credible classical formulas typically need extra assumptions "
            "or a different asymptotic argument than the mean case."
        )
    elif stat_label == "Mean":
        st.markdown(f"**{clt_label}**")
        clt_sim = rng.normal(loc=diag["mean"], scale=diag["std"] / np.sqrt(diag["n"]), size=6000)
        fig_clt = px.histogram(clt_sim, nbins=30, title="Reference: normal sampling model for x̄", color_discrete_sequence=[COLOR_CLT])
        fig_clt.add_vline(x=clt_lo, line_dash="dash", line_color="#c0392b")
        fig_clt.add_vline(x=clt_hi, line_dash="dash", line_color="#c0392b")
        fig_clt.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_clt, use_container_width=True)
        st.write(f"**Interval:** [{clt_lo:.4f}, {clt_hi:.4f}] {unit}")
        st.caption(f"Width: **{clt_hi - clt_lo:.4f}**")
    else:
        st.markdown(f"**{clt_label}**")
        p_hat = float(np.mean(data))
        fig_p = go.Figure()
        fig_p.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, line=dict(color="#bbb"))
        fig_p.add_trace(
            go.Scatter(
                x=[p_hat],
                y=[0],
                mode="markers+text",
                name="p̂",
                text=[f"p̂={p_hat:.3f}"],
                textposition="top center",
                marker=dict(size=14, color=COLOR_CLT),
            )
        )
        fig_p.add_shape(
            type="rect",
            x0=clt_lo,
            x1=clt_hi,
            y0=-0.15,
            y1=0.15,
            fillcolor=COLOR_CLT,
            opacity=0.25,
            line_width=0,
        )
        fig_p.update_xaxes(range=[-0.05, 1.05], title="Probability scale")
        fig_p.update_yaxes(visible=False, range=[-0.35, 0.35])
        fig_p.update_layout(height=260, title="Wilson interval on [0, 1]", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_p, use_container_width=True)
        st.write(f"**Interval:** [{clt_lo:.4f}, {clt_hi:.4f}] (proportion)")
        st.caption(f"Width: **{clt_hi - clt_lo:.4f}**")

with col_right:
    st.markdown("**Bootstrap (percentile)**")
    fig_b = px.histogram(
        boot_samples,
        nbins=30,
        title="Bootstrap distribution of the statistic",
        color_discrete_sequence=[COLOR_BOOTSTRAP],
    )
    fig_b.add_vline(x=boot_lo, line_dash="dash", line_color="#111")
    fig_b.add_vline(x=boot_hi, line_dash="dash", line_color="#111")
    fig_b.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_b, use_container_width=True)
    st.write(f"**Interval:** [{boot_lo:.4f}, {boot_hi:.4f}] {unit if not is_prop_mode else 'proportion'}")
    st.caption(f"Width: **{boot_width:.4f}**")

# Forest plot
st.subheader("Interval comparison")
rows = [{"method": "Bootstrap", "lo": boot_lo, "hi": boot_hi, "mid": (boot_lo + boot_hi) / 2}]
if clt_lo is not None:
    rows.insert(0, {"method": "Classical", "lo": clt_lo, "hi": clt_hi, "mid": (clt_lo + clt_hi) / 2})
forest = pd.DataFrame(rows)
fig_f = go.Figure()
for i, r in forest.iterrows():
    color = COLOR_CLT if r["method"] == "Classical" else COLOR_BOOTSTRAP
    fig_f.add_trace(
        go.Scatter(
            x=[r["lo"], r["hi"]],
            y=[r["method"], r["method"]],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=10, color=color),
            name=r["method"],
            showlegend=False,
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=[r["mid"]],
            y=[r["method"]],
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond"),
            showlegend=False,
        )
    )
fig_f.update_layout(
    title=f"{ci_label} endpoints",
    xaxis_title=unit if not is_prop_mode else "proportion",
    yaxis_title="",
    height=260,
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_f, use_container_width=True)

verdict = compare_methods(data, stat_label if not is_prop_mode else "Proportion")
st.subheader("Decision support (heuristic, not a proof)")
st.markdown(f"**{verdict.headline}**")
st.write(verdict.detail)
if verdict.checks:
    st.caption("Checklist used for scoring (mean/proportion paths):")
    st.json(verdict.checks)

if clt_lo is not None:
    clt_width = clt_hi - clt_lo
    width_pct = ((boot_width - clt_width) / clt_width) * 100 if clt_width > 0 else float("nan")
    if np.isfinite(width_pct):
        st.write(
            f"Bootstrap width is **{abs(width_pct):.1f}%** {'wider' if width_pct > 0 else 'narrower'} than classical."
        )

# -----------------------------------------------------------------------------
# Export + LLM
# -----------------------------------------------------------------------------
export_payload = {
    "target_column": target_col,
    "estimand": stat_label,
    "confidence": float(confidence),
    "n_bootstrap": int(n_bootstrap),
    "seed": int(seed),
    "diagnostics": diag,
    "bootstrap_ci": {"lo": boot_lo, "hi": boot_hi},
    "classical_ci": None if clt_lo is None else {"lo": clt_lo, "hi": clt_hi, "label": clt_label},
    "verdict": {
        "headline": verdict.headline,
        "detail": verdict.detail,
        "score": verdict.clt_support_score,
        "max_score": verdict.max_score,
        "checks": verdict.checks,
    },
}

csv_buf = io.StringIO()
pd.DataFrame([{**{f"d_{k}": v for k, v in diag.items()}, "boot_lo": boot_lo, "boot_hi": boot_hi}]).to_csv(csv_buf, index=False)
st.download_button(
    "Download summary CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="inference_summary.csv",
    mime="text/csv",
)
st.download_button(
    "Download summary JSON",
    data=json.dumps(export_payload, indent=2).encode("utf-8"),
    file_name="inference_summary.json",
    mime="application/json",
)

st.divider()
st.subheader("Optional: LLM interpretation")

api_key = None
model = "gpt-4o-mini"
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except FileNotFoundError:
    # Local dev without secrets file
    api_key = None

if not api_key:
    st.caption("No `OPENAI_API_KEY` in Streamlit secrets: LLM button will return setup instructions.")

summary_for_llm = {
    "n": int(diag["n"]),
    "mean": float(diag["mean"]),
    "median": float(diag["median"]),
    "skewness": float(diag["skewness"]),
    "kurtosis": float(diag["kurtosis"]),
    "estimand": stat_label,
    "unit": unit,
    "confidence": ci_label,
    "bootstrap_ci": (boot_lo, boot_hi),
    "bootstrap_width": float(boot_width),
}
if clt_lo is not None:
    summary_for_llm["classical_ci"] = (clt_lo, clt_hi)
    summary_for_llm["classical_width"] = float((clt_hi or 0) - (clt_lo or 0))
summary_for_llm["verdict"] = verdict.headline

if st.button("Generate LLM interpretation", type="primary", use_container_width=False):
    with st.spinner("Calling the model…"):
        st.session_state.ai_markdown = get_ai_insights(summary_for_llm, api_key=api_key, model=str(model))

if st.session_state.ai_markdown:
    st.markdown(st.session_state.ai_markdown)
