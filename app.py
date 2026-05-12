from __future__ import annotations

import hashlib
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
st.set_page_config(page_title="Inference Arena", layout="wide", page_icon="📊")

COLOR_DATA = "#4C78A8"
COLOR_CLT = "#59A8A0"
COLOR_BOOTSTRAP = "#E07B2F"
COLOR_MUTED = "#5C6370"

st.markdown(
    """
<style>
  .ia-hero { font-size: 1.05rem; line-height: 1.55; color: #333; max-width: 52rem; }
  .ia-step { font-weight: 600; color: #1a1a1a; margin-bottom: 0.25rem; }
  div.stButton > button[kind="primary"] {
    background-color: #59A8A0; color: white; border: none; border-radius: 8px; font-weight: 600;
  }
  div.stButton > button[kind="primary"]:hover { background-color: #E07B2F; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Inference Arena")
st.markdown(
    '<p class="ia-hero">Compare a <strong>classical</strong> confidence interval to a '
    "<strong>percentile bootstrap</strong> interval for the <em>same</em> quantity (mean, proportion, "
    "or other summaries). The goal is clarity: when a textbook normal approximation is reasonable, "
    "and when resampling is the more transparent choice.</p>",
    unsafe_allow_html=True,
)

with st.expander("How this app works (two-minute read)", expanded=False):
    st.markdown(
        """
**What you get**

| Step | What you see |
|------|----------------|
| 1 | **Diagnostics** on your numeric column: size, shape (skew), tail weight (excess kurtosis), and a **histogram**. For continuous data you also get a **normal QQ plot**: it checks whether *observations* look normal (not whether the mean is normal—that is a different idea). |
| 2 | **Intervals**: where a standard classical formula exists (mean → **Student t**; 0/1 proportion → **Wilson**), we show it next to a **bootstrap** interval for the same estimand. |
| 3 | **Comparison strip**: both intervals on one axis so you can see overlap and width at a glance. |
| 4 | **Heuristic verdict**: a small checklist-style hint, not a mathematical proof. |
| 5 | **Optional LLM text**: plain-language commentary using **only** the numbers we pass in—useful for portfolios and stakeholder notes. |

**Terms**

- **Estimand**: the population quantity you care about (mean, median, etc.).
- **Classical interval**: a formula-backed interval (here: *t* or Wilson).
- **Percentile bootstrap**: resample the dataset many times, recompute the statistic each time, and take percentiles of that cloud of values as the interval.

**Secrets (Streamlit Cloud)**  
Set `OPENAI_API_KEY` in app secrets. Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`).
"""
    )

# -----------------------------------------------------------------------------
# Session
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "ai_markdown" not in st.session_state:
    st.session_state.ai_markdown = ""
if "analysis_fingerprint" not in st.session_state:
    st.session_state.analysis_fingerprint = None


def _analysis_fingerprint(
    *,
    stat_label: str,
    target_col: str,
    confidence: float,
    n_bootstrap: int,
    seed: int,
    data: np.ndarray,
) -> str:
    h = hashlib.sha256()
    for part in (stat_label, target_col, str(confidence), str(n_bootstrap), str(seed)):
        h.update(part.encode())
    h.update(np.ascontiguousarray(data.astype(np.float64)).tobytes())
    return h.hexdigest()


def _reset_llm_if_analysis_changed(fp: str) -> None:
    if st.session_state.analysis_fingerprint != fp:
        st.session_state.ai_markdown = ""
        st.session_state.analysis_fingerprint = fp


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Analysis controls")

    stat_label = st.selectbox(
        "Estimand (what you want to infer about)",
        ["Mean", "Median", "Std Dev", "75th Percentile", "Proportion (0/1 column)"],
        help="Classical *t* / Wilson paths exist for Mean and Proportion. Other choices show bootstrap only, by design.",
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
    unit = st.text_input(
        "Axis label (units)",
        "units",
        help="Used on plots and in exports—for proportions you can type 'proportion' or leave as-is.",
    )

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, help="Nominal coverage before any finite-sample issues.")
    n_bootstrap = st.selectbox(
        "Bootstrap resamples",
        [2000, 5000, 10000, 20000],
        help="More resamples usually smooth the bootstrap histogram; cost is runtime.",
    )

    seed = st.number_input("Random seed", min_value=0, value=42, step=1, help="Fixes bootstrap resampling and any simulated reference curves.")

    rng = np.random.default_rng(int(seed))

    st.divider()
    st.caption("Templates for quick experiments")

    def template_csv() -> bytes:
        skewed = pd.DataFrame({"value": [10, 12, 15, 18, 20, 22, 25, 30, 10, 15, 45, 50, 60, 12, 14]})
        return skewed.to_csv(index=False).encode("utf-8")

    st.download_button("Skewed numeric template (CSV)", data=template_csv(), file_name="template_skewed.csv")

    def template_binary_csv() -> bytes:
        rng_local = np.random.default_rng(7)
        p = 0.22
        bits = (rng_local.random(60) < p).astype(int)
        return pd.DataFrame({"converted": bits}).to_csv(index=False).encode("utf-8")

    st.download_button("Binary 0/1 template (CSV)", data=template_binary_csv(), file_name="template_binary.csv")

    ci_label = f"{int(confidence * 100)}% confidence interval"

# -----------------------------------------------------------------------------
# Data ingest
# -----------------------------------------------------------------------------
st.markdown('<p class="ia-step">1 · Load data</p>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 2])
with c1:
    if st.button("Load demo dataset", use_container_width=True, help="Skewed gamma data to stress-test classical vs bootstrap."):
        st.session_state.df = pd.DataFrame({"value": rng.gamma(shape=2.0, scale=10.0, size=120)})
with c2:
    uploaded = st.file_uploader("Or upload a CSV", type=["csv"], help="One numeric column is enough; you will pick which column below.")

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)

if st.session_state.df is None:
    st.info(
        "**Start here:** click **Load demo dataset** or upload a CSV. "
        "Then choose your estimand in the sidebar and scroll through diagnostics → intervals → optional LLM notes."
    )
    st.stop()

df = st.session_state.df
num_cols = list(df.select_dtypes(include=[np.number]).columns)
if not num_cols:
    st.error("This CSV has no numeric columns. Add at least one numeric column and try again.")
    st.stop()

default_col = "value" if "value" in num_cols else num_cols[0]
target_col = st.selectbox(
    "Numeric column to analyze",
    num_cols,
    index=num_cols.index(default_col) if default_col in num_cols else 0,
    help="Non-numeric columns are ignored. Missing values are dropped.",
)
data = df[target_col].dropna().to_numpy(dtype=float)

if data.size == 0:
    st.error("After dropping missing values, this column is empty.")
    st.stop()

is_prop_mode = stat_label == "Proportion (0/1 column)"
if is_prop_mode and not is_binary_01(data):
    st.error(
        "**Proportion mode** needs values in **{0, 1}** only (success = 1, failure = 0). "
        "Recode your column or pick another estimand."
    )
    st.stop()

_fp = _analysis_fingerprint(
    stat_label=stat_label,
    target_col=target_col,
    confidence=float(confidence),
    n_bootstrap=int(n_bootstrap),
    seed=int(seed),
    data=data,
)
_reset_llm_if_analysis_changed(_fp)

diag = get_diagnostics(data)

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
st.markdown('<p class="ia-step">2 · Inspect the sample</p>', unsafe_allow_html=True)
st.caption(
    "These numbers describe **your uploaded column**, not the population. "
    "They help you decide how much to lean on asymptotic shortcuts."
)

alerts: list[tuple[str, str]] = []
if diag["n"] < 30:
    alerts.append(("Small n", f"n = {diag['n']}: classical approximations deserve extra skepticism."))
if abs(diag["skewness"]) > 1.0 and not is_prop_mode:
    alerts.append(("Skew", "Strong skew: the bootstrap interval is often easier to defend than a story that leans on normality."))
if diag["kurtosis"] > 3.5 and not is_prop_mode:
    alerts.append(("Tails", "Heavy tails or outliers: consider robust estimands and data validation, not only interval tweaks."))

for title, msg in alerts:
    st.warning(f"**{title}:** {msg}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Sample size", f"{diag['n']:,}")
m2.metric("Mean", f"{diag['mean']:.3f}")
m3.metric("Median", f"{diag['median']:.3f}")
m4.metric("Skewness", f"{diag['skewness']:.3f}")
m5.metric("Excess kurtosis", f"{diag['kurtosis']:.3f}")

left, right = st.columns(2)
with left:
    fig_hist = px.histogram(
        data,
        nbins=min(40, max(10, int(np.sqrt(diag["n"])) * 2)),
        title="Histogram of the sample",
        color_discrete_sequence=[COLOR_DATA],
    )
    fig_hist.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis_title=unit,
        yaxis_title="Count",
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    if is_prop_mode:
        st.caption("For binary data, a QQ plot against the normal is usually not informative—use the interval view below.")
        counts = pd.Series(data).value_counts().sort_index()
        fig_bar = px.bar(
            x=counts.index.astype(int).astype(str),
            y=counts.values,
            labels={"x": "Value", "y": "Count"},
            title="Class counts",
            color_discrete_sequence=[COLOR_DATA],
        )
        fig_bar.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        osm, osr = stats.probplot(data, dist="norm", fit=False)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Ordered data", marker=dict(color=COLOR_DATA, size=7)))
        lims = float(min(osm.min(), osr.min())), float(max(osm.max(), osr.max()))
        fig_qq.add_trace(
            go.Scatter(x=lims, y=lims, mode="lines", name="Perfect normal", line=dict(color=COLOR_MUTED, dash="dash"))
        )
        fig_qq.update_layout(
            title="Normal QQ plot (raw observations)",
            height=380,
            xaxis_title="Theoretical normal quantiles",
            yaxis_title="Ordered sample values",
            margin=dict(l=10, r=10, t=48, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption(
            "Points bending away from the diagonal mean the **raw data** are not normal. "
            "That does **not** automatically invalidate a **t** interval for the mean—it speaks to whether "
            "normality assumptions for other procedures are plausible."
        )

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
    clt_label = "Student t interval for the mean"
elif is_prop_mode:
    clt_lo, clt_hi = wilson_proportion_ci(data, confidence=float(confidence))
    clt_label = "Wilson interval for a binomial proportion"

st.markdown('<p class="ia-step">3 · Compare intervals</p>', unsafe_allow_html=True)
st.caption(
    f"**{ci_label}** for **{stat_label.lower()}**. "
    "Vertical lines mark interval endpoints; wider intervals reflect more uncertainty for this estimand and method."
)

col_left, col_right = st.columns(2)

with col_left:
    if clt_lo is None:
        st.markdown("##### Classical path")
        st.info(
            "There is no single “standard” classical interval for this estimand without adding strong "
            "parametric assumptions. The bootstrap panel on the right still targets the quantity you selected."
        )
    elif stat_label == "Mean":
        st.markdown("##### Classical path")
        st.caption(
            "**What this plot is:** a *reference* picture of a normal sampling model for the sample mean "
            "(same mean and SE as your data). Your raw histogram may still be skewed—that is allowed for "
            "many mean problems. The **t** interval below is the actual classical interval used."
        )
        clt_sim = rng.normal(loc=diag["mean"], scale=diag["std"] / np.sqrt(diag["n"]), size=6000)
        fig_clt = px.histogram(
            clt_sim,
            nbins=32,
            title="Reference: normal model for the mean’s sampling distribution",
            color_discrete_sequence=[COLOR_CLT],
        )
        fig_clt.add_vline(x=clt_lo, line_dash="dash", line_color="#a32121", annotation_text="Lower", annotation_position="top")
        fig_clt.add_vline(x=clt_hi, line_dash="dash", line_color="#a32121", annotation_text="Upper", annotation_position="top")
        fig_clt.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=52, b=10))
        st.plotly_chart(fig_clt, use_container_width=True)
        st.markdown(f"**{clt_label}:** [{clt_lo:.4f}, {clt_hi:.4f}] {unit}")
        st.caption(f"Interval width: **{clt_hi - clt_lo:.4f}** {unit}")
    else:
        st.markdown("##### Classical path")
        st.caption("Wilson is a standard alternative to the naive normal (Wald) interval for proportions, especially away from 0.5.")
        p_hat = float(np.mean(data))
        fig_p = go.Figure()
        fig_p.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, line=dict(color="#ccc"))
        fig_p.add_trace(
            go.Scatter(
                x=[p_hat],
                y=[0],
                mode="markers+text",
                name="Point estimate",
                text=[f"p̂ = {p_hat:.3f}"],
                textposition="top center",
                marker=dict(size=14, color=COLOR_CLT),
            )
        )
        fig_p.add_shape(
            type="rect",
            x0=clt_lo,
            x1=clt_hi,
            y0=-0.14,
            y1=0.14,
            fillcolor=COLOR_CLT,
            opacity=0.28,
            line_width=0,
        )
        fig_p.update_xaxes(range=[-0.02, 1.02], title="Probability scale [0, 1]")
        fig_p.update_yaxes(visible=False, range=[-0.32, 0.32])
        fig_p.update_layout(height=430, title="Wilson interval (shaded band)", margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown(f"**{clt_label}:** [{clt_lo:.4f}, {clt_hi:.4f}]")
        st.caption(f"Interval width: **{clt_hi - clt_lo:.4f}** on the probability scale")

with col_right:
    st.markdown("##### Bootstrap path")
    st.caption(
        "Each bar counts how often the resampled statistic landed in a bin. "
        "The dashed lines are the **percentile** endpoints (simple, transparent, widely used)."
    )
    fig_b = px.histogram(
        boot_samples,
        nbins=32,
        title="Bootstrap distribution of the statistic",
        color_discrete_sequence=[COLOR_BOOTSTRAP],
    )
    fig_b.add_vline(x=boot_lo, line_dash="dash", line_color="#222")
    fig_b.add_vline(x=boot_hi, line_dash="dash", line_color="#222")
    fig_b.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=52, b=10))
    st.plotly_chart(fig_b, use_container_width=True)
    axis_suffix = unit if not is_prop_mode else "probability"
    st.markdown(f"**Percentile bootstrap:** [{boot_lo:.4f}, {boot_hi:.4f}] {axis_suffix}")
    st.caption(f"Interval width: **{boot_width:.4f}**")

# Forest
st.markdown("##### One-axis comparison")
rows = [{"method": "Bootstrap", "lo": boot_lo, "hi": boot_hi, "mid": (boot_lo + boot_hi) / 2}]
if clt_lo is not None:
    rows.insert(0, {"method": "Classical", "lo": clt_lo, "hi": clt_hi, "mid": (clt_lo + clt_hi) / 2})
forest = pd.DataFrame(rows)
fig_f = go.Figure()
for _, r in forest.iterrows():
    color = COLOR_CLT if r["method"] == "Classical" else COLOR_BOOTSTRAP
    fig_f.add_trace(
        go.Scatter(
            x=[r["lo"], r["hi"]],
            y=[r["method"], r["method"]],
            mode="lines+markers",
            line=dict(color=color, width=5),
            marker=dict(size=10, color=color),
            showlegend=False,
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=[r["mid"]],
            y=[r["method"]],
            mode="markers",
            marker=dict(size=13, color=color, symbol="diamond"),
            showlegend=False,
        )
    )
fig_f.update_layout(
    title=f"{ci_label} — where each method puts its uncertainty",
    xaxis_title=unit if not is_prop_mode else "Probability",
    yaxis_title="",
    height=280,
    margin=dict(l=10, r=10, t=52, b=10),
)
st.plotly_chart(fig_f, use_container_width=True)

width_pct: float | None = None
width_abs_diff: float | None = None
if clt_lo is not None:
    clt_width = clt_hi - clt_lo
    width_abs_diff = float(boot_width - clt_width)
    width_pct = float(((boot_width - clt_width) / clt_width) * 100) if clt_width > 0 else None

verdict = compare_methods(data, stat_label if not is_prop_mode else "Proportion")
st.markdown('<p class="ia-step">4 · Decision support (heuristic)</p>', unsafe_allow_html=True)
st.info(verdict.headline)
st.write(verdict.detail)
if verdict.checks:
    st.caption("Checklist behind the headline (mean and proportion modes only):")
    chk = pd.DataFrame([{"Criterion": k, "Met": "Yes" if v else "No"} for k, v in verdict.checks.items()])
    st.dataframe(chk, hide_index=True, use_container_width=True)

if width_pct is not None and np.isfinite(width_pct):
    st.success(
        f"**Width comparison:** the bootstrap interval is **{abs(width_pct):.1f}%** "
        f"**{'wider' if width_pct > 0 else 'narrower'}** than the classical interval "
        f"(absolute width difference: **{abs(width_abs_diff or 0):.4f}** in the same units as the estimand)."
    )

# -----------------------------------------------------------------------------
# Export + LLM
# -----------------------------------------------------------------------------
export_payload: dict[str, object] = {
    "target_column": target_col,
    "estimand": stat_label,
    "confidence": float(confidence),
    "n_bootstrap": int(n_bootstrap),
    "seed": int(seed),
    "diagnostics": diag,
    "bootstrap_ci": {"lo": boot_lo, "hi": boot_hi, "width": boot_width},
    "classical_ci": None
    if clt_lo is None
    else {"lo": clt_lo, "hi": clt_hi, "label": clt_label, "width": float((clt_hi or 0) - (clt_lo or 0))},
    "width_comparison": None
    if width_pct is None or not np.isfinite(width_pct)
    else {
        "bootstrap_minus_classical_absolute": width_abs_diff,
        "bootstrap_vs_classical_pct": width_pct,
    },
    "verdict": {
        "headline": verdict.headline,
        "detail": verdict.detail,
        "score": verdict.clt_support_score,
        "max_score": verdict.max_score,
        "checks": verdict.checks,
    },
}

csv_buf = io.StringIO()
flat = {f"d_{k}": v for k, v in diag.items()}
flat.update({"boot_lo": boot_lo, "boot_hi": boot_hi, "boot_width": boot_width})
pd.DataFrame([flat]).to_csv(csv_buf, index=False)
d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Download summary (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="inference_summary.csv",
        mime="text/csv",
    )
with d2:
    st.download_button(
        "Download summary (JSON)",
        data=json.dumps(export_payload, indent=2).encode("utf-8"),
        file_name="inference_summary.json",
        mime="application/json",
    )

st.divider()
st.markdown('<p class="ia-step">5 · Optional narrative (LLM)</p>', unsafe_allow_html=True)
st.caption(
    "Uses your Streamlit secret `OPENAI_API_KEY`. The model only sees the structured summary below the hood—"
    "not your raw row-level data."
)

api_key = None
model = "gpt-4o-mini"
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except FileNotFoundError:
    api_key = None

if not api_key:
    st.warning("Add `OPENAI_API_KEY` under Streamlit **Settings → Secrets** to enable the button below.")

summary_for_llm: dict[str, object] = {
    "n": int(diag["n"]),
    "mean": float(diag["mean"]),
    "median": float(diag["median"]),
    "skewness": float(diag["skewness"]),
    "excess_kurtosis": float(diag["kurtosis"]),
    "estimand": stat_label,
    "unit_label": unit,
    "confidence_interval_nominal": ci_label,
    "bootstrap_ci": [float(boot_lo), float(boot_hi)],
    "bootstrap_width": float(boot_width),
    "heuristic_verdict_headline": verdict.headline,
}
if clt_lo is not None and clt_hi is not None:
    cw = float(clt_hi - clt_lo)
    summary_for_llm["classical_ci"] = [float(clt_lo), float(clt_hi)]
    summary_for_llm["classical_width"] = cw
    if width_abs_diff is not None and width_pct is not None and np.isfinite(width_pct):
        summary_for_llm["width_difference_absolute"] = float(width_abs_diff)
        summary_for_llm["width_difference_pct_vs_classical"] = float(width_pct)
        summary_for_llm["width_comparison_sentence"] = (
            f"Bootstrap width is {abs(width_pct):.1f}% {'wider' if width_pct > 0 else 'narrower'} than classical "
            f"(absolute difference bootstrap minus classical: {width_abs_diff:+.6f} in estimand units)."
        )

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Generate narrative", type="primary"):
        with st.spinner("Requesting a short interpretation…"):
            st.session_state.ai_markdown = get_ai_insights(summary_for_llm, api_key=api_key, model=str(model))
with col_b:
    if st.session_state.ai_markdown:
        if st.button("Clear narrative"):
            st.session_state.ai_markdown = ""

if st.session_state.ai_markdown:
    st.markdown(st.session_state.ai_markdown)

st.caption(
    "This tool supports learning and communication. It is not a substitute for study design, "
    "preregistration, or domain-specific review."
)
