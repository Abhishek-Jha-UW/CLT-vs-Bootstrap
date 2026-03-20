import numpy as np
from scipy import stats
from openai import OpenAI
import streamlit as st

# -----------------------------
# Diagnostics
# -----------------------------
def get_diagnostics(data):
    return {
        "n": len(data),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data, ddof=1)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data))
    }

# -----------------------------
# CLT Confidence Interval (Mean Only)
# -----------------------------
def clt_ci(data, confidence=0.95):
    # This formula ONLY applies to the Mean (Central Limit Theorem)
    mean = np.mean(data)
    se = stats.sem(data)
    # Using t-distribution for better accuracy with smaller n
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(data)-1)
    margin = t_crit * se
    return (float(mean - margin), float(mean + margin))

# -----------------------------
# Bootstrap Confidence Interval (Universal)
# -----------------------------
def bootstrap_ci(data, stat_func=np.mean, confidence=0.95, n_bootstrap=5000):
    boot_samples = []
    # Vectorized for speed
    resamples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    boot_samples = stat_func(resamples, axis=1)

    lower = np.percentile(boot_samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_samples, (1 + confidence) / 2 * 100)

    return (float(lower), float(upper)), boot_samples

# -----------------------------
# Decision Logic
# -----------------------------
def compare_methods(data, target_stat="Mean"):
    if target_stat != "Mean":
        return "Bootstrap is required (Theoretical CFF is unreliable for this statistic)."
    
    diag = get_diagnostics(data)
    score_clt = 0
    if diag["n"] >= 30: score_clt += 1
    if abs(diag["skewness"]) < 0.5: score_clt += 1
    if diag["std"] < abs(diag["mean"]): score_clt += 1

    if score_clt >= 2:
        return "CLT is reliable"
    elif score_clt == 1:
        return "CLT is acceptable, but Bootstrap preferred"
    else:
        return "Bootstrap is strongly recommended"

# -----------------------------
# AI Insights
# -----------------------------
def get_ai_insights(summary):
    try:
        # Check if key exists to avoid crash
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "Please set your OpenAI API Key in Streamlit Secrets."
            
        client = OpenAI(api_key=api_key)
        prompt = f"""You are a senior data scientist. Analyze these results: {summary}
        Respond in structure: 1. Key Insight, 2. Method Comparison, 3. Recommendation, 4. Practical interpretation ({summary.get('unit', 'units')}).
        Be precise about why the selected statistic matters."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
