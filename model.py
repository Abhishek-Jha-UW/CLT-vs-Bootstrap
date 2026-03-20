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
# CLT Confidence Interval
# -----------------------------
def clt_ci(data, confidence=0.95):
    mean = np.mean(data)
    se = stats.sem(data)

    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * se

    return (float(mean - margin), float(mean + margin))


# -----------------------------
# Bootstrap Confidence Interval
# -----------------------------
def bootstrap_ci(data, stat_func=np.mean, confidence=0.95, n_bootstrap=5000):
    boot_samples = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_samples.append(stat_func(sample))

    lower = np.percentile(boot_samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_samples, (1 + confidence) / 2 * 100)

    return (float(lower), float(upper)), boot_samples


# -----------------------------
# Decision Logic
# -----------------------------
def compare_methods(data):
    diag = get_diagnostics(data)

    score_clt = 0

    if diag["n"] >= 30:
        score_clt += 1

    if abs(diag["skewness"]) < 0.5:
        score_clt += 1

    if diag["std"] < diag["mean"]:
        score_clt += 1

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
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        prompt = f"""
You are a statistical expert helping a data analyst.

Here are the results:
{summary}

Explain in simple terms:
1. What the results mean
2. Which method is more reliable (CLT vs Bootstrap)
3. Why
4. What the user should do next

Keep it concise and practical.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating insights: {e}"
