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
    mean = np.mean(data)
    se = stats.sem(data)
    # Using t-distribution (more robust than Z-score for n < 100)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(data)-1)
    margin = t_crit * se
    return (float(mean - margin), float(mean + margin))

# -----------------------------
# Bootstrap Confidence Interval (Universal)
# -----------------------------
def bootstrap_ci(data, stat_func=np.mean, confidence=0.95, n_bootstrap=5000):
    # Vectorized for significant speed improvement
    resamples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    boot_samples = stat_func(resamples, axis=1)

    lower = np.percentile(boot_samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_samples, (1 + confidence) / 2 * 100)

    return (float(lower), float(upper)), boot_samples

# -----------------------------
# Improved Decision Logic
# -----------------------------
def compare_methods(data, target_stat="Mean"):
    if target_stat != "Mean":
        return "Bootstrap is required (Theoretical CFF is unreliable/complex for non-mean statistics)."
    
    diag = get_diagnostics(data)
    score_clt = 0
    
    # 1. Sample Size Check
    if diag["n"] >= 30: 
        score_clt += 1
    # 2. Symmetry Check
    if abs(diag["skewness"]) < 0.5: 
        score_clt += 1
    # 3. Tail Heaviness Check (Kurtosis < 1 suggests lighter tails)
    if diag["kurtosis"] < 1: 
        score_clt += 1

    if score_clt >= 2:
        return "✅ CLT is reliable: Assumptions of normality are likely met."
    elif score_clt == 1:
        return "⚠️ CLT is acceptable, but Bootstrap is preferred for better empirical coverage."
    else:
        return "🚨 Bootstrap is strongly recommended: Data distribution violates CLT assumptions."

# -----------------------------
# Structured AI Insights
# -----------------------------
def get_ai_insights(summary):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "Please set your OpenAI API Key in Streamlit Secrets."
            
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
You are a senior data scientist.

Analyze the statistical inference results below:

{summary}

Respond in this structured format:

### 1. Key Insight
What does the estimated statistic suggest about the population?

### 2. Method Comparison
Compare CLT vs Bootstrap results (if both exist). Are they similar or different?

### 3. Which Method to Trust?
Recommend one method and justify using:
- sample size
- skewness
- CI width

### 4. Practical Interpretation
Explain in simple real-world terms using unit: {summary.get('unit', 'units')}

### 5. Recommendation
What should the user do next? (e.g., increase sample size, investigate outliers, etc.)

Keep it concise, clear, and practical.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {e}"
