import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# Diagnostics
# -----------------------------
def get_diagnostics(data):
    return {
        "n": len(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data, ddof=1),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data)
    }


# -----------------------------
# CLT Confidence Interval
# -----------------------------
def clt_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)

    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * se

    return (mean - margin, mean + margin)


# -----------------------------
# Bootstrap
# -----------------------------
def bootstrap_ci(data, stat_func=np.mean, confidence=0.95, n_bootstrap=5000):
    boot_samples = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_samples.append(stat_func(sample))

    lower = np.percentile(boot_samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_samples, (1 + confidence) / 2 * 100)

    return (lower, upper), boot_samples


# -----------------------------
# Decision Logic
# -----------------------------
def compare_methods(data):
    diagnostics = get_diagnostics(data)

    score_clt = 0
    score_boot = 2  # bootstrap baseline

    if diagnostics["n"] >= 30:
        score_clt += 1

    if abs(diagnostics["skewness"]) < 0.5:
        score_clt += 1

    if diagnostics["std"] < np.mean(data):
        score_clt += 1

    if score_clt >= 2:
        verdict = "CLT is reliable"
    elif score_clt == 1:
        verdict = "CLT is acceptable, but Bootstrap preferred"
    else:
        verdict = "Bootstrap is strongly recommended"

    return verdict
